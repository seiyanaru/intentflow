"""E18: third-dataset check on BNCI2014_001 / BCI Competition IV 2a.

This is the external validity check for the source-side tangent subspace
selection story built on Stieger2021 and Lee2019_MI.

Protocol:

* dataset: BNCI2014_001, left-vs-right MI extracted by MOABB;
* source session: 0train;
* target session: 1test;
* target labels for the held subject are never used for selection or fitting;
* target prefix trials are used only as an unlabeled EA/tangent reference;
* source-side statistics are learned from all non-held subjects;
* the held subject trains LDA on source session and is evaluated on target.

The purpose is not to optimize BNCI2014_001.  It asks whether the two-session
LR result seen on Lee2019_MI is reproducible on a small, independent dataset:

* if source_only compact fractions win again, the task-family rule gains
  support;
* if longitudinal compact fractions win, the current Lee/Stieger interpretation
  needs to be revised.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MNE_DATA", "/home/islabshi/workspace-local2/mne_data")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore")

import mne
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    add_stats,
    bandpass_8_30,
    covariances,
    fit_predict_accuracy,
    longitudinal_score,
    lower_tail_loss,
    parse_subjects,
    session_metric_stats,
    source_only_score,
    tangent_features,
    top_fraction_indices,
)

mne.set_log_level("ERROR")


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260628_bnci2014_001_fraction_sweep_e18"
DEFAULT_FRACTIONS = (1.0, 0.50, 0.35, 0.25, 0.20, 0.15, 0.10, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-9")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Subject feature cache. Defaults to output-dir/subject_cache.",
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def fraction_token(fraction: float) -> str:
    return f"q{fraction:.2f}".replace(".", "p")


def method_name(family: str, fraction: float) -> str:
    return f"{family}_{fraction_token(float(fraction))}"


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(int(repeats), len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def load_subject_features(
    subject: int,
    cache_dir: Path,
    prefix: int,
    eval_start: int,
    force_cache: bool,
) -> dict[str, np.ndarray]:
    path = cache_dir / f"S{subject}.npz"
    if path.exists() and not force_cache:
        payload = np.load(path)
        return {key: payload[key] for key in payload.files}

    paradigm = LeftRightImagery(resample=250, fmin=1, fmax=45)
    dataset = BNCI2014_001()
    x, y_raw, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    labels = (np.asarray(y_raw).astype(str) == "right_hand").astype(np.int64)
    sessions = np.asarray(meta["session"].values).astype(str)

    source_session = "0train"
    target_session = "1test"
    available = set(sessions)
    if not {source_session, target_session}.issubset(available):
        raise ValueError(
            f"S{subject}: expected sessions {source_session!r}/{target_session!r}, "
            f"got {sorted(available)}"
        )
    if not (0 < int(prefix) <= int(eval_start)):
        raise ValueError(f"Require 0 < prefix <= eval_start, got {prefix}, {eval_start}")

    x = bandpass_8_30(x.astype(np.float64, copy=False))
    cov = covariances(x)

    source_indices = np.flatnonzero(sessions == source_session)
    target_indices = np.flatnonzero(sessions == target_session)
    if len(target_indices) <= int(eval_start):
        raise ValueError(f"S{subject}: eval_start={eval_start} leaves no target trials")

    prefix_indices = target_indices[: int(prefix)]
    eval_indices = target_indices[int(eval_start) :]

    output = {
        "source_features": tangent_features(cov, source_indices, source_indices).astype(np.float32),
        "source_labels": labels[source_indices].astype(np.int64),
        "target_features": tangent_features(cov, prefix_indices, eval_indices).astype(np.float32),
        "target_labels": labels[eval_indices].astype(np.int64),
        "all_target_features": tangent_features(cov, prefix_indices, target_indices).astype(np.float32),
        "all_target_labels": labels[target_indices].astype(np.int64),
        "n_source": np.asarray([len(source_indices)], dtype=np.int64),
        "n_prefix": np.asarray([len(prefix_indices)], dtype=np.int64),
        "n_eval": np.asarray([len(eval_indices)], dtype=np.int64),
        "source_session": np.asarray([source_session]),
        "target_session": np.asarray([target_session]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    return output


def aggregate_stats(
    payloads: Mapping[int, Mapping[str, np.ndarray]],
    held_subject: int,
) -> dict[str, np.ndarray]:
    total: dict[str, np.ndarray] = {}
    for subject, payload in payloads.items():
        if int(subject) == int(held_subject):
            continue
        stats = session_metric_stats(
            payload["source_features"],
            payload["source_labels"],
            payload["target_features"],
            payload["target_labels"],
        )
        add_stats(total, stats)
    count = float(total["count"][0])
    averaged = {key: value / count for key, value in total.items() if key != "count"}
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged


def summarize(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(seed)
    methods = sorted({str(row["method"]) for row in records})
    full = {
        int(row["subject"]): float(row["accuracy"])
        for row in records
        if str(row["method"]) == "full_q1p00"
    }
    output: dict[str, dict[str, object]] = {}
    for method in methods:
        by_subject = {
            int(row["subject"]): float(row["accuracy"])
            for row in records
            if str(row["method"]) == method
        }
        acc = np.asarray(
            [by_subject[subject] for subject in subjects if subject in by_subject],
            dtype=np.float64,
        )
        gains = np.asarray(
            [
                by_subject[subject] - full[subject]
                for subject in subjects
                if subject in by_subject and subject in full
            ],
            dtype=np.float64,
        )
        output[method] = {
            "n_subjects": int(len(acc)),
            "accuracy_mean": float(acc.mean()) if len(acc) else float("nan"),
            "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, int(bootstrap)),
            "gain_vs_full_mean_pp": float(gains.mean()) if len(gains) else float("nan"),
            "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, rng, int(bootstrap)),
            "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)) if len(gains) else float("nan"),
            "loss_r10_vs_full_pp": lower_tail_loss(gains),
            "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)) if len(gains) else float("nan"),
        }
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir is not None else args.output_dir / "subject_cache"
    subjects = parse_subjects(args.subjects)
    fractions = sorted(set(float(fraction) for fraction in args.fractions), reverse=True)
    if 1.0 not in fractions:
        fractions = [1.0, *fractions]

    payloads: dict[int, dict[str, np.ndarray]] = {}
    failures: list[dict[str, object]] = []
    for subject in subjects:
        try:
            payloads[subject] = load_subject_features(
                subject,
                cache_dir=cache_dir,
                prefix=int(args.prefix),
                eval_start=int(args.eval_start),
                force_cache=bool(args.force_cache),
            )
            if not args.quiet:
                payload = payloads[subject]
                print(
                    f"S{subject}: cached dim={payload['source_features'].shape[1]} "
                    f"n_source={int(payload['n_source'][0])} n_eval={int(payload['n_eval'][0])}",
                    flush=True,
                )
        except Exception as error:
            failures.append({"subject": int(subject), "stage": "load", "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)

    available_subjects = sorted(payloads)
    records: list[dict[str, object]] = []
    for held_subject in available_subjects:
        payload = payloads[held_subject]
        stats = aggregate_stats(payloads, held_subject)
        scores = {
            "longitudinal": longitudinal_score(stats),
            "source_only": source_only_score(stats),
        }
        dim = payload["source_features"].shape[1]
        index_by_method: dict[str, np.ndarray] = {
            "full_q1p00": np.arange(dim, dtype=np.int64),
        }
        for family, score in scores.items():
            for fraction in fractions:
                index_by_method[method_name(family, fraction)] = top_fraction_indices(score, fraction)

        for method, indices in index_by_method.items():
            accuracy = fit_predict_accuracy(
                payload["source_features"],
                payload["source_labels"],
                payload["target_features"],
                payload["target_labels"],
                indices,
            )
            records.append(
                {
                    "subject": int(held_subject),
                    "method": method,
                    "family": str(method).rsplit("_q", maxsplit=1)[0],
                    "fraction": (
                        1.0
                        if method == "full_q1p00"
                        else float(str(method).rsplit("_q", maxsplit=1)[1].replace("p", "."))
                    ),
                    "n_selected": int(len(indices)),
                    "accuracy": float(accuracy),
                    "n_source": int(payload["n_source"][0]),
                    "n_eval": int(payload["n_eval"][0]),
                    "prefix": int(args.prefix),
                    "eval_start": int(args.eval_start),
                    "n_metric_subjects": int(len(available_subjects) - 1),
                }
            )
        if not args.quiet:
            full = next(
                row["accuracy"]
                for row in records
                if int(row["subject"]) == int(held_subject) and row["method"] == "full_q1p00"
            )
            best_source = max(
                row["accuracy"]
                for row in records
                if int(row["subject"]) == int(held_subject)
                and str(row["method"]).startswith("source_only_")
            )
            best_long = max(
                row["accuracy"]
                for row in records
                if int(row["subject"]) == int(held_subject)
                and str(row["method"]).startswith("longitudinal_")
            )
            print(
                f"S{held_subject}: full={float(full):.1f} "
                f"best_source={float(best_source):.1f} best_long={float(best_long):.1f}",
                flush=True,
            )

    summary = summarize(records, available_subjects, int(args.bootstrap), int(args.seed))
    write_csv(args.output_dir / "selection_records.csv", records)
    write_csv(args.output_dir / "failures.csv", failures)
    write_csv(
        args.output_dir / "summary.csv",
        [{"method": method, **metrics} for method, metrics in sorted(summary.items())],
    )
    report = {
        "config": {
            "dataset": "BNCI2014_001",
            "paradigm": "LeftRightImagery",
            "subjects": available_subjects,
            "requested_subjects": subjects,
            "source_session": "0train",
            "target_session": "1test",
            "prefix": int(args.prefix),
            "eval_start": int(args.eval_start),
            "fractions": fractions,
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "summary": summary,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_records": len(records),
                "n_failures": len(failures),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
