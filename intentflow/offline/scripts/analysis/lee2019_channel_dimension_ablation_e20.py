"""E20: Lee2019 channel/dimension ablation.

E19 showed that Lee2019 source-side subspace selection stays useful even at
full source, while longitudinal compact selection gets stronger in source-scarce
settings.  E20 asks whether the Lee effect is mainly caused by high tangent
dimensionality.

We reduce Lee2019 from all 62 channels (p=1953) to the sensorimotor channels
available in Lee2019 (20 channels, p=210) and re-run the same source-side
selection protocol.

If the p/n story is correct, compact selection gains should shrink when the
feature space is already low-dimensional and physiologically focused.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MNE_DATA", "/home/islabshi/workspace-local2/mne_data")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore")

import mne
import numpy as np
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    add_stats,
    bandpass_8_30,
    bootstrap_ci,
    covariances,
    fit_predict_accuracy,
    longitudinal_score,
    parse_subjects,
    session_metric_stats,
    source_only_score,
    tangent_features,
    top_fraction_indices,
)

mne.set_log_level("ERROR")


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260629_lee2019_channel_dimension_ablation_e20"

LEE2019_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "TP9",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "TP10",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
    "FC3",
    "FC4",
    "C5",
    "C1",
    "C2",
    "C6",
    "CP3",
    "CPz",
    "CP4",
    "P1",
    "P2",
    "POz",
    "FT9",
    "FTT9h",
    "TTP7h",
    "TP7",
    "TPP9h",
    "FT10",
    "FTT10h",
    "TPP8h",
    "TP8",
    "TPP10h",
    "F9",
    "F10",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "PO3",
    "PO4",
]

SENSORIMOTOR_CANONICAL = [
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
]

FEATURE_CHANNELS = {
    "sensorimotor20": [ch for ch in SENSORIMOTOR_CANONICAL if ch in LEE2019_CHANNELS],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--feature", choices=sorted(FEATURE_CHANNELS), default="sensorimotor20")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.80, 0.50, 0.25, 0.10])
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=3)
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


def load_subject_features(
    subject: int,
    feature: str,
    cache_dir: Path,
    prefix: int,
    eval_start: int,
    force_cache: bool,
) -> dict[str, np.ndarray]:
    path = cache_dir / f"S{subject}.npz"
    if path.exists() and not force_cache:
        payload = np.load(path, allow_pickle=True)
        return {key: payload[key] for key in payload.files}

    channels = FEATURE_CHANNELS[feature]
    paradigm = LeftRightImagery(channels=channels, resample=250, fmin=1, fmax=45)
    dataset = Lee2019_MI()
    x, y_raw, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    labels = (np.asarray(y_raw).astype(str) == "right_hand").astype(np.int64)
    sessions = np.asarray(meta["session"].values).astype(str)
    if not {"0", "1"}.issubset(set(sessions)):
        raise ValueError(f"S{subject}: expected sessions 0/1, got {sorted(set(sessions))}")
    if not (0 < int(prefix) <= int(eval_start)):
        raise ValueError(f"Require 0 < prefix <= eval_start, got {prefix}, {eval_start}")

    x = bandpass_8_30(x.astype(np.float64, copy=False))
    cov = covariances(x)

    session0 = np.flatnonzero(sessions == "0")
    session1 = np.flatnonzero(sessions == "1")
    if len(session1) <= int(eval_start):
        raise ValueError(f"S{subject}: eval_start={eval_start} leaves no target trials")

    prefix_indices = session1[: int(prefix)]
    eval_indices = session1[int(eval_start) :]

    output = {
        "source_features": tangent_features(cov, session0, session0).astype(np.float32),
        "source_labels": labels[session0].astype(np.int64),
        "target_features": tangent_features(cov, prefix_indices, eval_indices).astype(np.float32),
        "target_labels": labels[eval_indices].astype(np.int64),
        "all_target_features": tangent_features(cov, prefix_indices, session1).astype(np.float32),
        "all_target_labels": labels[session1].astype(np.int64),
        "n_source": np.asarray([len(session0)], dtype=np.int64),
        "n_prefix": np.asarray([len(prefix_indices)], dtype=np.int64),
        "n_eval": np.asarray([len(eval_indices)], dtype=np.int64),
        "channels": np.asarray(channels),
        "feature": np.asarray([feature]),
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
            "feature": str(next(row["feature"] for row in records if str(row["method"]) == method)),
            "n_channels": int(next(row["n_channels"] for row in records if str(row["method"]) == method)),
            "p": int(next(row["p"] for row in records if str(row["method"]) == method)),
            "p_over_source_total": float(
                next(row["p"] for row in records if str(row["method"]) == method)
                / next(row["n_source"] for row in records if str(row["method"]) == method)
            ),
            "accuracy_mean": float(acc.mean()) if len(acc) else float("nan"),
            "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, int(bootstrap)),
            "gain_vs_full_mean_pp": float(gains.mean()) if len(gains) else float("nan"),
            "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, rng, int(bootstrap)),
            "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)) if len(gains) else float("nan"),
            "gain_vs_full_median_pp": float(np.median(gains)) if len(gains) else float("nan"),
            "loss_r10_vs_full_pp": float(
                -np.sort(gains)[: max(1, int(np.ceil(0.1 * len(gains))))].mean()
            )
            if len(gains)
            else float("nan"),
            "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)) if len(gains) else float("nan"),
            "n_subject_gain_pos": int(np.sum(gains > 0.0)),
            "n_subject_gain_neg": int(np.sum(gains < 0.0)),
        }
    return output


def main() -> None:
    args = parse_args()
    subjects = parse_subjects(args.subjects)
    output_dir = args.output_dir / args.feature
    cache_dir = args.cache_dir if args.cache_dir is not None else output_dir / "subject_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads: dict[int, dict[str, np.ndarray]] = {}
    failures: list[dict[str, object]] = []
    for subject in subjects:
        try:
            payloads[subject] = load_subject_features(
                subject,
                feature=args.feature,
                cache_dir=cache_dir,
                prefix=int(args.prefix),
                eval_start=int(args.eval_start),
                force_cache=bool(args.force_cache),
            )
            if not args.quiet:
                p = payloads[subject]["source_features"].shape[1]
                print(
                    f"S{subject}: loaded feature={args.feature} "
                    f"channels={len(payloads[subject]['channels'])} p={p}",
                    flush=True,
                )
        except Exception as error:
            failures.append({"subject": int(subject), "stage": "load", "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)

    available_subjects = sorted(payloads)
    fractions = sorted(set(float(value) for value in args.fractions), reverse=True)
    if 1.0 not in fractions:
        fractions = [1.0, *fractions]

    records: list[dict[str, object]] = []
    for held_subject in available_subjects:
        payload = payloads[held_subject]
        stats = aggregate_stats(payloads, held_subject)
        scores = {
            "source_only": source_only_score(stats),
            "longitudinal": longitudinal_score(stats),
        }
        dim = int(payload["source_features"].shape[1])
        index_by_method: dict[str, np.ndarray] = {"full_q1p00": np.arange(dim, dtype=np.int64)}
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
                    "feature": str(args.feature),
                    "n_channels": int(len(payload["channels"])),
                    "p": int(dim),
                    "n_source": int(payload["n_source"][0]),
                    "n_eval": int(payload["n_eval"][0]),
                    "method": method,
                    "n_selected": int(len(indices)),
                    "accuracy": float(accuracy),
                }
            )
        write_csv(output_dir / "selection_records.partial.csv", records)
        if not args.quiet:
            full = next(
                row["accuracy"]
                for row in records
                if int(row["subject"]) == int(held_subject) and row["method"] == "full_q1p00"
            )
            best = max(row["accuracy"] for row in records if int(row["subject"]) == int(held_subject))
            print(f"S{held_subject}: full={float(full):.1f} best={float(best):.1f}", flush=True)

    summary = summarize(records, available_subjects, int(args.bootstrap), int(args.seed))
    write_csv(output_dir / "selection_records.csv", records)
    write_csv(output_dir / "failures.csv", failures)
    write_csv(
        output_dir / "summary.csv",
        [{"method": method, **metrics} for method, metrics in sorted(summary.items())],
    )
    report = {
        "config": {
            "dataset": "Lee2019_MI",
            "feature": args.feature,
            "channels": FEATURE_CHANNELS[args.feature],
            "subjects": available_subjects,
            "requested_subjects": subjects,
            "prefix": int(args.prefix),
            "eval_start": int(args.eval_start),
            "fractions": fractions,
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "summary": summary,
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_records": len(records),
                "n_failures": len(failures),
                "summary_json": str(output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
