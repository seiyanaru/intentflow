"""E12: generic feature-selection baselines on Lee2019.

This audit asks whether E9/E10 is simply generic supervised dimensionality
reduction.  It compares the source-side population selectors against strong
allowed baselines that use the held subject's labeled calibration session only:

* self_source_fisher: univariate Fisher score on session 0;
* self_source_l1_c*: L1-logistic coefficient magnitude on session 0;
* self_source_l2_c1: L2-logistic coefficient magnitude on session 0;
* self_source_variance: source-session variance, label-free within source.

All methods still evaluate on session 1 with zero target labels.
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

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_LEE_OUTPUT,
    SCORE_FUNCTIONS,
    aggregate_stats,
    bootstrap_ci,
    class_centroids_and_within,
    fit_predict_accuracy,
    fraction_token,
    load_subject_features,
    lower_tail_loss,
    parse_subjects,
    top_fraction_indices,
)


RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_CACHE = RESULTS_DIR / "260627_lee2019_longitudinal_selection_pilot12" / "subject_cache"
DEFAULT_OUTPUT = RESULTS_DIR / "260628_e12_lee2019_generic_feature_selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[1.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50],
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--include-logistic", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


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


def self_source_fisher_score(source_features: np.ndarray, source_labels: np.ndarray) -> np.ndarray:
    classes = sorted(map(int, np.unique(source_labels)))
    centroids, within = class_centroids_and_within(source_features, source_labels, classes)
    score = np.maximum(centroids.var(axis=0), 0.0) / (0.25 * np.maximum(within, 0.0) + 1e-8)
    return np.where(np.isfinite(score), score, 0.0)


def self_source_variance_score(source_features: np.ndarray) -> np.ndarray:
    score = np.var(source_features, axis=0)
    return np.where(np.isfinite(score), score, 0.0)


def logistic_coef_score(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    penalty: str,
    c_value: float,
    seed: int,
) -> np.ndarray:
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=penalty,
            C=float(c_value),
            solver=solver,
            max_iter=3000,
            random_state=int(seed),
        ),
    )
    model.fit(source_features, source_labels)
    classifier = model.named_steps["logisticregression"]
    coef = np.asarray(classifier.coef_, dtype=np.float64)
    score = np.abs(coef).max(axis=0)
    return np.where(np.isfinite(score), score, 0.0)


def summarize_records(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(seed)
    methods = sorted({str(row["method"]) for row in records})
    full_by_subject = {
        int(row["subject"]): float(row["accuracy"])
        for row in records
        if str(row["method"]) == "full_q1p00"
    }
    summary: dict[str, dict[str, object]] = {}
    for method in methods:
        rows = [row for row in records if str(row["method"]) == method]
        by_subject = {int(row["subject"]): float(row["accuracy"]) for row in rows}
        acc = np.asarray([by_subject[subject] for subject in subjects if subject in by_subject])
        gains = np.asarray(
            [
                by_subject[subject] - full_by_subject[subject]
                for subject in subjects
                if subject in by_subject and subject in full_by_subject
            ],
            dtype=np.float64,
        )
        summary[method] = {
            "n_subjects": int(len(acc)),
            "accuracy_mean": float(acc.mean()) if len(acc) else float("nan"),
            "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, bootstrap),
            "gain_vs_full_mean_pp": float(gains.mean()) if len(gains) else float("nan"),
            "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, rng, bootstrap),
            "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)) if len(gains) else float("nan"),
            "loss_r10_vs_full_pp": lower_tail_loss(gains),
            "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)) if len(gains) else float("nan"),
        }
    return summary


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subjects = parse_subjects(args.subjects)
    payloads: dict[int, dict[str, np.ndarray]] = {}
    failures: list[dict[str, object]] = []
    for subject in subjects:
        try:
            payloads[subject] = load_subject_features(
                subject,
                cache_dir=args.cache_dir,
                prefix=args.prefix,
                eval_start=args.eval_start,
                force_cache=args.force_cache,
            )
            if not args.quiet:
                print(
                    f"S{subject}: loaded dim={payloads[subject]['source_features'].shape[1]} "
                    f"n_eval={int(payloads[subject]['n_eval'][0])}",
                    flush=True,
                )
        except Exception as error:
            failures.append({"subject": int(subject), "stage": "load", "error": repr(error)})
            print(f"S{subject}: FAIL load {error!r}", flush=True)

    records: list[dict[str, object]] = []
    available_subjects = sorted(payloads)
    for held_subject in available_subjects:
        payload = payloads[held_subject]
        dim = payload["source_features"].shape[1]
        index_by_method: dict[str, np.ndarray] = {"full_q1p00": np.arange(dim, dtype=np.int64)}

        population_stats = aggregate_stats(payloads, held_subject)
        for family in ("source_only", "longitudinal"):
            score = SCORE_FUNCTIONS[family](population_stats)
            for fraction in args.fractions:
                if float(fraction) >= 1.0:
                    continue
                index_by_method[f"{family}_{fraction_token(float(fraction))}"] = top_fraction_indices(
                    score, float(fraction)
                )

        generic_scores = {
            "self_source_fisher": self_source_fisher_score(
                payload["source_features"], payload["source_labels"]
            ),
            "self_source_variance": self_source_variance_score(payload["source_features"]),
        }
        if args.include_logistic:
            generic_scores.update(
                {
                    "self_source_l1_c0p1": logistic_coef_score(
                        payload["source_features"], payload["source_labels"], "l1", 0.1, args.seed
                    ),
                    "self_source_l1_c1": logistic_coef_score(
                        payload["source_features"], payload["source_labels"], "l1", 1.0, args.seed
                    ),
                    "self_source_l2_c1": logistic_coef_score(
                        payload["source_features"], payload["source_labels"], "l2", 1.0, args.seed
                    ),
                }
            )
        for family, score in generic_scores.items():
            for fraction in args.fractions:
                if float(fraction) >= 1.0:
                    continue
                index_by_method[f"{family}_{fraction_token(float(fraction))}"] = top_fraction_indices(
                    score, float(fraction)
                )

        for method, indices in index_by_method.items():
            try:
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
                        "accuracy": accuracy,
                        "n_selected": int(len(indices)),
                        "n_source": int(payload["n_source"][0]),
                        "n_eval": int(payload["n_eval"][0]),
                        "prefix": int(args.prefix),
                        "eval_start": int(args.eval_start),
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "subject": int(held_subject),
                        "stage": "evaluate",
                        "method": method,
                        "error": repr(error),
                    }
                )
        if not args.quiet:
            subject_rows = [row for row in records if int(row["subject"]) == int(held_subject)]
            full = next(row["accuracy"] for row in subject_rows if row["method"] == "full_q1p00")
            best = max(subject_rows, key=lambda row: float(row["accuracy"]))
            print(
                f"S{held_subject}: full={float(full):.1f} best={best['method']} {float(best['accuracy']):.1f}",
                flush=True,
            )
    return records, failures


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    records, failures = evaluate(args)
    write_csv(args.output_dir / "selection_records.csv", records)
    write_csv(args.output_dir / "failures.csv", failures)
    summary = summarize_records(records, subjects, args.bootstrap, args.seed)
    summary_rows = [{"method": method, **metrics} for method, metrics in sorted(summary.items())]
    write_csv(args.output_dir / "summary.csv", summary_rows)
    report = {
        "config": {
            "subjects": subjects,
            "cache_dir": str(args.cache_dir),
            "prefix": int(args.prefix),
            "eval_start": int(args.eval_start),
            "fractions": [float(value) for value in args.fractions],
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "include_logistic": bool(args.include_logistic),
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
    os.environ.setdefault("MNE_DATA", "/home/islabshi/workspace-local2/mne_data")
    main()
