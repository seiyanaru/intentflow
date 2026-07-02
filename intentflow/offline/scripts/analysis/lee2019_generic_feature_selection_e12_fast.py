"""E12-fast: generic feature-selection baselines on Lee2019 without refitting E9 methods.

The full E12 script recomputes full/source-side methods and is slow because
high-dimensional shrinkage LDA is expensive.  This fast audit reuses E9
selection_records for full/source_only/longitudinal and computes only generic
held-subject source-session selectors.

This directly tests the immediate reviewer question:

    Does a simple within-subject source-label feature selector beat
    source_only_q0.25 on Lee2019?
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
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_longitudinal_selection_pilot import (  # noqa: E402
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
DEFAULT_E9_RECORDS = RESULTS_DIR / "260628_lee2019_fraction_sweep_e9_full" / "selection_records.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "260628_e12_lee2019_generic_feature_selection_fast"
REFERENCE_METHODS = (
    "full_q1p00",
    "source_only_q0p25",
    "source_only_q0p50",
    "longitudinal_q0p10",
    "longitudinal_q0p20",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--e9-records", type=Path, default=DEFAULT_E9_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.25])
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
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


def reference_records(path: Path, subjects: Sequence[int]) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    frame = frame[frame["subject"].isin(subjects) & frame["method"].isin(REFERENCE_METHODS)]
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "subject": int(row["subject"]),
                "method": str(row["method"]),
                "accuracy": float(row["accuracy"]),
                "n_selected": int(row["n_selected"]),
                "n_source": int(row["n_source"]),
                "n_eval": int(row["n_eval"]),
                "prefix": int(row["prefix"]),
                "eval_start": int(row["eval_start"]),
                "source": "E9_reference",
            }
        )
    return rows


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
    records = reference_records(args.e9_records, subjects)
    failures: list[dict[str, object]] = []
    for subject in subjects:
        try:
            payload = load_subject_features(
                subject,
                cache_dir=args.cache_dir,
                prefix=args.prefix,
                eval_start=args.eval_start,
                force_cache=args.force_cache,
            )
            scores = {
                "self_source_fisher": self_source_fisher_score(
                    payload["source_features"], payload["source_labels"]
                ),
                "self_source_variance": self_source_variance_score(payload["source_features"]),
            }
            for family, score in scores.items():
                for fraction in args.fractions:
                    indices = top_fraction_indices(score, float(fraction))
                    accuracy = fit_predict_accuracy(
                        payload["source_features"],
                        payload["source_labels"],
                        payload["target_features"],
                        payload["target_labels"],
                        indices,
                    )
                    records.append(
                        {
                            "subject": int(subject),
                            "method": f"{family}_{fraction_token(float(fraction))}",
                            "accuracy": float(accuracy),
                            "n_selected": int(len(indices)),
                            "n_source": int(payload["n_source"][0]),
                            "n_eval": int(payload["n_eval"][0]),
                            "prefix": int(args.prefix),
                            "eval_start": int(args.eval_start),
                            "source": "E12_fast",
                        }
                    )
            if not args.quiet:
                print(f"S{subject}: done", flush=True)
        except Exception as error:
            failures.append({"subject": int(subject), "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)
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
            "e9_records": str(args.e9_records),
            "prefix": int(args.prefix),
            "eval_start": int(args.eval_start),
            "fractions": [float(value) for value in args.fractions],
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "reference_methods": list(REFERENCE_METHODS),
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
