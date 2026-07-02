"""Merge the four population-source repair folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=60.0)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def bootstrap_ci(
    values: np.ndarray,
    statistic,
    n_bootstrap: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        resampled = values[rng.integers(0, len(values), len(values))]
        samples[index] = statistic(resampled)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    paths = sorted(
        args.input_root.glob(f"fold_*/subjects/S*_seed{args.seed}.json")
    )
    payloads = [json.loads(path.read_text()) for path in paths]
    repaired = {
        int(payload["subject"]): float(payload["mean_target_source_accuracy"])
        for payload in payloads
    }
    baseline_report = json.loads(args.baseline_summary.read_text())
    baseline = {
        int(subject): float(accuracy)
        for subject, accuracy in baseline_report["source_accuracy_by_subject"].items()
    }
    subjects = sorted(set(repaired) & set(baseline))
    repaired_values = np.asarray([repaired[subject] for subject in subjects])
    baseline_values = np.asarray([baseline[subject] for subject in subjects])
    differences = repaired_values - baseline_values
    median_repaired = float(np.median(repaired_values))
    report = {
        "seed": args.seed,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "acceptance_threshold": args.threshold,
        "accepted": median_repaired >= args.threshold,
        "repaired": {
            "mean": float(np.mean(repaired_values)),
            "median": median_repaired,
            "range": [float(np.min(repaired_values)), float(np.max(repaired_values))],
            "median_subject_bootstrap_95ci": bootstrap_ci(
                repaired_values, np.median, args.bootstrap, args.seed
            ),
        },
        "baseline": {
            "mean": float(np.mean(baseline_values)),
            "median": float(np.median(baseline_values)),
        },
        "paired_change": {
            "mean": float(np.mean(differences)),
            "median": float(np.median(differences)),
            "mean_subject_bootstrap_95ci": bootstrap_ci(
                differences, np.mean, args.bootstrap, args.seed + 1
            ),
            "n_improved": int(np.sum(differences > 0)),
        },
        "source_accuracy_by_subject": repaired,
    }
    output = args.input_root / f"summary_16_seed{args.seed}.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
