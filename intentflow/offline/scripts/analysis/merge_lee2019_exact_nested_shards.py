"""Merge sharded Lee2019 exact nested subspace-selection outputs.

The sharded runs use the same --subjects source pool and disjoint
--target-subjects. This utility concatenates those shard outputs and recomputes
the summaries on the full outer-subject set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "research_outputs"
    / "260630_lee2019_exact_nested_subspace_selection_e25_full"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE / "merged")
    parser.add_argument("--expected-subjects", type=int, default=54)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(int(repeats), values.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize_records(
    records: pd.DataFrame,
    *,
    method: str,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    gains = records["gain_vs_full"].to_numpy(dtype=np.float64)
    acc = records["accuracy"].to_numpy(dtype=np.float64)
    full_acc = records["full_accuracy"].to_numpy(dtype=np.float64)
    tail = np.sort(gains)[: max(1, int(np.ceil(0.1 * gains.size)))]
    rng = np.random.default_rng(seed)
    row: dict[str, object] = {
        "method": method,
        "n_subjects": int(gains.size),
        "accuracy_mean": float(acc.mean()),
        "full_accuracy_mean": float(full_acc.mean()),
        "gain_vs_full_mean_pp": float(gains.mean()),
        "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, rng, bootstrap),
        "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)),
        "loss_r10_vs_full_pp": float(-tail.mean()),
        "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)),
    }
    if "chosen_candidate" in records:
        choices = Counter(records["chosen_candidate"].astype(str))
        row["chosen_candidate_counts"] = dict(sorted(choices.items()))
        row["chosen_candidate_rates"] = {
            key: float(value / gains.size) for key, value in sorted(choices.items())
        }
    return row


def summarize_paired_diff(
    nested: pd.DataFrame,
    fixed: pd.DataFrame,
    *,
    fixed_candidate: str,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    fixed_one = fixed[fixed["chosen_candidate"] == fixed_candidate].copy()
    merged = nested[["subject", "accuracy", "gain_vs_full"]].merge(
        fixed_one[["subject", "accuracy", "gain_vs_full"]],
        on="subject",
        suffixes=("_nested", "_fixed"),
        validate="one_to_one",
    )
    gain_diff = (
        merged["gain_vs_full_nested"].to_numpy(dtype=np.float64)
        - merged["gain_vs_full_fixed"].to_numpy(dtype=np.float64)
    )
    acc_diff = (
        merged["accuracy_nested"].to_numpy(dtype=np.float64)
        - merged["accuracy_fixed"].to_numpy(dtype=np.float64)
    )
    rng = np.random.default_rng(seed)
    return {
        "comparison": f"nested_minus_fixed__{fixed_candidate}",
        "n_subjects": int(gain_diff.size),
        "gain_diff_mean_pp": float(gain_diff.mean()),
        "gain_diff_subject_bootstrap_95ci": bootstrap_ci(gain_diff, rng, bootstrap),
        "accuracy_diff_mean_pp": float(acc_diff.mean()),
        "p_nested_worse_by_gt_5pp": float(np.mean(gain_diff < -5.0)),
        "p_nested_better_by_gt_5pp": float(np.mean(gain_diff > 5.0)),
        "nested_minus_fixed_q05_pp": float(np.quantile(gain_diff, 0.05)),
    }


def read_shards(base_dir: Path, filename: str) -> pd.DataFrame:
    paths = sorted(base_dir.glob(f"shard_*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No shard outputs found for {filename} under {base_dir}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.insert(0, "shard", path.parent.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nested = read_shards(args.base_dir, "exact_nested_selection_records.csv")
    fixed = read_shards(args.base_dir, "fixed_candidate_records.csv")
    validation = read_shards(args.base_dir, "exact_nested_source_validation.csv")

    nested = nested.sort_values("subject").reset_index(drop=True)
    fixed = fixed.sort_values(["subject", "chosen_candidate"]).reset_index(drop=True)
    validation = validation.sort_values(["held_subject", "inner_subject"]).reset_index(drop=True)

    n_unique = int(nested["subject"].nunique())
    if n_unique != len(nested):
        duplicates = nested["subject"][nested["subject"].duplicated()].tolist()
        raise ValueError(f"Duplicate nested subjects found: {duplicates}")
    if n_unique != int(args.expected_subjects):
        raise ValueError(f"Expected {args.expected_subjects} subjects, got {n_unique}")

    nested.to_csv(args.output_dir / "exact_nested_selection_records.csv", index=False)
    fixed.to_csv(args.output_dir / "fixed_candidate_records.csv", index=False)
    validation.to_csv(args.output_dir / "exact_nested_source_validation.csv", index=False)

    summary_rows: list[dict[str, object]] = [
        summarize_records(
            nested,
            method="nested_risk",
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        )
    ]
    for candidate, rows in fixed.groupby("chosen_candidate", sort=True):
        summary_rows.append(
            summarize_records(
                rows,
                method=f"fixed__{candidate}",
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "exact_nested_summary.csv", index=False)

    comparison_rows = [
        summarize_paired_diff(
            nested,
            fixed,
            fixed_candidate=str(candidate),
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        )
        for candidate in sorted(fixed["chosen_candidate"].astype(str).unique())
    ]
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(args.output_dir / "nested_vs_fixed_paired_comparisons.csv", index=False)

    payload = {
        "config": {
            "base_dir": str(args.base_dir),
            "output_dir": str(args.output_dir),
            "expected_subjects": int(args.expected_subjects),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "summary": summary_rows,
        "nested_vs_fixed_paired_comparisons": comparison_rows,
    }
    (args.output_dir / "exact_nested_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "n_subjects": n_unique,
                "summary_csv": str(args.output_dir / "exact_nested_summary.csv"),
                "paired_csv": str(args.output_dir / "nested_vs_fixed_paired_comparisons.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
