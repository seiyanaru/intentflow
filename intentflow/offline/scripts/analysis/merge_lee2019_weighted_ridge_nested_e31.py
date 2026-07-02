"""Merge sharded E31 weighted Ridge nested-validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_BASE = RESULTS_DIR / "260701_lee2019_weighted_ridge_nested_e31"
DEFAULT_OUTPUT = DEFAULT_BASE / "merged"
DEFAULT_LDA_FIXED = (
    RESULTS_DIR
    / "260630_lee2019_exact_nested_subspace_selection_e25_full"
    / "merged"
    / "fixed_candidate_records.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lda-fixed-records", type=Path, default=DEFAULT_LDA_FIXED)
    parser.add_argument("--expected-subjects", type=int, default=54)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_shards(base_dir: Path, filename: str) -> pd.DataFrame:
    paths = sorted(base_dir.glob(f"shard_*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No shard files for {filename} under {base_dir}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.insert(0, "shard", path.parent.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(arr, size=(int(repeats), arr.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize_accuracy(records: pd.DataFrame, *, method: str, bootstrap: int, seed: int) -> dict[str, object]:
    acc = records["accuracy"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = {
        "method": method,
        "n_subjects": int(len(records)),
        "accuracy_mean": float(acc.mean()),
        "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, bootstrap),
        "accuracy_q05": float(np.quantile(acc, 0.05)),
    }
    if "chosen_method" in records:
        out["chosen_method_counts"] = (
            records["chosen_method"].astype(str).value_counts().sort_index().to_dict()
        )
    return out


def summarize_diff(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    comparison: str,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    merged = left[["subject", "accuracy"]].merge(
        right[["subject", "accuracy"]],
        on="subject",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    diff = merged["accuracy_left"].to_numpy(dtype=np.float64) - merged["accuracy_right"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    return {
        "comparison": comparison,
        "n_subjects": int(len(diff)),
        "mean_diff_pp": float(diff.mean()),
        "subject_bootstrap_95ci": bootstrap_ci(diff, rng, bootstrap),
        "q05_diff_pp": float(np.quantile(diff, 0.05)),
        "p_diff_lt_minus5": float(np.mean(diff < -5.0)),
        "p_diff_gt_5": float(np.mean(diff > 5.0)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nested = read_shards(args.base_dir, "nested_selection_records.csv")
    fixed = read_shards(args.base_dir, "fixed_candidate_records.csv")
    validation = read_shards(args.base_dir, "source_validation_records.csv")

    nested = nested.sort_values(["policy", "subject"]).reset_index(drop=True)
    fixed = fixed.sort_values(["method", "subject"]).reset_index(drop=True)
    validation = validation.sort_values(["held_subject", "inner_subject"]).reset_index(drop=True)

    n_subjects = int(nested["subject"].nunique())
    if n_subjects != int(args.expected_subjects):
        raise ValueError(f"Expected {args.expected_subjects} subjects, got {n_subjects}")

    nested.to_csv(args.output_dir / "nested_selection_records.csv", index=False)
    fixed.to_csv(args.output_dir / "fixed_candidate_records.csv", index=False)
    validation.to_csv(args.output_dir / "source_validation_records.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for policy, rows in nested.groupby("policy", sort=True):
        summary_rows.append(
            summarize_accuracy(rows, method=str(policy), bootstrap=int(args.bootstrap), seed=int(args.seed))
        )
    for method, rows in fixed.groupby("method", sort=True):
        summary_rows.append(
            summarize_accuracy(
                rows,
                method=f"fixed__{method}",
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
        )
    summary = pd.DataFrame(summary_rows).sort_values("accuracy_mean", ascending=False)
    summary.to_csv(args.output_dir / "nested_weighted_ridge_summary.csv", index=False)

    lda = pd.read_csv(args.lda_fixed_records)
    lda_source = lda[lda["chosen_candidate"] == "source_only_q0p25"].copy()
    lda_long = lda[lda["chosen_candidate"] == "longitudinal_q0p10"].copy()
    lda_full = lda[lda["chosen_candidate"] == "full_q1p00"].copy()
    best_fixed_method = summary[summary["method"].astype(str).str.startswith("fixed__")].iloc[0]["method"]
    best_fixed = fixed[fixed["method"] == str(best_fixed_method).replace("fixed__", "")].copy()
    nested_mean = nested[nested["policy"] == "nested_mean"].copy()
    nested_risk = nested[nested["policy"] == "nested_risk20"].copy()

    comparisons = pd.DataFrame(
        [
            summarize_diff(
                nested_mean,
                lda_source,
                comparison="nested_mean_minus_lda_source_only_q0p25",
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            ),
            summarize_diff(
                nested_risk,
                lda_source,
                comparison="nested_risk20_minus_lda_source_only_q0p25",
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            ),
            summarize_diff(
                best_fixed,
                lda_source,
                comparison=f"{best_fixed_method}_minus_lda_source_only_q0p25",
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            ),
            summarize_diff(
                best_fixed,
                lda_full,
                comparison=f"{best_fixed_method}_minus_lda_full_q1p00",
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            ),
            summarize_diff(
                best_fixed,
                lda_long,
                comparison=f"{best_fixed_method}_minus_lda_longitudinal_q0p10",
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            ),
        ]
    )
    comparisons.to_csv(args.output_dir / "paired_comparisons.csv", index=False)

    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "base_dir": str(args.base_dir),
                    "output_dir": str(args.output_dir),
                    "lda_fixed_records": str(args.lda_fixed_records),
                    "expected_subjects": int(args.expected_subjects),
                    "bootstrap": int(args.bootstrap),
                    "seed": int(args.seed),
                },
                "summary": summary_rows,
                "paired_comparisons": comparisons.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_subjects": n_subjects,
                "summary_csv": str(args.output_dir / "nested_weighted_ridge_summary.csv"),
                "paired_csv": str(args.output_dir / "paired_comparisons.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
