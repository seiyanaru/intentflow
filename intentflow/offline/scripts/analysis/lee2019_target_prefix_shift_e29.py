"""E29 Lee2019 target-prefix feature-shift diagnostic.

Question:
    Can unlabeled target-prefix feature reliability/shift explain why
    zero-label subject-wise candidate selection fails on Lee2019?

This is a diagnostic, not a deployable method.  It reuses the E25 fixed
candidate records and Lee2019 feature cache.  Target prefix labels are never
used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_exact_nested_subspace_selection import (  # noqa: E402
    CANDIDATES,
    aggregate_excluding,
    candidate_indices,
)
from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    add_stats,
    load_subject_features,
    parse_subjects,
    session_metric_stats,
)


RESULTS_DIR = SCRIPT_DIR.parents[2] / "results" / "research_outputs"
DEFAULT_CACHE = RESULTS_DIR / "260627_lee2019_longitudinal_selection_pilot12" / "subject_cache"
DEFAULT_FIXED = (
    RESULTS_DIR
    / "260630_lee2019_exact_nested_subspace_selection_e25_full"
    / "merged"
    / "fixed_candidate_records.csv"
)
DEFAULT_NESTED = (
    RESULTS_DIR
    / "260630_lee2019_exact_nested_subspace_selection_e25_full"
    / "merged"
    / "exact_nested_selection_records.csv"
)
DEFAULT_OUTPUT = RESULTS_DIR / "260701_lee2019_target_prefix_shift_e29"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fixed-records", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--nested-records", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def spearmanr(x: Iterable[float], y: Iterable[float]) -> float:
    xs = pd.Series(list(x), dtype="float64").rank(method="average").to_numpy(dtype=np.float64)
    ys = pd.Series(list(y), dtype="float64").rank(method="average").to_numpy(dtype=np.float64)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    if xs.size < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def auroc(labels: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.int64)
    s = np.asarray(list(scores), dtype=np.float64)
    finite = np.isfinite(s)
    y = y[finite]
    s = s[finite]
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_pos = float(ranks[y == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_ci(values: Iterable[float], rng: np.random.Generator, repeats: int) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(arr, size=(int(repeats), arr.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize(values: Iterable[float], rng: np.random.Generator, repeats: int) -> dict[str, object]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q05": float("nan"),
            "q95": float("nan"),
            "bootstrap_95ci": [float("nan"), float("nan")],
        }
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q95": float(np.quantile(arr, 0.95)),
        "bootstrap_95ci": bootstrap_ci(arr, rng, repeats),
    }


def feature_shift_metrics(payload: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    source = np.asarray(payload["source_features"], dtype=np.float64)
    n_prefix = int(np.asarray(payload["n_prefix"])[0])
    target_prefix = np.asarray(payload["all_target_features"], dtype=np.float64)[:n_prefix]

    eps = 1e-8
    source_mean = source.mean(axis=0)
    prefix_mean = target_prefix.mean(axis=0)
    source_var = source.var(axis=0) + eps
    prefix_var = target_prefix.var(axis=0) + eps
    source_std = np.sqrt(source_var)

    return {
        "prefix_abs_z_shift": np.abs(prefix_mean - source_mean) / source_std,
        "prefix_log_var_ratio_abs": np.abs(np.log(prefix_var / source_var)),
        "prefix_var_ratio": prefix_var / source_var,
        "prefix_energy_z": np.abs(prefix_mean) / source_std,
        "prefix_std_z": np.sqrt(prefix_var) / source_std,
    }


def selected_summary(values: np.ndarray, indices: np.ndarray, prefix: str) -> dict[str, float]:
    selected = np.asarray(values, dtype=np.float64)[indices]
    return {
        f"{prefix}_mean": float(np.mean(selected)),
        f"{prefix}_median": float(np.median(selected)),
        f"{prefix}_q90": float(np.quantile(selected, 0.90)),
        f"{prefix}_q95": float(np.quantile(selected, 0.95)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    subjects = parse_subjects(args.subjects)
    fixed = pd.read_csv(args.fixed_records)
    nested = pd.read_csv(args.nested_records)
    fixed_gain = fixed.pivot(index="subject", columns="chosen_candidate", values="gain_vs_full")
    fixed_acc = fixed.pivot(index="subject", columns="chosen_candidate", values="accuracy")

    payloads: dict[int, dict[str, np.ndarray]] = {}
    for subject in subjects:
        payloads[subject] = load_subject_features(
            subject,
            cache_dir=args.cache_dir,
            prefix=int(args.prefix),
            eval_start=int(args.eval_start),
            force_cache=False,
        )

    stats_by_subject: dict[int, dict[str, np.ndarray]] = {}
    total_stats: dict[str, np.ndarray] = {}
    for subject, payload in payloads.items():
        stats = session_metric_stats(
            payload["source_features"],
            payload["source_labels"],
            payload["target_features"],
            payload["target_labels"],
        )
        stats_by_subject[subject] = stats
        add_stats(total_stats, stats)

    rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for held_subject in subjects:
        outer_stats = aggregate_excluding(total_stats, stats_by_subject, {held_subject})
        payload = payloads[held_subject]
        shifts = feature_shift_metrics(payload)
        dim = int(payload["source_features"].shape[1])
        nested_choice = str(nested[nested["subject"] == held_subject].iloc[0]["chosen_candidate"])

        for candidate in CANDIDATES:
            indices = candidate_indices(candidate, outer_stats, dim)
            row: dict[str, object] = {
                "subject": int(held_subject),
                "candidate": candidate,
                "n_selected": int(len(indices)),
                "selected_fraction": float(len(indices) / dim),
                "outer_gain": float(fixed_gain.loc[held_subject, candidate]),
                "outer_accuracy": float(fixed_acc.loc[held_subject, candidate]),
                "outer_harm_lt_minus5": int(float(fixed_gain.loc[held_subject, candidate]) < -5.0),
                "nested_chosen": int(candidate == nested_choice),
            }
            for metric_name, values in shifts.items():
                row.update(selected_summary(values, indices, metric_name))
            rows.append(row)

        source = "source_only_q0p25"
        long = "longitudinal_q0p10"
        source_row = next(r for r in rows if r["subject"] == held_subject and r["candidate"] == source)
        long_row = next(r for r in rows if r["subject"] == held_subject and r["candidate"] == long)
        pair_rows.append(
            {
                "subject": int(held_subject),
                "nested_choice": nested_choice,
                "outer_long_minus_source_pp": float(fixed_gain.loc[held_subject, long] - fixed_gain.loc[held_subject, source]),
                "nested_chose_long": int(nested_choice == long),
                "long_harm_lt_minus5": int(float(fixed_gain.loc[held_subject, long]) < -5.0),
                "source_harm_lt_minus5": int(float(fixed_gain.loc[held_subject, source]) < -5.0),
                "delta_prefix_abs_z_shift_mean_long_minus_source": float(
                    long_row["prefix_abs_z_shift_mean"] - source_row["prefix_abs_z_shift_mean"]
                ),
                "delta_prefix_log_var_ratio_abs_mean_long_minus_source": float(
                    long_row["prefix_log_var_ratio_abs_mean"] - source_row["prefix_log_var_ratio_abs_mean"]
                ),
                "delta_prefix_energy_z_mean_long_minus_source": float(
                    long_row["prefix_energy_z_mean"] - source_row["prefix_energy_z_mean"]
                ),
                "delta_prefix_std_z_mean_long_minus_source": float(
                    long_row["prefix_std_z_mean"] - source_row["prefix_std_z_mean"]
                ),
            }
        )

    df = pd.DataFrame(rows).sort_values(["subject", "candidate"])
    pair_df = pd.DataFrame(pair_rows).sort_values("subject")
    df.to_csv(args.output_dir / "candidate_prefix_shift_metrics.csv", index=False)
    pair_df.to_csv(args.output_dir / "long_vs_source_prefix_shift.csv", index=False)

    nonfull = df[df["candidate"] != "full_q1p00"].copy()
    candidate_level_metrics = [
        "prefix_abs_z_shift_mean",
        "prefix_abs_z_shift_q90",
        "prefix_log_var_ratio_abs_mean",
        "prefix_log_var_ratio_abs_q90",
        "prefix_energy_z_mean",
        "prefix_std_z_mean",
    ]
    metric_rows = []
    for metric in candidate_level_metrics:
        # High metric means more suspicious, so expected correlation with gain is negative.
        metric_rows.append(
            {
                "metric": metric,
                "spearman_metric_vs_outer_gain_nonfull": spearmanr(nonfull[metric], nonfull["outer_gain"]),
                "auroc_metric_predicts_outer_harm_nonfull": auroc(nonfull["outer_harm_lt_minus5"], nonfull[metric]),
                "within_subject_mean_spearman_neg_metric_vs_outer_gain": float(
                    np.nanmean(
                        [
                            spearmanr(-sub[metric], sub["outer_gain"])
                            for _, sub in nonfull.groupby("subject")
                        ]
                    )
                ),
            }
        )
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(args.output_dir / "prefix_metric_predictability.csv", index=False)

    pair_metric_rows = []
    for metric in [
        "delta_prefix_abs_z_shift_mean_long_minus_source",
        "delta_prefix_log_var_ratio_abs_mean_long_minus_source",
        "delta_prefix_energy_z_mean_long_minus_source",
        "delta_prefix_std_z_mean_long_minus_source",
    ]:
        # If long has larger suspicious metric than source, long should underperform source.
        pair_metric_rows.append(
            {
                "metric": metric,
                "spearman_delta_metric_vs_outer_long_minus_source": spearmanr(
                    pair_df[metric], pair_df["outer_long_minus_source_pp"]
                ),
                "spearman_neg_delta_metric_vs_outer_long_minus_source": spearmanr(
                    -pair_df[metric], pair_df["outer_long_minus_source_pp"]
                ),
                "auroc_delta_metric_predicts_long_harm": auroc(pair_df["long_harm_lt_minus5"], pair_df[metric]),
            }
        )
    pair_metric_df = pd.DataFrame(pair_metric_rows)
    pair_metric_df.to_csv(args.output_dir / "long_vs_source_shift_predictability.csv", index=False)

    long_chosen = pair_df[pair_df["nested_chose_long"] == 1].copy()
    summary = {
        "config": {
            "subjects": subjects,
            "cache_dir": str(args.cache_dir),
            "fixed_records": str(args.fixed_records),
            "nested_records": str(args.nested_records),
            "output_dir": str(args.output_dir),
            "prefix": int(args.prefix),
            "eval_start": int(args.eval_start),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "candidate_metric_predictability": metric_rows,
        "long_vs_source_predictability": pair_metric_rows,
        "long_vs_source_outer_delta_pp": summarize(
            pair_df["outer_long_minus_source_pp"], rng, int(args.bootstrap)
        ),
        "nested_chose_long_outer_delta_pp": summarize(
            long_chosen["outer_long_minus_source_pp"], rng, int(args.bootstrap)
        ),
        "nested_chose_long_delta_prefix_abs_z_shift_mean": summarize(
            long_chosen["delta_prefix_abs_z_shift_mean_long_minus_source"], rng, int(args.bootstrap)
        ),
        "nested_chose_long_delta_prefix_log_var_ratio_abs_mean": summarize(
            long_chosen["delta_prefix_log_var_ratio_abs_mean_long_minus_source"], rng, int(args.bootstrap)
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_rows": int(len(df)),
                "n_subjects": int(len(pair_df)),
                "metric_csv": str(args.output_dir / "prefix_metric_predictability.csv"),
                "pair_csv": str(args.output_dir / "long_vs_source_shift_predictability.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
