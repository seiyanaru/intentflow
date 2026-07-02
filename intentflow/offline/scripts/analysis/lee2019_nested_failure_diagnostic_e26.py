"""E26 diagnostic: why Lee2019 source-side nested selection fails.

This script analyzes the merged E25 exact nested outputs.  It does not rerun
models; it diagnoses whether source-validation candidate metrics predict the
held-out target outcome, and where the nested selector loses against the fixed
source_only_q0p25 policy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "research_outputs"
DEFAULT_INPUT = RESULTS_DIR / "260630_lee2019_exact_nested_subspace_selection_e25_full" / "merged"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_lee2019_nested_failure_diagnostic_e26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--risk-threshold", type=float, default=0.20)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def candidate_names(validation: pd.DataFrame) -> list[str]:
    candidates = []
    for column in validation.columns:
        if column.endswith("__gain_vs_full"):
            candidates.append(column.removesuffix("__gain_vs_full"))
    preferred_order = [
        "full_q1p00",
        "source_only_q0p25",
        "source_only_q0p10",
        "longitudinal_q0p25",
        "longitudinal_q0p10",
        "sep_no_drift_q0p25",
        "sep_no_drift_q0p10",
    ]
    return [c for c in preferred_order if c in candidates] + sorted(set(candidates) - set(preferred_order))


def tail_loss(values: Sequence[float], fraction: float = 0.10) -> float:
    arr = np.asarray(values, dtype=np.float64)
    tail = np.sort(arr)[: max(1, int(np.ceil(fraction * arr.size)))]
    return float(-tail.mean())


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(int(repeats), values.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def corr(x: Sequence[float], y: Sequence[float], *, rank: bool = False) -> float:
    x_arr = pd.Series(np.asarray(x, dtype=np.float64))
    y_arr = pd.Series(np.asarray(y, dtype=np.float64))
    if rank:
        x_arr = x_arr.rank(method="average")
        y_arr = y_arr.rank(method="average")
    if x_arr.nunique() <= 1 or y_arr.nunique() <= 1:
        return float("nan")
    return float(np.corrcoef(x_arr.to_numpy(), y_arr.to_numpy())[0, 1])


def corr_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rank: bool,
    rng: np.random.Generator,
    repeats: int,
) -> list[float]:
    if x.size == 0:
        return [float("nan"), float("nan")]
    vals = []
    for _ in range(int(repeats)):
        idx = rng.choice(np.arange(x.size), size=x.size, replace=True)
        vals.append(corr(x[idx], y[idx], rank=rank))
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def summarize_values(values: np.ndarray, *, bootstrap: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "mean_bootstrap_95ci": bootstrap_ci(values, rng, bootstrap),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "p_lt_minus5": float(np.mean(values < -5.0)),
        "r10_loss": tail_loss(values),
    }


def build_candidate_metric_table(
    validation: pd.DataFrame,
    fixed: pd.DataFrame,
    candidates: Sequence[str],
) -> pd.DataFrame:
    fixed_by_subject = fixed.pivot(index="subject", columns="chosen_candidate", values="gain_vs_full")
    fixed_acc = fixed.pivot(index="subject", columns="chosen_candidate", values="accuracy")
    rows = []
    for held_subject, group in validation.groupby("held_subject", sort=True):
        for candidate in candidates:
            gains = group[f"{candidate}__gain_vs_full"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "held_subject": int(held_subject),
                    "candidate": candidate,
                    "inner_mean_gain": float(gains.mean()),
                    "inner_median_gain": float(np.median(gains)),
                    "inner_q05_gain": float(np.quantile(gains, 0.05)),
                    "inner_q10_gain": float(np.quantile(gains, 0.10)),
                    "inner_r10_loss": tail_loss(gains),
                    "inner_p_gain_lt_minus5": float(np.mean(gains < -5.0)),
                    "outer_gain": float(fixed_by_subject.loc[int(held_subject), candidate]),
                    "outer_accuracy": float(fixed_acc.loc[int(held_subject), candidate]),
                }
            )
    return pd.DataFrame(rows)


def choose_from_inner(metrics: pd.DataFrame, risk_threshold: float) -> dict[str, object]:
    eligible = metrics[metrics["inner_p_gain_lt_minus5"] <= risk_threshold].copy()
    if eligible.empty:
        chosen = metrics[metrics["candidate"] == "full_q1p00"].iloc[0]
    else:
        chosen = eligible.sort_values(
            ["inner_mean_gain", "inner_r10_loss", "candidate"],
            ascending=[False, True, True],
        ).iloc[0]

    sorted_by_mean = metrics.sort_values("inner_mean_gain", ascending=False).reset_index(drop=True)
    top = sorted_by_mean.iloc[0]
    runner = sorted_by_mean.iloc[1] if len(sorted_by_mean) > 1 else sorted_by_mean.iloc[0]
    outer_best = metrics.sort_values("outer_gain", ascending=False).iloc[0]
    source_q025 = metrics[metrics["candidate"] == "source_only_q0p25"].iloc[0]
    long_q010 = metrics[metrics["candidate"] == "longitudinal_q0p10"].iloc[0]
    long_q025 = metrics[metrics["candidate"] == "longitudinal_q0p25"].iloc[0]
    return {
        "held_subject": int(metrics["held_subject"].iloc[0]),
        "chosen_candidate": str(chosen["candidate"]),
        "chosen_inner_mean_gain": float(chosen["inner_mean_gain"]),
        "chosen_inner_p_gain_lt_minus5": float(chosen["inner_p_gain_lt_minus5"]),
        "chosen_inner_r10_loss": float(chosen["inner_r10_loss"]),
        "chosen_outer_gain": float(chosen["outer_gain"]),
        "top_inner_candidate": str(top["candidate"]),
        "top_inner_mean_gain": float(top["inner_mean_gain"]),
        "runner_inner_candidate": str(runner["candidate"]),
        "runner_inner_mean_gain": float(runner["inner_mean_gain"]),
        "inner_mean_margin_top_minus_runner": float(top["inner_mean_gain"] - runner["inner_mean_gain"]),
        "outer_best_candidate": str(outer_best["candidate"]),
        "outer_best_gain": float(outer_best["outer_gain"]),
        "selected_is_outer_best": bool(str(chosen["candidate"]) == str(outer_best["candidate"])),
        "selected_minus_outer_best_gain": float(chosen["outer_gain"] - outer_best["outer_gain"]),
        "source_only_q0p25_outer_gain": float(source_q025["outer_gain"]),
        "selected_minus_source_only_q0p25_gain": float(chosen["outer_gain"] - source_q025["outer_gain"]),
        "longitudinal_q0p10_inner_mean_gain": float(long_q010["inner_mean_gain"]),
        "longitudinal_q0p10_inner_p_gain_lt_minus5": float(long_q010["inner_p_gain_lt_minus5"]),
        "longitudinal_q0p10_outer_gain": float(long_q010["outer_gain"]),
        "longitudinal_q0p10_minus_source_only_q0p25_inner_mean": float(
            long_q010["inner_mean_gain"] - source_q025["inner_mean_gain"]
        ),
        "longitudinal_q0p10_minus_source_only_q0p25_outer_gain": float(
            long_q010["outer_gain"] - source_q025["outer_gain"]
        ),
        "longitudinal_q0p10_minus_longitudinal_q0p25_inner_mean": float(
            long_q010["inner_mean_gain"] - long_q025["inner_mean_gain"]
        ),
        "longitudinal_q0p10_minus_longitudinal_q0p25_outer_gain": float(
            long_q010["outer_gain"] - long_q025["outer_gain"]
        ),
    }


def predictability_summary(
    candidate_metrics: pd.DataFrame,
    candidates: Sequence[str],
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for candidate in candidates:
        sub = candidate_metrics[candidate_metrics["candidate"] == candidate].copy()
        x_mean = sub["inner_mean_gain"].to_numpy(dtype=np.float64)
        x_q05 = sub["inner_q05_gain"].to_numpy(dtype=np.float64)
        x_r10 = sub["inner_r10_loss"].to_numpy(dtype=np.float64)
        x_pbad = sub["inner_p_gain_lt_minus5"].to_numpy(dtype=np.float64)
        y = sub["outer_gain"].to_numpy(dtype=np.float64)
        y_bad = (y < -5.0).astype(float)
        rows.append(
            {
                "candidate": candidate,
                "n_subjects": int(len(sub)),
                "inner_mean_vs_outer_pearson": corr(x_mean, y),
                "inner_mean_vs_outer_spearman": corr(x_mean, y, rank=True),
                "inner_mean_vs_outer_spearman_bootstrap_95ci": corr_bootstrap_ci(
                    x_mean, y, rank=True, rng=rng, repeats=bootstrap
                ),
                "inner_q05_vs_outer_spearman": corr(x_q05, y, rank=True),
                "inner_r10_loss_vs_outer_spearman": corr(x_r10, y, rank=True),
                "inner_pbad_vs_outer_bad_spearman": corr(x_pbad, y_bad, rank=True),
                "outer_gain_mean": float(y.mean()),
                "outer_gain_q05": float(np.quantile(y, 0.05)),
                "outer_p_gain_lt_minus5": float(np.mean(y < -5.0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(args.input_dir / "exact_nested_source_validation.csv")
    fixed = pd.read_csv(args.input_dir / "fixed_candidate_records.csv")
    nested = pd.read_csv(args.input_dir / "exact_nested_selection_records.csv")
    candidates = candidate_names(validation)

    candidate_metrics = build_candidate_metric_table(validation, fixed, candidates)
    candidate_metrics.to_csv(args.output_dir / "candidate_predictability_by_held_subject.csv", index=False)

    pred = predictability_summary(
        candidate_metrics,
        candidates,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    pred.to_csv(args.output_dir / "predictability_summary.csv", index=False)

    selection_rows = [
        choose_from_inner(group, risk_threshold=float(args.risk_threshold))
        for _, group in candidate_metrics.groupby("held_subject", sort=True)
    ]
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(args.output_dir / "selection_margin_and_outer_outcome.csv", index=False)

    long_selected = selection[selection["chosen_candidate"] == "longitudinal_q0p10"].copy()
    long_selected.to_csv(args.output_dir / "longitudinal_q0p10_selected_cases.csv", index=False)

    fixed_wide = fixed.pivot(index="subject", columns="chosen_candidate", values="gain_vs_full")
    nested_gain = nested.set_index("subject")["gain_vs_full"].sort_index().to_numpy(dtype=np.float64)
    source_q025 = fixed_wide["source_only_q0p25"].sort_index().to_numpy(dtype=np.float64)
    long_q010 = fixed_wide["longitudinal_q0p10"].sort_index().to_numpy(dtype=np.float64)

    summary = {
        "config": {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "risk_threshold": float(args.risk_threshold),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "candidates": list(candidates),
        },
        "nested_gain": summarize_values(nested_gain, bootstrap=int(args.bootstrap), seed=int(args.seed)),
        "fixed_source_only_q0p25_gain": summarize_values(
            source_q025,
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        ),
        "fixed_longitudinal_q0p10_gain": summarize_values(
            long_q010,
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        ),
        "selection": {
            "chosen_counts": dict(Counter(selection["chosen_candidate"])),
            "outer_best_match_rate": float(selection["selected_is_outer_best"].mean()),
            "selected_minus_outer_best_gain_mean": float(selection["selected_minus_outer_best_gain"].mean()),
            "selected_minus_source_only_q0p25_gain_mean": float(
                selection["selected_minus_source_only_q0p25_gain"].mean()
            ),
            "selected_minus_source_only_q0p25_gain_bootstrap_95ci": bootstrap_ci(
                selection["selected_minus_source_only_q0p25_gain"].to_numpy(dtype=np.float64),
                np.random.default_rng(int(args.seed)),
                int(args.bootstrap),
            ),
            "inner_mean_margin_top_minus_runner": summarize_values(
                selection["inner_mean_margin_top_minus_runner"].to_numpy(dtype=np.float64),
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            ),
        },
        "longitudinal_q0p10_selected": {
            "n": int(len(long_selected)),
            "outer_gain_mean": float(long_selected["chosen_outer_gain"].mean()),
            "outer_p_gain_lt_minus5": float(np.mean(long_selected["chosen_outer_gain"] < -5.0)),
            "minus_source_only_q0p25_outer_gain_mean": float(
                long_selected["selected_minus_source_only_q0p25_gain"].mean()
            ),
            "minus_source_only_q0p25_outer_gain_bootstrap_95ci": bootstrap_ci(
                long_selected["selected_minus_source_only_q0p25_gain"].to_numpy(dtype=np.float64),
                np.random.default_rng(int(args.seed)),
                int(args.bootstrap),
            ),
            "inner_mean_advantage_over_source_only_q0p25_mean": float(
                long_selected["longitudinal_q0p10_minus_source_only_q0p25_inner_mean"].mean()
            ),
            "outer_advantage_over_source_only_q0p25_mean": float(
                long_selected["longitudinal_q0p10_minus_source_only_q0p25_outer_gain"].mean()
            ),
            "worst_cases": long_selected.sort_values("chosen_outer_gain")
            .head(8)
            .to_dict(orient="records"),
        },
        "predictability_summary": pred.to_dict(orient="records"),
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_candidate_metric_rows": int(len(candidate_metrics)),
                "n_selection_rows": int(len(selection)),
                "chosen_counts": summary["selection"]["chosen_counts"],
                "outer_best_match_rate": summary["selection"]["outer_best_match_rate"],
                "longitudinal_q0p10_selected_n": summary["longitudinal_q0p10_selected"]["n"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
