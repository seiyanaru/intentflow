"""E17: task-family descriptor audit for fixed-policy recommendations.

E16 suggests that, once a good fixed policy is known, small-pool validation is
usually inferior.  The remaining scientific risk is that fixed policies look
post-hoc:

    Why Stieger -> longitudinal compact core, but Lee2019 -> source_only q0.25?

This script consolidates E11/E12/E15/E16 into descriptor tables that explain
which regime each dataset/task belongs to:

* Does source-side selection beat generic self-source feature selection?
* Is small/moderate-pool source-side validation stable?
* Should deployment use fixed policy or validation?

It does not add a new method.  It audits the conditions under which the current
method family is reliable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"

DEFAULT_E11 = RESULTS_DIR / "260628_e11_compactness_frontier" / "compactness_best_by_panel.csv"
DEFAULT_E12_STIEGER = RESULTS_DIR / "260628_e12_stieger_generic_feature_selection_fast" / "summary.json"
DEFAULT_E12_LEE = RESULTS_DIR / "260628_e12_lee2019_generic_feature_selection_fast" / "summary.json"
DEFAULT_E15 = RESULTS_DIR / "260628_e15_source_pool_size_ablation_full" / "summary.json"
DEFAULT_E16 = RESULTS_DIR / "260628_e16_fixed_policy_vs_validation" / "summary.json"
DEFAULT_OUTPUT = RESULTS_DIR / "260628_e17_task_family_descriptor_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e11-best", type=Path, default=DEFAULT_E11)
    parser.add_argument("--e12-stieger", type=Path, default=DEFAULT_E12_STIEGER)
    parser.add_argument("--e12-lee", type=Path, default=DEFAULT_E12_LEE)
    parser.add_argument("--e15-summary", type=Path, default=DEFAULT_E15)
    parser.add_argument("--e16-summary", type=Path, default=DEFAULT_E16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def family(method: str) -> str:
    if method.startswith("source_only"):
        return "source_only"
    if method.startswith("longitudinal"):
        return "longitudinal"
    if method.startswith("self_source_variance"):
        return "self_source_variance"
    if method.startswith("self_source_fisher"):
        return "self_source_fisher"
    if method.startswith("condition_long"):
        return "condition_longitudinal"
    if method.startswith("full"):
        return "full"
    return method.split("_q", maxsplit=1)[0]


def entropy(rates: Mapping[str, float]) -> float:
    values = [float(value) for value in rates.values() if float(value) > 0.0]
    if not values:
        return 0.0
    return float(-sum(value * math.log(value, 2) for value in values))


def best_generic_from_stieger(e12: Mapping[str, object], scope: str) -> dict[str, object]:
    scoped = e12["summary"][scope]
    rows = []
    for method, metrics in scoped.items():
        if method.startswith("self_source_") and metrics:
            rows.append((float(metrics["gain_vs_full_subject_pooled_pp"]), method, metrics))
    gain, method, metrics = max(rows, key=lambda item: item[0])
    ci = metrics["gain_vs_full_subject_bootstrap_95ci"]
    return {
        "best_generic_method": method,
        "best_generic_family": family(method),
        "best_generic_gain_pp": gain,
        "best_generic_ci_low": float(ci[0]),
        "best_generic_ci_high": float(ci[1]),
        "best_generic_p_loss_gt5": float(metrics["p_gain_vs_full_lt_minus5"]),
    }


def best_generic_from_lee(e12: Mapping[str, object]) -> dict[str, object]:
    rows = []
    for method, metrics in e12["summary"].items():
        if method.startswith("self_source_") and metrics:
            rows.append((float(metrics["gain_vs_full_mean_pp"]), method, metrics))
    gain, method, metrics = max(rows, key=lambda item: item[0])
    ci = metrics["gain_vs_full_subject_bootstrap_95ci"]
    return {
        "best_generic_method": method,
        "best_generic_family": family(method),
        "best_generic_gain_pp": gain,
        "best_generic_ci_low": float(ci[0]),
        "best_generic_ci_high": float(ci[1]),
        "best_generic_p_loss_gt5": float(metrics["p_gain_vs_full_lt_minus5"]),
    }


def task_panel_descriptors(e11_best: pd.DataFrame, e12_stieger: dict, e12_lee: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scope_map = {
        ("Stieger", "LR"): "pure_lr",
        ("Stieger", "UD"): "pure_ud",
        ("Stieger", "pooled LR+UD"): "primary_pooled",
    }
    for _, row in e11_best.iterrows():
        dataset = str(row["dataset"])
        task = str(row["task"])
        if dataset == "Stieger":
            generic = best_generic_from_stieger(e12_stieger, scope_map[(dataset, task)])
        else:
            generic = best_generic_from_lee(e12_lee)
        source_gain = float(row["gain_pp"])
        generic_gain = float(generic["best_generic_gain_pp"])
        gap = source_gain - generic_gain
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "best_source_side_method": str(row["method"]),
                "best_source_side_family": family(str(row["method"])),
                "best_source_side_gain_pp": source_gain,
                "best_source_side_ci_low": float(row["gain_ci_low"]),
                "best_source_side_ci_high": float(row["gain_ci_high"]),
                "best_source_side_p_loss_gt5": float(row["p_loss_gt5"]),
                **generic,
                "source_side_minus_generic_pp": gap,
                "generic_explains_most_of_gain": bool(generic_gain > 0 and generic_gain >= 0.7 * source_gain),
                "descriptor_interpretation": (
                    "generic compactness contributes strongly"
                    if generic_gain > 0 and generic_gain >= 0.7 * source_gain
                    else "source-side population/statistical ranking is necessary"
                ),
            }
        )
    return pd.DataFrame(rows)


def validation_descriptor_rows(e15: dict, e16: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    e16_rows = {
        (
            str(row["dataset"]),
            str(row["validation_policy"]),
            str(row["pool_size"]),
        ): row
        for row in e16["validation_vs_fixed"]
    }
    for key, item in e15["summary"].items():
        dataset, policy, pool = key.split("__", maxsplit=2)
        pool_size = pool.removeprefix("m")
        if pool_size == "all":
            continue
        fixed = e16_rows[(dataset, policy, pool_size)]
        family_rates = item["chosen_family_rates"]
        candidate_rates = item["chosen_candidate_rates"]
        top_candidate, top_rate = max(candidate_rates.items(), key=lambda kv: float(kv[1]))
        rows.append(
            {
                "dataset": dataset,
                "validation_policy": policy,
                "pool_size": pool_size,
                "validation_gain_pp": float(item["gain_vs_full_mean_pp"]),
                "validation_retention_vs_all": float(item["gain_retention_vs_all"]),
                "validation_regret_vs_all_pp": float(item["regret_vs_all_pp"]),
                "validation_p_loss_gt5": float(item["p_gain_vs_full_lt_minus5_unit"]),
                "chosen_family_entropy_bits": entropy(family_rates),
                "chosen_candidate_entropy_bits": entropy(candidate_rates),
                "top_candidate": top_candidate,
                "top_candidate_rate": float(top_rate),
                "longitudinal_family_rate": float(family_rates.get("longitudinal", 0.0)),
                "source_only_family_rate": float(family_rates.get("source_only", 0.0)),
                "full_rate": float(family_rates.get("full", 0.0)),
                "best_fixed_policy": str(fixed["best_fixed_by_gain"]),
                "best_fixed_gain_pp": float(fixed["best_fixed_gain_pp"]),
                "validation_minus_best_fixed_pp": float(fixed["validation_minus_best_fixed_pp"]),
                "best_fixed_p_loss_gt5": float(fixed["best_fixed_p_loss_gt5"]),
                "deployment_recommendation": (
                    "fixed_policy"
                    if float(fixed["validation_minus_best_fixed_pp"]) < -0.25
                    else "validation_or_fixed_both_reasonable"
                ),
            }
        )
    return pd.DataFrame(rows)


def regime_recommendations(task_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, policy), group in val_df.groupby(["dataset", "validation_policy"]):
        m10 = group[group["pool_size"] == "10"]
        m20 = group[group["pool_size"] == "20"]
        m40 = group[group["pool_size"] == "40"]
        m10_ret = float(m10["validation_retention_vs_all"].iloc[0]) if not m10.empty else float("nan")
        m20_ret = float(m20["validation_retention_vs_all"].iloc[0]) if not m20.empty else float("nan")
        m40_ret = float(m40["validation_retention_vs_all"].iloc[0]) if not m40.empty else float("nan")
        m20_entropy = float(m20["chosen_family_entropy_bits"].iloc[0]) if not m20.empty else float("nan")
        m20_diff_fixed = float(m20["validation_minus_best_fixed_pp"].iloc[0]) if not m20.empty else float("nan")
        if m20_ret >= 0.9 and m20_diff_fixed > -0.25:
            recommendation = "validation acceptable with m>=20"
        elif m20_ret >= 0.9:
            recommendation = "fixed preferred, validation acceptable for discovery"
        else:
            recommendation = "fixed policy preferred; validation unstable"
        rows.append(
            {
                "dataset": dataset,
                "validation_policy": policy,
                "m10_retention": m10_ret,
                "m20_retention": m20_ret,
                "m40_retention": m40_ret,
                "m20_family_entropy_bits": m20_entropy,
                "m20_validation_minus_fixed_pp": m20_diff_fixed,
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e11_best = pd.read_csv(args.e11_best)
    e12_stieger = load_json(args.e12_stieger)
    e12_lee = load_json(args.e12_lee)
    e15 = load_json(args.e15_summary)
    e16 = load_json(args.e16_summary)

    task_df = task_panel_descriptors(e11_best, e12_stieger, e12_lee)
    val_df = validation_descriptor_rows(e15, e16)
    regime_df = regime_recommendations(task_df, val_df)

    task_df.to_csv(args.output_dir / "task_panel_descriptors.csv", index=False)
    val_df.to_csv(args.output_dir / "validation_stability_descriptors.csv", index=False)
    regime_df.to_csv(args.output_dir / "regime_recommendations.csv", index=False)
    summary = {
        "task_panel_descriptors": task_df.to_dict("records"),
        "validation_stability_descriptors": val_df.to_dict("records"),
        "regime_recommendations": regime_df.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_task_panels": len(task_df),
                "n_validation_rows": len(val_df),
                "n_regime_rows": len(regime_df),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
