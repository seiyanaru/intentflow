"""E27 unified regime table for source-side tangent subspace selection.

This script consolidates the key Stieger, Lee2019, and BNCI results into a
single regime-level table.  It intentionally does not rerun classifiers; it
only reads previously generated CSV summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_e27_unified_regime_table"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def first_row(df: pd.DataFrame, **conditions: Any) -> pd.Series:
    mask = np.ones(len(df), dtype=bool)
    for key, value in conditions.items():
        mask &= df[key].astype(str).to_numpy() == str(value)
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {conditions}, got {len(rows)}")
    return rows.iloc[0]


def row_from_policy(
    *,
    dataset_regime: str,
    feature_regime: str,
    p: str | int | float,
    source_n: str | int | float,
    p_over_n: str | int | float,
    policy_type: str,
    selected_policy: str,
    row: pd.Series,
    ci_col: str = "gain_subject_bootstrap_95ci",
    q05_col: str = "gain_vs_full_q05_unit_pp",
    p_harm_col: str = "p_gain_vs_full_lt_minus5_unit",
    r10_col: str | None = "loss_r10_vs_full_unit_pp",
    source_validation_predictability: str = "",
    interpretation: str = "",
) -> dict[str, object]:
    return {
        "dataset_regime": dataset_regime,
        "feature_regime": feature_regime,
        "p": p,
        "source_n": source_n,
        "p_over_n": p_over_n,
        "policy_type": policy_type,
        "selected_policy": selected_policy,
        "accuracy_mean": float(row["accuracy_mean"]),
        "gain_vs_full_pp": float(row["gain_vs_full_mean_pp"]),
        "gain_95ci": str(row[ci_col]) if ci_col in row else "",
        "q05_gain_pp": float(row[q05_col]) if q05_col in row and pd.notna(row[q05_col]) else np.nan,
        "r10_loss_pp": float(row[r10_col]) if r10_col and r10_col in row and pd.notna(row[r10_col]) else np.nan,
        "p_gain_lt_minus5": float(row[p_harm_col]) if p_harm_col in row and pd.notna(row[p_harm_col]) else np.nan,
        "source_validation_predictability": source_validation_predictability,
        "interpretation": interpretation,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    e16 = read_csv(RESULTS_DIR / "260628_e16_fixed_policy_vs_validation" / "fixed_policy_summary.csv")
    e25 = read_csv(
        RESULTS_DIR
        / "260630_lee2019_exact_nested_subspace_selection_e25_full"
        / "merged"
        / "exact_nested_summary.csv"
    )
    e20 = read_csv(
        RESULTS_DIR
        / "260629_lee2019_channel_dimension_ablation_e20"
        / "sensorimotor20"
        / "summary.csv"
    )
    e18 = read_csv(RESULTS_DIR / "260628_bnci2014_001_fraction_sweep_e18_dense" / "summary.csv")
    e18b = read_csv(
        RESULTS_DIR
        / "260628_bnci2014_001_source_size_ablation_e18b_m8_r32"
        / "summary.csv"
    )
    e19 = read_csv(
        RESULTS_DIR
        / "260629_lee2019_source_size_ablation_e19_primary_chunks"
        / "merged"
        / "summary.csv"
    )
    e26 = json.loads(
        (
            RESULTS_DIR
            / "260701_lee2019_nested_predictability_e26"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )

    rows: list[dict[str, object]] = []

    # Stieger deployment-level policies.
    rows.append(
        row_from_policy(
            dataset_regime="Stieger2021 primary LR+UD",
            feature_regime="posterior ensemble: broad_all60 + fb_sensorimotor21",
            p="1830 + 693 ensemble",
            source_n="median 132-138",
            p_over_n="not single-p comparable",
            policy_type="condition-known fixed",
            selected_policy="LR longitudinal_q0p05 / UD longitudinal_q0p20",
            row=first_row(e16, dataset_policy="Stieger__fixed__condition_long_lr05_ud20"),
            source_validation_predictability="finite source-pool validation approaches but does not beat fixed",
            interpretation="longitudinal stability is useful in this multi-session/source-scarce ensemble regime",
        )
    )
    rows.append(
        row_from_policy(
            dataset_regime="Stieger2021 primary LR+UD",
            feature_regime="posterior ensemble: broad_all60 + fb_sensorimotor21",
            p="1830 + 693 ensemble",
            source_n="median 132-138",
            p_over_n="not single-p comparable",
            policy_type="global fixed",
            selected_policy="longitudinal_q0p10",
            row=first_row(e16, dataset_policy="Stieger__fixed__longitudinal_q0p10"),
            source_validation_predictability="finite source-pool validation not necessary once global policy is known",
            interpretation="single robust global Stieger policy; lower gain than condition-known fixed",
        )
    )

    # Lee all-channel fixed and nested.
    rows.append(
        row_from_policy(
            dataset_regime="Lee2019 LR all62",
            feature_regime="single tangent, 62 channels",
            p=1953,
            source_n=100,
            p_over_n=19.53,
            policy_type="fixed",
            selected_policy="source_only_q0p25",
            row=first_row(e25, method="fixed__source_only_q0p25"),
            ci_col="gain_vs_full_subject_bootstrap_95ci",
            q05_col="gain_vs_full_q05_pp",
            p_harm_col="p_gain_vs_full_lt_minus5",
            r10_col="loss_r10_vs_full_pp",
            source_validation_predictability=(
                "E26 inner/outer Spearman "
                f"{e26['inner_outer_spearman_nonfull']['mean']:.3f}; "
                f"oracle match {e26['nested_choice_matches_outer_rate']['mean']:.3f}"
            ),
            interpretation="broad all-channel denoising; subject-wise candidate selection is under-identified",
        )
    )
    rows.append(
        row_from_policy(
            dataset_regime="Lee2019 LR all62",
            feature_regime="single tangent, 62 channels",
            p=1953,
            source_n=100,
            p_over_n=19.53,
            policy_type="nested source-validation",
            selected_policy="nested risk-constrained candidate choice",
            row=first_row(e25, method="nested_risk"),
            ci_col="gain_vs_full_subject_bootstrap_95ci",
            q05_col="gain_vs_full_q05_pp",
            p_harm_col="p_gain_vs_full_lt_minus5",
            r10_col="loss_r10_vs_full_pp",
            source_validation_predictability="fails to beat fixed source_only_q0p25",
            interpretation="overreacts to small source-validation margins and selects longitudinal_q0p10 too often",
        )
    )

    # Lee sensorimotor low-dimensional ablation.
    best_e20 = e20[e20["method"] != "full_q1p00"].sort_values(
        "gain_vs_full_mean_pp", ascending=False
    ).iloc[0]
    rows.append(
        row_from_policy(
            dataset_regime="Lee2019 LR sensorimotor20",
            feature_regime="single tangent, 20 sensorimotor channels",
            p=210,
            source_n=100,
            p_over_n=2.10,
            policy_type="best mean fixed sweep",
            selected_policy=str(best_e20["method"]),
            row=best_e20,
            ci_col="gain_vs_full_subject_bootstrap_95ci",
            q05_col="gain_vs_full_q05_pp",
            p_harm_col="p_gain_vs_full_lt_minus5",
            r10_col="loss_r10_vs_full_pp",
            source_validation_predictability="not evaluated; compact selection loses robust signal after channel restriction",
            interpretation="full tangent already compact enough; all62 gain is not motor-channel restriction",
        )
    )

    # BNCI full-source and source-scarce simulation.
    best_e18 = e18[e18["method"] != "full_q1p00"].sort_values(
        "gain_vs_full_mean_pp", ascending=False
    ).iloc[0]
    rows.append(
        row_from_policy(
            dataset_regime="BNCI2014_001 LR full-source",
            feature_regime="single tangent, 22 channels",
            p=253,
            source_n=144,
            p_over_n=1.76,
            policy_type="best mean fixed sweep",
            selected_policy=str(best_e18["method"]),
            row=best_e18,
            ci_col="gain_vs_full_subject_bootstrap_95ci",
            q05_col="gain_vs_full_q05_pp",
            p_harm_col="p_gain_vs_full_lt_minus5",
            r10_col="loss_r10_vs_full_pp",
            source_validation_predictability="not evaluated; aggressive compactness is harmful",
            interpretation="low-dimensional/source-richer regime; full tangent already strong",
        )
    )
    best_e18b = e18b[e18b["method"] != "full_q1p00"].sort_values(
        "gain_vs_full_mean_pp", ascending=False
    ).iloc[0]
    stable_e18b = first_row(e18b, method="longitudinal_q0p70")
    rows.append(
        row_from_policy(
            dataset_regime="BNCI2014_001 LR m8/class",
            feature_regime="single tangent, 22 channels, source-scarce simulation",
            p=253,
            source_n=16,
            p_over_n=15.81,
            policy_type="best mean fixed sweep",
            selected_policy=str(best_e18b["method"]),
            row=best_e18b,
            ci_col="gain_vs_full_subject_bootstrap_95ci",
            q05_col="gain_vs_full_q05_pp",
            p_harm_col="p_subject_gain_lt_minus5",
            r10_col=None,
            source_validation_predictability="not evaluated; source-scarce simulation only",
            interpretation="compact selection becomes mildly useful again when p/n is high",
        )
    )
    rows.append(
        row_from_policy(
            dataset_regime="BNCI2014_001 LR m8/class",
            feature_regime="single tangent, 22 channels, source-scarce simulation",
            p=253,
            source_n=16,
            p_over_n=15.81,
            policy_type="stable conservative fixed sweep",
            selected_policy="longitudinal_q0p70",
            row=stable_e18b,
            ci_col="gain_vs_full_subject_bootstrap_95ci",
            q05_col="gain_vs_full_q05_pp",
            p_harm_col="p_subject_gain_lt_minus5",
            r10_col=None,
            source_validation_predictability="not evaluated; stable small positive effect",
            interpretation="smaller effect than q0.10 but lower-variance confirmation",
        )
    )

    regime_df = pd.DataFrame(rows)
    regime_df.to_csv(args.output_dir / "unified_regime_table.csv", index=False)

    # Compactness trend summaries from E19.
    trend_rows = []
    for method in ["source_only_q0p25", "longitudinal_q0p10"]:
        sub = e19[e19["method"] == method].sort_values("source_total")
        trend_rows.append(
            {
                "dataset_regime": "Lee2019 LR all62 source-size ablation",
                "method": method,
                "source_totals": ",".join(map(str, sub["source_total"].astype(int).tolist())),
                "p_over_n_values": ",".join(f"{v:.2f}" for v in sub["p_over_source_total"].tolist()),
                "gains_pp": ",".join(f"{v:.3f}" for v in sub["gain_vs_full_mean_pp"].tolist()),
                "p_harm_values": ",".join(f"{v:.3f}" for v in sub["p_subject_gain_lt_minus5"].tolist()),
                "interpretation": (
                    "source_only remains broadly strong"
                    if method == "source_only_q0p25"
                    else "longitudinal strengthens in source-scarce high-p/n settings but has worse tail risk"
                ),
            }
        )
    pd.DataFrame(trend_rows).to_csv(args.output_dir / "lee_source_size_trend_summary.csv", index=False)

    verdict = {
        "keep": [
            "source-side tangent subspace selection is a real accuracy lever in high-dimensional/source-scarce regimes",
            "Lee2019 all62 fixed source_only_q0p25 is the current robust policy",
            "Stieger condition-known longitudinal compactness is the strongest multi-session policy",
        ],
        "retract": [
            "subject-wise zero-label nested selection as a main Lee2019 method",
            "fixed task-family policies transfer across all LR datasets",
            "sensorimotor channel restriction explains the Lee2019 gain",
        ],
        "next_method_hypothesis": (
            "If we want a better method, the new information source must be unlabeled target-prefix "
            "feature reliability/shift, not another source-validation selector."
        ),
    }
    (args.output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_regime_rows": int(len(regime_df)),
                "table": str(args.output_dir / "unified_regime_table.csv"),
                "trend": str(args.output_dir / "lee_source_size_trend_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
