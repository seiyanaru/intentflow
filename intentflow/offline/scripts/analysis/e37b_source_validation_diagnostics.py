"""E37-B diagnostics: does source validation predict useful weighting?

This script links inner source-validation signals to outer held-subject gains
for the frozen E36 candidate set:

* lda_full
* ridge_source_rank_g2_a100
* ridge_longitudinal_rank_g2_a100

It answers:

1. Does inner source/longitudinal gain correlate with outer gain?
2. Does inner validation pick the outer-best weighted family?
3. Does the E36 guard switch when weighted Ridge is useful and stay anchored
   when it is not?
4. Where are the failure cases?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_e37b_source_validation_diagnostics"

DEFAULT_E33_DIRS = [
    RESULTS_DIR / "260701_e33_lee2019_sensorimotor20_nested_selector",
    RESULTS_DIR / "260701_e33_bnci2014_001_full_source_nested_selector",
    RESULTS_DIR / "260701_e33_bnci2014_001_m8_nested_selector",
]

LDA = "lda_full"
SOURCE = "ridge_source_rank_g2_a100"
LONG = "ridge_longitudinal_rank_g2_a100"
WEIGHTED = [SOURCE, LONG]
ALL_CANDIDATES = [LDA, SOURCE, LONG]
MARGIN_PP = 0.5
RISK_LIMIT = 0.20
HARM_THRESHOLD_PP = -5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e33-dirs", nargs="*", type=Path, default=DEFAULT_E33_DIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bootstrap_ci(values: Iterable[float], rng: np.random.Generator, repeats: int) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(arr, size=(int(repeats), arr.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def infer_regime_label(validation_df: pd.DataFrame, input_dir: Path) -> str:
    if "regime_label" in validation_df.columns and len(validation_df["regime_label"].dropna()) > 0:
        return str(validation_df["regime_label"].dropna().iloc[0])
    return input_dir.name


def gain_stats(gain: np.ndarray) -> dict[str, float]:
    gain = np.asarray(gain, dtype=np.float64)
    return {
        "mean": float(np.mean(gain)),
        "q05": float(np.quantile(gain, 0.05)),
        "p_harm": float(np.mean(gain < HARM_THRESHOLD_PP)),
    }


def choose_from_inner(source_gain_mean: float, source_p_harm: float, long_gain_mean: float, long_p_harm: float) -> tuple[str, str, float, float]:
    if source_gain_mean >= long_gain_mean:
        best_candidate = SOURCE
        best_gain = source_gain_mean
        best_p_harm = source_p_harm
    else:
        best_candidate = LONG
        best_gain = long_gain_mean
        best_p_harm = long_p_harm
    if best_gain >= MARGIN_PP and best_p_harm <= RISK_LIMIT:
        selected = best_candidate
    else:
        selected = LDA
    return selected, best_candidate, float(best_gain), float(best_p_harm)


def inner_summary(validation_df: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    validation_df = validation_df[validation_df["candidate"].isin(ALL_CANDIDATES)].copy()
    rows = []
    if mode == "repeat":
        grouped = validation_df.groupby(["outer_subject", "repeat"], sort=True)
    elif mode == "stable":
        grouped = validation_df.groupby(["outer_subject"], sort=True)
    else:
        raise ValueError(mode)

    for group_key, sub in grouped:
        if mode == "repeat":
            subject, repeat = group_key
            pivot_index = ["inner_subject"]
        else:
            subject = int(group_key[0] if isinstance(group_key, tuple) else group_key)
            repeat = -1
            pivot_index = ["repeat", "inner_subject"]
        pivot = sub.pivot_table(index=pivot_index, columns="candidate", values="accuracy", aggfunc="mean")
        if not all(candidate in pivot.columns for candidate in ALL_CANDIDATES):
            continue
        source_gain = pivot[SOURCE].to_numpy(dtype=np.float64) - pivot[LDA].to_numpy(dtype=np.float64)
        long_gain = pivot[LONG].to_numpy(dtype=np.float64) - pivot[LDA].to_numpy(dtype=np.float64)
        s = gain_stats(source_gain)
        l = gain_stats(long_gain)
        selected, best_candidate, best_gain, best_p_harm = choose_from_inner(
            s["mean"], s["p_harm"], l["mean"], l["p_harm"]
        )
        rows.append(
            {
                "audit_mode": mode,
                "subject": int(subject),
                "repeat": int(repeat),
                "inner_n_validation_points": int(len(pivot)),
                "inner_source_gain_mean_pp": s["mean"],
                "inner_source_gain_q05_pp": s["q05"],
                "inner_source_p_harm": s["p_harm"],
                "inner_long_gain_mean_pp": l["mean"],
                "inner_long_gain_q05_pp": l["q05"],
                "inner_long_p_harm": l["p_harm"],
                "inner_source_minus_long_gain_pp": float(s["mean"] - l["mean"]),
                "inner_best_weighted_candidate": best_candidate,
                "inner_best_weighted_gain_mean_pp": best_gain,
                "inner_best_weighted_p_harm": best_p_harm,
                "e36_selected_candidate_recomputed": selected,
                "e36_pass_margin_recomputed": bool(best_gain >= MARGIN_PP),
                "e36_pass_risk_recomputed": bool(best_p_harm <= RISK_LIMIT),
            }
        )
    return pd.DataFrame(rows)


def outer_summary(outer_df: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    outer_df = outer_df[outer_df["method"].isin(ALL_CANDIDATES)].copy()
    rows = []
    if mode == "repeat":
        grouped = outer_df.groupby(["subject", "repeat"], sort=True)
    elif mode == "stable":
        grouped = outer_df.groupby(["subject"], sort=True)
    else:
        raise ValueError(mode)

    for group_key, sub in grouped:
        if mode == "repeat":
            subject, repeat = group_key
            pivot = sub.pivot_table(index=["subject", "repeat"], columns="method", values="accuracy", aggfunc="mean")
        else:
            subject = int(group_key[0] if isinstance(group_key, tuple) else group_key)
            repeat = -1
            pivot = sub.groupby("method", as_index=True)["accuracy"].mean().to_frame().T
        if not all(candidate in pivot.columns for candidate in ALL_CANDIDATES):
            continue
        lda_acc = float(pivot[LDA].iloc[0])
        source_acc = float(pivot[SOURCE].iloc[0])
        long_acc = float(pivot[LONG].iloc[0])
        source_gain = source_acc - lda_acc
        long_gain = long_acc - lda_acc
        if source_gain >= long_gain:
            best_weighted = SOURCE
            best_weighted_gain = source_gain
        else:
            best_weighted = LONG
            best_weighted_gain = long_gain
        gains = {
            LDA: 0.0,
            SOURCE: source_gain,
            LONG: long_gain,
        }
        best_overall = max(gains, key=gains.get)
        rows.append(
            {
                "audit_mode": mode,
                "subject": int(subject),
                "repeat": int(repeat),
                "outer_lda_accuracy": lda_acc,
                "outer_source_accuracy": source_acc,
                "outer_long_accuracy": long_acc,
                "outer_source_gain_pp": float(source_gain),
                "outer_long_gain_pp": float(long_gain),
                "outer_source_minus_long_gain_pp": float(source_gain - long_gain),
                "outer_best_weighted_candidate": best_weighted,
                "outer_best_weighted_gain_pp": float(best_weighted_gain),
                "outer_best_overall_candidate": best_overall,
                "outer_best_overall_gain_pp": float(gains[best_overall]),
            }
        )
    return pd.DataFrame(rows)


def build_records(input_dir: Path) -> pd.DataFrame:
    validation_df = pd.read_csv(input_dir / "nested_inner_validation_records.csv")
    outer_df = pd.read_csv(input_dir / "nested_outer_records.csv")
    regime = infer_regime_label(validation_df, input_dir)
    frames = []
    for mode in ["repeat", "stable"]:
        inner = inner_summary(validation_df, mode=mode)
        outer = outer_summary(outer_df, mode=mode)
        merged = inner.merge(outer, on=["audit_mode", "subject", "repeat"], how="inner")
        merged["regime_label"] = regime
        selected_gain = []
        selected_acc = []
        for row in merged.itertuples():
            selected = str(row.e36_selected_candidate_recomputed)
            if selected == LDA:
                selected_gain.append(0.0)
                selected_acc.append(float(row.outer_lda_accuracy))
            elif selected == SOURCE:
                selected_gain.append(float(row.outer_source_gain_pp))
                selected_acc.append(float(row.outer_source_accuracy))
            elif selected == LONG:
                selected_gain.append(float(row.outer_long_gain_pp))
                selected_acc.append(float(row.outer_long_accuracy))
            else:
                selected_gain.append(float("nan"))
                selected_acc.append(float("nan"))
        merged["outer_selected_gain_pp"] = selected_gain
        merged["outer_selected_accuracy"] = selected_acc
        merged["inner_best_matches_outer_best_weighted"] = (
            merged["inner_best_weighted_candidate"] == merged["outer_best_weighted_candidate"]
        )
        merged["selected_matches_outer_best_overall"] = (
            merged["e36_selected_candidate_recomputed"] == merged["outer_best_overall_candidate"]
        )
        merged["selected_is_lda"] = merged["e36_selected_candidate_recomputed"] == LDA
        merged["selected_weighted"] = ~merged["selected_is_lda"]
        merged["selected_harm_lt_minus5"] = merged["outer_selected_gain_pp"] < HARM_THRESHOLD_PP
        merged["selected_negative"] = merged["outer_selected_gain_pp"] < 0.0
        merged["missed_large_weighted_opportunity"] = (
            merged["selected_is_lda"] & (merged["outer_best_weighted_gain_pp"] >= 1.5)
        )
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def corr(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return float("nan")
    return float(valid["x"].corr(valid["y"], method=method))


def summarize_correlations(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("source_gain", "inner_source_gain_mean_pp", "outer_source_gain_pp"),
        ("long_gain", "inner_long_gain_mean_pp", "outer_long_gain_pp"),
        ("best_weighted_gain", "inner_best_weighted_gain_mean_pp", "outer_best_weighted_gain_pp"),
        ("source_minus_long", "inner_source_minus_long_gain_pp", "outer_source_minus_long_gain_pp"),
    ]
    for (regime, mode), sub in records.groupby(["regime_label", "audit_mode"], sort=True):
        for name, inner_col, outer_col in pairs:
            rows.append(
                {
                    "regime_label": regime,
                    "audit_mode": mode,
                    "signal": name,
                    "n": int(sub[[inner_col, outer_col]].dropna().shape[0]),
                    "pearson": corr(sub[inner_col], sub[outer_col], "pearson"),
                    "spearman": corr(sub[inner_col], sub[outer_col], "spearman"),
                    "inner_mean": float(sub[inner_col].mean()),
                    "outer_mean": float(sub[outer_col].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_selection(records: pd.DataFrame, *, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (regime, mode), sub in records.groupby(["regime_label", "audit_mode"], sort=True):
        selected_gain = sub["outer_selected_gain_pp"].to_numpy(dtype=np.float64)
        best_weighted_gain = sub["outer_best_weighted_gain_pp"].to_numpy(dtype=np.float64)
        best_overall_gain = sub["outer_best_overall_gain_pp"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "regime_label": regime,
                "audit_mode": mode,
                "n": int(len(sub)),
                "selected_gain_mean_pp": float(np.mean(selected_gain)),
                "selected_gain_95ci": bootstrap_ci(selected_gain, rng, bootstrap),
                "selected_gain_q05_pp": float(np.quantile(selected_gain, 0.05)),
                "p_selected_gain_lt_minus5": float(np.mean(selected_gain < -5.0)),
                "p_selected_gain_lt_0": float(np.mean(selected_gain < 0.0)),
                "outer_best_weighted_gain_mean_pp": float(np.mean(best_weighted_gain)),
                "outer_best_overall_gain_mean_pp": float(np.mean(best_overall_gain)),
                "inner_best_matches_outer_best_weighted": float(
                    np.mean(sub["inner_best_matches_outer_best_weighted"])
                ),
                "selected_matches_outer_best_overall": float(
                    np.mean(sub["selected_matches_outer_best_overall"])
                ),
                "p_selected_lda": float(np.mean(sub["selected_is_lda"])),
                "p_selected_weighted": float(np.mean(sub["selected_weighted"])),
                "p_missed_large_weighted_opportunity": float(
                    np.mean(sub["missed_large_weighted_opportunity"])
                ),
                "selection_counts": json.dumps(
                    sub["e36_selected_candidate_recomputed"].value_counts().to_dict(),
                    sort_keys=True,
                ),
                "outer_best_overall_counts": json.dumps(
                    sub["outer_best_overall_candidate"].value_counts().to_dict(),
                    sort_keys=True,
                ),
                "outer_best_weighted_counts": json.dumps(
                    sub["outer_best_weighted_candidate"].value_counts().to_dict(),
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def failure_cases(records: pd.DataFrame) -> pd.DataFrame:
    failure = records[
        records["selected_harm_lt_minus5"]
        | records["selected_negative"]
        | records["missed_large_weighted_opportunity"]
        | (~records["selected_matches_outer_best_overall"])
    ].copy()
    priority_cols = [
        "regime_label",
        "audit_mode",
        "subject",
        "repeat",
        "e36_selected_candidate_recomputed",
        "outer_selected_gain_pp",
        "outer_best_overall_candidate",
        "outer_best_overall_gain_pp",
        "outer_best_weighted_candidate",
        "outer_best_weighted_gain_pp",
        "inner_best_weighted_candidate",
        "inner_best_weighted_gain_mean_pp",
        "inner_best_weighted_p_harm",
        "inner_source_gain_mean_pp",
        "inner_long_gain_mean_pp",
        "selected_harm_lt_minus5",
        "selected_negative",
        "missed_large_weighted_opportunity",
        "selected_matches_outer_best_overall",
    ]
    return failure[priority_cols].sort_values(
        ["regime_label", "audit_mode", "selected_harm_lt_minus5", "selected_negative"],
        ascending=[True, True, False, False],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = pd.concat([build_records(path) for path in args.e33_dirs], ignore_index=True)
    correlations = summarize_correlations(records)
    selection = summarize_selection(records, bootstrap=int(args.bootstrap), seed=int(args.seed))
    failures = failure_cases(records)

    records.to_csv(args.output_dir / "e37b_diagnostic_records.csv", index=False)
    correlations.to_csv(args.output_dir / "e37b_correlation_summary.csv", index=False)
    selection.to_csv(args.output_dir / "e37b_selection_summary.csv", index=False)
    failures.to_csv(args.output_dir / "e37b_failure_cases.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "inputs": [str(path) for path in args.e33_dirs],
                "frozen_rule": {
                    "margin_pp": MARGIN_PP,
                    "risk_limit": RISK_LIMIT,
                    "harm_threshold_pp": HARM_THRESHOLD_PP,
                    "candidates": ALL_CANDIDATES,
                },
                "correlations": correlations.to_dict(orient="records"),
                "selection_summary": selection.to_dict(orient="records"),
                "n_failure_cases": int(len(failures)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "diagnostic_records": str(args.output_dir / "e37b_diagnostic_records.csv"),
                "correlation_summary": str(args.output_dir / "e37b_correlation_summary.csv"),
                "selection_summary": str(args.output_dir / "e37b_selection_summary.csv"),
                "failure_cases": str(args.output_dir / "e37b_failure_cases.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

