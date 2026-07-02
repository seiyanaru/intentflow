"""E36 frozen audit for guarded reliability-weighted Ridge.

This is the frozen protocol after E35.  It intentionally does not sweep
margins, candidates, or risk limits.

Frozen rule:

* baseline/source anchor: lda_full
* weighted candidates:
  - ridge_source_rank_g2_a100
  - ridge_longitudinal_rank_g2_a100
* switch to the best weighted candidate only if source-validation shows:
  - mean gain over lda_full >= +0.5 percentage points
  - P(gain < -5pp) <= 0.20

The script reuses E33 inner/outer records.  It is a frozen audit over already
computed candidate scores, not a new hyperparameter search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_e36_frozen_guarded_reliability_audit"

BASELINE = "lda_full"
WEIGHTED_CANDIDATES = [
    "ridge_source_rank_g2_a100",
    "ridge_longitudinal_rank_g2_a100",
]
REFERENCE_METHODS = [
    BASELINE,
    "lda_long_q70",
    "ridge_source_rank_g2_a100",
    "ridge_longitudinal_rank_g2_a100",
]
FROZEN_MARGIN_PP = 0.5
FROZEN_RISK_LIMIT = 0.20
FROZEN_HARM_THRESHOLD_PP = -5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
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


def infer_regime_label(input_dir: Path, outer_df: pd.DataFrame) -> str:
    if "regime_label" in outer_df.columns and len(outer_df["regime_label"].dropna()) > 0:
        return str(outer_df["regime_label"].dropna().iloc[0])
    return input_dir.name


def choose_frozen(validation_df: pd.DataFrame) -> tuple[str, dict[str, float | str]]:
    keep = [BASELINE] + WEIGHTED_CANDIDATES
    validation_df = validation_df[validation_df["candidate"].isin(keep)].copy()
    merge_keys = ["inner_subject"]
    if "repeat" in validation_df.columns:
        merge_keys = ["repeat", "inner_subject"]
    baseline = validation_df[validation_df["candidate"] == BASELINE][
        merge_keys + ["accuracy"]
    ].rename(columns={"accuracy": "baseline_accuracy"})

    rows = []
    for candidate in WEIGHTED_CANDIDATES:
        sub = validation_df[validation_df["candidate"] == candidate]
        merged = sub.merge(baseline, on=merge_keys, how="left")
        gain = merged["accuracy"].to_numpy(dtype=np.float64) - merged[
            "baseline_accuracy"
        ].to_numpy(dtype=np.float64)
        rows.append(
            {
                "candidate": candidate,
                "inner_mean_accuracy": float(merged["accuracy"].mean()),
                "inner_gain_vs_baseline_mean_pp": float(np.mean(gain)),
                "inner_gain_vs_baseline_q05_pp": float(np.quantile(gain, 0.05)),
                "inner_p_gain_vs_baseline_lt_minus5": float(
                    np.mean(gain < FROZEN_HARM_THRESHOLD_PP)
                ),
            }
        )
    weighted = pd.DataFrame(rows).sort_values(
        ["inner_gain_vs_baseline_mean_pp", "inner_mean_accuracy"],
        ascending=[False, False],
    )
    best = weighted.iloc[0]
    pass_margin = float(best["inner_gain_vs_baseline_mean_pp"]) >= FROZEN_MARGIN_PP
    pass_risk = float(best["inner_p_gain_vs_baseline_lt_minus5"]) <= FROZEN_RISK_LIMIT
    if pass_margin and pass_risk:
        selected = str(best["candidate"])
    else:
        selected = BASELINE
    return selected, {
        "inner_best_weighted_candidate": str(best["candidate"]),
        "inner_best_weighted_gain_mean_pp": float(best["inner_gain_vs_baseline_mean_pp"]),
        "inner_best_weighted_gain_q05_pp": float(best["inner_gain_vs_baseline_q05_pp"]),
        "inner_best_weighted_p_harm": float(best["inner_p_gain_vs_baseline_lt_minus5"]),
        "pass_margin": float(pass_margin),
        "pass_risk": float(pass_risk),
    }


def build_frozen_records(
    validation_df: pd.DataFrame,
    outer_df: pd.DataFrame,
    *,
    regime_label: str,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_records: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []

    if mode == "repeat":
        grouped = validation_df.groupby(["outer_subject", "repeat"], sort=True)
    elif mode == "stable":
        grouped = validation_df.groupby("outer_subject", sort=True)
    else:
        raise ValueError(mode)

    for key, subject_validation in grouped:
        if mode == "repeat":
            subject, repeat = key
            candidate, meta = choose_frozen(subject_validation)
            selected_outer = outer_df[
                (outer_df["subject"] == int(subject))
                & (outer_df["repeat"] == int(repeat))
                & (outer_df["method"] == candidate)
            ].copy()
            repeats_for_log = [int(repeat)]
        else:
            subject = int(key)
            candidate, meta = choose_frozen(subject_validation)
            selected_outer = outer_df[
                (outer_df["subject"] == int(subject)) & (outer_df["method"] == candidate)
            ].copy()
            repeats_for_log = sorted(map(int, selected_outer["repeat"].unique()))

        method = f"e36_frozen_guarded_{mode}"
        selected_outer["method"] = method
        selected_outer["selected_candidate"] = candidate
        selected_outer["audit_mode"] = mode
        selected_records.extend(selected_outer.to_dict(orient="records"))
        for repeat_value in repeats_for_log:
            selection_records.append(
                {
                    "regime_label": regime_label,
                    "audit_mode": mode,
                    "subject": int(subject),
                    "repeat": int(repeat_value),
                    "method": method,
                    "selected_candidate": candidate,
                    **meta,
                }
            )

    return pd.DataFrame(selected_records), pd.DataFrame(selection_records)


def summarize(
    records: pd.DataFrame,
    *,
    regime_label: str,
    audit_mode: str,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = sorted(map(int, records["subject"].unique()))
    baseline_lookup = {
        (int(row.subject), int(row.repeat)): float(row.accuracy)
        for row in records[records["method"] == BASELINE].itertuples()
    }
    rows = []
    for method, sub in records.groupby("method", sort=True):
        by_subject_acc: dict[int, list[float]] = {}
        by_subject_gain: dict[int, list[float]] = {}
        selection_counts: dict[str, int] = {}
        for row in sub.itertuples():
            key = (int(row.subject), int(row.repeat))
            by_subject_acc.setdefault(int(row.subject), []).append(float(row.accuracy))
            by_subject_gain.setdefault(int(row.subject), []).append(
                float(row.accuracy) - baseline_lookup[key]
            )
            selected = getattr(row, "selected_candidate", "")
            if isinstance(selected, str) and selected:
                selection_counts[selected] = selection_counts.get(selected, 0) + 1
        acc = np.asarray([np.mean(by_subject_acc[s]) for s in subjects if s in by_subject_acc])
        gain = np.asarray([np.mean(by_subject_gain[s]) for s in subjects if s in by_subject_gain])
        rows.append(
            {
                "regime_label": regime_label,
                "audit_mode": audit_mode,
                "method": method,
                "n_subjects": int(len(acc)),
                "accuracy_mean": float(acc.mean()),
                "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, bootstrap),
                "gain_vs_lda_full_mean_pp": float(gain.mean()),
                "gain_vs_lda_full_95ci": bootstrap_ci(gain, rng, bootstrap),
                "gain_vs_lda_full_q05_pp": float(np.quantile(gain, 0.05)),
                "p_gain_vs_lda_full_lt_minus5": float(np.mean(gain < -5.0)),
                "selection_counts": json.dumps(selection_counts, sort_keys=True),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)


def pass_fail(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = summary[summary["method"].str.startswith("e36_frozen_guarded")]
    for row in selected.itertuples():
        regime = str(row.regime_label)
        mode = str(row.audit_mode)
        gain = float(row.gain_vs_lda_full_mean_pp)
        risk = float(row.p_gain_vs_lda_full_lt_minus5)
        if regime == "lee2019_sensorimotor20" and mode == "stable":
            passed = gain >= 1.5 and risk <= 0.12
            criterion = "gain>=+1.5pp and P(harm<-5)<=12%"
        elif regime == "bnci2014_001_m8_per_class" and mode == "repeat":
            passed = gain >= 1.5 and risk <= 0.05
            criterion = "repeat gain>=+1.5pp and P(harm<-5)<=5%"
        elif regime == "bnci2014_001_full_source" and mode == "stable":
            passed = gain >= -0.2 and risk <= 0.05
            criterion = "gain>=-0.2pp and P(harm<-5)<=5%"
        else:
            passed = True
            criterion = "reported, not primary pass criterion"
        rows.append(
            {
                "regime_label": regime,
                "audit_mode": mode,
                "method": str(row.method),
                "gain_vs_lda_full_mean_pp": gain,
                "p_gain_vs_lda_full_lt_minus5": risk,
                "criterion": criterion,
                "pass": bool(passed),
            }
        )
    return pd.DataFrame(rows)


def process_input(
    input_dir: Path,
    *,
    output_dir: Path,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_df = pd.read_csv(input_dir / "nested_inner_validation_records.csv")
    outer_df = pd.read_csv(input_dir / "nested_outer_records.csv")
    regime_label = infer_regime_label(input_dir, outer_df)
    fixed = outer_df[outer_df["method"].isin(REFERENCE_METHODS)].copy()
    fixed["selected_candidate"] = fixed.get("selected_candidate", "")
    fixed["audit_mode"] = "reference"

    all_summaries = []
    all_selected_records = []
    all_selection_records = []
    for mode in ["repeat", "stable"]:
        selected_df, selection_df = build_frozen_records(
            validation_df,
            outer_df,
            regime_label=regime_label,
            mode=mode,
        )
        all_selected_records.append(selected_df)
        all_selection_records.append(selection_df)
        records_for_summary = pd.concat([fixed, selected_df], ignore_index=True)
        all_summaries.append(
            summarize(
                records_for_summary,
                regime_label=regime_label,
                audit_mode=mode,
                bootstrap=int(bootstrap),
                seed=int(seed),
            )
        )

    regime_dir = output_dir / regime_label
    regime_dir.mkdir(parents=True, exist_ok=True)
    selected_records_df = pd.concat(all_selected_records, ignore_index=True)
    selection_records_df = pd.concat(all_selection_records, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    selected_records_df.to_csv(regime_dir / "e36_frozen_outer_records.csv", index=False)
    selection_records_df.to_csv(regime_dir / "e36_frozen_selection_records.csv", index=False)
    summary_df.to_csv(regime_dir / "e36_frozen_summary.csv", index=False)
    return summary_df, selected_records_df, selection_records_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    selected_records = []
    selection_records = []
    for input_dir in args.input_dirs:
        summary_df, selected_df, selection_df = process_input(
            input_dir,
            output_dir=args.output_dir,
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        )
        summaries.append(summary_df)
        selected_records.append(selected_df)
        selection_records.append(selection_df)

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_selected = pd.concat(selected_records, ignore_index=True)
    combined_selection = pd.concat(selection_records, ignore_index=True)
    pass_fail_df = pass_fail(combined_summary)

    combined_summary.to_csv(args.output_dir / "e36_frozen_combined_summary.csv", index=False)
    combined_selected.to_csv(args.output_dir / "e36_frozen_combined_outer_records.csv", index=False)
    combined_selection.to_csv(args.output_dir / "e36_frozen_combined_selection_records.csv", index=False)
    pass_fail_df.to_csv(args.output_dir / "e36_pass_fail.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "frozen_rule": {
                    "baseline": BASELINE,
                    "weighted_candidates": WEIGHTED_CANDIDATES,
                    "margin_pp": FROZEN_MARGIN_PP,
                    "risk_limit": FROZEN_RISK_LIMIT,
                    "harm_threshold_pp": FROZEN_HARM_THRESHOLD_PP,
                },
                "input_dirs": [str(path) for path in args.input_dirs],
                "combined_summary": combined_summary.to_dict(orient="records"),
                "pass_fail": pass_fail_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "combined_summary": str(args.output_dir / "e36_frozen_combined_summary.csv"),
                "pass_fail": str(args.output_dir / "e36_pass_fail.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

