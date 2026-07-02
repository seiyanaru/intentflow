"""E38 regime-level reliability-family rule.

E37-B showed that source validation is weak as a subject-level benefit
predictor, but useful as a regime/cohort-level signal.  E38 therefore tests a
simpler regime-level family rule:

1. Aggregate source-validation evidence at the regime level.
2. Choose one family among:
   - lda_full
   - ridge_source_rank_g2_a100
   - ridge_longitudinal_rank_g2_a100
3. Apply that family broadly, optionally keeping the E36 source-anchor guard
   but replacing local source-vs-long family switching with the regime-level
   weighted family.

This is a diagnostic/simplification experiment.  The global regime rule is not
an independent external validation because it is chosen on the same regime's
source-validation records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_e38_regime_level_family_rule"
DEFAULT_E33_DIRS = [
    RESULTS_DIR / "260701_e33_lee2019_sensorimotor20_nested_selector",
    RESULTS_DIR / "260701_e33_bnci2014_001_full_source_nested_selector",
    RESULTS_DIR / "260701_e33_bnci2014_001_m8_nested_selector",
]

LDA = "lda_full"
SOURCE = "ridge_source_rank_g2_a100"
LONG = "ridge_longitudinal_rank_g2_a100"
CANDIDATES = [LDA, SOURCE, LONG]
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


def infer_regime(validation_df: pd.DataFrame, path: Path) -> str:
    if "regime_label" in validation_df.columns and len(validation_df["regime_label"].dropna()) > 0:
        return str(validation_df["regime_label"].dropna().iloc[0])
    return path.name


def gain_stats(gain: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(gain)),
        "q05": float(np.quantile(gain, 0.05)),
        "p_harm": float(np.mean(gain < HARM_THRESHOLD_PP)),
    }


def choose_from_gains(source_gain: np.ndarray, long_gain: np.ndarray) -> dict[str, object]:
    source = gain_stats(source_gain)
    long = gain_stats(long_gain)
    if source["mean"] >= long["mean"]:
        best = SOURCE
        best_stats = source
    else:
        best = LONG
        best_stats = long
    if best_stats["mean"] >= MARGIN_PP and best_stats["p_harm"] <= RISK_LIMIT:
        selected = best
    else:
        selected = LDA
    return {
        "selected_candidate": selected,
        "best_weighted_candidate": best,
        "source_gain_mean_pp": source["mean"],
        "source_gain_q05_pp": source["q05"],
        "source_p_harm": source["p_harm"],
        "long_gain_mean_pp": long["mean"],
        "long_gain_q05_pp": long["q05"],
        "long_p_harm": long["p_harm"],
        "best_weighted_gain_mean_pp": best_stats["mean"],
        "best_weighted_q05_pp": best_stats["q05"],
        "best_weighted_p_harm": best_stats["p_harm"],
        "pass_margin": bool(best_stats["mean"] >= MARGIN_PP),
        "pass_risk": bool(best_stats["p_harm"] <= RISK_LIMIT),
    }


def validation_pivot(validation_df: pd.DataFrame, *, index_cols: Sequence[str]) -> pd.DataFrame:
    sub = validation_df[validation_df["candidate"].isin(CANDIDATES)].copy()
    return sub.pivot_table(index=list(index_cols), columns="candidate", values="accuracy", aggfunc="mean")


def global_choice(validation_df: pd.DataFrame) -> dict[str, object]:
    pivot = validation_pivot(validation_df, index_cols=["outer_subject", "repeat", "inner_subject"])
    pivot = pivot.dropna(subset=CANDIDATES)
    source_gain = pivot[SOURCE].to_numpy(dtype=np.float64) - pivot[LDA].to_numpy(dtype=np.float64)
    long_gain = pivot[LONG].to_numpy(dtype=np.float64) - pivot[LDA].to_numpy(dtype=np.float64)
    choice = choose_from_gains(source_gain, long_gain)
    choice["n_validation_points"] = int(len(pivot))
    return choice


def local_choices(validation_df: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    rows = []
    if mode == "repeat":
        grouped = validation_df.groupby(["outer_subject", "repeat"], sort=True)
        index_cols = ["inner_subject"]
    elif mode == "stable":
        grouped = validation_df.groupby("outer_subject", sort=True)
        index_cols = ["repeat", "inner_subject"]
    else:
        raise ValueError(mode)

    for key, sub in grouped:
        if mode == "repeat":
            subject, repeat = key
        else:
            subject = int(key)
            repeat = -1
        pivot = validation_pivot(sub, index_cols=index_cols)
        pivot = pivot.dropna(subset=CANDIDATES)
        source_gain = pivot[SOURCE].to_numpy(dtype=np.float64) - pivot[LDA].to_numpy(dtype=np.float64)
        long_gain = pivot[LONG].to_numpy(dtype=np.float64) - pivot[LDA].to_numpy(dtype=np.float64)
        choice = choose_from_gains(source_gain, long_gain)
        rows.append(
            {
                "subject": int(subject),
                "repeat": int(repeat),
                "mode": mode,
                **choice,
                "n_validation_points": int(len(pivot)),
            }
        )
    return pd.DataFrame(rows)


def apply_candidate(outer_df: pd.DataFrame, *, candidate: str, method_name: str, mode: str) -> pd.DataFrame:
    selected = outer_df[outer_df["method"] == candidate].copy()
    selected["selected_candidate"] = candidate
    selected["method"] = method_name
    selected["mode"] = mode
    return selected


def apply_local_guard_global_family(
    outer_df: pd.DataFrame,
    local: pd.DataFrame,
    *,
    global_candidate: str,
    method_name: str,
    mode: str,
) -> pd.DataFrame:
    rows = []
    if global_candidate == LDA:
        weighted_family = LDA
    else:
        weighted_family = global_candidate
    for row in local.itertuples():
        if mode == "repeat":
            subject_filter = (outer_df["subject"] == int(row.subject)) & (outer_df["repeat"] == int(row.repeat))
        else:
            subject_filter = outer_df["subject"] == int(row.subject)
        if weighted_family == LDA:
            candidate = LDA
        else:
            candidate = weighted_family if bool(row.pass_margin) and bool(row.pass_risk) else LDA
        selected = outer_df[subject_filter & (outer_df["method"] == candidate)].copy()
        selected["selected_candidate"] = candidate
        selected["method"] = method_name
        selected["mode"] = mode
        selected["local_best_weighted_candidate"] = row.best_weighted_candidate
        selected["local_best_weighted_gain_mean_pp"] = float(row.best_weighted_gain_mean_pp)
        selected["local_best_weighted_p_harm"] = float(row.best_weighted_p_harm)
        rows.extend(selected.to_dict(orient="records"))
    return pd.DataFrame(rows)


def summarize(records: pd.DataFrame, *, regime: str, level: str, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    baseline = records[records["selected_candidate"] == LDA][["subject", "repeat", "accuracy"]].copy()
    if len(baseline) == 0:
        # Create baseline lookup from explicit LDA records if this summary contains methods not selected as LDA.
        baseline = records[records["method_original"] == LDA][["subject", "repeat", "accuracy"]].copy()
    baseline_lookup = {
        (int(r.subject), int(r.repeat)): float(r.accuracy)
        for r in baseline.itertuples()
    }
    for method, sub in records.groupby("method", sort=True):
        gains_by_unit = []
        acc_by_unit = []
        if level == "unit":
            grouped = sub.groupby(["subject", "repeat"], sort=True)
        elif level == "subject":
            grouped = sub.groupby("subject", sort=True)
        else:
            raise ValueError(level)
        for key, g in grouped:
            if level == "unit":
                subject, repeat = key
                acc = float(g["accuracy"].mean())
                base = baseline_lookup.get((int(subject), int(repeat)))
                if base is None:
                    continue
                gain = acc - base
            else:
                subject = int(key)
                acc = float(g["accuracy"].mean())
                base_values = [
                    baseline_lookup[(int(r.subject), int(r.repeat))]
                    for r in g.itertuples()
                    if (int(r.subject), int(r.repeat)) in baseline_lookup
                ]
                if not base_values:
                    continue
                gain = acc - float(np.mean(base_values))
            acc_by_unit.append(acc)
            gains_by_unit.append(gain)
        acc_arr = np.asarray(acc_by_unit, dtype=np.float64)
        gain_arr = np.asarray(gains_by_unit, dtype=np.float64)
        rows.append(
            {
                "regime_label": regime,
                "summary_level": level,
                "method": method,
                "n_units": int(len(acc_arr)),
                "accuracy_mean": float(np.mean(acc_arr)),
                "accuracy_95ci": bootstrap_ci(acc_arr, rng, bootstrap),
                "gain_vs_lda_full_mean_pp": float(np.mean(gain_arr)),
                "gain_vs_lda_full_95ci": bootstrap_ci(gain_arr, rng, bootstrap),
                "gain_vs_lda_full_q05_pp": float(np.quantile(gain_arr, 0.05)),
                "p_gain_vs_lda_full_lt_minus5": float(np.mean(gain_arr < -5.0)),
                "p_gain_vs_lda_full_lt_0": float(np.mean(gain_arr < 0.0)),
                "selection_counts": json.dumps(
                    sub["selected_candidate"].value_counts().to_dict(),
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def process_dir(input_dir: Path, *, output_dir: Path, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation = pd.read_csv(input_dir / "nested_inner_validation_records.csv")
    outer = pd.read_csv(input_dir / "nested_outer_records.csv")
    regime = infer_regime(validation, input_dir)
    outer = outer[outer["method"].isin(CANDIDATES)].copy()
    outer["method_original"] = outer["method"]

    global_info = global_choice(validation)
    global_candidate = str(global_info["selected_candidate"])
    global_weighted_candidate = str(global_info["best_weighted_candidate"])

    records = []
    # Reference candidates
    for candidate in CANDIDATES:
        records.append(
            apply_candidate(
                outer,
                candidate=candidate,
                method_name=f"fixed_{candidate}",
                mode="reference",
            )
        )
    # Same global family for everyone.
    records.append(
        apply_candidate(
            outer,
            candidate=global_candidate,
            method_name="e38_global_regime_family",
            mode="global",
        )
    )
    # If global weighted family passes globally, use local guard only for weighted-vs-LDA,
    # but do not let local validation switch source-vs-long.
    selection_rows = []
    for mode in ["repeat", "stable"]:
        local = local_choices(validation, mode=mode)
        local["regime_label"] = regime
        local["global_selected_candidate"] = global_candidate
        local["global_best_weighted_candidate"] = global_weighted_candidate
        selection_rows.append(local)
        records.append(
            apply_local_guard_global_family(
                outer,
                local,
                global_candidate=global_candidate,
                method_name=f"e38_local_guard_global_family_{mode}",
                mode=mode,
            )
        )
    all_records = pd.concat(records, ignore_index=True, sort=False)
    all_records["regime_label"] = regime

    regime_dir = output_dir / regime
    regime_dir.mkdir(parents=True, exist_ok=True)
    all_records.to_csv(regime_dir / "e38_records.csv", index=False)
    selection_df = pd.concat(selection_rows, ignore_index=True, sort=False)
    selection_df.to_csv(regime_dir / "e38_selection_records.csv", index=False)
    global_df = pd.DataFrame([{ "regime_label": regime, **global_info }])
    global_df.to_csv(regime_dir / "e38_global_choice.csv", index=False)

    summaries = []
    for level in ["subject", "unit"]:
        summaries.append(
            summarize(
                all_records,
                regime=regime,
                level=level,
                bootstrap=bootstrap,
                seed=seed,
            )
        )
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(regime_dir / "e38_summary.csv", index=False)
    return summary, global_df, selection_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    globals_ = []
    selections = []
    for path in args.e33_dirs:
        summary, global_df, selection = process_dir(
            path,
            output_dir=args.output_dir,
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        )
        summaries.append(summary)
        globals_.append(global_df)
        selections.append(selection)

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_global = pd.concat(globals_, ignore_index=True)
    combined_selection = pd.concat(selections, ignore_index=True)
    combined_summary.to_csv(args.output_dir / "e38_combined_summary.csv", index=False)
    combined_global.to_csv(args.output_dir / "e38_global_choices.csv", index=False)
    combined_selection.to_csv(args.output_dir / "e38_combined_selection_records.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "frozen_rule": {
                    "candidates": CANDIDATES,
                    "margin_pp": MARGIN_PP,
                    "risk_limit": RISK_LIMIT,
                    "harm_threshold_pp": HARM_THRESHOLD_PP,
                },
                "inputs": [str(path) for path in args.e33_dirs],
                "global_choices": combined_global.to_dict(orient="records"),
                "combined_summary": combined_summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "combined_summary": str(args.output_dir / "e38_combined_summary.csv"),
                "global_choices": str(args.output_dir / "e38_global_choices.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

