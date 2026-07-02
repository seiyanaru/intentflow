"""E35 source-anchor guarded minimal selector.

E34 used a minimal candidate set but still selected weighted Ridge once in
BNCI full-source.  E35 adds a conservative source-anchor guard:

    default to lda_full;
    switch to the best weighted Ridge only if source-validation gain over
    lda_full exceeds a fixed margin and passes a harm-rate constraint.

This is a post-processing analysis over E33 records, so it is cheap and does
not use held target labels for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_e35_guarded_minimal_selector"
WEIGHTED_CANDIDATES = [
    "ridge_source_rank_g2_a100",
    "ridge_longitudinal_rank_g2_a100",
]
REFERENCE_METHODS = [
    "lda_full",
    "lda_long_q70",
    "ridge_source_rank_g2_a100",
    "ridge_longitudinal_rank_g2_a100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--margins", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 1.5])
    parser.add_argument("--risk-limit", type=float, default=0.20)
    parser.add_argument("--harm-threshold", type=float, default=-5.0)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bootstrap_ci(values: Sequence[float], rng: np.random.Generator, repeats: int) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(arr, size=(int(repeats), arr.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def infer_regime_label(input_dir: Path, outer_df: pd.DataFrame) -> str:
    if "regime_label" in outer_df.columns and len(outer_df["regime_label"].dropna()) > 0:
        return str(outer_df["regime_label"].dropna().iloc[0])
    return input_dir.name


def choose_guarded(
    validation_df: pd.DataFrame,
    *,
    margin: float,
    risk_limit: float,
    harm_threshold: float,
) -> tuple[str, dict[str, float]]:
    keep = ["lda_full"] + WEIGHTED_CANDIDATES
    validation_df = validation_df[validation_df["candidate"].isin(keep)].copy()
    merge_keys = ["inner_subject"]
    if "repeat" in validation_df.columns:
        merge_keys = ["repeat", "inner_subject"]
    lda = validation_df[validation_df["candidate"] == "lda_full"][
        merge_keys + ["accuracy"]
    ].rename(columns={"accuracy": "lda_full_accuracy"})

    rows = []
    for candidate in WEIGHTED_CANDIDATES:
        sub = validation_df[validation_df["candidate"] == candidate]
        merged = sub.merge(lda, on=merge_keys, how="left")
        gain = merged["accuracy"].to_numpy(dtype=np.float64) - merged[
            "lda_full_accuracy"
        ].to_numpy(dtype=np.float64)
        rows.append(
            {
                "candidate": candidate,
                "mean_accuracy": float(merged["accuracy"].mean()),
                "gain_vs_lda_full_mean": float(np.mean(gain)),
                "p_harm": float(np.mean(gain < float(harm_threshold))),
            }
        )
    weighted = pd.DataFrame(rows).sort_values(
        ["gain_vs_lda_full_mean", "mean_accuracy"],
        ascending=[False, False],
    )
    best = weighted.iloc[0]
    if (
        float(best["gain_vs_lda_full_mean"]) >= float(margin)
        and float(best["p_harm"]) <= float(risk_limit)
    ):
        return str(best["candidate"]), {
            "inner_best_weighted_gain": float(best["gain_vs_lda_full_mean"]),
            "inner_best_weighted_p_harm": float(best["p_harm"]),
        }
    return "lda_full", {
        "inner_best_weighted_gain": float(best["gain_vs_lda_full_mean"]),
        "inner_best_weighted_p_harm": float(best["p_harm"]),
    }


def summarize(
    records: pd.DataFrame,
    *,
    regime_label: str,
    margin: float,
    summary_kind: str,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = sorted(map(int, records["subject"].unique()))
    lda_lookup = {
        (int(row.subject), int(row.repeat)): float(row.accuracy)
        for row in records[records["method"] == "lda_full"].itertuples()
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
                float(row.accuracy) - lda_lookup[key]
            )
            selected = getattr(row, "selected_candidate", "")
            if isinstance(selected, str) and selected:
                selection_counts[selected] = selection_counts.get(selected, 0) + 1
        acc = np.asarray([np.mean(by_subject_acc[s]) for s in subjects if s in by_subject_acc])
        gain = np.asarray([np.mean(by_subject_gain[s]) for s in subjects if s in by_subject_gain])
        rows.append(
            {
                "regime_label": regime_label,
                "summary_kind": summary_kind,
                "margin": float(margin),
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


def process_one(
    input_dir: Path,
    *,
    output_dir: Path,
    margins: Sequence[float],
    risk_limit: float,
    harm_threshold: float,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    validation_df = pd.read_csv(input_dir / "nested_inner_validation_records.csv")
    outer_df = pd.read_csv(input_dir / "nested_outer_records.csv")
    regime_label = infer_regime_label(input_dir, outer_df)
    fixed = outer_df[outer_df["method"].isin(REFERENCE_METHODS)].copy()
    fixed["selected_candidate"] = fixed.get("selected_candidate", "")

    regime_dir = output_dir / regime_label
    regime_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    selected_outer_rows: list[dict[str, object]] = []

    for margin in margins:
        for kind in ["guarded_repeat", "guarded_stable"]:
            selected_records: list[dict[str, object]] = []
            if kind == "guarded_repeat":
                grouped = validation_df.groupby(["outer_subject", "repeat"], sort=True)
            else:
                grouped = validation_df.groupby("outer_subject", sort=True)

            for group_key, subject_validation in grouped:
                if kind == "guarded_repeat":
                    subject, repeat = group_key
                    candidate, meta = choose_guarded(
                        subject_validation,
                        margin=float(margin),
                        risk_limit=float(risk_limit),
                        harm_threshold=float(harm_threshold),
                    )
                    source_rows = outer_df[
                        (outer_df["subject"] == int(subject))
                        & (outer_df["repeat"] == int(repeat))
                        & (outer_df["method"] == candidate)
                    ].copy()
                    repeats_for_log = [int(repeat)]
                else:
                    subject = int(group_key)
                    candidate, meta = choose_guarded(
                        subject_validation,
                        margin=float(margin),
                        risk_limit=float(risk_limit),
                        harm_threshold=float(harm_threshold),
                    )
                    source_rows = outer_df[
                        (outer_df["subject"] == int(subject)) & (outer_df["method"] == candidate)
                    ].copy()
                    repeats_for_log = sorted(map(int, source_rows["repeat"].unique()))

                method = f"{kind}_m{str(margin).replace('.', 'p')}"
                source_rows["method"] = method
                source_rows["selected_candidate"] = candidate
                selected_records.extend(source_rows.to_dict(orient="records"))
                for repeat_value in repeats_for_log:
                    selection_rows.append(
                        {
                            "regime_label": regime_label,
                            "summary_kind": kind,
                            "margin": float(margin),
                            "subject": int(subject),
                            "repeat": int(repeat_value),
                            "method": method,
                            "selected_candidate": candidate,
                            **meta,
                        }
                    )

            selected_df = pd.DataFrame(selected_records)
            selected_outer_rows.extend(selected_records)
            all_records = pd.concat([fixed, selected_df], ignore_index=True)
            summaries.append(
                summarize(
                    all_records,
                    regime_label=regime_label,
                    margin=float(margin),
                    summary_kind=kind,
                    bootstrap=int(bootstrap),
                    seed=int(seed),
                )
            )

    selection_df = pd.DataFrame(selection_rows)
    selected_outer_df = pd.DataFrame(selected_outer_rows)
    selection_df.to_csv(regime_dir / "guarded_selection_records.csv", index=False)
    selected_outer_df.to_csv(regime_dir / "guarded_outer_records.csv", index=False)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(regime_dir / "guarded_summary.csv", index=False)
    return combined


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for input_dir in args.input_dirs:
        summaries.append(
            process_one(
                input_dir,
                output_dir=args.output_dir,
                margins=args.margins,
                risk_limit=float(args.risk_limit),
                harm_threshold=float(args.harm_threshold),
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
        )
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.output_dir / "e35_guarded_selector_combined_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "weighted_candidates": WEIGHTED_CANDIDATES,
                "reference_methods": REFERENCE_METHODS,
                "margins": list(map(float, args.margins)),
                "risk_limit": float(args.risk_limit),
                "harm_threshold": float(args.harm_threshold),
                "input_dirs": [str(path) for path in args.input_dirs],
                "combined_summary": combined.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "combined_summary": str(args.output_dir / "e35_guarded_selector_combined_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

