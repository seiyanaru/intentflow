"""E34 minimal stable source-validation over weighted Ridge families.

E33 showed that the full candidate set was too selector-like: weak candidates
added variance, especially in source-scarce repeats.  E34 therefore tests a
predefined minimal candidate set:

* lda_full
* ridge_source_rank_g2_a100
* ridge_longitudinal_rank_g2_a100

The script reuses E33 inner/outer records.  It does not re-fit models and does
not use held target labels for selection; it only restricts the candidate set
and recomputes nested/stable nested summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_e34_minimal_stable_selector"
MINIMAL_CANDIDATES = [
    "lda_full",
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
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="E33 output directories containing nested_inner_validation_records.csv and nested_outer_records.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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


def choose_candidate(
    validation_df: pd.DataFrame,
    *,
    risk_limit: float,
    harm_threshold: float,
) -> dict[str, str]:
    validation_df = validation_df[validation_df["candidate"].isin(MINIMAL_CANDIDATES)].copy()
    merge_keys = ["inner_subject"]
    if "repeat" in validation_df.columns:
        merge_keys = ["repeat", "inner_subject"]
    lda = validation_df[validation_df["candidate"] == "lda_full"][
        merge_keys + ["accuracy"]
    ].rename(columns={"accuracy": "lda_full_accuracy"})

    rows = []
    order = {candidate: idx for idx, candidate in enumerate(MINIMAL_CANDIDATES)}
    for candidate, sub in validation_df.groupby("candidate", sort=False):
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
                "order": int(order[candidate]),
            }
        )
    summary = pd.DataFrame(rows)
    selected_mean = summary.sort_values(
        ["mean_accuracy", "gain_vs_lda_full_mean", "order"],
        ascending=[False, False, True],
    ).iloc[0]["candidate"]
    safe = summary[summary["p_harm"] <= float(risk_limit)]
    if len(safe) == 0:
        safe = summary
    selected_risk = safe.sort_values(
        ["mean_accuracy", "gain_vs_lda_full_mean", "order"],
        ascending=[False, False, True],
    ).iloc[0]["candidate"]
    return {
        "minimal_nested_mean": str(selected_mean),
        "minimal_nested_risk20": str(selected_risk),
    }


def summarize(
    records: pd.DataFrame,
    *,
    regime_label: str,
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


def infer_regime_label(input_dir: Path, outer_df: pd.DataFrame) -> str:
    if "regime_label" in outer_df.columns and len(outer_df["regime_label"].dropna()) > 0:
        return str(outer_df["regime_label"].dropna().iloc[0])
    return input_dir.name


def process_input_dir(
    input_dir: Path,
    *,
    output_dir: Path,
    risk_limit: float,
    harm_threshold: float,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_df = pd.read_csv(input_dir / "nested_inner_validation_records.csv")
    outer_df = pd.read_csv(input_dir / "nested_outer_records.csv")
    regime_label = infer_regime_label(input_dir, outer_df)

    fixed = outer_df[outer_df["method"].isin(REFERENCE_METHODS)].copy()
    fixed["selected_candidate"] = fixed.get("selected_candidate", "")

    repeat_records: list[dict[str, object]] = []
    repeat_selection: list[dict[str, object]] = []
    for (subject, repeat), subject_validation in validation_df.groupby(
        ["outer_subject", "repeat"], sort=True
    ):
        selected = choose_candidate(
            subject_validation,
            risk_limit=float(risk_limit),
            harm_threshold=float(harm_threshold),
        )
        for method, candidate in selected.items():
            repeat_selection.append(
                {
                    "regime_label": regime_label,
                    "subject": int(subject),
                    "repeat": int(repeat),
                    "method": method,
                    "selected_candidate": candidate,
                }
            )
            selected_outer = outer_df[
                (outer_df["subject"] == int(subject))
                & (outer_df["repeat"] == int(repeat))
                & (outer_df["method"] == candidate)
            ].copy()
            selected_outer["method"] = method
            selected_outer["selected_candidate"] = candidate
            repeat_records.extend(selected_outer.to_dict(orient="records"))

    stable_records: list[dict[str, object]] = []
    stable_selection: list[dict[str, object]] = []
    for subject, subject_validation in validation_df.groupby("outer_subject", sort=True):
        selected = choose_candidate(
            subject_validation,
            risk_limit=float(risk_limit),
            harm_threshold=float(harm_threshold),
        )
        for method, candidate in [
            ("minimal_stable_mean", selected["minimal_nested_mean"]),
            ("minimal_stable_risk20", selected["minimal_nested_risk20"]),
        ]:
            stable_selection.append(
                {
                    "regime_label": regime_label,
                    "subject": int(subject),
                    "method": method,
                    "selected_candidate": candidate,
                }
            )
            selected_outer = outer_df[
                (outer_df["subject"] == int(subject)) & (outer_df["method"] == candidate)
            ].copy()
            selected_outer["method"] = method
            selected_outer["selected_candidate"] = candidate
            stable_records.extend(selected_outer.to_dict(orient="records"))

    repeat_df = pd.DataFrame(repeat_records)
    stable_df = pd.DataFrame(stable_records)
    repeat_all = pd.concat([fixed, repeat_df], ignore_index=True)
    stable_all = pd.concat([fixed, stable_df], ignore_index=True)

    regime_dir = output_dir / regime_label
    regime_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(repeat_selection).to_csv(regime_dir / "minimal_nested_selection_records.csv", index=False)
    repeat_df.to_csv(regime_dir / "minimal_nested_outer_records.csv", index=False)
    pd.DataFrame(stable_selection).to_csv(regime_dir / "minimal_stable_selection_records.csv", index=False)
    stable_df.to_csv(regime_dir / "minimal_stable_outer_records.csv", index=False)

    repeat_summary = summarize(
        repeat_all,
        regime_label=regime_label,
        bootstrap=int(bootstrap),
        seed=int(seed),
    )
    repeat_summary["summary_kind"] = "minimal_repeat_nested"
    stable_summary = summarize(
        stable_all,
        regime_label=regime_label,
        bootstrap=int(bootstrap),
        seed=int(seed),
    )
    stable_summary["summary_kind"] = "minimal_stable_nested"
    repeat_summary.to_csv(regime_dir / "minimal_nested_summary.csv", index=False)
    stable_summary.to_csv(regime_dir / "minimal_stable_summary.csv", index=False)
    return repeat_summary, stable_summary, pd.concat([repeat_df, stable_df], ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    for input_dir in args.input_dirs:
        repeat_summary, stable_summary, _ = process_input_dir(
            input_dir,
            output_dir=args.output_dir,
            risk_limit=float(args.risk_limit),
            harm_threshold=float(args.harm_threshold),
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
        )
        summaries.extend([repeat_summary, stable_summary])
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.output_dir / "e34_minimal_selector_combined_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "candidate_set": MINIMAL_CANDIDATES,
                "reference_methods": REFERENCE_METHODS,
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
                "combined_summary": str(args.output_dir / "e34_minimal_selector_combined_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

