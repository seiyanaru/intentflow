"""E33 honest source-side model-family selection for weighted Ridge.

E32 showed that soft reliability-weighted Ridge transfers to some regimes but
not all:

* Lee2019 sensorimotor20: source-weighted Ridge is strong.
* BNCI2014_001 m8/class: longitudinal-weighted Ridge is strong.
* BNCI2014_001 full-source: LDA remains better.

This script tests whether that regime dependence can be handled honestly using
only labeled source subjects/sessions.  For each outer held subject, it performs
inner source-subject validation, selects a candidate family, and evaluates once
on the held subject without using held target labels for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from weighted_ridge_regime_transfer_e32 import (
    RESULTS_DIR,
    add_stats,
    balanced_subset,
    bootstrap_ci,
    fit_lda,
    fit_ridge,
    load_payload,
    longitudinal_score,
    parse_subjects,
    rank_weights,
    session_metric_stats,
    source_score,
    top_fraction,
)


DEFAULT_OUTPUT = RESULTS_DIR / "260701_weighted_ridge_nested_selector_e33"

NESTED_CANDIDATES = [
    "lda_full",
    "lda_source_q25",
    "lda_long_q10",
    "lda_long_q70",
    "ridge_source_rank_g1_a100",
    "ridge_source_rank_g2_a100",
    "ridge_longitudinal_rank_g2_a100",
]

FIXED_METHODS = [
    "ridge_full_a100",
    "ridge_source_rank_g2_a100",
    "ridge_source_rank_g1_a100",
    "ridge_longitudinal_rank_g2_a100",
    "ridge_hard_source_q25_a100",
    "ridge_hard_long_q10_a100",
    "ridge_hard_long_q70_a100",
    "lda_full",
    "lda_source_q25",
    "lda_long_q10",
    "lda_long_q70",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regime-label", required=True)
    parser.add_argument("--per-class", type=int, default=0, help="0 uses all source trials.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--risk-limit", type=float, default=0.20)
    parser.add_argument("--harm-threshold", type=float, default=-5.0)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def aggregate_stats_excluding(
    payloads: Mapping[int, Mapping[str, np.ndarray]],
    excluded_subjects: Sequence[int],
) -> dict[str, np.ndarray]:
    excluded = {int(subject) for subject in excluded_subjects}
    total: dict[str, np.ndarray] = {}
    for subject, payload in payloads.items():
        if int(subject) in excluded:
            continue
        stats = session_metric_stats(
            np.asarray(payload["source_features"], dtype=np.float64),
            np.asarray(payload["source_labels"], dtype=np.int64),
            np.asarray(payload["target_features"], dtype=np.float64),
            np.asarray(payload["target_labels"], dtype=np.int64),
        )
        add_stats(total, stats)
    if "count" not in total:
        raise ValueError(f"no subjects remain after excluding {sorted(excluded)}")
    count = float(total["count"][0])
    averaged = {key: value / count for key, value in total.items() if key != "count"}
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged


def subject_source_indices(
    payload: Mapping[str, np.ndarray],
    *,
    subject: int,
    repeat: int,
    per_class: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(payload["source_labels"], dtype=np.int64)
    if int(per_class) <= 0:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 1000003 * int(subject) + int(repeat))
    return balanced_subset(labels, int(per_class), rng)


def feature_context(stats: Mapping[str, np.ndarray]) -> dict[str, object]:
    s_score = source_score(stats)
    l_score = longitudinal_score(stats)
    return {
        "w_source_g2": rank_weights(s_score, gamma=2.0),
        "w_source_g1": rank_weights(s_score, gamma=1.0),
        "w_long_g2": rank_weights(l_score, gamma=2.0),
        "idx_source_q25": top_fraction(s_score, 0.25),
        "idx_long_q10": top_fraction(l_score, 0.10),
        "idx_long_q70": top_fraction(l_score, 0.70),
        "p": int(len(s_score)),
    }


def evaluate_method(
    method: str,
    payload: Mapping[str, np.ndarray],
    context: Mapping[str, object],
    source_idx: np.ndarray,
) -> tuple[float, int]:
    xs_all = np.asarray(payload["source_features"], dtype=np.float64)
    ys_all = np.asarray(payload["source_labels"], dtype=np.int64)
    xt = np.asarray(payload["target_features"], dtype=np.float64)
    yt = np.asarray(payload["target_labels"], dtype=np.int64)
    xs = xs_all[source_idx]
    ys = ys_all[source_idx]
    p = int(context["p"])

    if method == "ridge_full_a100":
        return fit_ridge(xs, ys, xt, yt, weights=None, alpha=100.0), p
    if method == "ridge_source_rank_g2_a100":
        return fit_ridge(xs, ys, xt, yt, weights=np.asarray(context["w_source_g2"]), alpha=100.0), p
    if method == "ridge_source_rank_g1_a100":
        return fit_ridge(xs, ys, xt, yt, weights=np.asarray(context["w_source_g1"]), alpha=100.0), p
    if method == "ridge_longitudinal_rank_g2_a100":
        return fit_ridge(xs, ys, xt, yt, weights=np.asarray(context["w_long_g2"]), alpha=100.0), p
    if method == "ridge_hard_source_q25_a100":
        indices = np.asarray(context["idx_source_q25"], dtype=np.int64)
        return fit_ridge(xs[:, indices], ys, xt[:, indices], yt, weights=None, alpha=100.0), len(indices)
    if method == "ridge_hard_long_q10_a100":
        indices = np.asarray(context["idx_long_q10"], dtype=np.int64)
        return fit_ridge(xs[:, indices], ys, xt[:, indices], yt, weights=None, alpha=100.0), len(indices)
    if method == "ridge_hard_long_q70_a100":
        indices = np.asarray(context["idx_long_q70"], dtype=np.int64)
        return fit_ridge(xs[:, indices], ys, xt[:, indices], yt, weights=None, alpha=100.0), len(indices)
    if method == "lda_full":
        return fit_lda(xs, ys, xt, yt, None), p
    if method == "lda_source_q25":
        indices = np.asarray(context["idx_source_q25"], dtype=np.int64)
        return fit_lda(xs, ys, xt, yt, indices), len(indices)
    if method == "lda_long_q10":
        indices = np.asarray(context["idx_long_q10"], dtype=np.int64)
        return fit_lda(xs, ys, xt, yt, indices), len(indices)
    if method == "lda_long_q70":
        indices = np.asarray(context["idx_long_q70"], dtype=np.int64)
        return fit_lda(xs, ys, xt, yt, indices), len(indices)
    raise KeyError(method)


def choose_nested_candidates(
    validation_df: pd.DataFrame,
    *,
    risk_limit: float,
    harm_threshold: float,
) -> dict[str, str]:
    rows = []
    order = {name: i for i, name in enumerate(NESTED_CANDIDATES)}
    merge_keys = ["inner_subject"]
    if "repeat" in validation_df.columns:
        merge_keys = ["repeat", "inner_subject"]
    lda = validation_df[validation_df["candidate"] == "lda_full"][
        merge_keys + ["accuracy"]
    ].rename(columns={"accuracy": "lda_full_accuracy"})
    for candidate, sub in validation_df.groupby("candidate", sort=False):
        merged = sub.merge(lda, on=merge_keys, how="left")
        gain = merged["accuracy"].to_numpy(dtype=np.float64) - merged[
            "lda_full_accuracy"
        ].to_numpy(dtype=np.float64)
        rows.append(
            {
                "candidate": candidate,
                "inner_mean_accuracy": float(merged["accuracy"].mean()),
                "inner_gain_vs_lda_full_mean": float(np.mean(gain)),
                "inner_p_gain_vs_lda_full_lt_harm": float(np.mean(gain < float(harm_threshold))),
                "order": order[candidate],
            }
        )
    summary = pd.DataFrame(rows)
    selected_mean = summary.sort_values(
        ["inner_mean_accuracy", "inner_gain_vs_lda_full_mean", "order"],
        ascending=[False, False, True],
    ).iloc[0]["candidate"]
    safe = summary[summary["inner_p_gain_vs_lda_full_lt_harm"] <= float(risk_limit)]
    if len(safe) == 0:
        safe = summary
    selected_risk = safe.sort_values(
        ["inner_mean_accuracy", "inner_gain_vs_lda_full_mean", "order"],
        ascending=[False, False, True],
    ).iloc[0]["candidate"]
    return {
        "nested_mean": str(selected_mean),
        "nested_risk20": str(selected_risk),
    }


def summarize_outer(
    outer_records: pd.DataFrame,
    *,
    subjects: Sequence[int],
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lda_full_lookup = {
        (int(row.subject), int(row.repeat)): float(row.accuracy)
        for row in outer_records[outer_records["method"] == "lda_full"].itertuples()
    }
    rows = []
    for method, sub in outer_records.groupby("method", sort=True):
        by_subject_acc: dict[int, list[float]] = {}
        by_subject_gain_lda: dict[int, list[float]] = {}
        by_subject_selected: dict[int, list[str]] = {}
        for row in sub.itertuples():
            key = (int(row.subject), int(row.repeat))
            by_subject_acc.setdefault(int(row.subject), []).append(float(row.accuracy))
            by_subject_gain_lda.setdefault(int(row.subject), []).append(
                float(row.accuracy) - lda_full_lookup[key]
            )
            selected = getattr(row, "selected_candidate", "")
            if isinstance(selected, str) and selected:
                by_subject_selected.setdefault(int(row.subject), []).append(selected)
        acc = np.asarray([np.mean(by_subject_acc[s]) for s in subjects if s in by_subject_acc])
        gain_lda = np.asarray(
            [np.mean(by_subject_gain_lda[s]) for s in subjects if s in by_subject_gain_lda],
            dtype=np.float64,
        )
        selection_counts: dict[str, int] = {}
        for values in by_subject_selected.values():
            for value in values:
                selection_counts[value] = selection_counts.get(value, 0) + 1
        rows.append(
            {
                "method": method,
                "n_subjects": int(len(acc)),
                "accuracy_mean": float(acc.mean()),
                "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, bootstrap),
                "gain_vs_lda_full_mean_pp": float(gain_lda.mean()),
                "gain_vs_lda_full_95ci": bootstrap_ci(gain_lda, rng, bootstrap),
                "gain_vs_lda_full_q05_pp": float(np.quantile(gain_lda, 0.05)),
                "p_gain_vs_lda_full_lt_minus5": float(np.mean(gain_lda < -5.0)),
                "selection_counts": json.dumps(selection_counts, sort_keys=True),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)


def make_stable_nested_outputs(
    outer_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    subjects: Sequence[int],
    risk_limit: float,
    harm_threshold: float,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select once per outer subject after averaging all inner repeats.

    The ordinary nested methods select separately for each outer subject and
    repeat.  That is correct, but can overreact to source-scarce simulation
    noise.  The stable variant uses all inner repeats available for the outer
    subject to choose one candidate, then applies it to every outer repeat.
    This remains target-label-free for the held subject.
    """

    stable_records: list[dict[str, object]] = []
    stable_selection: list[dict[str, object]] = []
    for subject, subject_validation in validation_df.groupby("outer_subject", sort=True):
        selected = choose_nested_candidates(
            subject_validation,
            risk_limit=float(risk_limit),
            harm_threshold=float(harm_threshold),
        )
        for stable_method, selected_candidate in [
            ("stable_nested_mean", selected["nested_mean"]),
            ("stable_nested_risk20", selected["nested_risk20"]),
        ]:
            stable_selection.append(
                {
                    "subject": int(subject),
                    "stable_method": stable_method,
                    "selected_candidate": selected_candidate,
                }
            )
            selected_outer = outer_df[
                (outer_df["subject"] == int(subject))
                & (outer_df["method"] == selected_candidate)
            ].copy()
            selected_outer["method"] = stable_method
            selected_outer["selected_candidate"] = selected_candidate
            stable_records.extend(selected_outer.to_dict(orient="records"))

    stable_df = pd.DataFrame(stable_records)
    fixed_df = outer_df[outer_df["method"].isin(FIXED_METHODS)].copy()
    stable_summary = summarize_outer(
        pd.concat([fixed_df, stable_df], ignore_index=True),
        subjects=subjects,
        bootstrap=int(bootstrap),
        seed=int(seed),
    )
    return stable_df, pd.DataFrame(stable_selection), stable_summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    payloads = {subject: load_payload(args.cache_dir, subject) for subject in subjects}
    repeats = int(args.repeats) if int(args.per_class) > 0 else 1
    stats_cache: dict[tuple[int, ...], dict[str, np.ndarray]] = {}

    def get_context(excluded: Sequence[int]) -> dict[str, object]:
        key = tuple(sorted(map(int, excluded)))
        if key not in stats_cache:
            stats_cache[key] = aggregate_stats_excluding(payloads, key)
        return feature_context(stats_cache[key])

    outer_records: list[dict[str, object]] = []
    validation_records: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []

    for held_subject in subjects:
        outer_payload = payloads[held_subject]
        outer_context = get_context([held_subject])
        for repeat in range(repeats):
            inner_rows: list[dict[str, object]] = []
            for inner_subject in subjects:
                if int(inner_subject) == int(held_subject):
                    continue
                inner_payload = payloads[inner_subject]
                inner_context = get_context([held_subject, inner_subject])
                inner_source_idx = subject_source_indices(
                    inner_payload,
                    subject=int(inner_subject),
                    repeat=int(repeat),
                    per_class=int(args.per_class),
                    seed=int(args.seed),
                )
                for candidate in NESTED_CANDIDATES:
                    accuracy, n_selected = evaluate_method(
                        candidate, inner_payload, inner_context, inner_source_idx
                    )
                    record = {
                        "regime_label": args.regime_label,
                        "outer_subject": int(held_subject),
                        "repeat": int(repeat),
                        "inner_subject": int(inner_subject),
                        "candidate": candidate,
                        "n_selected": int(n_selected),
                        "accuracy": float(accuracy),
                    }
                    inner_rows.append(record)
                    validation_records.append(record)

            validation_df = pd.DataFrame(inner_rows)
            selected = choose_nested_candidates(
                validation_df,
                risk_limit=float(args.risk_limit),
                harm_threshold=float(args.harm_threshold),
            )
            for nested_method, selected_candidate in selected.items():
                selection_records.append(
                    {
                        "regime_label": args.regime_label,
                        "subject": int(held_subject),
                        "repeat": int(repeat),
                        "nested_method": nested_method,
                        "selected_candidate": selected_candidate,
                    }
                )

            outer_source_idx = subject_source_indices(
                outer_payload,
                subject=int(held_subject),
                repeat=int(repeat),
                per_class=int(args.per_class),
                seed=int(args.seed),
            )
            for fixed_method in FIXED_METHODS:
                accuracy, n_selected = evaluate_method(
                    fixed_method, outer_payload, outer_context, outer_source_idx
                )
                outer_records.append(
                    {
                        "regime_label": args.regime_label,
                        "subject": int(held_subject),
                        "repeat": int(repeat),
                        "method": fixed_method,
                        "selected_candidate": "",
                        "n_selected": int(n_selected),
                        "accuracy": float(accuracy),
                    }
                )
            for nested_method, selected_candidate in selected.items():
                accuracy, n_selected = evaluate_method(
                    selected_candidate, outer_payload, outer_context, outer_source_idx
                )
                outer_records.append(
                    {
                        "regime_label": args.regime_label,
                        "subject": int(held_subject),
                        "repeat": int(repeat),
                        "method": nested_method,
                        "selected_candidate": selected_candidate,
                        "n_selected": int(n_selected),
                        "accuracy": float(accuracy),
                    }
                )
        if not args.quiet:
            print(f"S{held_subject}: done", flush=True)

    outer_df = pd.DataFrame(outer_records)
    validation_df = pd.DataFrame(validation_records)
    selection_df = pd.DataFrame(selection_records)
    summary_df = summarize_outer(
        outer_df,
        subjects=subjects,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    stable_outer_df, stable_selection_df, stable_summary_df = make_stable_nested_outputs(
        outer_df,
        validation_df,
        subjects=subjects,
        risk_limit=float(args.risk_limit),
        harm_threshold=float(args.harm_threshold),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    outer_df.to_csv(args.output_dir / "nested_outer_records.csv", index=False)
    validation_df.to_csv(args.output_dir / "nested_inner_validation_records.csv", index=False)
    selection_df.to_csv(args.output_dir / "nested_selection_records.csv", index=False)
    summary_df.to_csv(args.output_dir / "nested_selector_summary.csv", index=False)
    stable_outer_df.to_csv(args.output_dir / "stable_nested_outer_records.csv", index=False)
    stable_selection_df.to_csv(args.output_dir / "stable_nested_selection_records.csv", index=False)
    stable_summary_df.to_csv(args.output_dir / "stable_nested_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "regime_label": args.regime_label,
                    "cache_dir": str(args.cache_dir),
                    "subjects": subjects,
                    "per_class": int(args.per_class),
                    "repeats": int(repeats),
                    "risk_limit": float(args.risk_limit),
                    "harm_threshold": float(args.harm_threshold),
                    "bootstrap": int(args.bootstrap),
                    "seed": int(args.seed),
                    "nested_candidates": NESTED_CANDIDATES,
                    "fixed_methods": FIXED_METHODS,
                },
                "summary": summary_df.to_dict(orient="records"),
                "stable_summary": stable_summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "regime_label": args.regime_label,
                "n_outer_records": int(len(outer_df)),
                "n_validation_records": int(len(validation_df)),
                "summary_csv": str(args.output_dir / "nested_selector_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
