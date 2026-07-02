"""E26 diagnostic audit for Lee2019 nested source-side selection.

This script quantifies why source-side nested candidate selection failed on
Lee2019.  It reuses the exact E25 merged outputs, so it does not retrain any
classifier.

Diagnostics:

1. Inner source-validation ranking vs outer target ranking.
2. Oracle regret of the nested choice.
3. Whether inner risk metrics predict outer target harm.
4. Margins for the failure mode where longitudinal_q0p10 is chosen over
   source_only_q0p25.
5. Bootstrap stability of the source-validation choice itself.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "research_outputs"
DEFAULT_INPUT = RESULTS_DIR / "260630_lee2019_exact_nested_subspace_selection_e25_full" / "merged"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_lee2019_nested_predictability_e26"

CANDIDATES = (
    "full_q1p00",
    "source_only_q0p25",
    "source_only_q0p10",
    "longitudinal_q0p25",
    "longitudinal_q0p10",
    "sep_no_drift_q0p25",
    "sep_no_drift_q0p10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--risk-threshold", type=float, default=0.20)
    parser.add_argument("--harm-threshold", type=float, default=-5.0)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--choice-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bootstrap_ci(values: Iterable[float], rng: np.random.Generator, repeats: int) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(arr, size=(int(repeats), arr.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize_vector(values: Iterable[float], *, rng: np.random.Generator, repeats: int) -> dict[str, float | list[float]]:
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


def candidate_metrics(values: np.ndarray, harm_threshold: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    tail = np.sort(arr)[: max(1, int(np.ceil(0.1 * arr.size)))]
    return {
        "inner_mean_gain": float(arr.mean()),
        "inner_p_gain_lt_harm": float(np.mean(arr < harm_threshold)),
        "inner_q05_gain": float(np.quantile(arr, 0.05)),
        "inner_r10_loss": float(-tail.mean()),
    }


def choose_candidate(metrics: dict[str, dict[str, float]], risk_threshold: float) -> str:
    eligible = [
        candidate
        for candidate, item in metrics.items()
        if float(item["inner_p_gain_lt_harm"]) <= risk_threshold
    ]
    if not eligible:
        return "full_q1p00"
    return sorted(
        eligible,
        key=lambda candidate: (
            float(metrics[candidate]["inner_mean_gain"]),
            -float(metrics[candidate]["inner_r10_loss"]),
            -CANDIDATES.index(candidate),
        ),
        reverse=True,
    )[0]


def spearmanr(x: Iterable[float], y: Iterable[float]) -> float:
    xs = pd.Series(list(x), dtype="float64").rank(method="average").to_numpy(dtype=np.float64)
    ys = pd.Series(list(y), dtype="float64").rank(method="average").to_numpy(dtype=np.float64)
    if np.std(xs) == 0 or np.std(ys) == 0:
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


def entropy_from_counts(counts: Counter[str], total: int) -> float:
    if total <= 1:
        return 0.0
    probs = np.asarray([count / total for count in counts.values() if count > 0], dtype=np.float64)
    entropy = float(-(probs * np.log(probs)).sum())
    return float(entropy / math.log(len(CANDIDATES)))


def bootstrap_choice_stability(
    source_rows: pd.DataFrame,
    *,
    rng: np.random.Generator,
    repeats: int,
    risk_threshold: float,
    harm_threshold: float,
) -> dict[str, object]:
    choices: list[str] = []
    values_by_candidate = {
        candidate: source_rows[f"{candidate}__gain_vs_full"].to_numpy(dtype=np.float64)
        for candidate in CANDIDATES
    }
    n_inner = len(source_rows)
    for _ in range(int(repeats)):
        idx = rng.integers(0, n_inner, size=n_inner)
        metrics = {
            candidate: candidate_metrics(values[idx], harm_threshold)
            for candidate, values in values_by_candidate.items()
        }
        choices.append(choose_candidate(metrics, risk_threshold))
    counts = Counter(choices)
    top_candidate, top_count = counts.most_common(1)[0]
    return {
        "choice_bootstrap_top_candidate": top_candidate,
        "choice_bootstrap_top_rate": float(top_count / repeats),
        "choice_bootstrap_entropy_norm": entropy_from_counts(counts, repeats),
        "choice_bootstrap_counts": dict(sorted(counts.items())),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    validation = pd.read_csv(args.input_dir / "exact_nested_source_validation.csv")
    nested = pd.read_csv(args.input_dir / "exact_nested_selection_records.csv")
    fixed = pd.read_csv(args.input_dir / "fixed_candidate_records.csv")

    outer_gain = fixed.pivot(index="subject", columns="chosen_candidate", values="gain_vs_full")
    outer_acc = fixed.pivot(index="subject", columns="chosen_candidate", values="accuracy")

    subject_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    choice_rows: list[dict[str, object]] = []

    for held_subject, source_rows in validation.groupby("held_subject", sort=True):
        held_subject = int(held_subject)
        metrics = {
            candidate: candidate_metrics(
                source_rows[f"{candidate}__gain_vs_full"].to_numpy(dtype=np.float64),
                float(args.harm_threshold),
            )
            for candidate in CANDIDATES
        }
        chosen = choose_candidate(metrics, float(args.risk_threshold))
        nested_row = nested[nested["subject"] == held_subject].iloc[0]
        if str(nested_row["chosen_candidate"]) != chosen:
            raise ValueError(f"Choice mismatch for S{held_subject}: {nested_row['chosen_candidate']} vs {chosen}")

        outer_gains = {candidate: float(outer_gain.loc[held_subject, candidate]) for candidate in CANDIDATES}
        outer_accs = {candidate: float(outer_acc.loc[held_subject, candidate]) for candidate in CANDIDATES}
        oracle_candidate = max(CANDIDATES, key=lambda candidate: outer_gains[candidate])
        oracle_gain = outer_gains[oracle_candidate]
        fixed_source_gain = outer_gains["source_only_q0p25"]

        inner_mean_values = [metrics[candidate]["inner_mean_gain"] for candidate in CANDIDATES]
        inner_mean_values_no_full = [metrics[candidate]["inner_mean_gain"] for candidate in CANDIDATES if candidate != "full_q1p00"]
        outer_gain_values = [outer_gains[candidate] for candidate in CANDIDATES]
        outer_gain_values_no_full = [outer_gains[candidate] for candidate in CANDIDATES if candidate != "full_q1p00"]
        source_candidate = "source_only_q0p25"
        long_candidate = "longitudinal_q0p10"

        stability = bootstrap_choice_stability(
            source_rows,
            rng=rng,
            repeats=int(args.choice_bootstrap),
            risk_threshold=float(args.risk_threshold),
            harm_threshold=float(args.harm_threshold),
        )

        subject_row = {
            "subject": held_subject,
            "chosen_candidate": chosen,
            "oracle_candidate": oracle_candidate,
            "chosen_outer_gain": outer_gains[chosen],
            "oracle_outer_gain": oracle_gain,
            "oracle_regret_pp": oracle_gain - outer_gains[chosen],
            "fixed_source_outer_gain": fixed_source_gain,
            "chosen_minus_fixed_source_pp": outer_gains[chosen] - fixed_source_gain,
            "inner_outer_spearman_all": spearmanr(inner_mean_values, outer_gain_values),
            "inner_outer_spearman_nonfull": spearmanr(inner_mean_values_no_full, outer_gain_values_no_full),
            "inner_top1_by_mean": max(CANDIDATES, key=lambda candidate: metrics[candidate]["inner_mean_gain"]),
            "outer_top1": oracle_candidate,
            "inner_top1_matches_outer": int(max(CANDIDATES, key=lambda candidate: metrics[candidate]["inner_mean_gain"]) == oracle_candidate),
            "nested_choice_matches_outer": int(chosen == oracle_candidate),
            "inner_margin_chosen_minus_source_q025": metrics[chosen]["inner_mean_gain"] - metrics[source_candidate]["inner_mean_gain"],
            "outer_margin_chosen_minus_source_q025": outer_gains[chosen] - fixed_source_gain,
            "inner_margin_long_q010_minus_source_q025": metrics[long_candidate]["inner_mean_gain"] - metrics[source_candidate]["inner_mean_gain"],
            "outer_margin_long_q010_minus_source_q025": outer_gains[long_candidate] - fixed_source_gain,
            **stability,
        }
        subject_rows.append(subject_row)
        choice_rows.append({"subject": held_subject, **stability})

        for candidate in CANDIDATES:
            item = metrics[candidate]
            pair_rows.append(
                {
                    "subject": held_subject,
                    "candidate": candidate,
                    **item,
                    "outer_gain": outer_gains[candidate],
                    "outer_accuracy": outer_accs[candidate],
                    "outer_harm": int(outer_gains[candidate] < float(args.harm_threshold)),
                    "chosen_by_nested": int(candidate == chosen),
                    "outer_oracle": int(candidate == oracle_candidate),
                }
            )

    subject_df = pd.DataFrame(subject_rows).sort_values("subject")
    pair_df = pd.DataFrame(pair_rows).sort_values(["subject", "candidate"])
    choice_df = pd.DataFrame(choice_rows).sort_values("subject")

    subject_df.to_csv(args.output_dir / "subject_predictability_diagnostics.csv", index=False)
    pair_df.to_csv(args.output_dir / "candidate_pair_predictability.csv", index=False)
    choice_df.to_csv(args.output_dir / "source_validation_choice_bootstrap.csv", index=False)

    nonfull_pair_df = pair_df[pair_df["candidate"] != "full_q1p00"].copy()
    chosen_long = subject_df[subject_df["chosen_candidate"] == "longitudinal_q0p10"].copy()
    chosen_source = subject_df[subject_df["chosen_candidate"] == "source_only_q0p25"].copy()

    summary: dict[str, object] = {
        "config": {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "risk_threshold": float(args.risk_threshold),
            "harm_threshold": float(args.harm_threshold),
            "bootstrap": int(args.bootstrap),
            "choice_bootstrap": int(args.choice_bootstrap),
            "seed": int(args.seed),
            "candidates": list(CANDIDATES),
        },
        "n_subjects": int(len(subject_df)),
        "choice_counts": dict(Counter(subject_df["chosen_candidate"].astype(str))),
        "outer_oracle_counts": dict(Counter(subject_df["oracle_candidate"].astype(str))),
        "inner_outer_spearman_all": summarize_vector(
            subject_df["inner_outer_spearman_all"], rng=rng, repeats=int(args.bootstrap)
        ),
        "inner_outer_spearman_nonfull": summarize_vector(
            subject_df["inner_outer_spearman_nonfull"], rng=rng, repeats=int(args.bootstrap)
        ),
        "nested_choice_matches_outer_rate": summarize_vector(
            subject_df["nested_choice_matches_outer"], rng=rng, repeats=int(args.bootstrap)
        ),
        "inner_top1_matches_outer_rate": summarize_vector(
            subject_df["inner_top1_matches_outer"], rng=rng, repeats=int(args.bootstrap)
        ),
        "oracle_regret_pp": summarize_vector(
            subject_df["oracle_regret_pp"], rng=rng, repeats=int(args.bootstrap)
        ),
        "chosen_minus_fixed_source_pp": summarize_vector(
            subject_df["chosen_minus_fixed_source_pp"], rng=rng, repeats=int(args.bootstrap)
        ),
        "choice_bootstrap_top_rate": summarize_vector(
            subject_df["choice_bootstrap_top_rate"], rng=rng, repeats=int(args.bootstrap)
        ),
        "choice_bootstrap_entropy_norm": summarize_vector(
            subject_df["choice_bootstrap_entropy_norm"], rng=rng, repeats=int(args.bootstrap)
        ),
        "inner_risk_predicts_outer_harm_auroc_nonfull": {
            "p_harm": auroc(nonfull_pair_df["outer_harm"], nonfull_pair_df["inner_p_gain_lt_harm"]),
            "r10_loss": auroc(nonfull_pair_df["outer_harm"], nonfull_pair_df["inner_r10_loss"]),
            "neg_q05": auroc(nonfull_pair_df["outer_harm"], -nonfull_pair_df["inner_q05_gain"]),
            "neg_inner_mean": auroc(nonfull_pair_df["outer_harm"], -nonfull_pair_df["inner_mean_gain"]),
        },
        "inner_mean_predicts_outer_gain_spearman_pairwise_nonfull": spearmanr(
            nonfull_pair_df["inner_mean_gain"], nonfull_pair_df["outer_gain"]
        ),
        "inner_mean_predicts_outer_gain_spearman_pairwise_all": spearmanr(
            pair_df["inner_mean_gain"], pair_df["outer_gain"]
        ),
        "longitudinal_choice_failure_mode": {
            "n_chosen_longitudinal_q0p10": int(len(chosen_long)),
            "chosen_longitudinal_outer_gain": summarize_vector(
                chosen_long["chosen_outer_gain"], rng=rng, repeats=int(args.bootstrap)
            ),
            "chosen_longitudinal_inner_margin_vs_source_q025": summarize_vector(
                chosen_long["inner_margin_chosen_minus_source_q025"],
                rng=rng,
                repeats=int(args.bootstrap),
            ),
            "chosen_longitudinal_outer_margin_vs_source_q025": summarize_vector(
                chosen_long["outer_margin_chosen_minus_source_q025"],
                rng=rng,
                repeats=int(args.bootstrap),
            ),
        },
        "source_choice_mode": {
            "n_chosen_source_only_q0p25": int(len(chosen_source)),
            "chosen_source_outer_gain": summarize_vector(
                chosen_source["chosen_outer_gain"], rng=rng, repeats=int(args.bootstrap)
            ),
        },
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    flat_rows = []
    for key, value in summary.items():
        if isinstance(value, dict) and {"mean", "median", "q05", "q95", "bootstrap_95ci"} <= set(value):
            flat_rows.append({"metric": key, **value})
    pd.DataFrame(flat_rows).to_csv(args.output_dir / "summary_key_metrics.csv", index=False)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_subjects": int(len(subject_df)),
                "choice_counts": summary["choice_counts"],
                "nested_choice_matches_outer_rate_mean": summary["nested_choice_matches_outer_rate"]["mean"],
                "inner_outer_spearman_nonfull_mean": summary["inner_outer_spearman_nonfull"]["mean"],
                "oracle_regret_pp_mean": summary["oracle_regret_pp"]["mean"],
                "chosen_minus_fixed_source_pp_mean": summary["chosen_minus_fixed_source_pp"]["mean"],
                "risk_auroc_nonfull": summary["inner_risk_predicts_outer_harm_auroc_nonfull"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
