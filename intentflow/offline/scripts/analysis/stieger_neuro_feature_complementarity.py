"""E1b complementarity audit for Neuro-LEMA feature branches.

This script consumes the E1 ``stieger_neuro_feature_baseline.py`` summary and
asks a deliberately oracle-only question:

* Do broad all-channel and neurophysiology-constrained feature branches win on
  different sessions?
* If an ideal selector could choose among feature branches, how much headroom
  exists over the best single branch?

The oracle numbers are not a deployable method. They are a go/no-go diagnostic
for whether a learned gate / LEMA-style fusion is worth implementing.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_E1_DIR = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260624_stieger_neuro_feature_baseline"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260624_stieger_neuro_feature_complementarity"
)
DEFAULT_FEATURES = (
    "broad_all60",
    "broad_sensorimotor21",
    "fb_all60_mu_beta",
    "fb_sensorimotor21_mu_beta",
)
PRIMARY_FEATURE = "broad_all60"
NEURO_FEATURE = "fb_sensorimotor21_mu_beta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_E1_DIR / "summary.json",
        help="Path to E1 summary.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter", default="prefix_ea")
    parser.add_argument("--primary-feature", default=PRIMARY_FEATURE)
    parser.add_argument("--neuro-feature", default=NEURO_FEATURE)
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def subject_balanced_weights(subjects: Iterable[int]) -> np.ndarray:
    subjects = [int(subject) for subject in subjects]
    counts = Counter(subjects)
    n_subjects = len(counts)
    return np.asarray(
        [1.0 / (n_subjects * counts[subject]) for subject in subjects],
        dtype=np.float64,
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = weights / weights.sum()
    return float(np.sum(values * weights))


def weighted_probability(mask: np.ndarray, weights: np.ndarray) -> float:
    weights = weights / weights.sum()
    return float(np.sum(weights[mask]))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order] / weights.sum()
    cumulative = np.cumsum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, quantile, side="left")])


def lower_tail_cvar(values: np.ndarray, weights: np.ndarray, alpha: float = 0.1) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order] / weights.sum()
    target = alpha
    used = 0.0
    total = 0.0
    for value, weight in zip(sorted_values, sorted_weights):
        take = min(float(weight), target - used)
        if take <= 0:
            break
        total += float(value) * take
        used += take
    if used <= 0:
        raise ValueError("empty weights")
    return total / used


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation with tie handling."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        # one-based average rank, matching scipy.stats.rankdata(method="average")
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = float(np.sqrt(np.sum(x0**2) * np.sum(y0**2)))
    if denom <= 0:
        return None
    return float(np.sum(x0 * y0) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def bootstrap_ci_by_subject(
    records: list[Mapping[str, object]],
    metric: Callable[[list[Mapping[str, object]]], float],
    n_bootstrap: int,
    seed: int,
) -> list[float] | None:
    subjects = sorted({int(record["subject"]) for record in records})
    if len(subjects) < 3:
        return None
    by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        by_subject[int(record["subject"])].append(record)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        sampled_records: list[Mapping[str, object]] = []
        for new_subject, original_subject in enumerate(sampled):
            for record in by_subject[int(original_subject)]:
                copied = dict(record)
                copied["subject"] = int(new_subject)
                sampled_records.append(copied)
        values.append(float(metric(sampled_records)))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def summarize_values(
    records: list[Mapping[str, object]],
    value_key: str,
    prefix: str,
) -> dict[str, float]:
    subjects = [int(record["subject"]) for record in records]
    weights = subject_balanced_weights(subjects)
    values = np.asarray([float(record[value_key]) for record in records])
    return {
        f"{prefix}_mean": weighted_mean(values, weights),
        f"{prefix}_q05": weighted_quantile(values, weights, 0.05),
        f"{prefix}_risk_r10": -lower_tail_cvar(values, weights, 0.1),
        f"{prefix}_p_lt_0": weighted_probability(values < 0, weights),
        f"{prefix}_p_lt_minus5": weighted_probability(values < -5, weights),
    }


def rows_for_arm(
    rows: list[Mapping[str, object]],
    condition: str,
    feature: str,
    adapter: str,
) -> dict[tuple[int, int], Mapping[str, object]]:
    selected: dict[tuple[int, int], Mapping[str, object]] = {}
    for row in rows:
        if (
            row["condition"] == condition
            and row["feature_config"] == feature
            and row["adapter"] == adapter
        ):
            key = (int(row["subject"]), int(row["session"]))
            selected[key] = row
    return selected


def pair_records(
    rows: list[Mapping[str, object]],
    condition: str,
    adapter: str,
    primary_feature: str,
    neuro_feature: str,
) -> list[dict[str, object]]:
    primary = rows_for_arm(rows, condition, primary_feature, adapter)
    neuro = rows_for_arm(rows, condition, neuro_feature, adapter)
    keys = sorted(set(primary) & set(neuro))
    records: list[dict[str, object]] = []
    for subject, session in keys:
        a = primary[(subject, session)]
        b = neuro[(subject, session)]
        primary_acc = float(a["adapted_acc"])
        neuro_acc = float(b["adapted_acc"])
        primary_delta = float(a["delta_pp"])
        neuro_delta = float(b["delta_pp"])
        records.append(
            {
                "subject": subject,
                "session": session,
                "condition": condition,
                "adapter": adapter,
                "primary_feature": primary_feature,
                "neuro_feature": neuro_feature,
                "primary_source_acc": float(a["source_acc"]),
                "primary_adapted_acc": primary_acc,
                "primary_delta_pp": primary_delta,
                "neuro_source_acc": float(b["source_acc"]),
                "neuro_adapted_acc": neuro_acc,
                "neuro_delta_pp": neuro_delta,
                "neuro_minus_primary_acc_pp": neuro_acc - primary_acc,
                "neuro_minus_primary_delta_pp": neuro_delta - primary_delta,
                "oracle_pair_adapted_acc": max(primary_acc, neuro_acc),
                "oracle_pair_delta_pp": max(primary_delta, neuro_delta),
                "oracle_pair_winner": (
                    neuro_feature
                    if neuro_acc > primary_acc
                    else primary_feature
                    if primary_acc > neuro_acc
                    else "tie"
                ),
            }
        )
    return records


def pair_summary(
    records: list[Mapping[str, object]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    subjects = [int(record["subject"]) for record in records]
    weights = subject_balanced_weights(subjects)
    primary_acc = np.asarray([float(record["primary_adapted_acc"]) for record in records])
    neuro_acc = np.asarray([float(record["neuro_adapted_acc"]) for record in records])
    primary_delta = np.asarray([float(record["primary_delta_pp"]) for record in records])
    neuro_delta = np.asarray([float(record["neuro_delta_pp"]) for record in records])
    diff_acc = neuro_acc - primary_acc
    diff_delta = neuro_delta - primary_delta
    oracle_acc = np.maximum(primary_acc, neuro_acc)
    oracle_delta = np.maximum(primary_delta, neuro_delta)
    abs_diff = np.abs(diff_acc)
    primary_harm5 = primary_delta < -5
    neuro_harm5 = neuro_delta < -5

    def oracle_gain_vs_primary(sample: list[Mapping[str, object]]) -> float:
        sample_weights = subject_balanced_weights(int(r["subject"]) for r in sample)
        primary = np.asarray([float(r["primary_adapted_acc"]) for r in sample])
        oracle = np.asarray([float(r["oracle_pair_adapted_acc"]) for r in sample])
        return weighted_mean(oracle - primary, sample_weights)

    def oracle_gain_vs_best_single(sample: list[Mapping[str, object]]) -> float:
        sample_weights = subject_balanced_weights(int(r["subject"]) for r in sample)
        primary = np.asarray([float(r["primary_adapted_acc"]) for r in sample])
        neuro = np.asarray([float(r["neuro_adapted_acc"]) for r in sample])
        oracle = np.asarray([float(r["oracle_pair_adapted_acc"]) for r in sample])
        best_single = max(weighted_mean(primary, sample_weights), weighted_mean(neuro, sample_weights))
        return weighted_mean(oracle, sample_weights) - best_single

    summary: dict[str, object] = {
        "n_subjects": len(set(subjects)),
        "n_sessions": len(records),
        "primary_adapted_acc_mean": weighted_mean(primary_acc, weights),
        "neuro_adapted_acc_mean": weighted_mean(neuro_acc, weights),
        "neuro_minus_primary_acc_mean": weighted_mean(diff_acc, weights),
        "neuro_minus_primary_delta_mean": weighted_mean(diff_delta, weights),
        "neuro_win_rate": weighted_probability(diff_acc > 0, weights),
        "primary_win_rate": weighted_probability(diff_acc < 0, weights),
        "tie_rate": weighted_probability(diff_acc == 0, weights),
        "neuro_win_ge3pp_rate": weighted_probability(diff_acc >= 3, weights),
        "primary_win_ge3pp_rate": weighted_probability(diff_acc <= -3, weights),
        "abs_acc_diff_mean": weighted_mean(abs_diff, weights),
        "abs_acc_diff_ge3pp_rate": weighted_probability(abs_diff >= 3, weights),
        "abs_acc_diff_ge5pp_rate": weighted_probability(abs_diff >= 5, weights),
        "spearman_adapted_acc": spearman(primary_acc, neuro_acc),
        "spearman_delta_pp": spearman(primary_delta, neuro_delta),
        "oracle_pair_adapted_acc_mean": weighted_mean(oracle_acc, weights),
        "oracle_pair_delta_mean": weighted_mean(oracle_delta, weights),
        "oracle_gain_vs_primary_acc_pp": weighted_mean(oracle_acc - primary_acc, weights),
        "oracle_gain_vs_neuro_acc_pp": weighted_mean(oracle_acc - neuro_acc, weights),
        "oracle_gain_vs_best_single_acc_pp": weighted_mean(oracle_acc, weights)
        - max(weighted_mean(primary_acc, weights), weighted_mean(neuro_acc, weights)),
        "oracle_gain_vs_primary_bootstrap_95ci": bootstrap_ci_by_subject(
            records, oracle_gain_vs_primary, n_bootstrap, seed
        ),
        "oracle_gain_vs_best_single_bootstrap_95ci": bootstrap_ci_by_subject(
            records, oracle_gain_vs_best_single, n_bootstrap, seed + 1
        ),
        "primary_harm5_rate": weighted_probability(primary_harm5, weights),
        "neuro_harm5_rate": weighted_probability(neuro_harm5, weights),
        "both_harm5_rate": weighted_probability(primary_harm5 & neuro_harm5, weights),
        "primary_only_harm5_rate": weighted_probability(primary_harm5 & ~neuro_harm5, weights),
        "neuro_only_harm5_rate": weighted_probability(~primary_harm5 & neuro_harm5, weights),
        "neither_harm5_rate": weighted_probability(~primary_harm5 & ~neuro_harm5, weights),
        "oracle_delta_harm5_rate": weighted_probability(oracle_delta < -5, weights),
    }
    summary.update(summarize_values(records, "neuro_minus_primary_acc_pp", "neuro_minus_primary_acc"))
    summary.update(summarize_values(records, "oracle_pair_delta_pp", "oracle_pair_delta"))
    return summary


def multi_arm_oracle_records(
    rows: list[Mapping[str, object]],
    condition: str,
    adapter: str,
    features: list[str],
) -> list[dict[str, object]]:
    arms = {
        feature: rows_for_arm(rows, condition, feature, adapter)
        for feature in features
    }
    common_keys = set.intersection(*(set(values) for values in arms.values()))
    records: list[dict[str, object]] = []
    for subject, session in sorted(common_keys):
        candidates = []
        for feature in features:
            row = arms[feature][(subject, session)]
            candidates.append(
                {
                    "feature": feature,
                    "adapted_acc": float(row["adapted_acc"]),
                    "delta_pp": float(row["delta_pp"]),
                    "source_acc": float(row["source_acc"]),
                }
            )
        best_acc = max(candidate["adapted_acc"] for candidate in candidates)
        best_delta = max(candidate["delta_pp"] for candidate in candidates)
        winners = [
            candidate["feature"]
            for candidate in candidates
            if candidate["adapted_acc"] == best_acc
        ]
        record: dict[str, object] = {
            "subject": subject,
            "session": session,
            "condition": condition,
            "adapter": adapter,
            "oracle_adapted_acc": best_acc,
            "oracle_delta_pp": best_delta,
            "oracle_winner": "|".join(winners),
        }
        for candidate in candidates:
            prefix = candidate["feature"]
            record[f"{prefix}_adapted_acc"] = candidate["adapted_acc"]
            record[f"{prefix}_delta_pp"] = candidate["delta_pp"]
            record[f"{prefix}_source_acc"] = candidate["source_acc"]
        records.append(record)
    return records


def multi_arm_oracle_summary(
    records: list[Mapping[str, object]],
    features: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    subjects = [int(record["subject"]) for record in records]
    weights = subject_balanced_weights(subjects)
    oracle_acc = np.asarray([float(record["oracle_adapted_acc"]) for record in records])
    oracle_delta = np.asarray([float(record["oracle_delta_pp"]) for record in records])
    single_means = {}
    single_r10 = {}
    for feature in features:
        acc = np.asarray([float(record[f"{feature}_adapted_acc"]) for record in records])
        delta = np.asarray([float(record[f"{feature}_delta_pp"]) for record in records])
        single_means[feature] = weighted_mean(acc, weights)
        single_r10[feature] = -lower_tail_cvar(delta, weights, 0.1)
    best_single_feature = max(single_means, key=single_means.get)
    best_single_acc = single_means[best_single_feature]
    best_single_r10_feature = min(single_r10, key=single_r10.get)

    def gain_vs_best_single(sample: list[Mapping[str, object]]) -> float:
        sample_weights = subject_balanced_weights(int(r["subject"]) for r in sample)
        sample_oracle = np.asarray([float(r["oracle_adapted_acc"]) for r in sample])
        sample_single_means = []
        for feature in features:
            sample_acc = np.asarray([float(r[f"{feature}_adapted_acc"]) for r in sample])
            sample_single_means.append(weighted_mean(sample_acc, sample_weights))
        return weighted_mean(sample_oracle, sample_weights) - max(sample_single_means)

    winner_counts = Counter(str(record["oracle_winner"]) for record in records)
    winner_rates = {}
    for feature in features:
        mask = np.asarray(
            [feature in str(record["oracle_winner"]).split("|") for record in records]
        )
        winner_rates[feature] = weighted_probability(mask, weights)

    summary: dict[str, object] = {
        "n_subjects": len(set(subjects)),
        "n_sessions": len(records),
        "features": features,
        "single_adapted_acc_mean": single_means,
        "single_r10": single_r10,
        "best_single_feature_by_acc": best_single_feature,
        "best_single_acc_mean": best_single_acc,
        "best_single_feature_by_r10": best_single_r10_feature,
        "oracle_adapted_acc_mean": weighted_mean(oracle_acc, weights),
        "oracle_delta_mean": weighted_mean(oracle_delta, weights),
        "oracle_r10": -lower_tail_cvar(oracle_delta, weights, 0.1),
        "oracle_q05_delta_pp": weighted_quantile(oracle_delta, weights, 0.05),
        "oracle_p_delta_lt_minus5": weighted_probability(oracle_delta < -5, weights),
        "oracle_gain_vs_best_single_acc_pp": weighted_mean(oracle_acc, weights)
        - best_single_acc,
        "oracle_gain_vs_best_single_bootstrap_95ci": bootstrap_ci_by_subject(
            records, gain_vs_best_single, n_bootstrap, seed
        ),
        "winner_counts_unweighted": dict(winner_counts),
        "winner_rates_subject_balanced": winner_rates,
    }
    return summary


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def decision_from_pair(summary: Mapping[str, object]) -> str:
    oracle_gain = float(summary["oracle_gain_vs_best_single_acc_pp"])
    ci = summary.get("oracle_gain_vs_best_single_bootstrap_95ci")
    ci_low = float(ci[0]) if isinstance(ci, list) else None
    abs_ge3 = float(summary["abs_acc_diff_ge3pp_rate"])
    corr = summary.get("spearman_adapted_acc")
    corr_value = float(corr) if corr is not None else 1.0
    if oracle_gain >= 1.0 and (ci_low is None or ci_low > 0.2) and abs_ge3 >= 0.25:
        return "strong_feature_selection_headroom"
    if oracle_gain >= 0.5 and abs_ge3 >= 0.20 and corr_value < 0.85:
        return "moderate_feature_selection_headroom"
    if oracle_gain >= 0.25:
        return "weak_headroom_diagnostic_only"
    return "little_feature_selection_headroom"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.summary_json.read_text())
    rows = payload["rows"]
    conditions = sorted({str(row["condition"]) for row in rows})

    pair_summaries: dict[str, object] = {}
    all_pair_records: list[dict[str, object]] = []
    multi_summaries: dict[str, object] = {}
    all_multi_records: list[dict[str, object]] = []

    for condition in conditions:
        records = pair_records(
            rows,
            condition,
            args.adapter,
            args.primary_feature,
            args.neuro_feature,
        )
        summary = pair_summary(records, args.bootstrap, args.seed)
        summary["decision"] = decision_from_pair(summary)
        pair_summaries[condition] = summary
        all_pair_records.extend(records)

        multi_records = multi_arm_oracle_records(
            rows,
            condition,
            args.adapter,
            list(args.features),
        )
        multi_summaries[condition] = multi_arm_oracle_summary(
            multi_records,
            list(args.features),
            args.bootstrap,
            args.seed + 100,
        )
        all_multi_records.extend(multi_records)

    report = {
        "config": {
            "summary_json": str(args.summary_json),
            "adapter": args.adapter,
            "primary_feature": args.primary_feature,
            "neuro_feature": args.neuro_feature,
            "features": list(args.features),
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        "pair_summaries": pair_summaries,
        "multi_arm_oracle_summaries": multi_summaries,
    }
    (args.output_dir / "complementarity_summary.json").write_text(
        json.dumps(report, indent=2)
    )
    write_csv(args.output_dir / "pair_records.csv", all_pair_records)
    write_csv(args.output_dir / "multi_arm_oracle_records.csv", all_multi_records)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
