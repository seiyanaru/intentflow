"""E30 Lee2019 weighted Ridge pilot.

Motivation:
    E29 found weak target-prefix risk signal, but hard candidate selection was
    not solved.  This pilot asks whether soft feature weighting can use the
    signal better than hard subset selection.

This is deliberately small:
    - no target labels are used for weighting or fitting;
    - source-side scores are learned from non-held subjects;
    - target-prefix metrics are computed from held target prefix without labels;
    - evaluation uses held target eval labels only after prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_exact_nested_subspace_selection import aggregate_excluding, candidate_indices  # noqa: E402
from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    SCORE_FUNCTIONS,
    add_stats,
    load_subject_features,
    parse_subjects,
    session_metric_stats,
    top_fraction_indices,
)
from lee2019_target_prefix_shift_e29 import feature_shift_metrics  # noqa: E402


RESULTS_DIR = SCRIPT_DIR.parents[2] / "results" / "research_outputs"
DEFAULT_CACHE = RESULTS_DIR / "260627_lee2019_longitudinal_selection_pilot12" / "subject_cache"
DEFAULT_LDA = (
    RESULTS_DIR
    / "260630_lee2019_exact_nested_subspace_selection_e25_full"
    / "merged"
    / "fixed_candidate_records.csv"
)
DEFAULT_OUTPUT = RESULTS_DIR / "260701_lee2019_weighted_ridge_e30"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--lda-fixed-records", type=Path, default=DEFAULT_LDA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def bootstrap_ci(values: Iterable[float], rng: np.random.Generator, repeats: int) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(arr, size=(int(repeats), arr.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize(records: pd.DataFrame, *, bootstrap: int, seed: int) -> dict[str, object]:
    acc = records["accuracy"].to_numpy(dtype=np.float64)
    full = records["full_accuracy"].to_numpy(dtype=np.float64)
    gain = acc - full
    tail = np.sort(gain)[: max(1, int(np.ceil(0.1 * gain.size)))]
    rng = np.random.default_rng(seed)
    return {
        "n_subjects": int(len(records)),
        "accuracy_mean": float(acc.mean()),
        "full_accuracy_mean": float(full.mean()),
        "gain_vs_ridge_full_mean_pp": float(gain.mean()),
        "gain_vs_ridge_full_subject_bootstrap_95ci": bootstrap_ci(gain, rng, bootstrap),
        "gain_vs_ridge_full_q05_pp": float(np.quantile(gain, 0.05)),
        "loss_r10_vs_ridge_full_pp": float(-tail.mean()),
        "p_gain_vs_ridge_full_lt_minus5": float(np.mean(gain < -5.0)),
    }


def percentile_rank(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    if len(ranks) <= 1:
        return np.ones_like(ranks)
    return (ranks - 1.0) / (len(ranks) - 1.0)


def weights_from_score(score: np.ndarray, *, gamma: float, floor: float = 0.05) -> np.ndarray:
    rank = percentile_rank(score)
    return np.clip(rank, floor, 1.0) ** float(gamma)


def fit_ridge_accuracy(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    *,
    weights: np.ndarray | None,
    alpha: float,
) -> float:
    scaler = StandardScaler()
    xs = scaler.fit_transform(source_features)
    xt = scaler.transform(target_features)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        xs = xs * w[None, :]
        xt = xt * w[None, :]
    clf = RidgeClassifier(alpha=float(alpha))
    clf.fit(xs, source_labels)
    pred = clf.predict(xt)
    return float(np.mean(pred == target_labels) * 100.0)


def fit_lda_subset_accuracy(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    indices: np.ndarray,
) -> float:
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(source_features[:, indices], source_labels)
    pred = clf.predict(target_features[:, indices])
    return float(np.mean(pred == target_labels) * 100.0)


def method_weight(
    method: str,
    *,
    source_score: np.ndarray,
    longitudinal_score: np.ndarray,
    prefix_risk: np.ndarray,
) -> np.ndarray | None:
    if method == "ridge_full":
        return None
    if method.startswith("ridge_weight_source_rank_g"):
        gamma = float(method.split("_g", maxsplit=1)[1].replace("p", "."))
        return weights_from_score(source_score, gamma=gamma)
    if method.startswith("ridge_weight_longitudinal_rank_g"):
        gamma = float(method.split("_g", maxsplit=1)[1].replace("p", "."))
        return weights_from_score(longitudinal_score, gamma=gamma)
    if method.startswith("ridge_weight_source_minus_prefix_l"):
        token = method.split("_l", maxsplit=1)[1]
        lam = float(token.replace("p", "."))
        adjusted = percentile_rank(source_score) - lam * percentile_rank(prefix_risk)
        return weights_from_score(adjusted, gamma=1.0)
    raise ValueError(f"Unknown weighted method: {method}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)

    payloads: dict[int, dict[str, np.ndarray]] = {}
    for subject in subjects:
        payloads[subject] = load_subject_features(
            subject,
            cache_dir=args.cache_dir,
            prefix=int(args.prefix),
            eval_start=int(args.eval_start),
            force_cache=False,
        )

    stats_by_subject: dict[int, dict[str, np.ndarray]] = {}
    total_stats: dict[str, np.ndarray] = {}
    for subject, payload in payloads.items():
        stats = session_metric_stats(
            payload["source_features"],
            payload["source_labels"],
            payload["target_features"],
            payload["target_labels"],
        )
        stats_by_subject[subject] = stats
        add_stats(total_stats, stats)

    weighted_methods = [
        "ridge_full",
        "ridge_weight_source_rank_g0p5",
        "ridge_weight_source_rank_g1p0",
        "ridge_weight_source_rank_g2p0",
        "ridge_weight_longitudinal_rank_g1p0",
        "ridge_weight_source_minus_prefix_l0p25",
        "ridge_weight_source_minus_prefix_l0p5",
        "ridge_weight_source_minus_prefix_l1p0",
        "ridge_weight_source_minus_prefix_l2p0",
    ]
    hard_methods = [
        "ridge_hard_source_only_q0p25",
        "ridge_hard_source_only_q0p50",
        "ridge_hard_longitudinal_q0p10",
        "ridge_hard_longitudinal_q0p25",
        "lda_hard_source_only_q0p25_recomputed",
    ]

    rows: list[dict[str, object]] = []
    for held_subject in subjects:
        payload = payloads[held_subject]
        outer_stats = aggregate_excluding(total_stats, stats_by_subject, {held_subject})
        source_score = SCORE_FUNCTIONS["source_only"](outer_stats)
        longitudinal_score = SCORE_FUNCTIONS["longitudinal"](outer_stats)
        prefix_risk = feature_shift_metrics(payload)["prefix_energy_z"]

        xs = np.asarray(payload["source_features"], dtype=np.float64)
        ys = np.asarray(payload["source_labels"], dtype=np.int64)
        xt = np.asarray(payload["target_features"], dtype=np.float64)
        yt = np.asarray(payload["target_labels"], dtype=np.int64)

        ridge_full_acc = fit_ridge_accuracy(
            xs, ys, xt, yt, weights=None, alpha=float(args.ridge_alpha)
        )
        for method in weighted_methods:
            weights = method_weight(
                method,
                source_score=source_score,
                longitudinal_score=longitudinal_score,
                prefix_risk=prefix_risk,
            )
            acc = fit_ridge_accuracy(
                xs, ys, xt, yt, weights=weights, alpha=float(args.ridge_alpha)
            )
            rows.append(
                {
                    "subject": int(held_subject),
                    "method": method,
                    "accuracy": acc,
                    "full_accuracy": ridge_full_acc,
                    "gain_vs_ridge_full": acc - ridge_full_acc,
                    "n_selected": int(xs.shape[1]),
                    "ridge_alpha": float(args.ridge_alpha),
                }
            )

        for method in hard_methods:
            if method == "ridge_hard_source_only_q0p25":
                indices = top_fraction_indices(source_score, 0.25)
                acc = fit_ridge_accuracy(
                    xs[:, indices],
                    ys,
                    xt[:, indices],
                    yt,
                    weights=None,
                    alpha=float(args.ridge_alpha),
                )
            elif method == "ridge_hard_source_only_q0p50":
                indices = top_fraction_indices(source_score, 0.50)
                acc = fit_ridge_accuracy(
                    xs[:, indices],
                    ys,
                    xt[:, indices],
                    yt,
                    weights=None,
                    alpha=float(args.ridge_alpha),
                )
            elif method == "ridge_hard_longitudinal_q0p10":
                indices = top_fraction_indices(longitudinal_score, 0.10)
                acc = fit_ridge_accuracy(
                    xs[:, indices],
                    ys,
                    xt[:, indices],
                    yt,
                    weights=None,
                    alpha=float(args.ridge_alpha),
                )
            elif method == "ridge_hard_longitudinal_q0p25":
                indices = top_fraction_indices(longitudinal_score, 0.25)
                acc = fit_ridge_accuracy(
                    xs[:, indices],
                    ys,
                    xt[:, indices],
                    yt,
                    weights=None,
                    alpha=float(args.ridge_alpha),
                )
            elif method == "lda_hard_source_only_q0p25_recomputed":
                indices = candidate_indices("source_only_q0p25", outer_stats, xs.shape[1])
                acc = fit_lda_subset_accuracy(xs, ys, xt, yt, indices)
            else:
                raise ValueError(method)
            rows.append(
                {
                    "subject": int(held_subject),
                    "method": method,
                    "accuracy": acc,
                    "full_accuracy": ridge_full_acc,
                    "gain_vs_ridge_full": acc - ridge_full_acc,
                    "n_selected": int(len(indices)),
                    "ridge_alpha": float(args.ridge_alpha),
                }
            )

    records = pd.DataFrame(rows).sort_values(["method", "subject"])
    records.to_csv(args.output_dir / "weighted_ridge_records.csv", index=False)
    summary_rows = [
        {"method": method, **summarize(sub, bootstrap=int(args.bootstrap), seed=int(args.seed))}
        for method, sub in records.groupby("method", sort=True)
    ]
    summary = pd.DataFrame(summary_rows).sort_values("accuracy_mean", ascending=False)

    lda_fixed = pd.read_csv(args.lda_fixed_records)
    lda_source = lda_fixed[lda_fixed["chosen_candidate"] == "source_only_q0p25"].copy()
    lda_full = lda_fixed[lda_fixed["chosen_candidate"] == "full_q1p00"].copy()
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "method": "reference_lda_full_q1p00",
                        "n_subjects": int(len(lda_full)),
                        "accuracy_mean": float(lda_full["accuracy"].mean()),
                        "full_accuracy_mean": float(lda_full["accuracy"].mean()),
                        "gain_vs_ridge_full_mean_pp": np.nan,
                        "gain_vs_ridge_full_subject_bootstrap_95ci": "",
                        "gain_vs_ridge_full_q05_pp": np.nan,
                        "loss_r10_vs_ridge_full_pp": np.nan,
                        "p_gain_vs_ridge_full_lt_minus5": np.nan,
                    },
                    {
                        "method": "reference_lda_source_only_q0p25",
                        "n_subjects": int(len(lda_source)),
                        "accuracy_mean": float(lda_source["accuracy"].mean()),
                        "full_accuracy_mean": float(lda_full["accuracy"].mean()),
                        "gain_vs_ridge_full_mean_pp": np.nan,
                        "gain_vs_ridge_full_subject_bootstrap_95ci": "",
                        "gain_vs_ridge_full_q05_pp": np.nan,
                        "loss_r10_vs_ridge_full_pp": np.nan,
                        "p_gain_vs_ridge_full_lt_minus5": np.nan,
                    },
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values("accuracy_mean", ascending=False)
    summary.to_csv(args.output_dir / "weighted_ridge_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "subjects": subjects,
                    "cache_dir": str(args.cache_dir),
                    "output_dir": str(args.output_dir),
                    "ridge_alpha": float(args.ridge_alpha),
                    "weighted_methods": weighted_methods,
                    "hard_methods": hard_methods,
                },
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_subjects": len(subjects),
                "summary_csv": str(args.output_dir / "weighted_ridge_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
