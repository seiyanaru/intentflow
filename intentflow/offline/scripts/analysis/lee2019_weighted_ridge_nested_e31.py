"""E31 exact nested validation for Lee2019 weighted Ridge.

E30 found a promising post-hoc result:

    RidgeClassifier + source-score soft weighting

This script tests whether that result survives honest source-side nested
hyperparameter validation.  For each held target subject H:

1. Build inner source-validation tasks from subjects S != H.
2. For each inner S, compute source-side scores using subjects excluding H and S.
3. Evaluate weighted Ridge hyperparameters on S session0 -> session1.
4. Select hyperparameters using only inner source-validation.
5. Recompute scores excluding H and evaluate once on H.

No held target labels are used to choose hyperparameters or feature weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lee2019_exact_nested_subspace_selection import aggregate_excluding  # noqa: E402
from lee2019_longitudinal_selection_pilot import (  # noqa: E402
    SCORE_FUNCTIONS,
    add_stats,
    load_subject_features,
    parse_subjects,
    session_metric_stats,
)
from lee2019_target_prefix_shift_e29 import feature_shift_metrics  # noqa: E402


RESULTS_DIR = SCRIPT_DIR.parents[2] / "results" / "research_outputs"
DEFAULT_CACHE = RESULTS_DIR / "260627_lee2019_longitudinal_selection_pilot12" / "subject_cache"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_lee2019_weighted_ridge_nested_e31"


@dataclass(frozen=True)
class CandidateSpec:
    method: str
    alpha: float
    gamma: float | None
    prefix_lambda: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--target-subjects", default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--prefix-lambdas", nargs="+", type=float, default=[0.0, 0.25])
    parser.add_argument("--risk-threshold", type=float, default=0.20)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_candidates(
    *, alphas: Sequence[float], gammas: Sequence[float], prefix_lambdas: Sequence[float]
) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    for alpha in alphas:
        candidates.append(CandidateSpec(method=f"ridge_full_a{token(alpha)}", alpha=float(alpha), gamma=None, prefix_lambda=None))
        for gamma in gammas:
            for lam in prefix_lambdas:
                candidates.append(
                    CandidateSpec(
                        method=f"ridge_source_rank_g{token(gamma)}_l{token(lam)}_a{token(alpha)}",
                        alpha=float(alpha),
                        gamma=float(gamma),
                        prefix_lambda=float(lam),
                    )
                )
    return candidates


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: Iterable[float], rng: np.random.Generator, repeats: int) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(arr, size=(int(repeats), arr.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize(records: Sequence[Mapping[str, object]], *, bootstrap: int, seed: int) -> dict[str, object]:
    acc = np.asarray([float(row["accuracy"]) for row in records], dtype=np.float64)
    rng = np.random.default_rng(seed)
    return {
        "n_subjects": int(len(records)),
        "accuracy_mean": float(acc.mean()),
        "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, bootstrap),
        "accuracy_q05": float(np.quantile(acc, 0.05)),
    }


def percentile_rank(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    if len(ranks) <= 1:
        return np.ones_like(ranks)
    return (ranks - 1.0) / (len(ranks) - 1.0)


def weights_from_spec(
    spec: CandidateSpec,
    *,
    stats: Mapping[str, np.ndarray],
    payload: Mapping[str, np.ndarray],
) -> np.ndarray | None:
    if spec.gamma is None:
        return None
    source_rank = percentile_rank(SCORE_FUNCTIONS["source_only"](stats))
    if spec.prefix_lambda and spec.prefix_lambda > 0:
        prefix_risk_rank = percentile_rank(feature_shift_metrics(payload)["prefix_energy_z"])
        adjusted = source_rank - float(spec.prefix_lambda) * prefix_risk_rank
        score = percentile_rank(adjusted)
    else:
        score = source_rank
    return np.clip(score, 0.05, 1.0) ** float(spec.gamma)


def scaled_payload(payload: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    scaler = StandardScaler()
    xs = scaler.fit_transform(np.asarray(payload["source_features"], dtype=np.float64))
    xt = scaler.transform(np.asarray(payload["target_features"], dtype=np.float64))
    return {
        "source_features": xs,
        "source_labels": np.asarray(payload["source_labels"], dtype=np.int64),
        "target_features": xt,
        "target_labels": np.asarray(payload["target_labels"], dtype=np.int64),
    }


def fit_scaled_ridge_accuracy(
    scaled: Mapping[str, np.ndarray],
    *,
    weights: np.ndarray | None,
    alpha: float,
) -> float:
    xs = np.asarray(scaled["source_features"], dtype=np.float64)
    xt = np.asarray(scaled["target_features"], dtype=np.float64)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        xs = xs * w[None, :]
        xt = xt * w[None, :]
    clf = RidgeClassifier(alpha=float(alpha))
    clf.fit(xs, scaled["source_labels"])
    pred = clf.predict(xt)
    return float(np.mean(pred == scaled["target_labels"]) * 100.0)


def metrics_from_inner_rows(
    rows: Sequence[Mapping[str, object]], candidates: Sequence[CandidateSpec]
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for spec in candidates:
        method = spec.method
        acc = np.asarray([float(row[f"{method}__accuracy"]) for row in rows], dtype=np.float64)
        full = np.asarray([float(row[f"ridge_full_a{token(spec.alpha)}__accuracy"]) for row in rows], dtype=np.float64)
        gain = acc - full
        metrics[method] = {
            "mean_accuracy": float(acc.mean()),
            "mean_gain_vs_same_alpha_full": float(gain.mean()),
            "p_gain_vs_same_alpha_full_lt_minus5": float(np.mean(gain < -5.0)),
            "q05_gain_vs_same_alpha_full": float(np.quantile(gain, 0.05)),
        }
    return metrics


def choose_mean(metrics: Mapping[str, Mapping[str, float]]) -> str:
    return sorted(metrics, key=lambda m: metrics[m]["mean_accuracy"], reverse=True)[0]


def choose_risk(metrics: Mapping[str, Mapping[str, float]], threshold: float) -> str:
    eligible = [
        method
        for method, item in metrics.items()
        if float(item["p_gain_vs_same_alpha_full_lt_minus5"]) <= float(threshold)
    ]
    if not eligible:
        eligible = list(metrics)
    return sorted(
        eligible,
        key=lambda m: (
            metrics[m]["mean_accuracy"],
            metrics[m]["mean_gain_vs_same_alpha_full"],
            metrics[m]["q05_gain_vs_same_alpha_full"],
        ),
        reverse=True,
    )[0]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    target_subjects = parse_subjects(args.target_subjects) if args.target_subjects else subjects
    unknown = sorted(set(target_subjects) - set(subjects))
    if unknown:
        raise ValueError(f"--target-subjects must be subset of --subjects: {unknown}")

    candidates = build_candidates(
        alphas=args.alphas,
        gammas=args.gammas,
        prefix_lambdas=args.prefix_lambdas,
    )

    payloads: dict[int, dict[str, np.ndarray]] = {}
    scaled_by_subject: dict[int, dict[str, np.ndarray]] = {}
    for subject in subjects:
        payload = load_subject_features(
            subject,
            cache_dir=args.cache_dir,
            prefix=int(args.prefix),
            eval_start=int(args.eval_start),
            force_cache=bool(args.force_cache),
        )
        payloads[subject] = payload
        scaled_by_subject[subject] = scaled_payload(payload)
        if not args.quiet:
            print(f"S{subject}: loaded", flush=True)

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

    validation_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []

    for held_subject in target_subjects:
        inner_rows: list[dict[str, object]] = []
        for inner_subject in subjects:
            if inner_subject == held_subject:
                continue
            stats = aggregate_excluding(total_stats, stats_by_subject, {held_subject, inner_subject})
            scaled = scaled_by_subject[inner_subject]
            payload = payloads[inner_subject]
            row: dict[str, object] = {
                "held_subject": int(held_subject),
                "inner_subject": int(inner_subject),
            }
            for spec in candidates:
                weights = weights_from_spec(spec, stats=stats, payload=payload)
                acc = fit_scaled_ridge_accuracy(scaled, weights=weights, alpha=spec.alpha)
                row[f"{spec.method}__accuracy"] = acc
            inner_rows.append(row)
            validation_rows.append(row)

        metrics = metrics_from_inner_rows(inner_rows, candidates)
        chosen_mean = choose_mean(metrics)
        chosen_risk = choose_risk(metrics, float(args.risk_threshold))

        outer_stats = aggregate_excluding(total_stats, stats_by_subject, {held_subject})
        outer_scaled = scaled_by_subject[held_subject]
        outer_payload = payloads[held_subject]
        outer_acc_by_method: dict[str, float] = {}
        for spec in candidates:
            weights = weights_from_spec(spec, stats=outer_stats, payload=outer_payload)
            acc = fit_scaled_ridge_accuracy(outer_scaled, weights=weights, alpha=spec.alpha)
            outer_acc_by_method[spec.method] = acc
            fixed_rows.append(
                {
                    "subject": int(held_subject),
                    "method": spec.method,
                    "accuracy": acc,
                    "alpha": spec.alpha,
                    "gamma": spec.gamma if spec.gamma is not None else "",
                    "prefix_lambda": spec.prefix_lambda if spec.prefix_lambda is not None else "",
                }
            )

        for policy, chosen in [("nested_mean", chosen_mean), ("nested_risk20", chosen_risk)]:
            selected_rows.append(
                {
                    "subject": int(held_subject),
                    "policy": policy,
                    "chosen_method": chosen,
                    "accuracy": outer_acc_by_method[chosen],
                    "source_validation_metrics": json.dumps(metrics[chosen], sort_keys=True),
                }
            )
        if not args.quiet:
            print(
                f"S{held_subject}: mean={chosen_mean} acc={outer_acc_by_method[chosen_mean]:.1f} "
                f"risk={chosen_risk} acc={outer_acc_by_method[chosen_risk]:.1f}",
                flush=True,
            )

    write_csv(args.output_dir / "nested_selection_records.csv", selected_rows)
    write_csv(args.output_dir / "fixed_candidate_records.csv", fixed_rows)
    write_csv(args.output_dir / "source_validation_records.csv", validation_rows)

    summary_rows: list[dict[str, object]] = []
    for policy in sorted({str(row["policy"]) for row in selected_rows}):
        rows = [row for row in selected_rows if row["policy"] == policy]
        choices = pd.Series([row["chosen_method"] for row in rows]).value_counts().sort_index().to_dict()
        summary_rows.append(
            {
                "method": policy,
                **summarize(rows, bootstrap=int(args.bootstrap), seed=int(args.seed)),
                "chosen_method_counts": choices,
            }
        )
    for method in sorted({str(row["method"]) for row in fixed_rows}):
        rows = [row for row in fixed_rows if row["method"] == method]
        summary_rows.append(
            {
                "method": f"fixed__{method}",
                **summarize(rows, bootstrap=int(args.bootstrap), seed=int(args.seed)),
                "chosen_method_counts": {method: len(rows)},
            }
        )
    write_csv(args.output_dir / "nested_weighted_ridge_summary.csv", summary_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "subjects": subjects,
                    "target_subjects": target_subjects,
                    "cache_dir": str(args.cache_dir),
                    "candidates": [spec.__dict__ for spec in candidates],
                    "risk_threshold": float(args.risk_threshold),
                    "prefix": int(args.prefix),
                    "eval_start": int(args.eval_start),
                    "bootstrap": int(args.bootstrap),
                    "seed": int(args.seed),
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
                "n_target_subjects": len(target_subjects),
                "n_candidates": len(candidates),
                "summary_csv": str(args.output_dir / "nested_weighted_ridge_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
