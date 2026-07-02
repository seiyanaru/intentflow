"""E32 regime-transfer check for source-score soft-weighted Ridge.

This script evaluates the E31 Lee-derived method

    ridge_source_rank_g2_a100

on additional regimes using cached tangent features:

* Lee2019 sensorimotor20
* BNCI2014_001 full-source
* BNCI2014_001 source-scarce simulation

The purpose is not to retune the method, but to test the regime hypothesis:
soft source-weighted Ridge should help most in high-dimensional/source-scarce
tangent regimes and less in compact low-p/n regimes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_weighted_ridge_regime_transfer_e32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regime-label", required=True)
    parser.add_argument("--per-class", type=int, default=0, help="0 uses all source trials.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_subjects(specification: str) -> list[int]:
    subjects: list[int] = []
    for chunk in specification.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", maxsplit=1)
            subjects.extend(range(int(start), int(end) + 1))
        else:
            subjects.append(int(chunk))
    return sorted(dict.fromkeys(subjects))


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


def class_centroids_and_within(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    variances = []
    for label in sorted(map(int, np.unique(labels))):
        selected = features[labels == label]
        if len(selected) == 0:
            raise ValueError(f"missing class {label}")
        centroids.append(selected.mean(axis=0))
        variances.append(selected.var(axis=0))
    return np.asarray(centroids, dtype=np.float64), np.mean(variances, axis=0)


def session_metric_stats(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    source_centroids, source_within = class_centroids_and_within(source_features, source_labels)
    target_centroids, target_within = class_centroids_and_within(target_features, target_labels)
    return {
        "source_between": source_centroids.var(axis=0),
        "target_between": target_centroids.var(axis=0),
        "same_class_drift": np.mean((target_centroids - source_centroids) ** 2, axis=0),
        "source_within": source_within,
        "target_within": target_within,
        "count": np.asarray([1.0], dtype=np.float64),
    }


def add_stats(target: dict[str, np.ndarray], stats: Mapping[str, np.ndarray]) -> None:
    for key, value in stats.items():
        if key not in target:
            target[key] = np.asarray(value, dtype=np.float64).copy()
        else:
            target[key] += np.asarray(value, dtype=np.float64)


def aggregate_stats(
    payloads: Mapping[int, Mapping[str, np.ndarray]],
    held_subject: int,
) -> dict[str, np.ndarray]:
    total: dict[str, np.ndarray] = {}
    for subject, payload in payloads.items():
        if int(subject) == int(held_subject):
            continue
        stats = session_metric_stats(
            np.asarray(payload["source_features"], dtype=np.float64),
            np.asarray(payload["source_labels"], dtype=np.int64),
            np.asarray(payload["target_features"], dtype=np.float64),
            np.asarray(payload["target_labels"], dtype=np.int64),
        )
        add_stats(total, stats)
    count = float(total["count"][0])
    averaged = {key: value / count for key, value in total.items() if key != "count"}
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged


def source_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.maximum(stats["source_between"], 0.0) / (
        np.maximum(stats["source_within"], 0.0) + 1e-8
    )


def longitudinal_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    signal = np.maximum(stats["source_between"], 0.0) + np.maximum(stats["target_between"], 0.0)
    nuisance = (
        np.maximum(stats["same_class_drift"], 0.0)
        + 0.25 * np.maximum(stats["source_within"], 0.0)
        + 0.25 * np.maximum(stats["target_within"], 0.0)
        + 1e-8
    )
    return signal / nuisance


def percentile_rank(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    if len(ranks) <= 1:
        return np.ones_like(ranks)
    return (ranks - 1.0) / (len(ranks) - 1.0)


def rank_weights(score: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(percentile_rank(score), 0.05, 1.0) ** float(gamma)


def top_fraction(score: np.ndarray, fraction: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    k = max(1, int(np.ceil(float(fraction) * len(score))))
    order = np.argsort(score, kind="mergesort")
    return np.sort(order[-k:]).astype(np.int64)


def load_payload(cache_dir: Path, subject: int) -> dict[str, np.ndarray]:
    path = cache_dir / f"S{subject}.npz"
    payload = np.load(path, allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def balanced_subset(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    chosen = []
    for label in sorted(map(int, np.unique(labels))):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < int(per_class):
            raise ValueError(f"class {label} has {len(candidates)} trials, need {per_class}")
        chosen.append(rng.choice(candidates, size=int(per_class), replace=False))
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices.astype(np.int64)


def fit_ridge(
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
        xs = xs * weights[None, :]
        xt = xt * weights[None, :]
    clf = RidgeClassifier(alpha=float(alpha))
    clf.fit(xs, source_labels)
    pred = clf.predict(xt)
    return float(np.mean(pred == target_labels) * 100.0)


def fit_lda(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    indices: np.ndarray | None,
) -> float:
    if indices is None:
        xs = source_features
        xt = target_features
    else:
        xs = source_features[:, indices]
        xt = target_features[:, indices]
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(xs, source_labels)
    pred = clf.predict(xt)
    return float(np.mean(pred == target_labels) * 100.0)


def summarize(records: pd.DataFrame, *, subjects: Sequence[int], bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    full_ridge_lookup = {
        (int(row.subject), int(row.repeat)): float(row.accuracy)
        for row in records[records["method"] == "ridge_full_a100"].itertuples()
    }
    lda_full_lookup = {
        (int(row.subject), int(row.repeat)): float(row.accuracy)
        for row in records[records["method"] == "lda_full"].itertuples()
    }
    for method, sub in records.groupby("method", sort=True):
        by_subject_acc: dict[int, list[float]] = {}
        by_subject_gain_ridge: dict[int, list[float]] = {}
        by_subject_gain_lda: dict[int, list[float]] = {}
        for row in sub.itertuples():
            key = (int(row.subject), int(row.repeat))
            by_subject_acc.setdefault(int(row.subject), []).append(float(row.accuracy))
            by_subject_gain_ridge.setdefault(int(row.subject), []).append(
                float(row.accuracy) - full_ridge_lookup[key]
            )
            by_subject_gain_lda.setdefault(int(row.subject), []).append(
                float(row.accuracy) - lda_full_lookup[key]
            )
        acc = np.asarray([np.mean(by_subject_acc[s]) for s in subjects if s in by_subject_acc], dtype=np.float64)
        gain_ridge = np.asarray(
            [np.mean(by_subject_gain_ridge[s]) for s in subjects if s in by_subject_gain_ridge],
            dtype=np.float64,
        )
        gain_lda = np.asarray(
            [np.mean(by_subject_gain_lda[s]) for s in subjects if s in by_subject_gain_lda],
            dtype=np.float64,
        )
        rows.append(
            {
                "method": method,
                "n_subjects": int(len(acc)),
                "accuracy_mean": float(acc.mean()),
                "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, bootstrap),
                "gain_vs_ridge_full_mean_pp": float(gain_ridge.mean()),
                "gain_vs_ridge_full_95ci": bootstrap_ci(gain_ridge, rng, bootstrap),
                "gain_vs_ridge_full_q05_pp": float(np.quantile(gain_ridge, 0.05)),
                "p_gain_vs_ridge_full_lt_minus5": float(np.mean(gain_ridge < -5.0)),
                "gain_vs_lda_full_mean_pp": float(gain_lda.mean()),
                "gain_vs_lda_full_95ci": bootstrap_ci(gain_lda, rng, bootstrap),
                "gain_vs_lda_full_q05_pp": float(np.quantile(gain_lda, 0.05)),
                "p_gain_vs_lda_full_lt_minus5": float(np.mean(gain_lda < -5.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    payloads = {subject: load_payload(args.cache_dir, subject) for subject in subjects}
    p = int(next(iter(payloads.values()))["source_features"].shape[1])
    source_n = int(next(iter(payloads.values()))["source_features"].shape[0])

    records: list[dict[str, object]] = []
    repeats = int(args.repeats) if int(args.per_class) > 0 else 1
    for held_subject in subjects:
        payload = payloads[held_subject]
        stats = aggregate_stats(payloads, held_subject)
        s_score = source_score(stats)
        l_score = longitudinal_score(stats)
        w_source_g2 = rank_weights(s_score, gamma=2.0)
        w_source_g1 = rank_weights(s_score, gamma=1.0)
        w_long_g2 = rank_weights(l_score, gamma=2.0)
        idx_source_q25 = top_fraction(s_score, 0.25)
        idx_long_q10 = top_fraction(l_score, 0.10)
        idx_long_q70 = top_fraction(l_score, 0.70)

        labels = np.asarray(payload["source_labels"], dtype=np.int64)
        for repeat in range(repeats):
            if int(args.per_class) > 0:
                rng = np.random.default_rng(
                    int(args.seed) + 1000003 * int(held_subject) + int(repeat)
                )
                source_idx = balanced_subset(labels, int(args.per_class), rng)
            else:
                source_idx = np.arange(len(labels), dtype=np.int64)

            xs = np.asarray(payload["source_features"], dtype=np.float64)[source_idx]
            ys = np.asarray(payload["source_labels"], dtype=np.int64)[source_idx]
            xt = np.asarray(payload["target_features"], dtype=np.float64)
            yt = np.asarray(payload["target_labels"], dtype=np.int64)

            method_specs = [
                ("ridge_full_a100", lambda: fit_ridge(xs, ys, xt, yt, weights=None, alpha=100.0), p),
                ("ridge_source_rank_g2_a100", lambda: fit_ridge(xs, ys, xt, yt, weights=w_source_g2, alpha=100.0), p),
                ("ridge_source_rank_g1_a100", lambda: fit_ridge(xs, ys, xt, yt, weights=w_source_g1, alpha=100.0), p),
                ("ridge_longitudinal_rank_g2_a100", lambda: fit_ridge(xs, ys, xt, yt, weights=w_long_g2, alpha=100.0), p),
                ("ridge_hard_source_q25_a100", lambda: fit_ridge(xs[:, idx_source_q25], ys, xt[:, idx_source_q25], yt, weights=None, alpha=100.0), len(idx_source_q25)),
                ("ridge_hard_long_q10_a100", lambda: fit_ridge(xs[:, idx_long_q10], ys, xt[:, idx_long_q10], yt, weights=None, alpha=100.0), len(idx_long_q10)),
                ("ridge_hard_long_q70_a100", lambda: fit_ridge(xs[:, idx_long_q70], ys, xt[:, idx_long_q70], yt, weights=None, alpha=100.0), len(idx_long_q70)),
                ("lda_full", lambda: fit_lda(xs, ys, xt, yt, None), p),
                ("lda_source_q25", lambda: fit_lda(xs, ys, xt, yt, idx_source_q25), len(idx_source_q25)),
                ("lda_long_q10", lambda: fit_lda(xs, ys, xt, yt, idx_long_q10), len(idx_long_q10)),
                ("lda_long_q70", lambda: fit_lda(xs, ys, xt, yt, idx_long_q70), len(idx_long_q70)),
            ]
            for method, fn, n_selected in method_specs:
                records.append(
                    {
                        "regime_label": args.regime_label,
                        "subject": int(held_subject),
                        "repeat": int(repeat),
                        "source_per_class": int(args.per_class),
                        "source_total": int(2 * args.per_class) if int(args.per_class) > 0 else int(len(source_idx)),
                        "p": int(p),
                        "p_over_source_total": float(p / len(source_idx)),
                        "method": method,
                        "n_selected": int(n_selected),
                        "accuracy": float(fn()),
                    }
                )
        if not args.quiet:
            print(f"S{held_subject}: done", flush=True)

    records_df = pd.DataFrame(records)
    summary_df = summarize(
        records_df,
        subjects=subjects,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    records_df.to_csv(args.output_dir / "weighted_ridge_transfer_records.csv", index=False)
    summary_df.to_csv(args.output_dir / "weighted_ridge_transfer_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "regime_label": args.regime_label,
                    "cache_dir": str(args.cache_dir),
                    "subjects": subjects,
                    "per_class": int(args.per_class),
                    "repeats": int(repeats),
                    "p": int(p),
                    "source_n_original": int(source_n),
                    "bootstrap": int(args.bootstrap),
                    "seed": int(args.seed),
                },
                "summary": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "regime_label": args.regime_label,
                "n_records": int(len(records_df)),
                "summary_csv": str(args.output_dir / "weighted_ridge_transfer_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
