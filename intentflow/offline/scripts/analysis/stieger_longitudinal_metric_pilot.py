"""E5b0: source-side longitudinal diagonal metric pilot.

This pilot tests the next logical step after E5a.  E5a found that
class-conditional geometry explains broad/neuro branch differences better than
global unlabeled shift, but is too weak as a safety gate.  E5b0 therefore stops
trying to route target sessions and instead learns a fixed feature metric from
*other participants'* longitudinal labels.

For each held-out subject:

* all sessions of that subject are excluded from metric learning;
* training subjects' session-1 labels and later-session labels are used only to
  estimate class-stable feature dimensions;
* the held-out subject still trains decoders from session 1 labels only;
* target sessions use only the unlabeled prefix for EA references;
* evaluation is on trial 65 onward, matching E4a.

The learned metric is deliberately simple: a diagonal feature scale based on
source/target class separation divided by same-class cross-session drift and
within-class variance.  This should be treated as a falsification pilot rather
than a polished final method.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from stieger_neuro_feature_baseline import (
    CONDITIONS,
    DEFAULT_CACHE,
    FEATURE_CONFIGS,
    concatenate_tangent_features,
    feature_covariances,
    load_subject,
    lower_tail_cvar,
    parse_subjects,
    references_for,
    weighted_quantile,
)
from stieger_neuro_fixed_fusion import weighted_log_probability_prediction


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260627_stieger_longitudinal_metric_pilot"
)
DEFAULT_FEATURES = ("broad_all60", "fb_sensorimotor21_mu_beta")
DEFAULT_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_FRACTIONS = (1.0, 0.75, 0.5, 0.25, 0.1)
PRIMARY_CONDITIONS = ("pure_lr", "pure_ud")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--features", nargs=2, default=list(DEFAULT_FEATURES))
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument(
        "--eval-start",
        type=int,
        default=64,
        help="Zero-based evaluation start. Default 64 means trial 65 onward.",
    )
    parser.add_argument("--min-eval-trials", type=int, default=40)
    parser.add_argument("--strengths", nargs="+", type=float, default=list(DEFAULT_STRENGTHS))
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_FRACTIONS),
        help="Top fractions of longitudinally stable features to keep. Must include 1.0.",
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-stats", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    unknown = [feature for feature in args.features if feature not in FEATURE_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}")
    if args.features[0] == args.features[1]:
        raise ValueError("--features must name two different branches.")
    if not (0 < args.prefix <= args.eval_start):
        raise ValueError("Require 0 < prefix <= eval-start.")
    if args.min_eval_trials <= 0:
        raise ValueError("min-eval-trials must be positive.")
    if not args.strengths or any(strength < 0 or strength > 1 for strength in args.strengths):
        raise ValueError("--strengths must be non-empty values in [0, 1].")
    if 0.0 not in set(float(value) for value in args.strengths):
        raise ValueError("--strengths must include 0.0 as the E4a-equivalent baseline.")
    if not args.fractions or any(fraction <= 0 or fraction > 1 for fraction in args.fractions):
        raise ValueError("--fractions must be non-empty values in (0, 1].")
    if 1.0 not in set(float(value) for value in args.fractions):
        raise ValueError("--fractions must include 1.0 as the E4a-equivalent baseline.")


def strength_token(strength: float) -> str:
    return f"s{strength:.2f}".replace(".", "p")


def fraction_token(fraction: float) -> str:
    return f"q{fraction:.2f}".replace(".", "p")


def posterior_candidate(strength: float) -> str:
    return f"metric__{strength_token(strength)}__posterior_equal__prefix_ea"


def single_candidate(strength: float, feature: str) -> str:
    return f"metric__{strength_token(strength)}__single__{feature}__prefix_ea"


def select_posterior_candidate(fraction: float) -> str:
    return f"select__{fraction_token(fraction)}__posterior_equal__prefix_ea"


def select_single_candidate(fraction: float, feature: str) -> str:
    return f"select__{fraction_token(fraction)}__single__{feature}__prefix_ea"


def source_select_posterior_candidate(fraction: float) -> str:
    return f"source_select__{fraction_token(fraction)}__posterior_equal__prefix_ea"


def source_select_single_candidate(fraction: float, feature: str) -> str:
    return f"source_select__{fraction_token(fraction)}__single__{feature}__prefix_ea"


def candidate_acc_key(candidate: str) -> str:
    return f"candidate__{candidate}__acc"


def candidate_correct_key(candidate: str) -> str:
    return f"candidate__{candidate}__correct"


def lda() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def class_centroids_and_within(
    features: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    variances = []
    for label in classes:
        selected = features[labels == label]
        if len(selected) == 0:
            raise ValueError(f"Missing class {label}")
        centroids.append(selected.mean(axis=0))
        variances.append(selected.var(axis=0))
    return np.asarray(centroids, dtype=np.float64), np.mean(variances, axis=0)


def session_metric_stats(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
) -> dict[str, np.ndarray] | None:
    classes = sorted(set(map(int, np.unique(source_labels))).intersection(map(int, np.unique(target_labels))))
    if len(classes) < 2:
        return None
    source_centroids, source_within = class_centroids_and_within(
        source_features, source_labels, classes
    )
    target_centroids, target_within = class_centroids_and_within(
        target_features, target_labels, classes
    )
    source_between = source_centroids.var(axis=0)
    target_between = target_centroids.var(axis=0)
    same_class_drift = np.mean((target_centroids - source_centroids) ** 2, axis=0)
    return {
        "source_between": source_between,
        "target_between": target_between,
        "same_class_drift": same_class_drift,
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


def compute_subject_stats(
    subject: int,
    data: Mapping[str, np.ndarray],
    features: Sequence[str],
    prefix: int,
    eval_start: int,
    min_eval_trials: int,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Sufficient statistics for metric learning from one subject."""
    output: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    for condition, specification in CONDITIONS.items():
        mask = np.isin(data["task"], specification["tasks"]) & np.isin(
            data["target"], specification["targets"]
        )
        sessions = sorted(np.unique(data["session"][mask]).tolist())
        if len(sessions) < 2:
            continue
        source_session = int(sessions[0])
        source_indices = np.flatnonzero(mask & (data["session"] == source_session))
        source_labels = data["target"][source_indices]
        if len(np.unique(source_labels)) < 2:
            continue
        for feature in features:
            config = FEATURE_CONFIGS[feature]
            bands = tuple(str(band) for band in config["bands"])
            covariances = feature_covariances(data, config)
            source_references = references_for(covariances, bands, source_indices)
            source_features = concatenate_tangent_features(
                covariances, bands, source_indices, source_references
            )
            total: dict[str, np.ndarray] = {}
            for session in sessions[1:]:
                target_indices = np.flatnonzero(mask & (data["session"] == session))
                if len(target_indices) <= eval_start:
                    continue
                eval_indices = target_indices[eval_start:]
                if len(eval_indices) < min_eval_trials:
                    continue
                prefix_indices = target_indices[:prefix]
                prefix_references = references_for(covariances, bands, prefix_indices)
                target_features = concatenate_tangent_features(
                    covariances, bands, eval_indices, prefix_references
                )
                stats = session_metric_stats(
                    source_features,
                    source_labels,
                    target_features,
                    data["target"][eval_indices],
                )
                if stats is not None:
                    add_stats(total, stats)
            if total:
                output[condition][feature] = total
    return {condition: dict(by_feature) for condition, by_feature in output.items()}


def stats_npz_path(output_dir: Path, subject: int) -> Path:
    return output_dir / "metric_stats" / f"S{subject}.npz"


def flatten_stats(stats: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for condition, by_feature in stats.items():
        for feature, by_metric in by_feature.items():
            for metric, value in by_metric.items():
                arrays[f"{condition}__{feature}__{metric}"] = np.asarray(value)
    return arrays


def unflatten_stats(payload: Mapping[str, np.ndarray]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    output: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for key, value in payload.items():
        condition, feature, metric = key.split("__", maxsplit=2)
        output[condition][feature][metric] = np.asarray(value, dtype=np.float64)
    return {
        condition: {feature: dict(metrics) for feature, metrics in by_feature.items()}
        for condition, by_feature in output.items()
    }


def build_metric_stats(
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, dict[str, dict[str, np.ndarray]]]], list[dict[str, object]]]:
    stats_dir = args.output_dir / "metric_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    all_stats: dict[int, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    failures: list[dict[str, object]] = []
    for subject in parse_subjects(args.subjects):
        path = stats_npz_path(args.output_dir, subject)
        if path.exists() and not args.force and not args.force_stats:
            payload = np.load(path)
            all_stats[subject] = unflatten_stats({key: payload[key] for key in payload.files})
            if not args.quiet:
                print(f"S{subject}: metric stats resume", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            stats = compute_subject_stats(
                subject,
                data,
                args.features,
                args.prefix,
                args.eval_start,
                args.min_eval_trials,
            )
            np.savez_compressed(path, **flatten_stats(stats))
            all_stats[subject] = stats
            if not args.quiet:
                n_items = sum(len(by_feature) for by_feature in stats.values())
                print(f"S{subject}: metric stats {n_items} condition-feature items", flush=True)
        except Exception as error:
            failures.append({"subject": int(subject), "stage": "metric_stats", "error": repr(error)})
            print(f"S{subject}: metric stats FAIL {error!r}", flush=True)
    return all_stats, failures


def aggregate_stats(
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
    held_subject: int,
    condition: str,
    feature: str,
) -> dict[str, np.ndarray] | None:
    total: dict[str, np.ndarray] = {}
    for subject, subject_stats in all_stats.items():
        if int(subject) == int(held_subject):
            continue
        stats = subject_stats.get(condition, {}).get(feature)
        if not stats:
            continue
        add_stats(total, stats)
    if not total or float(total.get("count", np.asarray([0.0]))[0]) <= 0:
        return None
    count = float(total["count"][0])
    averaged = {
        key: value / count
        for key, value in total.items()
        if key != "count"
    }
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged


def metric_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    signal = np.maximum(stats["source_between"], 0.0) + np.maximum(stats["target_between"], 0.0)
    nuisance = (
        np.maximum(stats["same_class_drift"], 0.0)
        + 0.25 * np.maximum(stats["source_within"], 0.0)
        + 0.25 * np.maximum(stats["target_within"], 0.0)
        + 1e-8
    )
    raw_score = signal / nuisance
    return np.where(np.isfinite(raw_score), raw_score, 0.0)


def source_only_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    raw_score = np.maximum(stats["source_between"], 0.0) / (
        0.25 * np.maximum(stats["source_within"], 0.0) + 1e-8
    )
    return np.where(np.isfinite(raw_score), raw_score, 0.0)


def metric_scale(stats: Mapping[str, np.ndarray], strength: float) -> np.ndarray:
    raw_score = metric_score(stats)
    if np.all(raw_score <= 0):
        normalized = np.ones_like(raw_score)
    else:
        high = np.quantile(raw_score, 0.95)
        clipped = np.clip(raw_score, 0.0, high if high > 0 else None)
        mean = float(np.mean(clipped))
        normalized = clipped / mean if mean > 1e-12 else np.ones_like(clipped)
    blended = (1.0 - float(strength)) + float(strength) * normalized
    return np.sqrt(np.maximum(blended, 1e-6))


def metric_indices(stats: Mapping[str, np.ndarray], fraction: float) -> np.ndarray:
    return top_fraction_indices(metric_score(stats), fraction)


def source_only_indices(stats: Mapping[str, np.ndarray], fraction: float) -> np.ndarray:
    return top_fraction_indices(source_only_score(stats), fraction)


def top_fraction_indices(score: np.ndarray, fraction: float) -> np.ndarray:
    dim = len(score)
    if fraction >= 1.0:
        return np.arange(dim, dtype=np.int64)
    score = np.where(np.isfinite(score), score, -np.inf)
    k = max(2, int(np.ceil(float(fraction) * dim)))
    k = min(k, dim)
    selected = np.argpartition(score, -k)[-k:]
    return np.sort(selected.astype(np.int64))


def evaluate_subject(
    subject: int,
    data: Mapping[str, np.ndarray],
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
    features: Sequence[str],
    strengths: Sequence[float],
    fractions: Sequence[float],
    prefix: int,
    eval_start: int,
    min_eval_trials: int,
) -> list[dict[str, object]]:
    broad, neuro = features
    rows: list[dict[str, object]] = []
    for condition, specification in CONDITIONS.items():
        mask = np.isin(data["task"], specification["tasks"]) & np.isin(
            data["target"], specification["targets"]
        )
        sessions = sorted(np.unique(data["session"][mask]).tolist())
        if len(sessions) < 2:
            continue
        source_session = int(sessions[0])
        source_indices = np.flatnonzero(mask & (data["session"] == source_session))
        source_labels = data["target"][source_indices]
        if len(np.unique(source_labels)) < 2:
            continue

        branch_data: dict[str, dict[str, object]] = {}
        scales: dict[tuple[str, float], np.ndarray] = {}
        indices: dict[tuple[str, float], np.ndarray] = {}
        source_only_select_indices: dict[tuple[str, float], np.ndarray] = {}
        for feature in features:
            config = FEATURE_CONFIGS[feature]
            bands = tuple(str(band) for band in config["bands"])
            covariances = feature_covariances(data, config)
            source_references = references_for(covariances, bands, source_indices)
            source_features = concatenate_tangent_features(
                covariances, bands, source_indices, source_references
            )
            branch_data[feature] = {
                "bands": bands,
                "covariances": covariances,
                "source_features": source_features,
            }
            aggregated = aggregate_stats(all_stats, subject, condition, feature)
            if aggregated is None:
                for strength in strengths:
                    scales[(feature, float(strength))] = np.ones(source_features.shape[1])
                for fraction in fractions:
                    indices[(feature, float(fraction))] = np.arange(
                        source_features.shape[1], dtype=np.int64
                    )
                    source_only_select_indices[(feature, float(fraction))] = np.arange(
                        source_features.shape[1], dtype=np.int64
                    )
            else:
                for strength in strengths:
                    scales[(feature, float(strength))] = metric_scale(aggregated, float(strength))
                for fraction in fractions:
                    indices[(feature, float(fraction))] = metric_indices(aggregated, float(fraction))
                    source_only_select_indices[(feature, float(fraction))] = source_only_indices(
                        aggregated, float(fraction)
                    )

        models: dict[tuple[str, float], LinearDiscriminantAnalysis] = {}
        select_models: dict[tuple[str, float], LinearDiscriminantAnalysis] = {}
        source_select_models: dict[tuple[str, float], LinearDiscriminantAnalysis] = {}
        for feature in features:
            source_features = branch_data[feature]["source_features"]  # type: ignore[assignment]
            for strength in strengths:
                scale = scales[(feature, float(strength))]
                models[(feature, float(strength))] = lda().fit(
                    source_features * scale,  # type: ignore[operator]
                    source_labels,
                )
            for fraction in fractions:
                index = indices[(feature, float(fraction))]
                select_models[(feature, float(fraction))] = lda().fit(
                    source_features[:, index],  # type: ignore[index]
                    source_labels,
                )
                source_index = source_only_select_indices[(feature, float(fraction))]
                source_select_models[(feature, float(fraction))] = lda().fit(
                    source_features[:, source_index],  # type: ignore[index]
                    source_labels,
                )

        for session in sessions[1:]:
            target_indices = np.flatnonzero(mask & (data["session"] == session))
            if len(target_indices) <= eval_start:
                continue
            eval_indices = target_indices[eval_start:]
            if len(eval_indices) < min_eval_trials:
                continue
            prefix_indices = target_indices[:prefix]
            eval_labels = data["target"][eval_indices]
            row: dict[str, object] = {
                "subject": int(subject),
                "session": int(session),
                "condition": condition,
                "source_session": source_session,
                "n_source": int(len(source_indices)),
                "n_prefix": int(len(prefix_indices)),
                "eval_start": int(eval_start),
                "n_eval": int(len(eval_indices)),
            }
            target_features: dict[str, np.ndarray] = {}
            for feature in features:
                branch = branch_data[feature]
                prefix_references = references_for(
                    branch["covariances"],  # type: ignore[arg-type]
                    branch["bands"],  # type: ignore[arg-type]
                    prefix_indices,
                )
                target_features[feature] = concatenate_tangent_features(
                    branch["covariances"],  # type: ignore[arg-type]
                    branch["bands"],  # type: ignore[arg-type]
                    eval_indices,
                    prefix_references,
                )

            for strength in strengths:
                logp: dict[str, np.ndarray] = {}
                for feature in features:
                    candidate = single_candidate(float(strength), feature)
                    model = models[(feature, float(strength))]
                    transformed = target_features[feature] * scales[(feature, float(strength))]
                    predictions = model.predict(transformed)
                    correct = int(np.sum(predictions == eval_labels))
                    row[candidate_correct_key(candidate)] = correct
                    row[candidate_acc_key(candidate)] = float(100.0 * correct / len(eval_labels))
                    logp[feature] = model.predict_log_proba(transformed)
                broad_model = models[(broad, float(strength))]
                neuro_model = models[(neuro, float(strength))]
                if not np.array_equal(broad_model.classes_, neuro_model.classes_):
                    raise RuntimeError("Broad/neuro models have different class order.")
                predictions = weighted_log_probability_prediction(
                    logp[broad],
                    logp[neuro],
                    broad_model.classes_,
                    broad_weight=0.5,
                )
                candidate = posterior_candidate(float(strength))
                correct = int(np.sum(predictions == eval_labels))
                row[candidate_correct_key(candidate)] = correct
                row[candidate_acc_key(candidate)] = float(100.0 * correct / len(eval_labels))
            for fraction in fractions:
                logp = {}
                for feature in features:
                    candidate = select_single_candidate(float(fraction), feature)
                    model = select_models[(feature, float(fraction))]
                    index = indices[(feature, float(fraction))]
                    transformed = target_features[feature][:, index]
                    predictions = model.predict(transformed)
                    correct = int(np.sum(predictions == eval_labels))
                    row[candidate_correct_key(candidate)] = correct
                    row[candidate_acc_key(candidate)] = float(100.0 * correct / len(eval_labels))
                    logp[feature] = model.predict_log_proba(transformed)
                broad_model = select_models[(broad, float(fraction))]
                neuro_model = select_models[(neuro, float(fraction))]
                if not np.array_equal(broad_model.classes_, neuro_model.classes_):
                    raise RuntimeError("Broad/neuro selected models have different class order.")
                predictions = weighted_log_probability_prediction(
                    logp[broad],
                    logp[neuro],
                    broad_model.classes_,
                    broad_weight=0.5,
                )
                candidate = select_posterior_candidate(float(fraction))
                correct = int(np.sum(predictions == eval_labels))
                row[candidate_correct_key(candidate)] = correct
                row[candidate_acc_key(candidate)] = float(100.0 * correct / len(eval_labels))
                source_logp = {}
                for feature in features:
                    candidate = source_select_single_candidate(float(fraction), feature)
                    model = source_select_models[(feature, float(fraction))]
                    index = source_only_select_indices[(feature, float(fraction))]
                    transformed = target_features[feature][:, index]
                    predictions = model.predict(transformed)
                    correct = int(np.sum(predictions == eval_labels))
                    row[candidate_correct_key(candidate)] = correct
                    row[candidate_acc_key(candidate)] = float(100.0 * correct / len(eval_labels))
                    source_logp[feature] = model.predict_log_proba(transformed)
                broad_model = source_select_models[(broad, float(fraction))]
                neuro_model = source_select_models[(neuro, float(fraction))]
                if not np.array_equal(broad_model.classes_, neuro_model.classes_):
                    raise RuntimeError(
                        "Broad/neuro source-only selected models have different class order."
                    )
                predictions = weighted_log_probability_prediction(
                    source_logp[broad],
                    source_logp[neuro],
                    broad_model.classes_,
                    broad_weight=0.5,
                )
                candidate = source_select_posterior_candidate(float(fraction))
                correct = int(np.sum(predictions == eval_labels))
                row[candidate_correct_key(candidate)] = correct
                row[candidate_acc_key(candidate)] = float(100.0 * correct / len(eval_labels))
            rows.append(row)
    return rows


def build_evaluation_rows(
    args: argparse.Namespace,
    all_stats: Mapping[int, Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    table_path = args.output_dir / "metric_table.json"
    if table_path.exists() and not args.force and not args.force_eval:
        with table_path.open() as handle:
            payload = json.load(handle)
        return list(payload["rows"]), list(payload.get("failures", []))

    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    strengths = sorted(set(float(value) for value in args.strengths))
    fractions = sorted(set(float(value) for value in args.fractions), reverse=True)
    for subject in parse_subjects(args.subjects):
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force and not args.force_eval:
            with output.open() as handle:
                payload = json.load(handle)
            rows.extend(payload["rows"])
            if not args.quiet:
                print(f"S{subject}: evaluation resume ({len(payload['rows'])} rows)", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            subject_rows = evaluate_subject(
                subject,
                data,
                all_stats,
                args.features,
                strengths,
                fractions,
                args.prefix,
                args.eval_start,
                args.min_eval_trials,
            )
            output.write_text(json.dumps({"subject": subject, "rows": subject_rows}, indent=2))
            rows.extend(subject_rows)
            if not args.quiet:
                print(f"S{subject}: evaluation {len(subject_rows)} rows", flush=True)
        except Exception as error:
            failures.append({"subject": int(subject), "stage": "evaluation", "error": repr(error)})
            print(f"S{subject}: evaluation FAIL {error!r}", flush=True)
    table_path.write_text(json.dumps({"rows": rows, "failures": failures}, indent=2))
    write_csv(args.output_dir / "metric_table.csv", rows)
    return rows, failures


def choose_by_accuracy(scored: Mapping[str, float], preferred: str) -> str:
    maximum = max(scored.values())
    tied = [key for key, score in scored.items() if np.isclose(score, maximum)]
    if preferred in tied:
        return preferred
    return sorted(tied)[0]


def candidate_accuracy(rows: Sequence[Mapping[str, object]], candidate: str) -> float:
    by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_subject[int(row["subject"])].append(row)
    values = []
    for subject_rows in by_subject.values():
        correct = sum(int(row[candidate_correct_key(candidate)]) for row in subject_rows)
        total = sum(int(row["n_eval"]) for row in subject_rows)
        values.append(100.0 * correct / total)
    return float(np.mean(values))


def build_selection_records(
    rows: Sequence[Mapping[str, object]],
    strengths: Sequence[float],
    fractions: Sequence[float],
) -> list[dict[str, object]]:
    strengths = sorted(set(float(value) for value in strengths))
    fractions = sorted(set(float(value) for value in fractions), reverse=True)
    baseline = posterior_candidate(0.0)
    select_baseline = select_posterior_candidate(1.0)
    posterior_candidates = [posterior_candidate(strength) for strength in strengths]
    select_candidates = [select_posterior_candidate(fraction) for fraction in fractions]
    source_select_candidates = [
        source_select_posterior_candidate(fraction) for fraction in fractions
    ]
    records: list[dict[str, object]] = []
    methods = [f"fixed_{strength_token(strength)}" for strength in strengths]
    select_methods = [f"select_{fraction_token(fraction)}" for fraction in fractions]
    source_select_methods = [
        f"source_select_{fraction_token(fraction)}" for fraction in fractions
    ]
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        subjects = sorted({int(row["subject"]) for row in condition_rows})
        for held_subject in subjects:
            train_rows = [row for row in condition_rows if int(row["subject"]) != held_subject]
            test_rows = [row for row in condition_rows if int(row["subject"]) == held_subject]
            if not train_rows or not test_rows:
                continue
            scores = {
                candidate: candidate_accuracy(train_rows, candidate)
                for candidate in posterior_candidates
            }
            selected = choose_by_accuracy(scores, preferred=baseline)
            nonzero_scores = {
                candidate: score for candidate, score in scores.items() if candidate != baseline
            }
            selected_nonzero = (
                choose_by_accuracy(nonzero_scores, preferred=posterior_candidate(0.5))
                if nonzero_scores
                else selected
            )
            select_scores = {
                candidate: candidate_accuracy(train_rows, candidate)
                for candidate in select_candidates
            }
            selected_fraction = choose_by_accuracy(
                select_scores, preferred=select_baseline
            )
            nonfull_scores = {
                candidate: score
                for candidate, score in select_scores.items()
                if candidate != select_baseline
            }
            selected_nonfull_fraction = (
                choose_by_accuracy(nonfull_scores, preferred=select_posterior_candidate(0.5))
                if nonfull_scores
                else selected_fraction
            )
            source_select_scores = {
                candidate: candidate_accuracy(train_rows, candidate)
                for candidate in source_select_candidates
            }
            selected_source_fraction = choose_by_accuracy(
                source_select_scores,
                preferred=source_select_posterior_candidate(1.0),
            )
            source_nonfull_scores = {
                candidate: score
                for candidate, score in source_select_scores.items()
                if candidate != source_select_posterior_candidate(1.0)
            }
            selected_source_nonfull_fraction = (
                choose_by_accuracy(
                    source_nonfull_scores,
                    preferred=source_select_posterior_candidate(0.5),
                )
                if source_nonfull_scores
                else selected_source_fraction
            )
            for row in test_rows:
                for strength, method in zip(strengths, methods):
                    candidate = posterior_candidate(strength)
                    records.append(
                        {
                            "subject": int(row["subject"]),
                            "session": int(row["session"]),
                            "condition": str(row["condition"]),
                            "method": method,
                            "candidate": candidate,
                            "test_correct": int(row[candidate_correct_key(candidate)]),
                            "n_eval": int(row["n_eval"]),
                            "test_acc": float(row[candidate_acc_key(candidate)]),
                        }
                    )
                for fraction, method in zip(fractions, select_methods):
                    candidate = select_posterior_candidate(fraction)
                    records.append(
                        {
                            "subject": int(row["subject"]),
                            "session": int(row["session"]),
                            "condition": str(row["condition"]),
                            "method": method,
                            "candidate": candidate,
                            "test_correct": int(row[candidate_correct_key(candidate)]),
                            "n_eval": int(row["n_eval"]),
                            "test_acc": float(row[candidate_acc_key(candidate)]),
                        }
                    )
                for fraction, method in zip(fractions, source_select_methods):
                    candidate = source_select_posterior_candidate(fraction)
                    records.append(
                        {
                            "subject": int(row["subject"]),
                            "session": int(row["session"]),
                            "condition": str(row["condition"]),
                            "method": method,
                            "candidate": candidate,
                            "test_correct": int(row[candidate_correct_key(candidate)]),
                            "n_eval": int(row["n_eval"]),
                            "test_acc": float(row[candidate_acc_key(candidate)]),
                        }
                    )
                for method, candidate in (
                    ("outer_best_strength", selected),
                    ("outer_best_nonzero_strength", selected_nonzero),
                    ("outer_best_fraction", selected_fraction),
                    ("outer_best_nonfull_fraction", selected_nonfull_fraction),
                    ("outer_best_source_fraction", selected_source_fraction),
                    (
                        "outer_best_source_nonfull_fraction",
                        selected_source_nonfull_fraction,
                    ),
                ):
                    records.append(
                        {
                            "subject": int(row["subject"]),
                            "session": int(row["session"]),
                            "condition": str(row["condition"]),
                            "method": method,
                            "candidate": candidate,
                            "test_correct": int(row[candidate_correct_key(candidate)]),
                            "n_eval": int(row["n_eval"]),
                            "test_acc": float(row[candidate_acc_key(candidate)]),
                        }
                    )
    return records


def _group_by_subject(rows: Sequence[Mapping[str, object]]) -> dict[int, list[Mapping[str, object]]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["subject"])].append(row)
    return grouped


def per_subject_paired_gains(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
) -> np.ndarray:
    fixed_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in fixed
    }
    gains = []
    for subject, subject_rows in _group_by_subject(selected).items():
        selected_correct = sum(int(row["test_correct"]) for row in subject_rows)
        fixed_correct = sum(
            int(fixed_by_key[(subject, int(row["session"]), str(row["condition"]))]["test_correct"])
            for row in subject_rows
        )
        total = sum(int(row["n_eval"]) for row in subject_rows)
        gains.append(100.0 * (selected_correct - fixed_correct) / total)
    return np.asarray(gains, dtype=np.float64)


def bootstrap_ci(values: np.ndarray, bootstrap: int, seed: int) -> list[float] | None:
    if len(values) < 3:
        return None
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(bootstrap, len(values)))
    estimates = values[sampled].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def session_risk(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    fixed_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in fixed
    }
    differences = []
    subjects = []
    for row in selected:
        key = (int(row["subject"]), int(row["session"]), str(row["condition"]))
        differences.append(float(row["test_acc"]) - float(fixed_by_key[key]["test_acc"]))
        subjects.append(int(row["subject"]))
    values = np.asarray(differences, dtype=np.float64)
    counts = Counter(subjects)
    weights = np.asarray(
        [1.0 / (len(counts) * counts[subject]) for subject in subjects],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return {
        "gain_vs_e4a_equal_session_mean_pp": float(np.sum(weights * values)),
        "loss_r10_vs_e4a_equal_pp": float(-lower_tail_cvar(values, weights, 0.1)),
        "gain_vs_e4a_equal_q05_pp": weighted_quantile(values, weights, 0.05),
        "p_gain_vs_e4a_equal_lt_minus5": float(np.sum(weights[values < -5.0])),
    }


def summarize_method(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    if not selected:
        return {}
    gains = per_subject_paired_gains(selected, fixed)
    candidates = Counter(str(row["candidate"]) for row in selected)
    return {
        "n_subjects": len({int(row["subject"]) for row in selected}),
        "n_sessions": len(selected),
        "test_accuracy_subject_pooled_mean": float(
            np.mean(
                [
                    100.0
                    * sum(int(row["test_correct"]) for row in subject_rows)
                    / sum(int(row["n_eval"]) for row in subject_rows)
                    for subject_rows in _group_by_subject(selected).values()
                ]
            )
        ),
        "gain_vs_e4a_equal_subject_pooled_pp": float(np.mean(gains)),
        "gain_vs_e4a_equal_subject_bootstrap_95ci": bootstrap_ci(
            gains, bootstrap, seed
        ),
        "selected_candidate_counts": dict(candidates),
        "selected_candidate_rates": {
            candidate: float(count / len(selected))
            for candidate, count in sorted(candidates.items())
        },
        **session_risk(selected, fixed),
    }


def summarize_records(
    records: Sequence[Mapping[str, object]],
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    report: dict[str, object] = {}
    methods = sorted({str(row["method"]) for row in records})
    for condition in [*CONDITIONS, "primary_pooled"]:
        scoped = [
            row
            for row in records
            if (
                row["condition"] == condition
                if condition != "primary_pooled"
                else row["condition"] in PRIMARY_CONDITIONS
            )
        ]
        fixed = [row for row in scoped if row["method"] == "select_q1p00"]
        if not fixed:
            fixed = [row for row in scoped if row["method"] == "fixed_s0p00"]
        if not fixed:
            continue
        condition_report = {}
        for method in methods:
            selected = [row for row in scoped if row["method"] == method]
            condition_report[method] = summarize_method(
                selected, fixed, bootstrap, seed + len(method) + len(condition)
            )
        report[condition] = condition_report
    return report


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_stats, stats_failures = build_metric_stats(args)
    rows, eval_failures = build_evaluation_rows(args, all_stats)
    strengths = sorted(set(float(value) for value in args.strengths))
    fractions = sorted(set(float(value) for value in args.fractions), reverse=True)
    records = build_selection_records(rows, strengths, fractions)
    summary = summarize_records(records, args.bootstrap, args.seed)
    report = {
        "config": {
            "subjects": args.subjects,
            "features": list(args.features),
            "strengths": strengths,
            "fractions": fractions,
            "prefix": args.prefix,
            "eval_start": args.eval_start,
            "min_eval_trials": args.min_eval_trials,
            "cache_dir": str(args.cache_dir),
            "primary_conditions": list(PRIMARY_CONDITIONS),
            "comparator": "select_q1p00, equivalent to E4a equal posterior + prefix-EA in this script",
            "outer_selection": "outer_best_fraction selects retained feature fraction by other subjects only",
        },
        "n_subjects": len({int(row["subject"]) for row in rows}),
        "n_rows": len(rows),
        "failures": stats_failures + eval_failures,
        "summaries": summary,
    }
    (args.output_dir / "selection_records.json").write_text(
        json.dumps({"records": records}, indent=2)
    )
    write_csv(args.output_dir / "selection_records.csv", records)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    if args.quiet:
        print(
            json.dumps(
                {
                    "n_subjects": report["n_subjects"],
                    "n_rows": report["n_rows"],
                    "n_failures": len(report["failures"]),
                    "summary_json": str(args.output_dir / "summary.json"),
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
