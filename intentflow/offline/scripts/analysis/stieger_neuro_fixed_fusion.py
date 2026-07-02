"""E4a: fixed source-trained broad/neuro fusion for longitudinal EEG-MI.

This is a deliberately small falsification experiment.  E1 found that the
all-channel broadband and sensorimotor filter-bank branches can be
complementary, while E2 and G1a showed that target-prefix statistics do not
reliably reveal which branch will win.  E4a therefore removes the target-side
decision completely:

* each decoder is trained only on the held participant's session 1 labels;
* posterior-fusion weights and feature-concatenation hyperparameters are
  selected only by five-fold out-of-fold predictions inside that source
  session;
* a target session is evaluated either with the source reference or with a
  prefix-EA reference estimated from its first ``prefix`` *unlabelled* trials;
* neither fusion weights nor classifier hyperparameters are selected from the
  target session.

The outer leave-one-subject-out layer is used only to choose a strong fixed
single-branch comparator and a population-level fixed fusion configuration.
It never exposes the held participant's target labels to a decision.

The experiment is not a proposed paper method.  Its purpose is to decide
whether representation-level integration has enough zero-target-label signal
to justify a subsequent learned longitudinal representation project.
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
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260626_stieger_neuro_fixed_fusion"
)
DEFAULT_FEATURES = ("broad_all60", "fb_sensorimotor21_mu_beta")
DEFAULT_POSTERIOR_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_CONCAT_WEIGHTS = (0.25, 0.5, 0.75)
DEFAULT_RIDGE_ALPHAS = (1.0, 10.0, 100.0)
PRIMARY_CONDITIONS = ("pure_lr", "pure_ud")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--features",
        nargs=2,
        default=list(DEFAULT_FEATURES),
        metavar=("BROAD", "NEURO"),
        help="Exactly two branches: broad first, neuro second.",
    )
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument(
        "--eval-start",
        type=int,
        default=64,
        help="Zero-based evaluation start.  Default 64 means trial 65 onward.",
    )
    parser.add_argument("--min-eval-trials", type=int, default=40)
    parser.add_argument("--source-cv-folds", type=int, default=5)
    parser.add_argument(
        "--posterior-weights",
        nargs="+",
        type=float,
        default=list(DEFAULT_POSTERIOR_WEIGHTS),
    )
    parser.add_argument(
        "--concat-weights",
        nargs="+",
        type=float,
        default=list(DEFAULT_CONCAT_WEIGHTS),
    )
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_RIDGE_ALPHAS),
    )
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.features[0] == args.features[1]:
        raise ValueError("--features must name two different branches.")
    unknown = [feature for feature in args.features if feature not in FEATURE_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}")
    if not (0 < args.prefix <= args.eval_start):
        raise ValueError("Require 0 < prefix <= eval-start.")
    if args.min_eval_trials <= 0 or args.source_cv_folds < 2:
        raise ValueError("min-eval-trials must be positive and source-cv-folds >= 2.")
    for name, values, allow_endpoint in (
        ("posterior-weights", args.posterior_weights, True),
        ("concat-weights", args.concat_weights, False),
    ):
        if not values or any((weight < 0 or weight > 1) for weight in values):
            raise ValueError(f"{name} must be non-empty values in [0, 1].")
        if not allow_endpoint and any(weight <= 0 or weight >= 1 for weight in values):
            raise ValueError(f"{name} must be strictly inside (0, 1).")
    if not args.ridge_alphas or any(alpha <= 0 for alpha in args.ridge_alphas):
        raise ValueError("ridge-alphas must be positive.")


def single_name(feature: str, adapter: str) -> str:
    return f"single__{feature}__{adapter}"


def posterior_name(kind: str, adapter: str) -> str:
    return f"posterior__{kind}__{adapter}"


def concat_name(kind: str, adapter: str) -> str:
    return f"concat__{kind}__{adapter}"


def candidate_correct_key(candidate: str) -> str:
    return f"candidate__{candidate}__correct"


def candidate_acc_key(candidate: str) -> str:
    return f"candidate__{candidate}__acc"


def lda() -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def source_cv_splits(
    labels: np.ndarray,
    requested_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = np.unique(labels, return_counts=True)[1]
    n_splits = min(int(requested_folds), int(counts.min()))
    if n_splits < 2:
        raise ValueError("Source session has too few examples per class for OOF fusion.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels))


def block_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    return mean, np.maximum(scale, 1e-8)


def concat_blocks(
    broad: np.ndarray,
    neuro: np.ndarray,
    broad_stats: tuple[np.ndarray, np.ndarray],
    neuro_stats: tuple[np.ndarray, np.ndarray],
    broad_weight: float,
) -> np.ndarray:
    """Standardize blocks, then give each block its requested total energy."""
    broad_mean, broad_scale = broad_stats
    neuro_mean, neuro_scale = neuro_stats
    broad_z = (broad - broad_mean) / broad_scale
    neuro_z = (neuro - neuro_mean) / neuro_scale
    broad_factor = np.sqrt(float(broad_weight) / broad_z.shape[1])
    neuro_factor = np.sqrt(float(1.0 - broad_weight) / neuro_z.shape[1])
    return np.concatenate(
        [broad_factor * broad_z, neuro_factor * neuro_z], axis=1
    )


def weighted_log_probability_prediction(
    broad_logp: np.ndarray,
    neuro_logp: np.ndarray,
    classes: np.ndarray,
    broad_weight: float,
) -> np.ndarray:
    scores = broad_weight * broad_logp + (1.0 - broad_weight) * neuro_logp
    return classes[np.argmax(scores, axis=1)]


def choose_by_accuracy(
    scored: Mapping[object, float],
    preferred: object,
) -> object:
    maximum = max(scored.values())
    tied = [key for key, score in scored.items() if np.isclose(score, maximum)]
    if preferred in tied:
        return preferred
    return sorted(tied, key=str)[0]


def select_source_posterior_weight(
    broad_features: np.ndarray,
    neuro_features: np.ndarray,
    labels: np.ndarray,
    weights: Sequence[float],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> float:
    classes = np.unique(labels)
    broad_logp = np.empty((len(labels), len(classes)), dtype=np.float64)
    neuro_logp = np.empty((len(labels), len(classes)), dtype=np.float64)
    for train, valid in splits:
        broad_model = lda().fit(broad_features[train], labels[train])
        neuro_model = lda().fit(neuro_features[train], labels[train])
        if not np.array_equal(broad_model.classes_, classes) or not np.array_equal(
            neuro_model.classes_, classes
        ):
            raise RuntimeError("Unexpected source-fold class mismatch.")
        broad_logp[valid] = broad_model.predict_log_proba(broad_features[valid])
        neuro_logp[valid] = neuro_model.predict_log_proba(neuro_features[valid])
    scores = {
        float(weight): float(
            np.mean(
                weighted_log_probability_prediction(
                    broad_logp, neuro_logp, classes, float(weight)
                )
                == labels
            )
        )
        for weight in weights
    }
    return float(choose_by_accuracy(scores, preferred=0.5))


def select_source_concat_hyperparameters(
    broad_features: np.ndarray,
    neuro_features: np.ndarray,
    labels: np.ndarray,
    weights: Sequence[float],
    alphas: Sequence[float],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    scores: dict[tuple[float, float], float] = {}
    for weight in weights:
        for alpha in alphas:
            predictions = np.empty_like(labels)
            for train, valid in splits:
                broad_stats = block_stats(broad_features[train])
                neuro_stats = block_stats(neuro_features[train])
                train_features = concat_blocks(
                    broad_features[train],
                    neuro_features[train],
                    broad_stats,
                    neuro_stats,
                    float(weight),
                )
                valid_features = concat_blocks(
                    broad_features[valid],
                    neuro_features[valid],
                    broad_stats,
                    neuro_stats,
                    float(weight),
                )
                model = RidgeClassifier(alpha=float(alpha), solver="lsqr")
                predictions[valid] = model.fit(train_features, labels[train]).predict(
                    valid_features
                )
            scores[(float(weight), float(alpha))] = float(
                np.mean(predictions == labels)
            )
    preferred = (0.5, min(alphas, key=lambda alpha: abs(alpha - 10.0)))
    chosen = choose_by_accuracy(scores, preferred=preferred)
    return float(chosen[0]), float(chosen[1])


def fit_concat_model(
    broad_features: np.ndarray,
    neuro_features: np.ndarray,
    labels: np.ndarray,
    broad_weight: float,
    alpha: float,
) -> dict[str, object]:
    broad_stats = block_stats(broad_features)
    neuro_stats = block_stats(neuro_features)
    train_features = concat_blocks(
        broad_features,
        neuro_features,
        broad_stats,
        neuro_stats,
        broad_weight,
    )
    return {
        "model": RidgeClassifier(alpha=alpha, solver="lsqr").fit(
            train_features, labels
        ),
        "broad_stats": broad_stats,
        "neuro_stats": neuro_stats,
        "weight": broad_weight,
        "alpha": alpha,
    }


def predict_concat(
    model: Mapping[str, object],
    broad_features: np.ndarray,
    neuro_features: np.ndarray,
) -> np.ndarray:
    features = concat_blocks(
        broad_features,
        neuro_features,
        model["broad_stats"],  # type: ignore[arg-type]
        model["neuro_stats"],  # type: ignore[arg-type]
        float(model["weight"]),
    )
    return model["model"].predict(features)  # type: ignore[union-attr]


def evaluate_subject_condition(
    subject: int,
    condition: str,
    specification: Mapping[str, tuple[int, ...]],
    data: Mapping[str, np.ndarray],
    features: Sequence[str],
    prefix: int,
    eval_start: int,
    min_eval_trials: int,
    source_cv_folds: int,
    posterior_weights: Sequence[float],
    concat_weights: Sequence[float],
    ridge_alphas: Sequence[float],
    seed: int,
) -> list[dict[str, object]]:
    """Evaluate all fixed fusion candidates for one subject and condition."""
    broad_name, neuro_name = features
    mask = np.isin(data["task"], specification["tasks"]) & np.isin(
        data["target"], specification["targets"]
    )
    sessions = sorted(np.unique(data["session"][mask]).tolist())
    if len(sessions) < 2:
        return []
    source_session = int(sessions[0])
    source_indices = np.flatnonzero(mask & (data["session"] == source_session))
    labels = data["target"][source_indices]
    if len(np.unique(labels)) < 2:
        return []

    branch_data: dict[str, dict[str, object]] = {}
    for feature in features:
        config = FEATURE_CONFIGS[feature]
        bands = tuple(str(band) for band in config["bands"])
        covariances = feature_covariances(data, config)
        source_references = references_for(covariances, bands, source_indices)
        source_features = concatenate_tangent_features(
            covariances, bands, source_indices, source_references
        )
        model = lda().fit(source_features, labels)
        branch_data[feature] = {
            "bands": bands,
            "covariances": covariances,
            "source_references": source_references,
            "source_features": source_features,
            "model": model,
        }

    broad_source = branch_data[broad_name]["source_features"]
    neuro_source = branch_data[neuro_name]["source_features"]
    splits = source_cv_splits(labels, source_cv_folds, seed + subject)
    selected_posterior_weight = select_source_posterior_weight(
        broad_source,  # type: ignore[arg-type]
        neuro_source,  # type: ignore[arg-type]
        labels,
        posterior_weights,
        splits,
    )
    selected_concat_weight, selected_concat_alpha = select_source_concat_hyperparameters(
        broad_source,  # type: ignore[arg-type]
        neuro_source,  # type: ignore[arg-type]
        labels,
        concat_weights,
        ridge_alphas,
        splits,
    )
    concat_equal = fit_concat_model(
        broad_source,  # type: ignore[arg-type]
        neuro_source,  # type: ignore[arg-type]
        labels,
        broad_weight=0.5,
        alpha=10.0,
    )
    concat_s1_oof = fit_concat_model(
        broad_source,  # type: ignore[arg-type]
        neuro_source,  # type: ignore[arg-type]
        labels,
        broad_weight=selected_concat_weight,
        alpha=selected_concat_alpha,
    )

    rows: list[dict[str, object]] = []
    for session in sessions[1:]:
        target_indices = np.flatnonzero(mask & (data["session"] == session))
        if len(target_indices) <= eval_start:
            continue
        eval_indices = target_indices[eval_start:]
        if len(eval_indices) < min_eval_trials:
            continue
        eval_labels = data["target"][eval_indices]
        prefix_indices = target_indices[:prefix]
        if len(prefix_indices) < prefix:
            continue

        row: dict[str, object] = {
            "subject": int(subject),
            "session": int(session),
            "condition": condition,
            "source_session": source_session,
            "n_source": int(len(source_indices)),
            "n_prefix": int(prefix),
            "eval_start": int(eval_start),
            "n_eval": int(len(eval_indices)),
            "posterior_s1_oof_broad_weight": selected_posterior_weight,
            "concat_s1_oof_broad_weight": selected_concat_weight,
            "concat_s1_oof_ridge_alpha": selected_concat_alpha,
        }
        transformed: dict[str, dict[str, np.ndarray]] = {"source": {}, "prefix_ea": {}}
        log_probabilities: dict[str, dict[str, np.ndarray]] = {"source": {}, "prefix_ea": {}}
        for adapter, reference_indices in (
            ("source", source_indices),
            ("prefix_ea", prefix_indices),
        ):
            for feature in features:
                branch = branch_data[feature]
                if adapter == "source":
                    references = branch["source_references"]
                else:
                    references = references_for(
                        branch["covariances"],  # type: ignore[arg-type]
                        branch["bands"],  # type: ignore[arg-type]
                        reference_indices,
                    )
                target_features = concatenate_tangent_features(
                    branch["covariances"],  # type: ignore[arg-type]
                    branch["bands"],  # type: ignore[arg-type]
                    eval_indices,
                    references,  # type: ignore[arg-type]
                )
                transformed[adapter][feature] = target_features
                branch_model = branch["model"]
                log_probabilities[adapter][feature] = branch_model.predict_log_proba(  # type: ignore[union-attr]
                    target_features
                )
                predictions = branch_model.predict(target_features)  # type: ignore[union-attr]
                candidate = single_name(feature, adapter)
                correct = int(np.sum(predictions == eval_labels))
                row[candidate_correct_key(candidate)] = correct
                row[candidate_acc_key(candidate)] = float(100.0 * correct / len(eval_labels))

            broad_model = branch_data[broad_name]["model"]
            neuro_model = branch_data[neuro_name]["model"]
            if not np.array_equal(broad_model.classes_, neuro_model.classes_):  # type: ignore[union-attr]
                raise RuntimeError("Broad/neuro models have different class order.")
            for kind, weight in (
                ("equal", 0.5),
                ("s1_oof", selected_posterior_weight),
            ):
                predictions = weighted_log_probability_prediction(
                    log_probabilities[adapter][broad_name],
                    log_probabilities[adapter][neuro_name],
                    broad_model.classes_,  # type: ignore[union-attr]
                    weight,
                )
                candidate = posterior_name(kind, adapter)
                correct = int(np.sum(predictions == eval_labels))
                row[candidate_correct_key(candidate)] = correct
                row[candidate_acc_key(candidate)] = float(100.0 * correct / len(eval_labels))

            for kind, model in (("equal", concat_equal), ("s1_oof", concat_s1_oof)):
                predictions = predict_concat(
                    model,
                    transformed[adapter][broad_name],
                    transformed[adapter][neuro_name],
                )
                candidate = concat_name(kind, adapter)
                correct = int(np.sum(predictions == eval_labels))
                row[candidate_correct_key(candidate)] = correct
                row[candidate_acc_key(candidate)] = float(100.0 * correct / len(eval_labels))
        rows.append(row)
    return rows


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    table_path = args.output_dir / "fusion_table.json"
    if table_path.exists() and not args.force:
        payload = json.loads(table_path.read_text())
        return list(payload["rows"]), list(payload.get("failures", []))

    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for subject in parse_subjects(args.subjects):
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force:
            payload = json.loads(output.read_text())
            rows.extend(payload["rows"])
            if not args.quiet:
                print(f"S{subject}: resume ({len(payload['rows'])} rows)", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            subject_rows: list[dict[str, object]] = []
            for condition, specification in CONDITIONS.items():
                subject_rows.extend(
                    evaluate_subject_condition(
                        subject,
                        condition,
                        specification,
                        data,
                        args.features,
                        args.prefix,
                        args.eval_start,
                        args.min_eval_trials,
                        args.source_cv_folds,
                        args.posterior_weights,
                        args.concat_weights,
                        args.ridge_alphas,
                        args.seed,
                    )
                )
            output.write_text(
                json.dumps({"subject": subject, "rows": subject_rows}, indent=2)
            )
            rows.extend(subject_rows)
            if not args.quiet:
                print(f"S{subject}: {len(subject_rows)} rows", flush=True)
        except Exception as error:  # Preserve completed subjects during a long run.
            failure = {"subject": int(subject), "error": repr(error)}
            failures.append(failure)
            print(f"S{subject}: FAIL {error!r}", flush=True)
    table_path.write_text(json.dumps({"rows": rows, "failures": failures}, indent=2))
    write_csv(args.output_dir / "fusion_table.csv", rows)
    return rows, failures


def candidate_lists(features: Sequence[str]) -> tuple[list[str], list[str]]:
    single = [
        single_name(feature, adapter)
        for feature in features
        for adapter in ("source", "prefix_ea")
    ]
    fusion = [
        posterior_name(kind, adapter)
        for kind in ("equal", "s1_oof")
        for adapter in ("source", "prefix_ea")
    ] + [
        concat_name(kind, adapter)
        for kind in ("equal", "s1_oof")
        for adapter in ("source", "prefix_ea")
    ]
    return single, fusion


def subject_pooled_accuracy(rows: Sequence[Mapping[str, object]], candidate: str) -> float:
    by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_subject[int(row["subject"])].append(row)
    values = []
    for subject_rows in by_subject.values():
        correct = sum(int(row[candidate_correct_key(candidate)]) for row in subject_rows)
        total = sum(int(row["n_eval"]) for row in subject_rows)
        values.append(100.0 * correct / total)
    return float(np.mean(values))


def choose_outer_candidate(rows: Sequence[Mapping[str, object]], candidates: Sequence[str]) -> str:
    scores = {candidate: subject_pooled_accuracy(rows, candidate) for candidate in candidates}
    return str(choose_by_accuracy(scores, preferred=candidates[0]))


def selection_record(
    row: Mapping[str, object],
    method: str,
    candidate: str,
    outer_reference: str,
) -> dict[str, object]:
    return {
        "subject": int(row["subject"]),
        "session": int(row["session"]),
        "condition": str(row["condition"]),
        "method": method,
        "candidate": candidate,
        "outer_reference": outer_reference,
        "test_correct": int(row[candidate_correct_key(candidate)]),
        "n_eval": int(row["n_eval"]),
        "test_acc": float(row[candidate_acc_key(candidate)]),
        "posterior_s1_oof_broad_weight": row["posterior_s1_oof_broad_weight"],
        "concat_s1_oof_broad_weight": row["concat_s1_oof_broad_weight"],
        "concat_s1_oof_ridge_alpha": row["concat_s1_oof_ridge_alpha"],
    }


def build_selection_records(
    rows: Sequence[Mapping[str, object]],
    features: Sequence[str],
) -> list[dict[str, object]]:
    single, fusion = candidate_lists(features)
    records: list[dict[str, object]] = []
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        for held_subject in sorted({int(row["subject"]) for row in condition_rows}):
            train_rows = [row for row in condition_rows if int(row["subject"]) != held_subject]
            test_rows = [row for row in condition_rows if int(row["subject"]) == held_subject]
            if not train_rows or not test_rows:
                continue
            outer_reference = choose_outer_candidate(train_rows, single)
            outer_fusion = choose_outer_candidate(train_rows, fusion)
            for row in test_rows:
                records.append(
                    selection_record(row, "outer_best_single", outer_reference, outer_reference)
                )
                records.append(
                    selection_record(row, "outer_best_fusion", outer_fusion, outer_reference)
                )
                for candidate in fusion:
                    records.append(
                        selection_record(row, candidate, candidate, outer_reference)
                    )
    return records


def per_subject_paired_gains(
    selected: Sequence[Mapping[str, object]],
    fixed: Sequence[Mapping[str, object]],
) -> np.ndarray:
    fixed_by_key = {
        (int(row["subject"]), int(row["session"]), str(row["condition"])): row
        for row in fixed
    }
    selected_by_subject: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        key = (int(row["subject"]), int(row["session"]), str(row["condition"]))
        if key not in fixed_by_key:
            raise RuntimeError(f"Missing comparator for {key}")
        selected_by_subject[int(row["subject"])].append(row)
    gains = []
    for subject, subject_rows in selected_by_subject.items():
        selected_correct = sum(int(row["test_correct"]) for row in subject_rows)
        fixed_correct = sum(
            int(
                fixed_by_key[(subject, int(row["session"]), str(row["condition"]))][
                    "test_correct"
                ]
            )
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
        [1.0 / (len(counts) * counts[subject]) for subject in subjects], dtype=np.float64
    )
    weights /= weights.sum()
    return {
        "gain_vs_outer_best_single_session_mean_pp": float(np.sum(weights * values)),
        "loss_r10_vs_outer_best_single_pp": float(-lower_tail_cvar(values, weights, 0.1)),
        "gain_vs_outer_best_single_q05_pp": weighted_quantile(values, weights, 0.05),
        "p_gain_vs_outer_best_single_lt_minus5": float(
            np.sum(weights[values < -5.0])
        ),
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
    posterior_weights = [
        float(row["posterior_s1_oof_broad_weight"])
        for row in selected
        if "s1_oof" in str(row["method"])
    ]
    concat_weights = [
        float(row["concat_s1_oof_broad_weight"])
        for row in selected
        if "concat__s1_oof" in str(row["method"])
    ]
    concat_alphas = [
        float(row["concat_s1_oof_ridge_alpha"])
        for row in selected
        if "concat__s1_oof" in str(row["method"])
    ]
    return {
        "n_subjects": len({int(row["subject"]) for row in selected}),
        "n_sessions": len(selected),
        "test_accuracy_subject_pooled_mean": float(
            np.mean(
                [
                    100.0
                    * sum(int(row["test_correct"]) for row in subject_rows)
                    / sum(int(row["n_eval"]) for row in subject_rows)
                    for _, subject_rows in _group_by_subject(selected).items()
                ]
            )
        ),
        "gain_vs_outer_best_single_subject_pooled_pp": float(np.mean(gains)),
        "gain_vs_outer_best_single_subject_bootstrap_95ci": bootstrap_ci(
            gains, bootstrap, seed
        ),
        "selected_candidate_counts": dict(candidates),
        "selected_candidate_rates": {
            candidate: float(count / len(selected))
            for candidate, count in sorted(candidates.items())
        },
        "source_oof_posterior_weight_counts": dict(Counter(map(str, posterior_weights))),
        "source_oof_concat_weight_counts": dict(Counter(map(str, concat_weights))),
        "source_oof_concat_alpha_counts": dict(Counter(map(str, concat_alphas))),
        **session_risk(selected, fixed),
    }


def _group_by_subject(
    rows: Sequence[Mapping[str, object]],
) -> dict[int, list[Mapping[str, object]]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["subject"])].append(row)
    return grouped


def apply_e4a_gate(summary: Mapping[str, object]) -> dict[str, object]:
    ci = summary.get("gain_vs_outer_best_single_subject_bootstrap_95ci")
    lower_ci = None if ci is None else float(ci[0])
    passed = bool(
        float(summary["gain_vs_outer_best_single_subject_pooled_pp"]) >= 1.0
        and lower_ci is not None
        and lower_ci > 0.0
        and float(summary["loss_r10_vs_outer_best_single_pp"]) <= 1.0
        and float(summary["p_gain_vs_outer_best_single_lt_minus5"]) <= 0.05
    )
    return {
        "thresholds": {
            "gain_vs_outer_best_single_subject_pooled_pp_min": 1.0,
            "bootstrap_95ci_lower_strictly_gt": 0.0,
            "loss_r10_vs_outer_best_single_pp_max": 1.0,
            "p_gain_vs_outer_best_single_lt_minus5_max": 0.05,
        },
        "pass": passed,
        "reason": (
            "Proceed only if a fixed source-trained fusion improves primary pooled "
            "accuracy by >=1pp without material lower-tail loss."
        ),
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
        fixed = [row for row in scoped if row["method"] == "outer_best_single"]
        if not fixed:
            continue
        condition_report: dict[str, object] = {}
        for method in methods:
            selected = [row for row in scoped if row["method"] == method]
            condition_report[method] = summarize_method(
                selected, fixed, bootstrap, seed + len(method) + len(condition)
            )
        if condition == "primary_pooled":
            condition_report["e4a_gate"] = {
                method: apply_e4a_gate(summary)
                for method, summary in condition_report.items()
                if method != "outer_best_single" and summary
            }
        report[condition] = condition_report
    return report


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = build_rows(args)
    records = build_selection_records(rows, args.features)
    summary = summarize_records(records, args.bootstrap, args.seed)
    single, fusion = candidate_lists(args.features)
    report = {
        "config": {
            "subjects": args.subjects,
            "features": list(args.features),
            "prefix": args.prefix,
            "eval_start": args.eval_start,
            "min_eval_trials": args.min_eval_trials,
            "source_cv_folds": args.source_cv_folds,
            "posterior_weights": sorted(set(args.posterior_weights)),
            "concat_weights": sorted(set(args.concat_weights)),
            "ridge_alphas": sorted(set(args.ridge_alphas)),
            "outer": "leave-one-subject-out comparator/configuration selection",
            "primary_conditions": list(PRIMARY_CONDITIONS),
            "cache_dir": str(args.cache_dir),
        },
        "candidates": {"single": single, "fusion": fusion},
        "n_subjects": len({int(row["subject"]) for row in rows}),
        "n_rows": len(rows),
        "failures": failures,
        "summaries": summary,
    }
    (args.output_dir / "selection_records.json").write_text(
        json.dumps({"records": records}, indent=2)
    )
    write_csv(args.output_dir / "selection_records.csv", records)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
