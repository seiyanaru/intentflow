"""E2 label-free branch selector for Neuro-LEMA.

E1/E1b showed that broad all-channel and sensorimotor mu/beta branches have
large session-level complementarity, but the E1b oracle used target labels.
This script tests the next question:

Can a selector trained on other subjects recover any of that oracle headroom
using only target-prefix label-free signals?

Default arms are intentionally minimal:

* broad_all60 source
* broad_all60 prefix_ea
* fb_sensorimotor21_mu_beta source
* fb_sensorimotor21_mu_beta prefix_ea

The evaluation is grouped by subject.  The ridge selector predicts each arm's
absolute target-suffix accuracy from prefix-only signals and chooses the arm
with the largest predicted accuracy.  Hyperparameter selection is done inside
the training subjects of each outer fold.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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
    tangent_features,
    weighted_quantile,
)
from stieger_neuro_feature_complementarity import (
    bootstrap_ci_by_subject,
    subject_balanced_weights,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260624_stieger_neuro_branch_selector"
)
DEFAULT_E1_SUMMARY = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260624_stieger_neuro_feature_baseline"
    / "summary.json"
)
DEFAULT_BRANCH_FEATURES = ("broad_all60", "fb_sensorimotor21_mu_beta")
DEFAULT_ADAPTERS = ("source", "prefix_ea")
DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--e1-summary-json",
        type=Path,
        default=DEFAULT_E1_SUMMARY,
        help="Use E1 arm accuracies instead of recomputing suffix predictions.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--eval-start", type=int, default=64)
    parser.add_argument(
        "--branch-features",
        nargs="+",
        default=list(DEFAULT_BRANCH_FEATURES),
        help="Feature branches used as selectable arms.",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=list(DEFAULT_ADAPTERS),
        choices=["source", "prefix_ea"],
    )
    parser.add_argument("--bootstrap-instability", type=int, default=8)
    parser.add_argument(
        "--outer",
        choices=["loso", "group5"],
        default="loso",
        help="Outer grouped evaluation. LOSO is stricter but slower.",
    )
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_ALPHAS),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_e1_lookup(path: Path | None) -> dict[tuple[int, int, str, str, str], dict]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    lookup = {}
    for row in payload["rows"]:
        key = (
            int(row["subject"]),
            int(row["session"]),
            str(row["condition"]),
            str(row["feature_config"]),
            str(row["adapter"]),
        )
        lookup[key] = row
    return lookup


def arm_name(feature_config: str, adapter: str) -> str:
    return f"{feature_config}__{adapter}"


def entropy_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    if probabilities.shape[1] > 1:
        entropy /= np.log(probabilities.shape[1])
    return entropy


def probability_summary(probabilities: np.ndarray) -> dict[str, float]:
    confidence = np.max(probabilities, axis=1)
    if probabilities.shape[1] >= 2:
        sorted_probabilities = np.sort(probabilities, axis=1)
        margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    else:
        margin = np.ones(len(probabilities), dtype=np.float64)
    entropy = entropy_from_probabilities(probabilities)
    return {
        "mean_conf": float(np.mean(confidence)),
        "std_conf": float(np.std(confidence)),
        "mean_margin": float(np.mean(margin)),
        "std_margin": float(np.std(margin)),
        "mean_entropy": float(np.mean(entropy)),
        "std_entropy": float(np.std(entropy)),
    }


def prediction_distribution_entropy(
    predictions: np.ndarray,
    classes: np.ndarray,
) -> float:
    counts = np.asarray([np.sum(predictions == label) for label in classes], dtype=float)
    probabilities = counts / max(float(np.sum(counts)), 1.0)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    if len(classes) > 1:
        entropy /= float(np.log(len(classes)))
    return entropy


def log_spd_vector(covariance: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return tangent_features(covariance[None, :, :], reference)[0]


def covariance_shift_summaries(
    band_covariances: Mapping[str, np.ndarray],
    bands: tuple[str, ...],
    source_indices: np.ndarray,
    prefix_indices: np.ndarray,
    source_references: Mapping[str, np.ndarray],
    target_references: Mapping[str, np.ndarray],
) -> dict[str, float]:
    values: dict[str, float] = {}
    tangent_norms = []
    reference_norms = []
    for band in bands:
        source_mean = band_covariances[band][source_indices].mean(axis=0)
        prefix_mean = band_covariances[band][prefix_indices].mean(axis=0)
        tangent = log_spd_vector(prefix_mean, source_references[band])
        tangent_norm = float(np.linalg.norm(tangent))
        reference_norm = float(
            np.linalg.norm(target_references[band] - source_references[band])
            / max(np.linalg.norm(source_references[band]), 1e-12)
        )
        values[f"band_{band}_cov_shift_norm"] = tangent_norm
        values[f"band_{band}_reference_shift_rel_norm"] = reference_norm
        values[f"band_{band}_trace_ratio"] = float(
            np.trace(prefix_mean) / max(np.trace(source_mean), 1e-12)
        )
        tangent_norms.append(tangent_norm)
        reference_norms.append(reference_norm)
    values["cov_shift_norm_mean"] = float(np.mean(tangent_norms))
    values["cov_shift_norm_max"] = float(np.max(tangent_norms))
    values["reference_shift_rel_norm_mean"] = float(np.mean(reference_norms))
    values["reference_shift_rel_norm_max"] = float(np.max(reference_norms))
    if len(tangent_norms) >= 2:
        values["cov_shift_norm_std"] = float(np.std(tangent_norms))
        values["reference_shift_rel_norm_std"] = float(np.std(reference_norms))
    else:
        values["cov_shift_norm_std"] = 0.0
        values["reference_shift_rel_norm_std"] = 0.0
    return values


def bootstrap_reference_instability(
    band_covariances: Mapping[str, np.ndarray],
    bands: tuple[str, ...],
    prefix_indices: np.ndarray,
    full_references: Mapping[str, np.ndarray],
    classifier,
    full_prefix_predictions: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    if n_bootstrap <= 0:
        return {
            "bootstrap_pred_disagreement_mean": 0.0,
            "bootstrap_pred_disagreement_std": 0.0,
            "bootstrap_reference_shift_mean": 0.0,
            "bootstrap_reference_shift_std": 0.0,
        }
    rng = np.random.default_rng(seed)
    disagreements = []
    shifts = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(prefix_indices, size=len(prefix_indices), replace=True)
        references = references_for(band_covariances, bands, sampled)
        features = concatenate_tangent_features(
            band_covariances,
            bands,
            prefix_indices,
            references,
        )
        predictions = classifier.predict(features)
        disagreements.append(float(np.mean(predictions != full_prefix_predictions)))
        band_shifts = [
            float(
                np.linalg.norm(references[band] - full_references[band])
                / max(np.linalg.norm(full_references[band]), 1e-12)
            )
            for band in bands
        ]
        shifts.append(float(np.mean(band_shifts)))
    return {
        "bootstrap_pred_disagreement_mean": float(np.mean(disagreements)),
        "bootstrap_pred_disagreement_std": float(np.std(disagreements)),
        "bootstrap_reference_shift_mean": float(np.mean(shifts)),
        "bootstrap_reference_shift_std": float(np.std(shifts)),
    }


def add_prefixed(
    output: dict[str, float],
    prefix: str,
    values: Mapping[str, float],
) -> None:
    for key, value in values.items():
        output[f"lf__{prefix}__{key}"] = float(value)


def evaluate_subject_condition(
    subject: int,
    condition: str,
    specification: Mapping[str, tuple[int, ...]],
    data: dict[str, np.ndarray],
    branch_features: list[str],
    adapters: list[str],
    prefix: int,
    eval_start: int,
    bootstrap_instability: int,
    seed: int,
    e1_lookup: Mapping[tuple[int, int, str, str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    if prefix > eval_start:
        raise ValueError("prefix must be <= eval_start")
    mask = np.isin(data["task"], specification["tasks"]) & np.isin(
        data["target"],
        specification["targets"],
    )
    sessions = sorted(np.unique(data["session"][mask]).tolist())
    if len(sessions) < 2:
        return []
    source_session = sessions[0]
    source_indices = np.flatnonzero(mask & (data["session"] == source_session))
    source_labels = data["target"][source_indices]
    if len(np.unique(source_labels)) < 2:
        return []

    branch_models: dict[str, dict[str, object]] = {}
    for feature_config in branch_features:
        config = FEATURE_CONFIGS[feature_config]
        bands = tuple(str(band) for band in config["bands"])
        band_covariances = feature_covariances(data, config)
        source_references = references_for(band_covariances, bands, source_indices)
        source_features = concatenate_tangent_features(
            band_covariances,
            bands,
            source_indices,
            source_references,
        )
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        classifier = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        ).fit(source_features, source_labels)
        source_train_predictions = classifier.predict(source_features)
        branch_models[feature_config] = {
            "bands": bands,
            "band_covariances": band_covariances,
            "source_references": source_references,
            "classifier": classifier,
            "source_train_accuracy": float(
                np.mean(source_train_predictions == source_labels) * 100.0
            ),
            "source_class_balance_entropy": prediction_distribution_entropy(
                source_labels,
                classifier.classes_,
            ),
        }

    rows: list[dict[str, object]] = []
    for session in sessions[1:]:
        target_indices = np.flatnonzero(mask & (data["session"] == session))
        labels = data["target"][target_indices]
        if len(labels) <= eval_start or len(np.unique(labels)) < 2:
            continue
        prefix_indices = target_indices[:prefix]
        eval_indices = target_indices[eval_start:]
        n_eval = int(len(eval_indices))
        row: dict[str, object] = {
            "subject": int(subject),
            "session": int(session),
            "condition": condition,
            "n_prefix": int(prefix),
            "eval_start": int(eval_start),
            "n_eval": n_eval,
            "lf__session_number": float(session),
            "lf__n_eval": float(n_eval),
        }
        prefix_predictions: dict[str, np.ndarray] = {}
        prefix_probabilities: dict[str, np.ndarray] = {}

        for feature_config in branch_features:
            model = branch_models[feature_config]
            bands = model["bands"]
            band_covariances = model["band_covariances"]
            source_references = model["source_references"]
            classifier = model["classifier"]
            assert isinstance(bands, tuple)
            assert isinstance(band_covariances, dict)
            assert isinstance(source_references, dict)

            source_prefix_features = concatenate_tangent_features(
                band_covariances,
                bands,
                prefix_indices,
                source_references,
            )
            source_prefix_probabilities = classifier.predict_proba(
                source_prefix_features
            )
            source_prefix_predictions = classifier.predict(source_prefix_features)
            source_lookup = e1_lookup.get(
                (subject, int(session), condition, feature_config, "source")
            )
            if source_lookup is not None:
                source_accuracy = float(source_lookup["adapted_acc"])
            else:
                eval_labels = data["target"][eval_indices]
                source_eval_features = concatenate_tangent_features(
                    band_covariances,
                    bands,
                    eval_indices,
                    source_references,
                )
                source_eval_predictions = classifier.predict(source_eval_features)
                source_accuracy = float(
                    np.mean(source_eval_predictions == eval_labels) * 100.0
                )
            source_arm = arm_name(feature_config, "source")
            row[f"arm__{source_arm}__acc"] = source_accuracy
            row[f"arm__{source_arm}__delta_vs_feature_source_pp"] = 0.0
            row[f"arm__{source_arm}__feature_source_acc"] = source_accuracy
            prefix_predictions[source_arm] = source_prefix_predictions
            prefix_probabilities[source_arm] = source_prefix_probabilities
            add_prefixed(
                row,
                f"{feature_config}__source",
                {
                    **probability_summary(source_prefix_probabilities),
                    "prediction_balance_entropy": prediction_distribution_entropy(
                        source_prefix_predictions,
                        classifier.classes_,
                    ),
                    "source_train_accuracy": float(model["source_train_accuracy"]),
                    "source_class_balance_entropy": float(
                        model["source_class_balance_entropy"]
                    ),
                },
            )

            target_references = references_for(
                band_covariances,
                bands,
                prefix_indices,
            )
            prefix_ea_prefix_features = concatenate_tangent_features(
                band_covariances,
                bands,
                prefix_indices,
                target_references,
            )
            prefix_ea_prefix_probabilities = classifier.predict_proba(
                prefix_ea_prefix_features
            )
            prefix_ea_prefix_predictions = classifier.predict(prefix_ea_prefix_features)
            prefix_lookup = e1_lookup.get(
                (subject, int(session), condition, feature_config, "prefix_ea")
            )
            if prefix_lookup is not None:
                prefix_ea_accuracy = float(prefix_lookup["adapted_acc"])
            else:
                eval_labels = data["target"][eval_indices]
                prefix_ea_eval_features = concatenate_tangent_features(
                    band_covariances,
                    bands,
                    eval_indices,
                    target_references,
                )
                prefix_ea_eval_predictions = classifier.predict(prefix_ea_eval_features)
                prefix_ea_accuracy = float(
                    np.mean(prefix_ea_eval_predictions == eval_labels) * 100.0
                )
            prefix_ea_arm = arm_name(feature_config, "prefix_ea")
            row[f"arm__{prefix_ea_arm}__acc"] = prefix_ea_accuracy
            row[f"arm__{prefix_ea_arm}__delta_vs_feature_source_pp"] = (
                prefix_ea_accuracy - source_accuracy
            )
            row[f"arm__{prefix_ea_arm}__feature_source_acc"] = source_accuracy
            prefix_predictions[prefix_ea_arm] = prefix_ea_prefix_predictions
            prefix_probabilities[prefix_ea_arm] = prefix_ea_prefix_probabilities
            add_prefixed(
                row,
                f"{feature_config}__prefix_ea",
                {
                    **probability_summary(prefix_ea_prefix_probabilities),
                    "prediction_balance_entropy": prediction_distribution_entropy(
                        prefix_ea_prefix_predictions,
                        classifier.classes_,
                    ),
                    "source_to_prefix_ea_pred_disagreement": float(
                        np.mean(
                            source_prefix_predictions != prefix_ea_prefix_predictions
                        )
                    ),
                    "source_to_prefix_ea_conf_change": float(
                        np.max(prefix_ea_prefix_probabilities, axis=1).mean()
                        - np.max(source_prefix_probabilities, axis=1).mean()
                    ),
                },
            )
            add_prefixed(
                row,
                f"{feature_config}__shift",
                covariance_shift_summaries(
                    band_covariances,
                    bands,
                    source_indices,
                    prefix_indices,
                    source_references,
                    target_references,
                ),
            )
            add_prefixed(
                row,
                f"{feature_config}__bootstrap",
                bootstrap_reference_instability(
                    band_covariances,
                    bands,
                    prefix_indices,
                    target_references,
                    classifier,
                    prefix_ea_prefix_predictions,
                    bootstrap_instability,
                    seed + subject * 1000 + int(session) * 10 + len(rows),
                ),
            )

        if len(branch_features) >= 2:
            first, second = branch_features[:2]
            for adapter in adapters:
                first_arm = arm_name(first, adapter)
                second_arm = arm_name(second, adapter)
                if first_arm in prefix_predictions and second_arm in prefix_predictions:
                    add_prefixed(
                        row,
                        f"{first}_vs_{second}__{adapter}",
                        {
                            "pred_disagreement": float(
                                np.mean(
                                    prefix_predictions[first_arm]
                                    != prefix_predictions[second_arm]
                                )
                            ),
                            "conf_diff_second_minus_first": float(
                                np.max(prefix_probabilities[second_arm], axis=1).mean()
                                - np.max(prefix_probabilities[first_arm], axis=1).mean()
                            ),
                            "entropy_diff_second_minus_first": float(
                                entropy_from_probabilities(
                                    prefix_probabilities[second_arm]
                                ).mean()
                                - entropy_from_probabilities(
                                    prefix_probabilities[first_arm]
                                ).mean()
                            ),
                        },
                    )

        broad_source_key = f"arm__{arm_name(branch_features[0], 'source')}__acc"
        broad_source_acc = float(row[broad_source_key])
        best_acc = -np.inf
        best_arm = ""
        for feature_config in branch_features:
            for adapter in adapters:
                arm = arm_name(feature_config, adapter)
                acc = float(row[f"arm__{arm}__acc"])
                row[f"arm__{arm}__delta_vs_primary_source_pp"] = (
                    acc - broad_source_acc
                )
                if acc > best_acc:
                    best_acc = acc
                    best_arm = arm
        row["oracle_arm"] = best_arm
        row["oracle_acc"] = float(best_acc)
        row["oracle_delta_vs_primary_source_pp"] = float(best_acc - broad_source_acc)
        rows.append(row)
    return rows


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


def load_or_build_feature_table(args: argparse.Namespace) -> list[dict[str, object]]:
    table_path = args.output_dir / "branch_selector_table.json"
    if table_path.exists() and not args.force:
        return json.loads(table_path.read_text())["rows"]

    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    e1_lookup = load_e1_lookup(args.e1_summary_json)
    for subject in parse_subjects(args.subjects):
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force:
            payload = json.loads(output.read_text())
            all_rows.extend(payload["rows"])
            print(f"S{subject}: resume ({len(payload['rows'])} rows)", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            rows: list[dict[str, object]] = []
            for condition, specification in CONDITIONS.items():
                rows.extend(
                    evaluate_subject_condition(
                        subject,
                        condition,
                        specification,
                        data,
                        list(args.branch_features),
                        list(args.adapters),
                        args.prefix,
                        args.eval_start,
                        args.bootstrap_instability,
                        args.seed,
                        e1_lookup,
                    )
                )
            output.write_text(json.dumps({"subject": subject, "rows": rows}, indent=2))
            all_rows.extend(rows)
            print(f"S{subject}: {len(rows)} rows", flush=True)
        except Exception as error:
            failures.append({"subject": subject, "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)
    payload = {
        "config": {
            "subjects": args.subjects,
            "cache_dir": str(args.cache_dir),
            "e1_summary_json": str(args.e1_summary_json),
            "prefix": args.prefix,
            "eval_start": args.eval_start,
            "branch_features": list(args.branch_features),
            "adapters": list(args.adapters),
            "bootstrap_instability": args.bootstrap_instability,
            "seed": args.seed,
        },
        "n_subjects": len({int(row["subject"]) for row in all_rows}),
        "n_rows": len(all_rows),
        "failures": failures,
        "rows": all_rows,
    }
    table_path.write_text(json.dumps(payload, indent=2))
    write_csv(args.output_dir / "branch_selector_table.csv", all_rows)
    return all_rows


def subject_balanced_mean(values: np.ndarray, subjects: np.ndarray) -> float:
    weights = subject_balanced_weights([int(subject) for subject in subjects])
    return float(np.sum((weights / weights.sum()) * values))


def metric_summary(
    records: list[Mapping[str, object]],
    method: str,
) -> dict[str, object]:
    selected = [record for record in records if record["method"] == method]
    subjects = np.asarray([int(record["subject"]) for record in selected])
    weights = subject_balanced_weights(subjects.tolist())
    weights = weights / weights.sum()
    acc = np.asarray([float(record["selected_acc"]) for record in selected])
    delta_primary = np.asarray(
        [float(record["selected_delta_vs_primary_source_pp"]) for record in selected]
    )
    delta_feature = np.asarray(
        [float(record["selected_delta_vs_feature_source_pp"]) for record in selected]
    )
    arm_counts = Counter(str(record["selected_arm"]) for record in selected)
    arm_rates = {
        arm: float(
            np.sum(
                weights[
                    np.asarray([str(record["selected_arm"]) == arm for record in selected])
                ]
            )
        )
        for arm in sorted(arm_counts)
    }
    return {
        "method": method,
        "n_subjects": len(set(subjects.tolist())),
        "n_sessions": len(selected),
        "adapted_acc_mean": float(np.sum(weights * acc)),
        "delta_vs_primary_source_mean": float(np.sum(weights * delta_primary)),
        "delta_vs_primary_source_r10": float(-lower_tail_cvar(delta_primary, weights)),
        "delta_vs_primary_source_q05": weighted_quantile(
            delta_primary,
            weights,
            0.05,
        ),
        "p_delta_vs_primary_source_lt_minus5": float(
            np.sum(weights[delta_primary < -5.0])
        ),
        "delta_vs_feature_source_mean": float(np.sum(weights * delta_feature)),
        "delta_vs_feature_source_r10": float(-lower_tail_cvar(delta_feature, weights)),
        "p_delta_vs_feature_source_lt_minus5": float(
            np.sum(weights[delta_feature < -5.0])
        ),
        "selected_arm_counts": dict(arm_counts),
        "selected_arm_rates": arm_rates,
    }


def outer_splits(subjects: np.ndarray, mode: str, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_subjects = np.asarray(sorted(set(int(subject) for subject in subjects)))
    if mode == "loso":
        return [
            (
                np.flatnonzero(subjects != subject),
                np.flatnonzero(subjects == subject),
            )
            for subject in unique_subjects
        ]
    if mode == "group5":
        splitter = GroupKFold(n_splits=min(5, len(unique_subjects)))
        dummy = np.zeros(len(subjects))
        return [(train, test) for train, test in splitter.split(dummy, groups=subjects)]
    raise ValueError(mode)


def fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_subjects: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    weights = subject_balanced_weights(train_subjects.tolist())
    model = Ridge(alpha=alpha)
    model.fit(x_train_scaled, y_train, sample_weight=weights)
    return model.predict(x_test_scaled)


def choose_alpha_inner_cv(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    arms: list[str],
    alphas: list[float],
    inner_splits: int,
) -> float:
    unique_subjects = sorted(set(int(subject) for subject in subjects))
    n_splits = min(inner_splits, len(unique_subjects))
    if n_splits < 2:
        return float(alphas[0])
    splitter = GroupKFold(n_splits=n_splits)
    scores: dict[float, list[float]] = {float(alpha): [] for alpha in alphas}
    dummy = np.zeros(len(subjects))
    for train_idx, valid_idx in splitter.split(dummy, groups=subjects):
        for alpha in alphas:
            predictions = fit_predict_ridge(
                x[train_idx],
                y[train_idx],
                subjects[train_idx],
                x[valid_idx],
                float(alpha),
            )
            chosen = np.argmax(predictions, axis=1)
            selected = y[valid_idx, chosen]
            score = subject_balanced_mean(selected, subjects[valid_idx])
            scores[float(alpha)].append(score)
    return max(scores, key=lambda alpha: float(np.mean(scores[alpha])))


def evaluate_condition(
    rows: list[Mapping[str, object]],
    condition: str,
    arms: list[str],
    outer_mode: str,
    inner_splits: int,
    alphas: list[float],
    seed: int,
) -> dict[str, object]:
    selected_rows = [row for row in rows if row["condition"] == condition]
    if not selected_rows:
        return {}
    feature_columns = sorted(
        key
        for key in selected_rows[0]
        if key.startswith("lf__")
        and all(isinstance(row.get(key), (int, float)) for row in selected_rows)
    )
    x = np.asarray(
        [[float(row[column]) for column in feature_columns] for row in selected_rows],
        dtype=np.float64,
    )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(
        [[float(row[f"arm__{arm}__acc"]) for arm in arms] for row in selected_rows],
        dtype=np.float64,
    )
    subjects = np.asarray([int(row["subject"]) for row in selected_rows])
    splits = outer_splits(subjects, outer_mode, seed)

    prediction_records: list[dict[str, object]] = []
    prefix_arm_indices = [
        index for index, arm in enumerate(arms) if arm.endswith("__prefix_ea")
    ]
    for train_idx, test_idx in splits:
        train_subjects = subjects[train_idx]
        test_rows = [selected_rows[int(index)] for index in test_idx]
        train_weights = subject_balanced_weights(train_subjects.tolist())
        train_weights = train_weights / train_weights.sum()
        train_arm_means = np.sum(y[train_idx] * train_weights[:, None], axis=0)
        train_best_arm_index = int(np.argmax(train_arm_means))
        alpha = choose_alpha_inner_cv(
            x[train_idx],
            y[train_idx],
            train_subjects,
            arms,
            alphas,
            inner_splits,
        )
        ridge_predictions = fit_predict_ridge(
            x[train_idx],
            y[train_idx],
            train_subjects,
            x[test_idx],
            alpha,
        )
        ridge_chosen = np.argmax(ridge_predictions, axis=1)
        prefix_ridge_chosen: np.ndarray | None = None
        if len(prefix_arm_indices) >= 2:
            prefix_alpha = choose_alpha_inner_cv(
                x[train_idx],
                y[train_idx][:, prefix_arm_indices],
                train_subjects,
                [arms[index] for index in prefix_arm_indices],
                alphas,
                inner_splits,
            )
            prefix_predictions = fit_predict_ridge(
                x[train_idx],
                y[train_idx][:, prefix_arm_indices],
                train_subjects,
                x[test_idx],
                prefix_alpha,
            )
            prefix_ridge_chosen = np.asarray(
                [
                    prefix_arm_indices[int(local_index)]
                    for local_index in np.argmax(prefix_predictions, axis=1)
                ],
                dtype=int,
            )
        oracle_chosen = np.argmax(y[test_idx], axis=1)

        for local, row in enumerate(test_rows):
            broad_source_acc = float(row[f"arm__{arms[0]}__acc"])
            methods = {
                "train_best_fixed": train_best_arm_index,
                "ridge_acc_selector": int(ridge_chosen[local]),
                "oracle": int(oracle_chosen[local]),
            }
            if prefix_ridge_chosen is not None:
                methods["ridge_prefix_pair_selector"] = int(
                    prefix_ridge_chosen[local]
                )
            for arm_index, arm in enumerate(arms):
                methods[f"fixed__{arm}"] = arm_index
            for method, arm_index in methods.items():
                arm = arms[int(arm_index)]
                acc = float(row[f"arm__{arm}__acc"])
                prediction_records.append(
                    {
                        "subject": int(row["subject"]),
                        "session": int(row["session"]),
                        "condition": condition,
                        "method": method,
                        "selected_arm": arm,
                        "selected_acc": acc,
                        "selected_delta_vs_primary_source_pp": acc - broad_source_acc,
                        "selected_delta_vs_feature_source_pp": float(
                            row[f"arm__{arm}__delta_vs_feature_source_pp"]
                        ),
                        "alpha": float(alpha),
                        "oracle_arm": arms[int(oracle_chosen[local])],
                        "oracle_acc": float(y[test_idx[local], oracle_chosen[local]]),
                    }
                )

    methods = sorted({record["method"] for record in prediction_records})
    summaries = {method: metric_summary(prediction_records, method) for method in methods}
    fixed_methods = [method for method in methods if method.startswith("fixed__")]
    best_fixed_method = max(
        fixed_methods,
        key=lambda method: float(summaries[method]["adapted_acc_mean"]),
    )
    best_fixed_acc = float(summaries[best_fixed_method]["adapted_acc_mean"])

    def gain_vs_baseline(
        sample: list[Mapping[str, object]],
        method: str,
        baseline_method: str,
    ) -> float:
        by_key: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
        subjects_for_keys = {}
        for record in sample:
            if record["method"] in {method, baseline_method}:
                key = (int(record["subject"]), int(record["session"]))
                by_key[key][str(record["method"])] = float(record["selected_acc"])
                subjects_for_keys[key] = int(record["subject"])
        values = []
        subjects_local = []
        for key, values_by_method in by_key.items():
            if method in values_by_method and baseline_method in values_by_method:
                values.append(
                    values_by_method[method] - values_by_method[baseline_method]
                )
                subjects_local.append(subjects_for_keys[key])
        return subject_balanced_mean(np.asarray(values), np.asarray(subjects_local))

    for method in methods:
        summaries[method]["best_fixed_method"] = best_fixed_method
        summaries[method]["gain_vs_best_fixed_acc_pp"] = (
            float(summaries[method]["adapted_acc_mean"]) - best_fixed_acc
        )

    for method in ["ridge_acc_selector", "ridge_prefix_pair_selector", "oracle"]:
        if method not in summaries:
            continue
        summaries[method]["gain_vs_train_best_fixed_acc_pp"] = gain_vs_baseline(
            prediction_records,
            method,
            "train_best_fixed",
        )
        summaries[method]["gain_vs_train_best_fixed_bootstrap_95ci"] = (
            bootstrap_ci_by_subject(
                prediction_records,
                lambda sample, method=method: gain_vs_baseline(
                    sample,
                    method,
                    "train_best_fixed",
                ),
                1000,
                seed + (1 if method == "ridge_acc_selector" else 2),
            )
        )
        summaries[method]["gain_vs_best_fixed_bootstrap_95ci"] = (
            bootstrap_ci_by_subject(
                prediction_records,
                lambda sample, method=method: gain_vs_baseline(
                    sample,
                    method,
                    best_fixed_method,
                ),
                1000,
                seed + (11 if method == "ridge_acc_selector" else 12),
            )
        )

    return {
        "condition": condition,
        "arms": arms,
        "feature_columns": feature_columns,
        "n_feature_columns": len(feature_columns),
        "outer": outer_mode,
        "inner_splits": inner_splits,
        "alphas": alphas,
        "summaries": summaries,
        "prediction_records": prediction_records,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_or_build_feature_table(args)
    arms = [
        arm_name(feature_config, adapter)
        for feature_config in args.branch_features
        for adapter in args.adapters
    ]
    condition_reports = {}
    all_prediction_records: list[dict[str, object]] = []
    for condition in CONDITIONS:
        report = evaluate_condition(
            rows,
            condition,
            arms,
            args.outer,
            args.inner_splits,
            list(args.alphas),
            args.seed,
        )
        condition_reports[condition] = {
            key: value for key, value in report.items() if key != "prediction_records"
        }
        all_prediction_records.extend(report["prediction_records"])

    summary = {
        "config": {
            "subjects": args.subjects,
            "cache_dir": str(args.cache_dir),
            "output_dir": str(args.output_dir),
            "prefix": args.prefix,
            "eval_start": args.eval_start,
            "branch_features": list(args.branch_features),
            "adapters": list(args.adapters),
            "arms": arms,
            "bootstrap_instability": args.bootstrap_instability,
            "outer": args.outer,
            "inner_splits": args.inner_splits,
            "alphas": list(args.alphas),
            "seed": args.seed,
        },
        "n_rows": len(rows),
        "condition_reports": condition_reports,
    }
    (args.output_dir / "selector_summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(args.output_dir / "selector_predictions.csv", all_prediction_records)
    if args.quiet:
        print(args.output_dir / "selector_summary.json", flush=True)
    else:
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
