"""E5a: class-conditional geometry audit for longitudinal EEG-MI.

This is a *diagnostic* experiment, not a deployable adaptation method.  E1/E1b
showed that broad and neurophysiology-constrained branches are complementary.
E2/G1a then showed that unlabeled target-prefix statistics do not reliably tell
which branch or channel subset will be safe.  E4a finally showed that fixed
trial-level fusion recovers average accuracy but still has a large lower tail.

E5a asks whether the missing signal is class-conditional:

    Are harmful sessions and fusion failures explained by source-target
    shifts of class centroids, margins, prototype identity, or LR
    lateralization, rather than by global unlabeled shift alone?

Target labels from the evaluation suffix are deliberately used to measure this
mechanism.  Therefore any positive result supports a next representation
learning experiment; it does *not* imply that the same metrics can be used as a
test-time gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score

from stieger_neuro_feature_baseline import (
    CONDITIONS,
    DEFAULT_CACHE,
    FEATURE_CONFIGS,
    concatenate_tangent_features,
    feature_covariances,
    load_subject,
    parse_subjects,
    references_for,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_E4A_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260626_stieger_neuro_fixed_fusion"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260627_stieger_class_conditional_geometry_audit"
)
DEFAULT_FEATURES = ("broad_all60", "fb_sensorimotor21_mu_beta")
PRIMARY_CONDITIONS = ("pure_lr", "pure_ud")
MULTIVARIATE_SCOPES = ("primary_pooled",)
MULTIVARIATE_CONTINUOUS_TARGETS = (
    "fusion_gain_vs_outer_best_single_pp",
    "branch_gap_neuro_minus_broad_pp",
)
MULTIVARIATE_BINARY_TARGETS = (
    "fusion_harm5_vs_outer_best_single",
    "broad_ea_harm5",
    "neuro_ea_harm5",
)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fusion-table-json",
        type=Path,
        default=DEFAULT_E4A_OUTPUT / "fusion_table.json",
    )
    parser.add_argument(
        "--selection-records-json",
        type=Path,
        default=DEFAULT_E4A_OUTPUT / "selection_records.json",
    )
    parser.add_argument("--features", nargs=2, default=list(DEFAULT_FEATURES))
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument(
        "--eval-start",
        type=int,
        default=64,
        help="Zero-based evaluation start. Default 64 means trial 65 onward.",
    )
    parser.add_argument("--min-eval-trials", type=int, default=40)
    parser.add_argument("--force", action="store_true")
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


def candidate_acc_key(candidate: str) -> str:
    return f"candidate__{candidate}__acc"


def single_name(feature: str, adapter: str) -> str:
    return f"single__{feature}__{adapter}"


def posterior_name(kind: str, adapter: str) -> str:
    return f"posterior__{kind}__{adapter}"


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


def load_json(path: Path) -> object:
    with path.open() as handle:
        return json.load(handle)


def load_e4a_outcomes(
    fusion_table_json: Path,
    selection_records_json: Path,
    features: Sequence[str],
) -> dict[tuple[int, int, str], dict[str, object]]:
    """Load E4a accuracies and derive per-session outcome variables."""
    payload = load_json(fusion_table_json)
    fusion_rows = list(payload["rows"])  # type: ignore[index]
    selection_payload = load_json(selection_records_json)
    selection_rows = list(selection_payload["records"])  # type: ignore[index]
    selected_by_key: dict[tuple[int, int, str, str], Mapping[str, object]] = {}
    for record in selection_rows:
        key = (
            int(record["subject"]),
            int(record["session"]),
            str(record["condition"]),
            str(record["method"]),
        )
        selected_by_key[key] = record

    broad, neuro = features
    broad_source = single_name(broad, "source")
    broad_prefix = single_name(broad, "prefix_ea")
    neuro_source = single_name(neuro, "source")
    neuro_prefix = single_name(neuro, "prefix_ea")
    equal_prefix = posterior_name("equal", "prefix_ea")

    outcomes: dict[tuple[int, int, str], dict[str, object]] = {}
    for row in fusion_rows:
        key = (int(row["subject"]), int(row["session"]), str(row["condition"]))
        broad_source_acc = float(row[candidate_acc_key(broad_source)])
        broad_prefix_acc = float(row[candidate_acc_key(broad_prefix)])
        neuro_source_acc = float(row[candidate_acc_key(neuro_source)])
        neuro_prefix_acc = float(row[candidate_acc_key(neuro_prefix)])
        equal_prefix_acc = float(row[candidate_acc_key(equal_prefix)])
        best_prefix_branch_acc = max(broad_prefix_acc, neuro_prefix_acc)
        outer_record = selected_by_key.get((*key, "outer_best_single"))
        outer_acc = (
            float(outer_record["test_acc"])
            if outer_record is not None
            else best_prefix_branch_acc
        )
        fusion_vs_outer = equal_prefix_acc - outer_acc
        fusion_vs_best_prefix = equal_prefix_acc - best_prefix_branch_acc
        outcomes[key] = {
            "broad_source_acc": broad_source_acc,
            "broad_prefix_acc": broad_prefix_acc,
            "neuro_source_acc": neuro_source_acc,
            "neuro_prefix_acc": neuro_prefix_acc,
            "posterior_equal_prefix_acc": equal_prefix_acc,
            "best_prefix_branch_acc": best_prefix_branch_acc,
            "outer_best_single_acc": outer_acc,
            "broad_ea_delta_pp": broad_prefix_acc - broad_source_acc,
            "neuro_ea_delta_pp": neuro_prefix_acc - neuro_source_acc,
            "branch_gap_neuro_minus_broad_pp": neuro_prefix_acc - broad_prefix_acc,
            "branch_abs_gap_pp": abs(neuro_prefix_acc - broad_prefix_acc),
            "fusion_gain_vs_best_prefix_branch_pp": fusion_vs_best_prefix,
            "fusion_gain_vs_outer_best_single_pp": fusion_vs_outer,
            "broad_ea_harm5": int((broad_prefix_acc - broad_source_acc) < -5.0),
            "neuro_ea_harm5": int((neuro_prefix_acc - neuro_source_acc) < -5.0),
            "fusion_harm5_vs_best_prefix_branch": int(fusion_vs_best_prefix < -5.0),
            "fusion_harm5_vs_outer_best_single": int(fusion_vs_outer < -5.0),
        }
    return outcomes


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _pairwise_distances(centroids: np.ndarray) -> np.ndarray:
    n_rows = centroids.shape[0]
    distances = np.full((n_rows, n_rows), np.nan, dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_rows):
            distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    return distances


def class_geometry_metrics(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
) -> dict[str, float | int]:
    """Class-conditional source-target geometry in a common feature space."""
    dim = int(source_features.shape[1])
    scale = float(np.sqrt(max(dim, 1)))
    metrics: dict[str, float | int] = {"feature_dim": dim}

    source_var = np.var(source_features, axis=0) + 1e-8
    target_var = np.var(target_features, axis=0) + 1e-8
    source_spread = float(np.sqrt(np.mean(source_var)))
    target_spread = float(np.sqrt(np.mean(target_var)))
    metrics.update(
        {
            "global_shift_rms": float(
                np.linalg.norm(target_features.mean(axis=0) - source_features.mean(axis=0))
                / scale
            ),
            "source_spread_rms": source_spread,
            "target_spread_rms": target_spread,
            "spread_ratio": _safe_ratio(target_spread, source_spread),
            "logvar_shift_rms": float(
                np.linalg.norm(np.log(target_var) - np.log(source_var)) / scale
            ),
        }
    )

    classes = sorted(set(map(int, np.unique(source_labels))).intersection(map(int, np.unique(target_labels))))
    metrics["n_common_classes"] = len(classes)
    if len(classes) < 2:
        return metrics

    source_centroids = np.asarray(
        [source_features[source_labels == label].mean(axis=0) for label in classes],
        dtype=np.float64,
    )
    target_centroids = np.asarray(
        [target_features[target_labels == label].mean(axis=0) for label in classes],
        dtype=np.float64,
    )
    class_shifts = np.linalg.norm(target_centroids - source_centroids, axis=1) / scale
    source_distances = _pairwise_distances(source_centroids) / scale
    target_distances = _pairwise_distances(target_centroids) / scale
    non_diag = ~np.eye(len(classes), dtype=bool)
    source_margin = float(np.nanmin(source_distances[non_diag]))
    target_margin = float(np.nanmin(target_distances[non_diag]))

    cross_distances = np.empty((len(classes), len(classes)), dtype=np.float64)
    for target_row in range(len(classes)):
        for source_col in range(len(classes)):
            cross_distances[target_row, source_col] = (
                np.linalg.norm(target_centroids[target_row] - source_centroids[source_col])
                / scale
            )
    same_distances = np.diag(cross_distances)
    nearest_source = np.argmin(cross_distances, axis=1)
    prototype_correct = nearest_source == np.arange(len(classes))
    wrong_gaps: list[float] = []
    for row_index in range(len(classes)):
        wrong = np.delete(cross_distances[row_index], row_index)
        wrong_gaps.append(float(np.min(wrong) - same_distances[row_index]))
    wrong_gaps_array = np.asarray(wrong_gaps, dtype=np.float64)

    metrics.update(
        {
            "class_shift_mean_rms": float(np.mean(class_shifts)),
            "class_shift_max_rms": float(np.max(class_shifts)),
            "class_shift_std_rms": float(np.std(class_shifts)),
            "class_shift_cv": _safe_ratio(float(np.std(class_shifts)), float(np.mean(class_shifts))),
            "source_margin_rms": source_margin,
            "target_margin_rms": target_margin,
            "margin_ratio": _safe_ratio(target_margin, source_margin),
            "relative_class_shift_mean": _safe_ratio(float(np.mean(class_shifts)), source_margin),
            "relative_class_shift_max": _safe_ratio(float(np.max(class_shifts)), source_margin),
            "target_margin_over_shift": _safe_ratio(target_margin, float(np.mean(class_shifts))),
            "prototype_accuracy": float(np.mean(prototype_correct)),
            "prototype_error_rate": float(1.0 - np.mean(prototype_correct)),
            "prototype_gap_mean_rms": float(np.mean(wrong_gaps_array)),
            "prototype_gap_min_rms": float(np.min(wrong_gaps_array)),
            "prototype_gap_mean_over_margin": _safe_ratio(float(np.mean(wrong_gaps_array)), source_margin),
            "prototype_gap_min_over_margin": _safe_ratio(float(np.min(wrong_gaps_array)), source_margin),
        }
    )
    return metrics


def prefix_metrics(row: dict[str, object], prefix: str, metrics: Mapping[str, object]) -> None:
    for key, value in metrics.items():
        row[f"{prefix}__{key}"] = value


def add_metric_differences(
    row: dict[str, object],
    feature: str,
    metric_names: Sequence[str],
) -> None:
    source_prefix = f"geom__{feature}__source_ref"
    ea_prefix = f"geom__{feature}__prefix_ea"
    diff_prefix = f"geom__{feature}__prefix_minus_source_ref"
    for metric in metric_names:
        source_value = float(row.get(f"{source_prefix}__{metric}", np.nan))
        ea_value = float(row.get(f"{ea_prefix}__{metric}", np.nan))
        row[f"{diff_prefix}__{metric}"] = (
            ea_value - source_value
            if np.isfinite(source_value) and np.isfinite(ea_value)
            else float("nan")
        )


def lateralization_metrics(
    data: Mapping[str, np.ndarray],
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    condition: str,
) -> dict[str, object]:
    """Simple LR C3/C4 log-power contrast diagnostic."""
    output: dict[str, object] = {}
    if condition != "pure_lr":
        return output
    channels = [str(channel) for channel in data["channels"].tolist()]
    if "C3" not in channels or "C4" not in channels:
        return output
    c3_index = channels.index("C3")
    c4_index = channels.index("C4")
    labels = sorted(set(map(int, np.unique(source_labels))).intersection(map(int, np.unique(target_labels))))
    if len(labels) != 2:
        return output
    first, second = labels
    for band in ("mu_8_13", "low_beta_13_20", "high_beta_20_30"):
        source_diag = np.diagonal(data[f"cov_{band}"][source_indices], axis1=1, axis2=2)
        target_diag = np.diagonal(data[f"cov_{band}"][target_indices], axis1=1, axis2=2)
        source_lr = np.log(source_diag[:, c3_index] + 1e-12) - np.log(
            source_diag[:, c4_index] + 1e-12
        )
        target_lr = np.log(target_diag[:, c3_index] + 1e-12) - np.log(
            target_diag[:, c4_index] + 1e-12
        )
        source_contrast = float(
            source_lr[source_labels == first].mean()
            - source_lr[source_labels == second].mean()
        )
        target_contrast = float(
            target_lr[target_labels == first].mean()
            - target_lr[target_labels == second].mean()
        )
        output[f"lat__{band}__source_c3_c4_contrast"] = source_contrast
        output[f"lat__{band}__target_c3_c4_contrast"] = target_contrast
        output[f"lat__{band}__contrast_abs_ratio"] = _safe_ratio(
            abs(target_contrast), abs(source_contrast)
        )
        output[f"lat__{band}__contrast_diff_abs"] = abs(target_contrast - source_contrast)
        output[f"lat__{band}__sign_agreement"] = int(
            np.sign(source_contrast) == np.sign(target_contrast)
        )
    return output


def build_geometry_rows(
    args: argparse.Namespace,
    outcomes: Mapping[tuple[int, int, str], Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    table_path = args.output_dir / "geometry_table.json"
    if table_path.exists() and not args.force:
        payload = load_json(table_path)
        return list(payload["rows"]), list(payload.get("failures", []))  # type: ignore[index]

    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    broad, neuro = args.features
    for subject in parse_subjects(args.subjects):
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force:
            payload = load_json(output)
            all_rows.extend(payload["rows"])  # type: ignore[index]
            if not args.quiet:
                print(f"S{subject}: resume ({len(payload['rows'])} rows)", flush=True)  # type: ignore[index]
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            subject_rows: list[dict[str, object]] = []
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
                for feature in (broad, neuro):
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
                        "source_references": source_references,
                        "source_features": source_features,
                    }

                for session in sessions[1:]:
                    target_indices = np.flatnonzero(mask & (data["session"] == session))
                    if len(target_indices) <= args.eval_start:
                        continue
                    eval_indices = target_indices[args.eval_start :]
                    if len(eval_indices) < args.min_eval_trials:
                        continue
                    key = (int(subject), int(session), condition)
                    if key not in outcomes:
                        continue
                    eval_labels = data["target"][eval_indices]
                    prefix_indices = target_indices[: args.prefix]
                    row: dict[str, object] = {
                        "subject": int(subject),
                        "session": int(session),
                        "condition": condition,
                        "source_session": source_session,
                        "n_source": int(len(source_indices)),
                        "n_prefix": int(len(prefix_indices)),
                        "eval_start": int(args.eval_start),
                        "n_eval": int(len(eval_indices)),
                    }
                    row.update(outcomes[key])
                    row.update(
                        lateralization_metrics(
                            data,
                            source_indices,
                            eval_indices,
                            source_labels,
                            eval_labels,
                            condition,
                        )
                    )
                    for feature in (broad, neuro):
                        branch = branch_data[feature]
                        source_features = branch["source_features"]  # type: ignore[assignment]
                        source_ref_target = concatenate_tangent_features(
                            branch["covariances"],  # type: ignore[arg-type]
                            branch["bands"],  # type: ignore[arg-type]
                            eval_indices,
                            branch["source_references"],  # type: ignore[arg-type]
                        )
                        prefix_references = references_for(
                            branch["covariances"],  # type: ignore[arg-type]
                            branch["bands"],  # type: ignore[arg-type]
                            prefix_indices,
                        )
                        prefix_target = concatenate_tangent_features(
                            branch["covariances"],  # type: ignore[arg-type]
                            branch["bands"],  # type: ignore[arg-type]
                            eval_indices,
                            prefix_references,
                        )
                        prefix_metrics(
                            row,
                            f"geom__{feature}__source_ref",
                            class_geometry_metrics(
                                source_features,  # type: ignore[arg-type]
                                source_labels,
                                source_ref_target,
                                eval_labels,
                            ),
                        )
                        prefix_metrics(
                            row,
                            f"geom__{feature}__prefix_ea",
                            class_geometry_metrics(
                                source_features,  # type: ignore[arg-type]
                                source_labels,
                                prefix_target,
                                eval_labels,
                            ),
                        )
                        add_metric_differences(
                            row,
                            feature,
                            (
                                "global_shift_rms",
                                "class_shift_mean_rms",
                                "relative_class_shift_mean",
                                "margin_ratio",
                                "prototype_gap_min_over_margin",
                                "prototype_error_rate",
                                "spread_ratio",
                                "logvar_shift_rms",
                            ),
                        )
                    subject_rows.append(row)
            output.write_text(json.dumps({"subject": subject, "rows": subject_rows}, indent=2))
            all_rows.extend(subject_rows)
            if not args.quiet:
                print(f"S{subject}: {len(subject_rows)} rows", flush=True)
        except Exception as error:
            failure = {"subject": int(subject), "error": repr(error)}
            failures.append(failure)
            print(f"S{subject}: FAIL {error!r}", flush=True)
    (args.output_dir / "geometry_table.json").write_text(
        json.dumps({"rows": all_rows, "failures": failures}, indent=2)
    )
    write_csv(args.output_dir / "geometry_table.csv", all_rows)
    return all_rows, failures


def is_numeric(value: object) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def numeric_columns(rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not rows:
        return []
    blocked = {
        "subject",
        "session",
        "source_session",
        "n_source",
        "n_prefix",
        "eval_start",
        "n_eval",
    }
    candidates = sorted(
        key
        for key in rows[0]
        if key not in blocked and key != "condition"
    )
    output = []
    for column in candidates:
        values = [row.get(column) for row in rows]
        if sum(is_numeric(value) for value in values) >= max(10, int(0.5 * len(rows))):
            output.append(column)
    return output


def predictor_groups(columns: Sequence[str]) -> dict[str, list[str]]:
    geometry = [column for column in columns if column.startswith("geom__") or column.startswith("lat__")]
    label_free_tokens = (
        "global_shift_rms",
        "source_spread_rms",
        "target_spread_rms",
        "spread_ratio",
        "logvar_shift_rms",
    )
    class_tokens = (
        "class_shift",
        "margin",
        "prototype",
        "target_margin_over_shift",
        "lat__",
    )
    return {
        "global_unlabeled_geometry": [
            column for column in geometry if any(token in column for token in label_free_tokens)
        ],
        "class_conditional_geometry": [
            column for column in geometry if any(token in column for token in class_tokens)
        ],
        "all_geometry": geometry,
    }


def rows_for_scope(rows: Sequence[Mapping[str, object]], scope: str) -> list[Mapping[str, object]]:
    if scope == "primary_pooled":
        return [row for row in rows if row["condition"] in PRIMARY_CONDITIONS]
    return [row for row in rows if row["condition"] == scope]


def finite_arrays(
    rows: Sequence[Mapping[str, object]],
    x_column: str,
    y_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        x = row.get(x_column)
        y = row.get(y_column)
        if is_numeric(x) and is_numeric(y):
            xs.append(float(x))
            ys.append(float(y))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def univariate_spearman(
    rows: Sequence[Mapping[str, object]],
    predictor_columns: Sequence[str],
    target: str,
    top_k: int = 12,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for predictor in predictor_columns:
        x, y = finite_arrays(rows, predictor, target)
        if len(x) < 20 or len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
            continue
        rho, pvalue = spearmanr(x, y)
        if not np.isfinite(rho):
            continue
        results.append(
            {
                "predictor": predictor,
                "n": int(len(x)),
                "spearman_rho": float(rho),
                "pvalue": float(pvalue),
                "abs_spearman_rho": float(abs(rho)),
            }
        )
    return sorted(results, key=lambda item: item["abs_spearman_rho"], reverse=True)[:top_k]


def univariate_auroc(
    rows: Sequence[Mapping[str, object]],
    predictor_columns: Sequence[str],
    target: str,
    top_k: int = 12,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for predictor in predictor_columns:
        x, y = finite_arrays(rows, predictor, target)
        if len(x) < 20 or len(np.unique(y)) != 2 or len(np.unique(x)) < 2:
            continue
        auc = float(roc_auc_score(y.astype(int), x))
        oriented_auc = max(auc, 1.0 - auc)
        direction = "high_risk" if auc >= 0.5 else "low_risk"
        results.append(
            {
                "predictor": predictor,
                "n": int(len(x)),
                "positive_rate": float(np.mean(y)),
                "auroc": auc,
                "oriented_auroc": float(oriented_auc),
                "direction": direction,
            }
        )
    return sorted(results, key=lambda item: item["oriented_auroc"], reverse=True)[:top_k]


def design_matrix(
    rows: Sequence[Mapping[str, object]],
    feature_columns: Sequence[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    kept_rows = [row for row in rows if is_numeric(row.get(target))]
    kept_columns = []
    for column in feature_columns:
        finite_count = sum(is_numeric(row.get(column)) for row in kept_rows)
        if finite_count >= max(20, int(0.7 * len(kept_rows))):
            kept_columns.append(column)
    x = np.full((len(kept_rows), len(kept_columns)), np.nan, dtype=np.float64)
    for row_index, row in enumerate(kept_rows):
        for column_index, column in enumerate(kept_columns):
            value = row.get(column)
            if is_numeric(value):
                x[row_index, column_index] = float(value)
    y = np.asarray([float(row[target]) for row in kept_rows], dtype=np.float64)
    groups = np.asarray([int(row["subject"]) for row in kept_rows], dtype=np.int64)
    return x, y, groups, kept_columns


def impute_and_standardize(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train = np.where(np.isfinite(x_train), x_train, medians)
    test = np.where(np.isfinite(x_test), x_test, medians)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def loso_multivariate(
    rows: Sequence[Mapping[str, object]],
    feature_columns: Sequence[str],
    target: str,
    kind: str,
) -> dict[str, object]:
    x, y, groups, kept_columns = design_matrix(rows, feature_columns, target)
    if len(y) < 30 or not kept_columns:
        return {"n": int(len(y)), "n_features": 0, "status": "not_enough_data"}
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    for held_subject in sorted(np.unique(groups).tolist()):
        train_mask = groups != held_subject
        test_mask = groups == held_subject
        if not np.any(test_mask) or not np.any(train_mask):
            continue
        x_train, x_test = impute_and_standardize(x[train_mask], x[test_mask])
        y_train = y[train_mask]
        if kind == "binary":
            if len(np.unique(y_train)) != 2:
                continue
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=0,
            )
            model.fit(x_train, y_train.astype(int))
            predictions[test_mask] = model.predict_proba(x_test)[:, 1]
        elif kind == "continuous":
            model = RidgeCV(alphas=np.asarray(RIDGE_ALPHAS, dtype=np.float64))
            model.fit(x_train, y_train)
            predictions[test_mask] = model.predict(x_test)
        else:
            raise ValueError(f"Unknown kind={kind}")

    valid = np.isfinite(predictions)
    result: dict[str, object] = {
        "n": int(np.sum(valid)),
        "n_features": int(len(kept_columns)),
        "status": "ok" if np.sum(valid) >= 30 else "not_enough_valid_predictions",
    }
    if result["status"] != "ok":
        return result
    if kind == "binary":
        y_valid = y[valid].astype(int)
        if len(np.unique(y_valid)) != 2:
            result["status"] = "single_class"
            return result
        auc = float(roc_auc_score(y_valid, predictions[valid]))
        result.update(
            {
                "positive_rate": float(np.mean(y_valid)),
                "auroc": auc,
                "oriented_auroc": float(max(auc, 1.0 - auc)),
            }
        )
    else:
        rho, pvalue = spearmanr(predictions[valid], y[valid])
        result.update(
            {
                "spearman_rho": float(rho) if np.isfinite(rho) else None,
                "abs_spearman_rho": float(abs(rho)) if np.isfinite(rho) else None,
                "pvalue": float(pvalue) if np.isfinite(pvalue) else None,
            }
        )
    return result


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    columns = numeric_columns(rows)
    groups = predictor_groups(columns)
    continuous_targets = (
        "broad_ea_delta_pp",
        "neuro_ea_delta_pp",
        "branch_gap_neuro_minus_broad_pp",
        "fusion_gain_vs_best_prefix_branch_pp",
        "fusion_gain_vs_outer_best_single_pp",
    )
    binary_targets = (
        "broad_ea_harm5",
        "neuro_ea_harm5",
        "fusion_harm5_vs_best_prefix_branch",
        "fusion_harm5_vs_outer_best_single",
    )
    report: dict[str, object] = {"predictor_groups": {key: len(value) for key, value in groups.items()}}
    for scope in (*CONDITIONS.keys(), "primary_pooled"):
        scoped = rows_for_scope(rows, scope)
        if not scoped:
            continue
        scope_report: dict[str, object] = {
            "n_sessions": len(scoped),
            "n_subjects": len({int(row["subject"]) for row in scoped}),
        }
        top_continuous: dict[str, object] = {}
        top_binary: dict[str, object] = {}
        multivariate: dict[str, object] = {}
        for target in continuous_targets:
            if target not in columns:
                continue
            top_continuous[target] = univariate_spearman(
                scoped,
                groups["all_geometry"],
                target,
            )
            if scope in MULTIVARIATE_SCOPES and target in MULTIVARIATE_CONTINUOUS_TARGETS:
                multivariate[target] = {
                    group_name: loso_multivariate(scoped, group_columns, target, "continuous")
                    for group_name, group_columns in groups.items()
                }
        for target in binary_targets:
            if target not in columns:
                continue
            top_binary[target] = univariate_auroc(
                scoped,
                groups["all_geometry"],
                target,
            )
            if scope in MULTIVARIATE_SCOPES and target in MULTIVARIATE_BINARY_TARGETS:
                multivariate[target] = {
                    group_name: loso_multivariate(scoped, group_columns, target, "binary")
                    for group_name, group_columns in groups.items()
                }
        scope_report["top_univariate_spearman"] = top_continuous
        scope_report["top_univariate_auroc"] = top_binary
        scope_report["loso_multivariate"] = multivariate
        report[scope] = scope_report
    return report


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = load_e4a_outcomes(
        args.fusion_table_json,
        args.selection_records_json,
        args.features,
    )
    rows, failures = build_geometry_rows(args, outcomes)
    summary = summarize_rows(rows)
    report = {
        "config": {
            "subjects": args.subjects,
            "features": list(args.features),
            "prefix": args.prefix,
            "eval_start": args.eval_start,
            "min_eval_trials": args.min_eval_trials,
            "cache_dir": str(args.cache_dir),
            "fusion_table_json": str(args.fusion_table_json),
            "selection_records_json": str(args.selection_records_json),
            "primary_conditions": list(PRIMARY_CONDITIONS),
            "note": (
                "Class-conditional metrics use evaluation-suffix target labels. "
                "They are mechanism diagnostics, not deployable gates."
            ),
        },
        "n_subjects": len({int(row["subject"]) for row in rows}),
        "n_rows": len(rows),
        "failures": failures,
        "summaries": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    if args.quiet:
        print(
            json.dumps(
                {
                    "n_subjects": report["n_subjects"],
                    "n_rows": report["n_rows"],
                    "n_failures": len(failures),
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
