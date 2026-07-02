"""E1 neuro feature baseline for Neuro-LEMA.

This evaluator compares all-channel broadband Riemann features against
sensorimotor and filter-bank Riemann features under the same prospective
cross-session protocol:

* source decoder: session 1 labels
* target prefix: first m unlabeled trials for prefix EA
* evaluation: fixed suffix, trial ``eval_start + 1`` onward

It is intentionally a viability screen, not the final Neuro-LEMA method.
If physiology-informed features collapse here, the learned adapter should not
be built on top of them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_CACHE = Path(
    "/home/islabshi/workspace-local2/mne_data/stieger_neuro_band_cov_cache"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260624_stieger_neuro_feature_baseline"
)
CONDITIONS = {
    "pure_lr": {"tasks": (1,), "targets": (1, 2)},
    "pure_ud": {"tasks": (2,), "targets": (3, 4)},
    "two_d": {"tasks": (3,), "targets": (1, 2, 3, 4)},
}
BAND_NAMES = ("broad_8_30", "mu_8_13", "low_beta_13_20", "high_beta_20_30")
SENSORIMOTOR21 = (
    "FC5",
    "FC3",
    "FC1",
    "C5",
    "C3",
    "C1",
    "CP5",
    "CP3",
    "CP1",
    "FCz",
    "Cz",
    "CPz",
    "FC2",
    "FC4",
    "FC6",
    "C2",
    "C4",
    "C6",
    "CP2",
    "CP4",
    "CP6",
)
FEATURE_CONFIGS = {
    "broad_all60": {
        "bands": ("broad_8_30",),
        "channels": "all",
    },
    "broad_sensorimotor21": {
        "bands": ("broad_8_30",),
        "channels": "sensorimotor21",
    },
    "fb_all60_mu_beta": {
        "bands": ("mu_8_13", "low_beta_13_20", "high_beta_20_30"),
        "channels": "all",
    },
    "fb_sensorimotor21_mu_beta": {
        "bands": ("mu_8_13", "low_beta_13_20", "high_beta_20_30"),
        "channels": "sensorimotor21",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument(
        "--eval-start",
        type=int,
        default=64,
        help="Evaluate from this zero-based trial index onward. "
        "Default 64 means trial 65 onward.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_subjects(value: str) -> list[int]:
    subjects: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", maxsplit=1))
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


def invsqrt_spd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    values, vectors = eigh(matrix)
    values = np.clip(values, eps, None)
    return (vectors * values**-0.5) @ vectors.T


def tangent_features(covariances: np.ndarray, reference: np.ndarray) -> np.ndarray:
    transformed = np.einsum(
        "ij,njk,lk->nil",
        reference,
        covariances,
        reference,
        optimize=True,
    )
    transformed = 0.5 * (transformed + transformed.transpose(0, 2, 1))
    values, vectors = np.linalg.eigh(transformed)
    values = np.clip(values, 1e-10, None)
    logs = np.einsum(
        "nij,nj,nkj->nik",
        vectors,
        np.log(values),
        vectors,
        optimize=True,
    )
    upper = np.triu_indices(covariances.shape[1])
    scale = np.sqrt(2.0) * np.ones(
        (covariances.shape[1], covariances.shape[1])
    )
    np.fill_diagonal(scale, 1.0)
    return logs[:, upper[0], upper[1]] * scale[upper]


def subject_balanced_weights(subjects: list[int]) -> np.ndarray:
    counts = Counter(int(subject) for subject in subjects)
    n_subjects = len(counts)
    return np.asarray(
        [1.0 / (n_subjects * counts[int(subject)]) for subject in subjects],
        dtype=np.float64,
    )


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = quantile * sorted_weights.sum()
    return float(sorted_values[np.searchsorted(cumulative, cutoff, side="left")])


def lower_tail_cvar(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.1,
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    target_mass = alpha * sorted_weights.sum()
    used_mass = 0.0
    total = 0.0
    for value, weight in zip(sorted_values, sorted_weights):
        take = min(float(weight), target_mass - used_mass)
        if take <= 0:
            break
        total += float(value) * take
        used_mass += take
    if used_mass <= 0:
        raise ValueError("weights must contain positive mass")
    return total / used_mass


def risk_utility_summary(
    rows: list[Mapping[str, float | int | str]],
    adapter: str,
    cvar_alpha: float = 0.1,
) -> dict[str, float | int | None | str]:
    selected = [row for row in rows if row["adapter"] == adapter]
    if not selected:
        raise ValueError(f"No rows found for adapter={adapter}")
    subjects = [int(row["subject"]) for row in selected]
    deltas = np.asarray([float(row["delta_pp"]) for row in selected])
    source_acc = np.asarray([float(row["source_acc"]) for row in selected])
    adapted_acc = np.asarray([float(row["adapted_acc"]) for row in selected])
    weights = subject_balanced_weights(subjects)
    weights /= weights.sum()
    lower_cvar = lower_tail_cvar(deltas, weights, alpha=cvar_alpha)
    eligible_70 = source_acc >= 70.0
    crossing_70 = eligible_70 & (adapted_acc < 70.0)
    crossing_probability: float | None = (
        float(np.sum(weights[crossing_70]) / np.sum(weights[eligible_70]))
        if np.any(eligible_70)
        else None
    )
    return {
        "adapter": adapter,
        "n_subjects": len(set(subjects)),
        "n_sessions": len(selected),
        "utility_mean_delta_pp": float(np.sum(weights * deltas)),
        "lower_cvar10_delta_pp": float(lower_cvar),
        "risk_r10": float(-lower_cvar),
        "weighted_q05_delta_pp": weighted_quantile(deltas, weights, 0.05),
        "p_delta_lt_0": float(np.sum(weights[deltas < 0])),
        "p_delta_lt_minus3": float(np.sum(weights[deltas < -3])),
        "p_delta_lt_minus5": float(np.sum(weights[deltas < -5])),
        "p_delta_lt_minus10": float(np.sum(weights[deltas < -10])),
        "p_cross_below_70_given_source_ge_70": crossing_probability,
    }


def one_way_icc(rows: list[dict], adapter: str) -> float | None:
    selected = [row for row in rows if row["adapter"] == adapter]
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in selected:
        grouped[int(row["subject"])].append(float(row["delta_pp"]))
    grouped = {key: value for key, value in grouped.items() if len(value) >= 2}
    if len(grouped) < 3:
        return None
    values = np.asarray([value for group in grouped.values() for value in group])
    counts = np.asarray([len(group) for group in grouped.values()], dtype=float)
    means = np.asarray([np.mean(group) for group in grouped.values()])
    grand = float(np.mean(values))
    n_observations = len(values)
    n_subjects = len(grouped)
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    ss_within = float(
        sum(
            np.sum((np.asarray(group) - np.mean(group)) ** 2)
            for group in grouped.values()
        )
    )
    ms_between = ss_between / (n_subjects - 1)
    ms_within = ss_within / (n_observations - n_subjects)
    n0 = (
        n_observations - float(np.sum(counts**2)) / n_observations
    ) / (n_subjects - 1)
    subject_variance = max((ms_between - ms_within) / n0, 0.0)
    denominator = subject_variance + ms_within
    if denominator <= 0:
        return None
    return float(subject_variance / denominator)


def bootstrap_icc(
    rows: list[dict],
    adapter: str,
    n_bootstrap: int = 1_000,
    seed: int = 0,
) -> list[float] | None:
    selected = [row for row in rows if row["adapter"] == adapter]
    subjects = sorted({int(row["subject"]) for row in selected})
    if len(subjects) < 3:
        return None
    by_subject = {
        subject: [row for row in selected if int(row["subject"]) == subject]
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        bootstrap_rows: list[dict] = []
        for new_subject, original_subject in enumerate(sampled):
            for row in by_subject[int(original_subject)]:
                copied = dict(row)
                copied["subject"] = new_subject
                bootstrap_rows.append(copied)
        value = one_way_icc(bootstrap_rows, adapter)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _channel_indices(channels: np.ndarray, channel_set: str) -> np.ndarray:
    names = [str(channel) for channel in channels.tolist()]
    if channel_set == "all":
        return np.arange(len(names), dtype=np.int64)
    if channel_set != "sensorimotor21":
        raise ValueError(f"Unknown channel_set={channel_set}")
    missing = [channel for channel in SENSORIMOTOR21 if channel not in names]
    if missing:
        raise RuntimeError(f"Missing sensorimotor channels: {missing}")
    return np.asarray([names.index(channel) for channel in SENSORIMOTOR21], dtype=np.int64)


def _subselect_covariances(covariances: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return covariances[:, indices][:, :, indices]


def load_subject(cache_dir: Path, subject: int) -> dict[str, np.ndarray]:
    paths = sorted(
        (cache_dir / f"S{subject}").glob("session_*.npz"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not paths:
        raise FileNotFoundError(cache_dir / f"S{subject}")
    arrays: dict[str, list[np.ndarray]] = {
        "task": [],
        "target": [],
        "session": [],
        "run": [],
        "trial": [],
    }
    for band_name in BAND_NAMES:
        arrays[f"cov_{band_name}"] = []
    channels: np.ndarray | None = None
    for path in paths:
        payload = np.load(path)
        if channels is None:
            channels = payload["channels"]
        n_trials = len(payload["task"])
        arrays["task"].append(payload["task"].astype(np.int64))
        arrays["target"].append(payload["target"].astype(np.int64))
        arrays["session"].append(
            np.full(n_trials, int(payload["session"]), dtype=np.int64)
        )
        arrays["run"].append(payload["run"].astype(np.int64))
        arrays["trial"].append(payload["trial"].astype(np.int64))
        for band_name in BAND_NAMES:
            arrays[f"cov_{band_name}"].append(
                payload[f"cov_{band_name}"].astype(np.float64)
            )
    if channels is None:
        raise RuntimeError(f"No channels found for S{subject}")
    merged = {key: np.concatenate(value) for key, value in arrays.items()}
    merged["channels"] = channels
    return merged


def references_for(
    band_covariances: Mapping[str, np.ndarray],
    bands: tuple[str, ...],
    trial_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        band: invsqrt_spd(band_covariances[band][trial_indices].mean(axis=0))
        for band in bands
    }


def concatenate_tangent_features(
    band_covariances: Mapping[str, np.ndarray],
    bands: tuple[str, ...],
    trial_indices: np.ndarray,
    references: Mapping[str, np.ndarray],
) -> np.ndarray:
    return np.concatenate(
        [
            tangent_features(band_covariances[band][trial_indices], references[band])
            for band in bands
        ],
        axis=1,
    )


def feature_covariances(
    data: dict[str, np.ndarray],
    config: Mapping[str, object],
) -> dict[str, np.ndarray]:
    channel_indices = _channel_indices(data["channels"], str(config["channels"]))
    return {
        band: _subselect_covariances(
            data[f"cov_{band}"],
            channel_indices,
        )
        for band in config["bands"]
    }


def evaluate_feature_condition(
    subject: int,
    feature_config: str,
    condition: str,
    specification: Mapping[str, tuple[int, ...]],
    data: dict[str, np.ndarray],
    prefix: int,
    eval_start: int,
) -> list[dict]:
    if prefix > eval_start:
        raise ValueError("prefix must be <= eval_start for fixed-suffix evaluation")
    config = FEATURE_CONFIGS[feature_config]
    bands = tuple(str(band) for band in config["bands"])
    band_covariances = feature_covariances(data, config)
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
    source_references = references_for(band_covariances, bands, source_indices)
    source_features = concatenate_tangent_features(
        band_covariances,
        bands,
        source_indices,
        source_references,
    )
    classifier = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
    ).fit(source_features, source_labels)

    rows: list[dict] = []
    for session in sessions[1:]:
        target_indices = np.flatnonzero(mask & (data["session"] == session))
        labels = data["target"][target_indices]
        if len(labels) <= eval_start or len(np.unique(labels)) < 2:
            continue
        prefix_indices = target_indices[:prefix]
        eval_indices = target_indices[eval_start:]
        eval_labels = data["target"][eval_indices]
        source_predictions = classifier.predict(
            concatenate_tangent_features(
                band_covariances,
                bands,
                eval_indices,
                source_references,
            )
        )
        source_accuracy = float(np.mean(source_predictions == eval_labels) * 100)
        rows.append(
            {
                "subject": subject,
                "session": int(session),
                "condition": condition,
                "feature_config": feature_config,
                "adapter": "source",
                "source_acc": source_accuracy,
                "adapted_acc": source_accuracy,
                "delta_pp": 0.0,
                "n_prefix": prefix,
                "eval_start": eval_start,
                "n_eval": int(len(eval_labels)),
                "n_bands": len(bands),
                "channel_set": str(config["channels"]),
            }
        )
        for adapter, reference_indices in (
            ("prefix_ea", prefix_indices),
            ("full_ea", target_indices),
        ):
            target_references = references_for(
                band_covariances,
                bands,
                reference_indices,
            )
            predictions = classifier.predict(
                concatenate_tangent_features(
                    band_covariances,
                    bands,
                    eval_indices,
                    target_references,
                )
            )
            adapted_accuracy = float(np.mean(predictions == eval_labels) * 100)
            rows.append(
                {
                    "subject": subject,
                    "session": int(session),
                    "condition": condition,
                    "feature_config": feature_config,
                    "adapter": adapter,
                    "source_acc": source_accuracy,
                    "adapted_acc": adapted_accuracy,
                    "delta_pp": adapted_accuracy - source_accuracy,
                    "n_prefix": prefix,
                    "eval_start": eval_start,
                    "n_eval": int(len(eval_labels)),
                    "n_bands": len(bands),
                    "channel_set": str(config["channels"]),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for condition in CONDITIONS:
        output[condition] = {}
        for feature_config in FEATURE_CONFIGS:
            selected = [
                row
                for row in rows
                if row["condition"] == condition
                and row["feature_config"] == feature_config
            ]
            if not selected:
                continue
            output[condition][feature_config] = {}
            for adapter in ("source", "prefix_ea", "full_ea"):
                if not any(row["adapter"] == adapter for row in selected):
                    continue
                summary = risk_utility_summary(selected, adapter)
                if adapter == "source":
                    summary["icc_delta"] = None
                    summary["icc_subject_bootstrap_95ci"] = None
                else:
                    summary["icc_delta"] = one_way_icc(selected, adapter)
                    summary["icc_subject_bootstrap_95ci"] = bootstrap_icc(
                        selected,
                        adapter,
                    )
                output[condition][feature_config][adapter] = summary
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(exist_ok=True)
    all_rows: list[dict] = []
    failures: list[dict] = []
    for subject in parse_subjects(args.subjects):
        output = subject_dir / f"S{subject}.json"
        if output.exists() and not args.force:
            payload = json.loads(output.read_text())
            all_rows.extend(payload["rows"])
            print(f"S{subject}: resume ({len(payload['rows'])} rows)", flush=True)
            continue
        try:
            data = load_subject(args.cache_dir, subject)
            rows: list[dict] = []
            for feature_config in FEATURE_CONFIGS:
                for condition, specification in CONDITIONS.items():
                    rows.extend(
                        evaluate_feature_condition(
                            subject,
                            feature_config,
                            condition,
                            specification,
                            data,
                            args.prefix,
                            args.eval_start,
                        )
                    )
            output.write_text(json.dumps({"subject": subject, "rows": rows}, indent=2))
            all_rows.extend(rows)
            print(f"S{subject}: {len(rows)} rows", flush=True)
        except Exception as error:
            failures.append({"subject": subject, "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)
    report = {
        "config": {
            "subjects": args.subjects,
            "prefix": args.prefix,
            "eval_start": args.eval_start,
            "cache_dir": str(args.cache_dir),
            "feature_configs": FEATURE_CONFIGS,
            "conditions": CONDITIONS,
        },
        "n_subjects": len({int(row["subject"]) for row in all_rows}),
        "n_rows": len(all_rows),
        "failures": failures,
        "summaries": summarize(all_rows),
        "rows": all_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summaries"], indent=2), flush=True)


if __name__ == "__main__":
    main()
