"""V3 pilot: source-side longitudinal subspace selection on Lee2019_MI.

This is the second-dataset check for the Stieger E5b result.

Protocol:

* dataset: Lee2019_MI, left-vs-right MI, two sessions per subject;
* held-out unit: subject;
* target labels for the held-out subject's session 1 are never used for
  selecting the subspace or fitting the classifier;
* source-side longitudinal statistics are learned from all non-held subjects'
  session0 -> session1 labeled structure;
* the held-out subject trains an LDA on labeled session0 and is evaluated on
  session1 trial eval_start onward;
* target session1 prefix trials are used only as unlabeled EA reference.

The goal is not to optimize Lee2019.  The goal is to ask whether the
"longitudinally stable tangent subspace" effect is Stieger-specific or
transfers to a second multi-session MI dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path
from typing import Mapping, Sequence

warnings.filterwarnings("ignore")
os.environ.setdefault("MNE_DATA", "/home/islabshi/workspace-local2/mne_data")

import mne
import numpy as np
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

mne.set_log_level("ERROR")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parents[1]
    / "results"
    / "research_outputs"
    / "260627_lee2019_longitudinal_selection_pilot"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Subject feature cache. Defaults to output-dir/subject_cache.",
    )
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=20)
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.25, 0.1])
    parser.add_argument("--random-repeats", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
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


def fraction_token(fraction: float) -> str:
    return f"q{fraction:.2f}".replace(".", "p")


def invsqrtm(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    return (eigenvectors * (eigenvalues**-0.5)) @ eigenvectors.T


def logm_spd(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    return (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T


def covariances(x: np.ndarray) -> np.ndarray:
    cov = np.einsum("nct,ndt->ncd", x, x, optimize=True) / x.shape[-1]
    ridge = 1e-6 * np.trace(cov, axis1=1, axis2=2) / cov.shape[-1]
    cov[:, np.arange(cov.shape[-1]), np.arange(cov.shape[-1])] += ridge[:, None]
    return cov


def tangent_features(cov: np.ndarray, reference_indices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    reference = cov[reference_indices].mean(axis=0)
    projection = invsqrtm(reference)
    aligned = np.einsum("ij,njk,kl->nil", projection, cov[indices], projection, optimize=True)
    dim = aligned.shape[-1]
    upper = np.triu_indices(dim)
    scale = np.sqrt(2.0) * np.ones((dim, dim), dtype=np.float64)
    scale[np.diag_indices(dim)] = 1.0
    scale = scale[upper]
    output = np.empty((len(indices), len(scale)), dtype=np.float64)
    for row, matrix in enumerate(aligned):
        output[row] = logm_spd(matrix)[upper] * scale
    return output


def bandpass_8_30(x: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    b, a = butter(5, [8.0 / (sfreq / 2.0), 30.0 / (sfreq / 2.0)], btype="band")
    return filtfilt(b, a, x, axis=-1).astype(np.float64, copy=False)


def load_subject_features(
    subject: int,
    cache_dir: Path,
    prefix: int,
    eval_start: int,
    force_cache: bool,
) -> dict[str, np.ndarray]:
    path = cache_dir / f"S{subject}.npz"
    if path.exists() and not force_cache:
        payload = np.load(path)
        return {key: payload[key] for key in payload.files}

    paradigm = LeftRightImagery(resample=250, fmin=1, fmax=45)
    dataset = Lee2019_MI()
    x, y_raw, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    labels = (y_raw == "right_hand").astype(np.int64)
    sessions = np.asarray(meta["session"].values).astype(str)
    if not {"0", "1"}.issubset(set(sessions)):
        raise ValueError(f"S{subject}: expected sessions 0 and 1, got {sorted(set(sessions))}")

    x = bandpass_8_30(x.astype(np.float64, copy=False))
    cov = covariances(x)

    session0 = np.flatnonzero(sessions == "0")
    session1 = np.flatnonzero(sessions == "1")
    if len(session1) <= eval_start:
        raise ValueError(f"S{subject}: eval_start={eval_start} leaves no target trials")
    if not (0 < prefix <= eval_start):
        raise ValueError(f"Require 0 < prefix <= eval_start, got {prefix}, {eval_start}")

    prefix_indices = session1[:prefix]
    eval_indices = session1[eval_start:]

    output = {
        "source_features": tangent_features(cov, session0, session0).astype(np.float32),
        "source_labels": labels[session0].astype(np.int64),
        "target_features": tangent_features(cov, prefix_indices, eval_indices).astype(np.float32),
        "target_labels": labels[eval_indices].astype(np.int64),
        "all_target_features": tangent_features(cov, prefix_indices, session1).astype(np.float32),
        "all_target_labels": labels[session1].astype(np.int64),
        "n_source": np.asarray([len(session0)], dtype=np.int64),
        "n_prefix": np.asarray([len(prefix_indices)], dtype=np.int64),
        "n_eval": np.asarray([len(eval_indices)], dtype=np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    return output


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
    classes = sorted(
        set(map(int, np.unique(source_labels))).intersection(map(int, np.unique(target_labels)))
    )
    if len(classes) < 2:
        raise ValueError("need at least two shared classes")
    source_centroids, source_within = class_centroids_and_within(
        source_features, source_labels, classes
    )
    target_centroids, target_within = class_centroids_and_within(
        target_features, target_labels, classes
    )
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
    subject_payloads: Mapping[int, Mapping[str, np.ndarray]],
    held_subject: int,
) -> dict[str, np.ndarray]:
    total: dict[str, np.ndarray] = {}
    for subject, payload in subject_payloads.items():
        if int(subject) == int(held_subject):
            continue
        stats = session_metric_stats(
            payload["source_features"],
            payload["source_labels"],
            payload["target_features"],
            payload["target_labels"],
        )
        add_stats(total, stats)
    count = float(total["count"][0])
    averaged = {key: value / count for key, value in total.items() if key != "count"}
    averaged["count"] = np.asarray([count], dtype=np.float64)
    return averaged


def longitudinal_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    signal = np.maximum(stats["source_between"], 0.0) + np.maximum(stats["target_between"], 0.0)
    nuisance = (
        np.maximum(stats["same_class_drift"], 0.0)
        + 0.25 * np.maximum(stats["source_within"], 0.0)
        + 0.25 * np.maximum(stats["target_within"], 0.0)
        + 1e-8
    )
    score = signal / nuisance
    return np.where(np.isfinite(score), score, 0.0)


def source_only_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    score = np.maximum(stats["source_between"], 0.0) / (
        0.25 * np.maximum(stats["source_within"], 0.0) + 1e-8
    )
    return np.where(np.isfinite(score), score, 0.0)


def sep_no_drift_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    signal = np.maximum(stats["source_between"], 0.0) + np.maximum(stats["target_between"], 0.0)
    nuisance = (
        0.25 * np.maximum(stats["source_within"], 0.0)
        + 0.25 * np.maximum(stats["target_within"], 0.0)
        + 1e-8
    )
    score = signal / nuisance
    return np.where(np.isfinite(score), score, 0.0)


def target_only_score(stats: Mapping[str, np.ndarray]) -> np.ndarray:
    score = np.maximum(stats["target_between"], 0.0) / (
        0.25 * np.maximum(stats["target_within"], 0.0) + 1e-8
    )
    return np.where(np.isfinite(score), score, 0.0)


SCORE_FUNCTIONS = {
    "longitudinal": longitudinal_score,
    "source_only": source_only_score,
    "sep_no_drift": sep_no_drift_score,
    "target_only": target_only_score,
}


def top_fraction_indices(score: np.ndarray, fraction: float) -> np.ndarray:
    dim = len(score)
    if fraction >= 1.0:
        return np.arange(dim, dtype=np.int64)
    score = np.where(np.isfinite(score), score, -np.inf)
    k = max(2, int(np.ceil(float(fraction) * dim)))
    k = min(k, dim)
    selected = np.argpartition(score, -k)[-k:]
    return np.sort(selected.astype(np.int64))


def fit_predict_accuracy(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    indices: np.ndarray,
) -> float:
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(source_features[:, indices], source_labels)
    predictions = model.predict(target_features[:, indices])
    return float(100.0 * np.mean(predictions == target_labels))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(int(repeats), len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def lower_tail_loss(values: np.ndarray, fraction: float = 0.1) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    k = max(1, int(np.ceil(float(fraction) * len(values))))
    return float(-np.sort(values)[:k].mean())


def summarize_records(
    records: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(seed)
    methods = sorted({str(row["method"]) for row in records})
    full_by_subject = {
        int(row["subject"]): float(row["accuracy"])
        for row in records
        if str(row["method"]) == "full_q1p00"
    }
    summary: dict[str, dict[str, object]] = {}
    for method in methods:
        rows = [row for row in records if str(row["method"]) == method]
        by_subject = {int(row["subject"]): float(row["accuracy"]) for row in rows}
        acc = np.asarray([by_subject[subject] for subject in subjects if subject in by_subject])
        gains = np.asarray(
            [
                by_subject[subject] - full_by_subject[subject]
                for subject in subjects
                if subject in by_subject and subject in full_by_subject
            ],
            dtype=np.float64,
        )
        summary[method] = {
            "n_subjects": int(len(acc)),
            "accuracy_mean": float(acc.mean()) if len(acc) else float("nan"),
            "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, bootstrap),
            "gain_vs_full_mean_pp": float(gains.mean()) if len(gains) else float("nan"),
            "gain_vs_full_subject_bootstrap_95ci": bootstrap_ci(gains, rng, bootstrap),
            "gain_vs_full_q05_pp": float(np.quantile(gains, 0.05)) if len(gains) else float("nan"),
            "loss_r10_vs_full_pp": lower_tail_loss(gains),
            "p_gain_vs_full_lt_minus5": float(np.mean(gains < -5.0)) if len(gains) else float("nan"),
        }
    return summary


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(args.seed)
    subjects = parse_subjects(args.subjects)
    cache_dir = args.cache_dir if args.cache_dir is not None else args.output_dir / "subject_cache"
    payloads: dict[int, dict[str, np.ndarray]] = {}
    failures: list[dict[str, object]] = []

    for subject in subjects:
        try:
            payloads[subject] = load_subject_features(
                subject,
                cache_dir=cache_dir,
                prefix=args.prefix,
                eval_start=args.eval_start,
                force_cache=args.force_cache,
            )
            if not args.quiet:
                print(
                    f"S{subject}: cached dim={payloads[subject]['source_features'].shape[1]} "
                    f"n_eval={int(payloads[subject]['n_eval'][0])}",
                    flush=True,
                )
        except Exception as error:
            failures.append({"subject": int(subject), "stage": "load", "error": repr(error)})
            print(f"S{subject}: FAIL {error!r}", flush=True)

    records: list[dict[str, object]] = []
    available_subjects = sorted(payloads)
    for held_subject in available_subjects:
        payload = payloads[held_subject]
        stats = aggregate_stats(payloads, held_subject)
        dim = payload["source_features"].shape[1]
        score_by_family = {name: fn(stats) for name, fn in SCORE_FUNCTIONS.items()}
        index_by_method: dict[str, np.ndarray] = {
            "full_q1p00": np.arange(dim, dtype=np.int64),
        }
        for family, score in score_by_family.items():
            for fraction in args.fractions:
                index_by_method[f"{family}_{fraction_token(float(fraction))}"] = top_fraction_indices(
                    score, float(fraction)
                )
        for fraction in args.fractions:
            if float(fraction) >= 1.0:
                continue
            k = len(top_fraction_indices(np.arange(dim), float(fraction)))
            for repeat in range(args.random_repeats):
                index_by_method[f"random_{fraction_token(float(fraction))}_r{repeat:02d}"] = np.sort(
                    rng.choice(dim, size=k, replace=False).astype(np.int64)
                )

        for method, indices in index_by_method.items():
            accuracy = fit_predict_accuracy(
                payload["source_features"],
                payload["source_labels"],
                payload["target_features"],
                payload["target_labels"],
                indices,
            )
            records.append(
                {
                    "subject": int(held_subject),
                    "method": method,
                    "accuracy": accuracy,
                    "n_selected": int(len(indices)),
                    "n_source": int(payload["n_source"][0]),
                    "n_eval": int(payload["n_eval"][0]),
                    "prefix": int(args.prefix),
                    "eval_start": int(args.eval_start),
                    "n_metric_subjects": int(len(available_subjects) - 1),
                }
            )
        if not args.quiet:
            full = next(
                row["accuracy"]
                for row in records
                if int(row["subject"]) == int(held_subject) and row["method"] == "full_q1p00"
            )
            best_longitudinal = max(
                row["accuracy"]
                for row in records
                if int(row["subject"]) == int(held_subject)
                and str(row["method"]).startswith("longitudinal_")
            )
            print(
                f"S{held_subject}: full={float(full):.1f} best_long={float(best_longitudinal):.1f}",
                flush=True,
            )

    return records, failures


def collapse_random_methods(summary: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    collapsed: dict[str, dict[str, object]] = {}
    for method, metrics in summary.items():
        if not method.startswith("random_") or "_r" not in method:
            collapsed[method] = dict(metrics)
    random_groups: dict[str, list[Mapping[str, object]]] = {}
    for method, metrics in summary.items():
        if method.startswith("random_") and "_r" in method:
            prefix = method.rsplit("_r", maxsplit=1)[0]
            random_groups.setdefault(prefix, []).append(metrics)
    for prefix, rows in random_groups.items():
        collapsed[f"{prefix}_mean"] = {
            "n_repeats": int(len(rows)),
            "accuracy_mean": float(np.mean([float(row["accuracy_mean"]) for row in rows])),
            "gain_vs_full_mean_pp": float(np.mean([float(row["gain_vs_full_mean_pp"]) for row in rows])),
            "loss_r10_vs_full_pp": float(np.mean([float(row["loss_r10_vs_full_pp"]) for row in rows])),
            "p_gain_vs_full_lt_minus5": float(
                np.mean([float(row["p_gain_vs_full_lt_minus5"]) for row in rows])
            ),
        }
    return collapsed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    records, failures = evaluate(args)
    write_csv(args.output_dir / "selection_records.csv", records)
    write_csv(args.output_dir / "failures.csv", failures)
    summary = summarize_records(records, subjects, args.bootstrap, args.seed)
    collapsed = collapse_random_methods(summary)
    write_csv(
        args.output_dir / "summary.csv",
        [{"method": method, **metrics} for method, metrics in collapsed.items()],
    )
    report = {
        "config": {
            "subjects": subjects,
            "prefix": int(args.prefix),
            "eval_start": int(args.eval_start),
            "fractions": [float(value) for value in args.fractions],
            "random_repeats": int(args.random_repeats),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "summary": summary,
        "collapsed_summary": collapsed,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_records": len(records),
                "n_failures": len(failures),
                "summary_json": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
