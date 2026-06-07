"""Build subject-level label-free features for adaptation portfolio selection.

The table produced here is for designing a selector that chooses among a small
set of source / EA / shrink-EA portfolio weights. Labels are included only as
evaluation targets; all feature columns are computable at test time without
ground-truth labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EPS = 1e-12


CANDIDATES = {
    "safe_652510": np.asarray([0.65, 0.25, 0.10], dtype=np.float64),
    "balanced_454510": np.asarray([0.45, 0.45, 0.10], dtype=np.float64),
    "accuracy_482923": np.asarray([0.48, 0.29, 0.23], dtype=np.float64),
    "ea_heavy_304030": np.asarray([0.30, 0.40, 0.30], dtype=np.float64),
    "uniform_333333": np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
}


def parse_array_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Array spec must be name:path, got {spec}")
        name, path = spec.split(":", 1)
        out.append((name, Path(path)))
    return out


def soft_entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, EPS, 1.0))).sum(axis=-1)


def margin(probs: np.ndarray) -> np.ndarray:
    top2 = np.sort(probs, axis=-1)[..., -2:]
    return top2[..., 1] - top2[..., 0]


def kl_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p * (np.log(np.clip(p, EPS, 1.0)) - np.log(np.clip(q, EPS, 1.0)))).sum(axis=-1)


def prior_kl(preds: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(preds, minlength=n_classes).astype(np.float64)
    prior = counts / np.clip(counts.sum(), 1.0, None)
    return float((prior * np.log(np.clip(prior, EPS, 1.0) / (1.0 / n_classes))).sum())


def load_rows_by_subject(path: str | None) -> dict[int, dict]:
    if not path:
        return {}
    obj = json.load(open(path))
    return {int(row["subject"]): row for row in obj.get("rows", [])}


def add_cov_features(row: dict, cov: dict | None) -> None:
    if not cov:
        return
    row["cov_distance"] = cov.get("cov_distance")
    for domain in ("train", "test"):
        values = cov.get(domain, {})
        for key in ("condition", "eig_entropy", "eig_min", "eig_max", "diag_cv", "diag_ratio"):
            if key in values:
                row[f"cov_{domain}_{key}"] = values[key]


def summarize_vector(row: dict, prefix: str, values: list[float] | np.ndarray | None) -> None:
    if values is None:
        return
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        row[f"{prefix}_mean"] = 0.0
        row[f"{prefix}_max"] = 0.0
        row[f"{prefix}_nonzero"] = 0
        return
    row[f"{prefix}_mean"] = float(arr.mean())
    row[f"{prefix}_max"] = float(arr.max())
    row[f"{prefix}_std"] = float(arr.std())
    row[f"{prefix}_nonzero"] = int((arr > 1e-8).sum())


def add_reliability_features(row: dict, rel: dict | None) -> None:
    if not rel:
        return
    for mode, values in rel.get("weight_summary", {}).items():
        for key in ("min", "mean", "max"):
            row[f"rel_{mode}_{key}"] = values.get(key)
        row[f"rel_{mode}_low_count"] = len(values.get("low_channels", []))
    metrics = rel.get("metrics", {})
    for key in (
        "relative_cov_leverage",
        "badness_full",
        "badness_var",
        "badness_artifact",
        "cov_leverage",
        "line_noise_z",
        "corr_abnormal_z",
    ):
        summarize_vector(row, f"rel_{key}", metrics.get(key))


def combine_probs(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, probs, axes=(0, 0))


def add_prediction_features(
    row: dict,
    probs: np.ndarray,
    expert_names: list[str],
    prefix: str,
    n_classes: int,
) -> None:
    preds = probs.argmax(axis=-1)
    consensus = probs.mean(axis=0)
    per_expert_kl = kl_div(probs, consensus[None, :, :])
    row[f"{prefix}_consensus_entropy"] = float(soft_entropy(consensus).mean())
    row[f"{prefix}_expert_disagreement_kl"] = float(per_expert_kl.mean())
    row[f"{prefix}_any_pred_disagreement"] = float((preds != preds[0:1]).any(axis=0).mean())
    for i in range(len(expert_names)):
        for j in range(i + 1, len(expert_names)):
            row[f"{prefix}_disagree_{expert_names[i]}_{expert_names[j]}"] = float(
                (preds[i] != preds[j]).mean()
            )
    for idx, name in enumerate(expert_names):
        p = probs[idx]
        row[f"{prefix}_{name}_entropy"] = float(soft_entropy(p).mean())
        row[f"{prefix}_{name}_confidence"] = float(p.max(axis=-1).mean())
        row[f"{prefix}_{name}_margin"] = float(margin(p).mean())
        row[f"{prefix}_{name}_prior_kl"] = prior_kl(preds[idx], n_classes)


def subject_accuracy(probs: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    return float((combine_probs(probs, weights).argmax(axis=-1) == labels).mean() * 100.0)


def build_rows(args: argparse.Namespace) -> list[dict]:
    cov_by_subject = load_rows_by_subject(args.cov_diagnostics)
    rel_by_subject = load_rows_by_subject(args.reliability_diagnostics)
    rows: list[dict] = []

    for seed_name, array_path in parse_array_specs(args.array):
        data = np.load(array_path, allow_pickle=True)
        all_experts = [str(x) for x in data["experts"].tolist()]
        wanted = ["source", "full_ea", "shrink_0.1"]
        expert_indices = [all_experts.index(name) for name in wanted]
        probs_all = data["probs"][:, expert_indices]
        labels_all = data["labels"]
        subjects = data["subjects"].astype(int)

        for sidx, sid in enumerate(subjects):
            probs = probs_all[sidx]
            labels = labels_all[sidx]
            row: dict = {"seed": seed_name, "subject": int(sid)}
            add_cov_features(row, cov_by_subject.get(int(sid)))
            add_reliability_features(row, rel_by_subject.get(int(sid)))
            add_prediction_features(row, probs[:, : args.first_n], wanted, f"first{args.first_n}", args.n_classes)
            add_prediction_features(row, probs, wanted, "all", args.n_classes)

            row["acc_source"] = subject_accuracy(probs, labels, np.asarray([1.0, 0.0, 0.0]))
            row["acc_full_ea"] = subject_accuracy(probs, labels, np.asarray([0.0, 1.0, 0.0]))
            row["acc_shrink_0.1"] = subject_accuracy(probs, labels, np.asarray([0.0, 0.0, 1.0]))
            for name, weights in CANDIDATES.items():
                row[f"acc_{name}"] = subject_accuracy(probs, labels, weights)
            candidate_accs = {name: row[f"acc_{name}"] for name in CANDIDATES}
            best_name, best_acc = max(candidate_accs.items(), key=lambda item: item[1])
            row["target_best_candidate"] = best_name
            row["target_best_candidate_acc"] = best_acc
            row["target_best_candidate_delta_source"] = best_acc - row["acc_source"]
            row["delta_full_vs_source"] = row["acc_full_ea"] - row["acc_source"]
            row["delta_shrink_vs_source"] = row["acc_shrink_0.1"] - row["acc_source"]
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--array",
        action="append",
        default=[
            "seed0:intentflow/offline/results/research_outputs/260602_expert_portfolio_table/expert_portfolio_arrays.npz",
            "seed1:intentflow/offline/results/research_outputs/260602_expert_portfolio_table_seed1/expert_portfolio_arrays.npz",
        ],
        help="Repeatable name:path to expert_portfolio_arrays.npz",
    )
    parser.add_argument(
        "--output_csv",
        default="intentflow/offline/results/research_outputs/260602_selector_feature_table_seed0_seed1.csv",
    )
    parser.add_argument("--cov_diagnostics", default="intentflow/offline/results/research_outputs/260601_ea_cov_diagnostics.json")
    parser.add_argument(
        "--reliability_diagnostics",
        default="intentflow/offline/results/research_outputs/260602_channel_reliability_all9_relative.json",
    )
    parser.add_argument("--first_n", type=int, default=32)
    parser.add_argument("--n_classes", type=int, default=4)
    args = parser.parse_args()

    rows = build_rows(args)
    write_csv(Path(args.output_csv), rows)
    counts = {}
    for row in rows:
        counts[row["target_best_candidate"]] = counts.get(row["target_best_candidate"], 0) + 1
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
