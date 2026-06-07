"""Evaluate convex expert portfolio weights with global and LOSO selection.

This script asks whether a static source/full-EA/shrink-EA mixture found on
BCIC2a is a robust portfolio principle or just a dataset-specific fit.

Labels are used only for offline analysis:

1. global grid search over simplex weights
2. leave-one-subject-out weight selection on the other subjects
3. held-out evaluation for each subject
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np


EPS = 1e-12


def parse_experts(arg: str, all_experts: list[str]) -> list[int]:
    wanted = [x.strip() for x in arg.split(",") if x.strip()]
    missing = [name for name in wanted if name not in all_experts]
    if missing:
        raise ValueError(f"Unknown experts: {missing}; available={all_experts}")
    return [all_experts.index(name) for name in wanted]


def simplex_grid(n: int, step: float) -> np.ndarray:
    units = int(round(1.0 / step))
    if abs(units * step - 1.0) > 1e-8:
        raise ValueError(f"--step must divide 1.0, got {step}")
    rows = []
    for vals in product(range(units + 1), repeat=n - 1):
        used = sum(vals)
        if used <= units:
            rows.append(list(vals) + [units - used])
    return np.asarray(rows, dtype=np.float64) / float(units)


def combine_probs(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, probs, axes=(0, 0))


def subject_acc(probs: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    pred = combine_probs(probs, weights).argmax(axis=-1)
    return float((pred == labels).mean() * 100.0)


def evaluate_weights(
    probs: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    subject_indices: np.ndarray,
    source_acc: np.ndarray,
    full_acc: np.ndarray,
) -> dict:
    accs = np.asarray(
        [subject_acc(probs[sidx], labels[sidx], weights) for sidx in subject_indices],
        dtype=np.float64,
    )
    src = source_acc[subject_indices]
    full = full_acc[subject_indices]
    return {
        "mean_acc": float(accs.mean()),
        "std_acc": float(accs.std()),
        "worst_acc": float(accs.min()),
        "best_acc": float(accs.max()),
        "harmed_vs_source": int((accs < src - 1e-9).sum()),
        "harmed_vs_full": int((accs < full - 1e-9).sum()),
        "mean_delta_vs_source": float((accs - src).mean()),
        "mean_delta_vs_full": float((accs - full).mean()),
        "accs": accs,
    }


def tie_break_score(metrics: dict, weights: np.ndarray, uniform: np.ndarray, objective: str) -> tuple:
    if objective == "mean":
        primary = metrics["mean_acc"]
    elif objective == "worst":
        primary = metrics["worst_acc"]
    elif objective == "mean_minus_harm":
        primary = metrics["mean_acc"] - metrics["harmed_vs_source"]
    else:
        raise ValueError(f"Unknown objective: {objective}")
    distance_to_uniform = float(np.linalg.norm(weights - uniform))
    # Maximize primary, then worst, minimize harm, then prefer smoother weights.
    return (
        primary,
        metrics["worst_acc"],
        -metrics["harmed_vs_source"],
        -metrics["harmed_vs_full"],
        -distance_to_uniform,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arrays",
        default="intentflow/offline/results/research_outputs/260602_expert_portfolio_table/expert_portfolio_arrays.npz",
    )
    parser.add_argument(
        "--output_dir",
        default="intentflow/offline/results/research_outputs/260602_portfolio_weight_grid",
    )
    parser.add_argument("--experts", default="source,full_ea,shrink_0.1")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--objective",
        choices=["mean", "worst", "mean_minus_harm"],
        default="mean",
    )
    args = parser.parse_args()

    data = np.load(args.arrays, allow_pickle=True)
    subjects = data["subjects"].astype(int)
    all_experts = [str(x) for x in data["experts"].tolist()]
    expert_indices = parse_experts(args.experts, all_experts)
    expert_names = [all_experts[idx] for idx in expert_indices]
    probs_all = data["probs"][:, expert_indices]
    labels = data["labels"]
    available = data["available"][:, expert_indices].astype(bool)

    valid_subject_mask = available.all(axis=1)
    valid_subject_indices = np.where(valid_subject_mask)[0]
    if valid_subject_indices.size < 2:
        raise ValueError("Need at least two subjects with all selected experts available.")

    source_idx = expert_names.index("source") if "source" in expert_names else None
    full_idx = expert_names.index("full_ea") if "full_ea" in expert_names else None
    if source_idx is None or full_idx is None:
        raise ValueError("This analysis currently expects source and full_ea experts.")

    source_acc = np.asarray(
        [
            subject_acc(probs_all[sidx], labels[sidx], np.eye(len(expert_names))[source_idx])
            for sidx in range(len(subjects))
        ]
    )
    full_acc = np.asarray(
        [
            subject_acc(probs_all[sidx], labels[sidx], np.eye(len(expert_names))[full_idx])
            for sidx in range(len(subjects))
        ]
    )

    weights_grid = simplex_grid(len(expert_names), args.step)
    uniform = np.ones(len(expert_names), dtype=np.float64) / len(expert_names)
    static_304030 = np.asarray([0.3, 0.4, 0.3], dtype=np.float64)
    static_454510 = np.asarray([0.45, 0.45, 0.10], dtype=np.float64)
    if len(expert_names) != 3:
        static_304030 = uniform
        static_454510 = uniform

    grid_rows = []
    best_global = None
    best_global_score = None
    for weights in weights_grid:
        metrics = evaluate_weights(
            probs_all, labels, weights, valid_subject_indices, source_acc, full_acc
        )
        row = {
            "weights": json.dumps(dict(zip(expert_names, weights.tolist()))),
            "mean_acc": metrics["mean_acc"],
            "std_acc": metrics["std_acc"],
            "worst_acc": metrics["worst_acc"],
            "harmed_vs_source": metrics["harmed_vs_source"],
            "harmed_vs_full": metrics["harmed_vs_full"],
            "mean_delta_vs_source": metrics["mean_delta_vs_source"],
            "mean_delta_vs_full": metrics["mean_delta_vs_full"],
        }
        for name, weight in zip(expert_names, weights):
            row[f"w_{name}"] = float(weight)
        grid_rows.append(row)
        score = tie_break_score(metrics, weights, uniform, args.objective)
        if best_global_score is None or score > best_global_score:
            best_global_score = score
            best_global = (weights.copy(), metrics)

    loso_rows = []
    heldout_accs = []
    static_accs = []
    static_454510_accs = []
    uniform_accs = []
    global_accs = []
    for heldout in valid_subject_indices:
        train_indices = valid_subject_indices[valid_subject_indices != heldout]
        best = None
        best_score = None
        for weights in weights_grid:
            metrics = evaluate_weights(
                probs_all, labels, weights, train_indices, source_acc, full_acc
            )
            score = tie_break_score(metrics, weights, uniform, args.objective)
            if best_score is None or score > best_score:
                best_score = score
                best = (weights.copy(), metrics)

        weights, train_metrics = best
        heldout_subject = int(subjects[heldout])
        heldout_acc = subject_acc(probs_all[heldout], labels[heldout], weights)
        static_acc = subject_acc(probs_all[heldout], labels[heldout], static_304030)
        static_454510_acc = subject_acc(probs_all[heldout], labels[heldout], static_454510)
        uniform_acc = subject_acc(probs_all[heldout], labels[heldout], uniform)
        global_acc = subject_acc(probs_all[heldout], labels[heldout], best_global[0])
        heldout_accs.append(heldout_acc)
        static_accs.append(static_acc)
        static_454510_accs.append(static_454510_acc)
        uniform_accs.append(uniform_acc)
        global_accs.append(global_acc)

        row = {
            "subject": heldout_subject,
            "heldout_acc": heldout_acc,
            "static_304030_acc": static_acc,
            "static_454510_acc": static_454510_acc,
            "uniform_acc": uniform_acc,
            "global_grid_acc": global_acc,
            "source_acc": source_acc[heldout],
            "full_ea_acc": full_acc[heldout],
            "train_mean_acc": train_metrics["mean_acc"],
            "train_worst_acc": train_metrics["worst_acc"],
            "train_harmed_vs_source": train_metrics["harmed_vs_source"],
            "selected_weights": json.dumps(dict(zip(expert_names, weights.tolist()))),
        }
        for name, weight in zip(expert_names, weights):
            row[f"w_{name}"] = float(weight)
        loso_rows.append(row)
        print(
            f"S{heldout_subject}: loso={heldout_acc:.2f}, "
            f"static304030={static_acc:.2f}, static454510={static_454510_acc:.2f}, "
            f"uniform={uniform_acc:.2f}, "
            f"w={dict(zip(expert_names, weights.round(2).tolist()))}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "grid_summary.csv", grid_rows)
    write_csv(out_dir / "loso_summary.csv", loso_rows)

    summary = {
        "arrays": args.arrays,
        "experts": expert_names,
        "step": args.step,
        "objective": args.objective,
        "n_weight_vectors": int(len(weights_grid)),
        "subjects": subjects[valid_subject_indices].astype(int).tolist(),
        "global_best": {
            "weights": dict(zip(expert_names, best_global[0].tolist())),
            "metrics": {
                key: value
                for key, value in best_global[1].items()
                if key != "accs"
            },
        },
        "mean": {
            "loso_acc": float(np.mean(heldout_accs)),
            "static_304030_acc": float(np.mean(static_accs)),
            "static_454510_acc": float(np.mean(static_454510_accs)),
            "uniform_acc": float(np.mean(uniform_accs)),
            "global_grid_acc": float(np.mean(global_accs)),
            "source_acc": float(np.mean(source_acc[valid_subject_indices])),
            "full_ea_acc": float(np.mean(full_acc[valid_subject_indices])),
        },
        "rows": loso_rows,
    }
    with open(out_dir / "weight_grid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 80)
    print(json.dumps(summary["mean"], indent=2))
    print("global_best:", json.dumps(summary["global_best"], indent=2))


if __name__ == "__main__":
    main()
