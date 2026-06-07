"""Evaluate source / EA / shrink-EA portfolios when subjects have variable trials.

BCIC-IV 2b has a different number of evaluation trials for S2, so the BCIC2a
fixed-array portfolio scripts are intentionally not reused here.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np


EPS = 1e-12


STATIC_WEIGHTS = {
    "source": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    "full_ea": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    "shrink_0.1": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    "source_full_mean": np.asarray([0.5, 0.5, 0.0], dtype=np.float64),
    "uniform": np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64),
    "static_304030": np.asarray([0.30, 0.40, 0.30], dtype=np.float64),
    "static_454510": np.asarray([0.45, 0.45, 0.10], dtype=np.float64),
    "safe_652510": np.asarray([0.65, 0.25, 0.10], dtype=np.float64),
    "accuracy_482923": np.asarray([0.48, 0.29, 0.23], dtype=np.float64),
}


def subjects_from_arg(arg: str) -> list[int]:
    if arg == "all":
        return list(range(1, 10))
    return [int(x) for x in arg.split(",") if x.strip()]


def latest_match(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), EPS, None)


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


def evaluate_weights(subjects: list[dict], weights: np.ndarray, indices: np.ndarray) -> dict:
    accs = np.asarray(
        [subject_acc(subjects[i]["probs"], subjects[i]["labels"], weights) for i in indices],
        dtype=np.float64,
    )
    source = np.asarray([subjects[i]["acc_source"] for i in indices], dtype=np.float64)
    full = np.asarray([subjects[i]["acc_full_ea"] for i in indices], dtype=np.float64)
    return {
        "mean_acc": float(accs.mean()),
        "std_acc": float(accs.std()),
        "worst_acc": float(accs.min()),
        "best_acc": float(accs.max()),
        "harmed_vs_source": int((accs < source - 1e-9).sum()),
        "harmed_vs_full": int((accs < full - 1e-9).sum()),
        "mean_delta_vs_source": float((accs - source).mean()),
        "mean_delta_vs_full": float((accs - full).mean()),
        "accs": accs,
    }


def tie_break(metrics: dict, weights: np.ndarray, objective: str) -> tuple:
    if objective == "mean":
        primary = metrics["mean_acc"]
    elif objective == "worst":
        primary = metrics["worst_acc"]
    elif objective == "mean_minus_harm":
        primary = metrics["mean_acc"] - metrics["harmed_vs_source"]
    else:
        raise ValueError(f"Unknown objective: {objective}")
    uniform = np.ones_like(weights) / len(weights)
    return (
        primary,
        metrics["worst_acc"],
        -metrics["harmed_vs_source"],
        -metrics["harmed_vs_full"],
        -float(np.linalg.norm(weights - uniform)),
    )


def load_subjects(args: argparse.Namespace) -> list[dict]:
    root = Path(args.results_root)
    rows = []
    for sid in subjects_from_arg(args.subjects):
        label_path = latest_match(root, args.labels_pattern.format(sid=sid))
        source_path = latest_match(root, args.source_logits_pattern.format(sid=sid))
        full_path = latest_match(root, args.full_ea_logits_pattern.format(sid=sid))
        shrink_path = latest_match(root, args.shrink01_logits_pattern.format(sid=sid))
        missing = [
            name
            for name, path in (
                ("labels", label_path),
                ("source", source_path),
                ("full_ea", full_path),
                ("shrink_0.1", shrink_path),
            )
            if path is None or not path.exists()
        ]
        if missing:
            if args.allow_missing:
                print(f"S{sid}: skip missing {missing}")
                continue
            raise FileNotFoundError(f"S{sid}: missing {missing}")

        labels = np.load(label_path)["labels"].astype(np.int64)
        logits = [np.load(source_path), np.load(full_path), np.load(shrink_path)]
        shapes = [z.shape for z in logits]
        if len(set(shapes)) != 1:
            raise ValueError(f"S{sid}: logits shape mismatch {shapes}")
        if logits[0].shape[0] != labels.shape[0]:
            raise ValueError(f"S{sid}: labels {labels.shape}, logits {logits[0].shape}")

        probs = np.stack([softmax(z) for z in logits], axis=0)
        row = {
            "subject": sid,
            "n_trials": int(labels.shape[0]),
            "n_classes": int(logits[0].shape[1]),
            "labels": labels,
            "probs": probs,
            "paths": {
                "labels": str(label_path),
                "source": str(source_path),
                "full_ea": str(full_path),
                "shrink_0.1": str(shrink_path),
            },
        }
        row["acc_source"] = subject_acc(probs, labels, STATIC_WEIGHTS["source"])
        row["acc_full_ea"] = subject_acc(probs, labels, STATIC_WEIGHTS["full_ea"])
        row["acc_shrink_0.1"] = subject_acc(probs, labels, STATIC_WEIGHTS["shrink_0.1"])
        rows.append(row)
    if not rows:
        raise ValueError("No subjects loaded.")
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
    parser.add_argument("--results_root", default="intentflow/offline/results")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subjects", default="all")
    parser.add_argument(
        "--source_logits_pattern",
        required=True,
        help="Glob pattern relative to results_root, supports {sid}",
    )
    parser.add_argument("--full_ea_logits_pattern", required=True)
    parser.add_argument("--shrink01_logits_pattern", required=True)
    parser.add_argument(
        "--labels_pattern",
        required=True,
        help="Glob pattern to features npz with labels, supports {sid}",
    )
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--objective", choices=["mean", "worst", "mean_minus_harm"], default="mean_minus_harm")
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    subjects = load_subjects(args)
    indices = np.arange(len(subjects), dtype=int)
    grid = simplex_grid(3, args.step)

    best_global = None
    best_score = None
    grid_rows = []
    for weights in grid:
        metrics = evaluate_weights(subjects, weights, indices)
        score = tie_break(metrics, weights, args.objective)
        row = {
            "w_source": float(weights[0]),
            "w_full_ea": float(weights[1]),
            "w_shrink_0.1": float(weights[2]),
            **{k: v for k, v in metrics.items() if k != "accs"},
        }
        grid_rows.append(row)
        if best_score is None or score > best_score:
            best_score = score
            best_global = (weights.copy(), metrics)

    subject_rows = []
    for i, item in enumerate(subjects):
        row = {
            "subject": item["subject"],
            "n_trials": item["n_trials"],
            "acc_source": item["acc_source"],
            "acc_full_ea": item["acc_full_ea"],
            "acc_shrink_0.1": item["acc_shrink_0.1"],
        }
        candidate_accs = {}
        for name, weights in STATIC_WEIGHTS.items():
            acc = subject_acc(item["probs"], item["labels"], weights)
            row[f"acc_{name}"] = acc
            candidate_accs[name] = acc
        row["acc_global_grid"] = subject_acc(item["probs"], item["labels"], best_global[0])
        candidate_accs["global_grid"] = row["acc_global_grid"]
        oracle_name, oracle_acc = max(candidate_accs.items(), key=lambda kv: kv[1])
        row["oracle_candidate"] = oracle_name
        row["oracle_acc"] = oracle_acc
        row["oracle_gap_vs_source"] = oracle_acc - item["acc_source"]
        subject_rows.append(row)

    loso_rows = []
    for heldout in indices:
        train_indices = indices[indices != heldout]
        best = None
        score = None
        for weights in grid:
            metrics = evaluate_weights(subjects, weights, train_indices)
            this_score = tie_break(metrics, weights, args.objective)
            if score is None or this_score > score:
                score = this_score
                best = (weights.copy(), metrics)
        weights, train_metrics = best
        item = subjects[heldout]
        heldout_acc = subject_acc(item["probs"], item["labels"], weights)
        loso_rows.append(
            {
                "subject": item["subject"],
                "heldout_acc": heldout_acc,
                "source_acc": item["acc_source"],
                "full_ea_acc": item["acc_full_ea"],
                "shrink_0.1_acc": item["acc_shrink_0.1"],
                "train_mean_acc": train_metrics["mean_acc"],
                "train_harmed_vs_source": train_metrics["harmed_vs_source"],
                "w_source": float(weights[0]),
                "w_full_ea": float(weights[1]),
                "w_shrink_0.1": float(weights[2]),
            }
        )
        print(
            f"S{item['subject']}: loso={heldout_acc:.2f}, "
            f"source={item['acc_source']:.2f}, full={item['acc_full_ea']:.2f}, "
            f"w={{source:{weights[0]:.2f}, full:{weights[1]:.2f}, shrink:{weights[2]:.2f}}}"
        )

    static_summary = {}
    for name, weights in STATIC_WEIGHTS.items():
        metrics = evaluate_weights(subjects, weights, indices)
        static_summary[name] = {k: v for k, v in metrics.items() if k != "accs"}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "subject_summary.csv", subject_rows)
    write_csv(out_dir / "loso_summary.csv", loso_rows)
    write_csv(out_dir / "grid_summary.csv", grid_rows)

    summary = {
        "subjects": [int(row["subject"]) for row in subjects],
        "n_trials": {str(row["subject"]): int(row["n_trials"]) for row in subjects},
        "objective": args.objective,
        "step": args.step,
        "static": static_summary,
        "global_best": {
            "weights": {
                "source": float(best_global[0][0]),
                "full_ea": float(best_global[0][1]),
                "shrink_0.1": float(best_global[0][2]),
            },
            "metrics": {k: v for k, v in best_global[1].items() if k != "accs"},
        },
        "loso": {
            "mean_acc": float(np.mean([row["heldout_acc"] for row in loso_rows])),
            "harmed_vs_source": int(
                sum(row["heldout_acc"] < row["source_acc"] - 1e-9 for row in loso_rows)
            ),
            "mean_delta_vs_source": float(
                np.mean([row["heldout_acc"] - row["source_acc"] for row in loso_rows])
            ),
        },
        "paths": {str(row["subject"]): row["paths"] for row in subjects},
    }
    with open(out_dir / "portfolio_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 80)
    print(json.dumps({
        "source": static_summary["source"]["mean_acc"],
        "full_ea": static_summary["full_ea"]["mean_acc"],
        "shrink_0.1": static_summary["shrink_0.1"]["mean_acc"],
        "static_454510": static_summary["static_454510"]["mean_acc"],
        "safe_652510": static_summary["safe_652510"]["mean_acc"],
        "global_best": summary["global_best"],
        "loso": summary["loso"],
    }, indent=2))


if __name__ == "__main__":
    main()
