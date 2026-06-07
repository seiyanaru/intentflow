"""Evaluate a label-free online adaptation portfolio over saved expert logits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EPS = 1e-12


def entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, EPS, 1.0))).sum(axis=-1)


def kl_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p * (np.log(np.clip(p, EPS, 1.0)) - np.log(np.clip(q, EPS, 1.0)))).sum(axis=-1)


def prior_kl(counts: np.ndarray) -> np.ndarray:
    prior = counts / np.clip(counts.sum(axis=-1, keepdims=True), 1, None)
    n_classes = counts.shape[-1]
    return (prior * np.log(np.clip(prior, EPS, 1.0) / (1.0 / n_classes))).sum(axis=-1)


def parse_experts(arg: str, all_experts: list[str]) -> list[int]:
    if arg == "all":
        return list(range(len(all_experts)))
    wanted = [x.strip() for x in arg.split(",") if x.strip()]
    missing = [name for name in wanted if name not in all_experts]
    if missing:
        raise ValueError(f"Unknown experts: {missing}; available={all_experts}")
    return [all_experts.index(name) for name in wanted]


def normalize_scores(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = scores.copy()
    valid = out[mask]
    if valid.size <= 1:
        return out
    std = valid.std()
    if std > 1e-8:
        out[mask] = (valid - valid.mean()) / std
    else:
        out[mask] = 0.0
    return out


def run_subject(
    probs: np.ndarray,
    labels: np.ndarray,
    expert_names: list[str],
    expert_indices: list[int],
    available: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    n_experts_total, n_trials, n_classes = probs.shape
    use = np.zeros(n_experts_total, dtype=bool)
    use[expert_indices] = True
    use &= available
    if not use.any():
        raise ValueError("No available experts for subject")

    alpha = np.zeros(n_experts_total, dtype=np.float64)
    alpha[use] = 1.0 / use.sum()
    counts = np.zeros((n_experts_total, n_classes), dtype=np.float64)

    portfolio_preds = []
    equal_preds = []
    alpha_history = []
    score_history = []
    for trial in range(n_trials):
        p = probs[:, trial, :]
        mask = use & np.isfinite(p).all(axis=-1)
        if not mask.any():
            raise ValueError(f"No valid experts at trial {trial}")

        consensus = p[mask].mean(axis=0)
        pred_k = p.argmax(axis=-1)
        counts[mask, pred_k[mask]] += 1.0

        ent_score = entropy(p)
        disagree_score = kl_div(p, consensus[None, :])
        collapse_score = prior_kl(counts)
        score = (
            args.lambda_entropy * ent_score
            + args.lambda_disagreement * disagree_score
            + args.lambda_collapse * collapse_score
        )
        if args.normalize_scores:
            score = normalize_scores(score, mask)

        if args.update_before_predict:
            alpha[mask] *= np.exp(-args.eta * score[mask])
            alpha[~mask] = 0.0
            alpha = alpha / np.clip(alpha.sum(), EPS, None)

        blended = (alpha[:, None] * np.nan_to_num(p, nan=0.0)).sum(axis=0)
        equal = p[mask].mean(axis=0)
        portfolio_preds.append(int(blended.argmax()))
        equal_preds.append(int(equal.argmax()))
        alpha_history.append(alpha.copy())
        score_history.append(np.where(mask, score, np.nan))

        if not args.update_before_predict:
            alpha[mask] *= np.exp(-args.eta * score[mask])
            alpha[~mask] = 0.0
            alpha = alpha / np.clip(alpha.sum(), EPS, None)

    portfolio_preds = np.asarray(portfolio_preds, dtype=np.int16)
    equal_preds = np.asarray(equal_preds, dtype=np.int16)
    alpha_history = np.stack(alpha_history)
    score_history = np.stack(score_history)

    expert_acc = {}
    for idx in np.where(use)[0]:
        pred = probs[idx].argmax(axis=-1)
        expert_acc[expert_names[idx]] = float((pred == labels).mean() * 100.0)
    oracle_name, oracle_acc = max(expert_acc.items(), key=lambda item: item[1])

    return {
        "portfolio_acc": float((portfolio_preds == labels).mean() * 100.0),
        "equal_acc": float((equal_preds == labels).mean() * 100.0),
        "oracle_expert": oracle_name,
        "oracle_acc": oracle_acc,
        "expert_acc": expert_acc,
        "portfolio_preds": portfolio_preds,
        "equal_preds": equal_preds,
        "alpha_history": alpha_history,
        "score_history": score_history,
        "final_alpha": alpha,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arrays",
        default="intentflow/offline/results/research_outputs/260602_expert_portfolio_table/expert_portfolio_arrays.npz",
    )
    parser.add_argument(
        "--output_dir",
        default="intentflow/offline/results/research_outputs/260602_online_portfolio_eval",
    )
    parser.add_argument("--experts", default="source,full_ea,shrink_0.1,full_shrink_mean,source_full_mean")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--lambda_entropy", type=float, default=1.0)
    parser.add_argument("--lambda_disagreement", type=float, default=0.5)
    parser.add_argument("--lambda_collapse", type=float, default=0.1)
    parser.add_argument("--normalize_scores", action="store_true")
    parser.add_argument("--update_before_predict", action="store_true")
    args = parser.parse_args()

    data = np.load(args.arrays, allow_pickle=True)
    subjects = data["subjects"].astype(int).tolist()
    expert_names = [str(x) for x in data["experts"].tolist()]
    expert_indices = parse_experts(args.experts, expert_names)
    probs = data["probs"]
    labels = data["labels"]
    available = data["available"].astype(bool)

    rows = []
    alpha_dump = {}
    pred_dump = {}
    for sidx, sid in enumerate(subjects):
        result = run_subject(
            probs[sidx],
            labels[sidx],
            expert_names,
            expert_indices,
            available[sidx],
            args,
        )
        row = {
            "subject": sid,
            "portfolio_acc": result["portfolio_acc"],
            "equal_acc": result["equal_acc"],
            "oracle_expert": result["oracle_expert"],
            "oracle_acc": result["oracle_acc"],
            "oracle_gap_portfolio": result["oracle_acc"] - result["portfolio_acc"],
            "oracle_gap_equal": result["oracle_acc"] - result["equal_acc"],
        }
        for name, acc in result["expert_acc"].items():
            row[f"acc_{name}"] = acc
        for idx, value in enumerate(result["final_alpha"]):
            if idx in expert_indices:
                row[f"final_alpha_{expert_names[idx]}"] = float(value)
        rows.append(row)
        alpha_dump[f"s{sid}"] = result["alpha_history"].astype(np.float32)
        pred_dump[f"s{sid}_portfolio"] = result["portfolio_preds"]
        pred_dump[f"s{sid}_equal"] = result["equal_preds"]
        print(
            f"S{sid}: portfolio={row['portfolio_acc']:.2f}, equal={row['equal_acc']:.2f}, "
            f"oracle={row['oracle_expert']} {row['oracle_acc']:.2f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "online_portfolio_summary.csv", rows)
    np.savez_compressed(output_dir / "online_portfolio_traces.npz", **alpha_dump, **pred_dump)

    summary = {
        "arrays": args.arrays,
        "experts": [expert_names[idx] for idx in expert_indices],
        "params": {
            "eta": args.eta,
            "lambda_entropy": args.lambda_entropy,
            "lambda_disagreement": args.lambda_disagreement,
            "lambda_collapse": args.lambda_collapse,
            "normalize_scores": args.normalize_scores,
            "update_before_predict": args.update_before_predict,
        },
        "mean": {
            "portfolio_acc": float(np.mean([row["portfolio_acc"] for row in rows])),
            "equal_acc": float(np.mean([row["equal_acc"] for row in rows])),
            "oracle_acc": float(np.mean([row["oracle_acc"] for row in rows])),
            "oracle_gap_portfolio": float(np.mean([row["oracle_gap_portfolio"] for row in rows])),
            "oracle_gap_equal": float(np.mean([row["oracle_gap_equal"] for row in rows])),
        },
        "rows": rows,
    }
    with open(output_dir / "online_portfolio_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("-" * 80)
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
