"""Population-pretrained EEGNet source repair pilot for Stieger.

For each subject fold, the model is pretrained using session 1 from the other
pilot subjects, then fine-tuned using only the held-out subject's session 1.
No target-session sample is used for training or model selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
OFFLINE_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(OFFLINE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from cross_adapter_core import (
    apply_reference,
    clone_model,
    ea_reference,
    predict_labels,
    seed_everything,
    state_dict_cpu,
    state_digest,
)
from stieger_cross_adapter_pilot import build_eegnet, parse_subjects


DEFAULT_CACHE = Path("/home/islabshi/workspace-local2/mne_data/stieger_epochs_cache")
DEFAULT_OUTPUT = (
    OFFLINE_DIR
    / "results"
    / "research_outputs"
    / "260622_population_source_pilot"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-16")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backbone",
        choices=("braindecode_eegnet", "local_eegnet"),
        default="braindecode_eegnet",
    )
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def subject_folds(subjects: list[int], n_folds: int) -> list[list[int]]:
    if n_folds < 2 or n_folds > len(subjects):
        raise ValueError("n_folds must be between 2 and number of subjects")
    return [chunk.tolist() for chunk in np.array_split(np.asarray(subjects), n_folds)]


def load_source_session(
    subject: int,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = np.load(cache_dir / f"S{subject}_epochs.npz")
    x = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.int64)
    sessions = cache["sess"].astype(np.int64)
    source_session = int(np.min(sessions))
    source_mask = sessions == source_session
    reference = ea_reference(x[source_mask])
    source_x = apply_reference(reference, x[source_mask])
    return source_x, y[source_mask], x, y, sessions


def optimize_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    warmup_epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    seed_everything(seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    rng = np.random.default_rng(seed)
    start_time = time.time()
    last_loss = float("nan")
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            multiplier = (epoch + 1) / max(1, warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            multiplier = 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * multiplier
        permutation = rng.permutation(len(y))
        model.train()
        losses: list[float] = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = torch.from_numpy(x[indices]).to(device)
            batch_y = torch.from_numpy(y[indices]).long().to(device)
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses))
    model.eval()
    accuracy = float(
        np.mean(predict_labels(model, x, device, batch_size) == y) * 100
    )
    return {
        "loss": last_loss,
        "accuracy": accuracy,
        "seconds": time.time() - start_time,
    }


def evaluate_target_source(
    model: nn.Module,
    subject: int,
    cache_dir: Path,
    prefix: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, list[dict]]:
    source_x, source_y, x, y, sessions = load_source_session(subject, cache_dir)
    del source_x, source_y
    source_mask = sessions == int(np.min(sessions))
    reference = ea_reference(x[source_mask])
    rows: list[dict] = []
    for session in sorted(np.unique(sessions).tolist())[1:]:
        mask = sessions == session
        if int(np.sum(mask)) <= prefix:
            continue
        session_x = apply_reference(reference, x[mask])[prefix:]
        session_y = y[mask][prefix:]
        predictions = predict_labels(model, session_x, device, batch_size)
        rows.append(
            {
                "subject": subject,
                "session": int(session),
                "source_acc": float(np.mean(predictions == session_y) * 100),
                "n_eval": int(len(session_y)),
            }
        )
    return float(np.mean([row["source_acc"] for row in rows])), rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    subjects = parse_subjects(args.subjects)
    folds = subject_folds(subjects, args.n_folds)
    if not 0 <= args.fold_index < len(folds):
        raise ValueError(f"fold-index must be in [0, {len(folds) - 1}]")
    held_out = folds[args.fold_index]
    training = [subject for subject in subjects if subject not in held_out]
    fold_dir = args.output_dir / f"fold_{args.fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    subject_dir = fold_dir / "subjects"
    checkpoint_dir = fold_dir / "checkpoints"
    subject_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    pooled_x: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    for subject in training:
        source_x, source_y, _, _, _ = load_source_session(subject, args.cache_dir)
        pooled_x.append(source_x)
        pooled_y.append(source_y)
    pretrain_x = np.concatenate(pooled_x)
    pretrain_y = np.concatenate(pooled_y)
    seed_everything(args.seed + 10_000 * args.fold_index)
    model = build_eegnet(pretrain_x, pretrain_y, device, args.backbone)
    pretrain_metrics = optimize_model(
        model,
        pretrain_x,
        pretrain_y,
        epochs=args.pretrain_epochs,
        learning_rate=args.pretrain_learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed + 10_000 * args.fold_index,
        device=device,
    )
    print(
        f"fold={args.fold_index} held_out={held_out} "
        f"pretrain_acc={pretrain_metrics['accuracy']:.1f}%",
        flush=True,
    )

    completed: list[dict] = []
    for subject in held_out:
        output_path = subject_dir / f"S{subject}_seed{args.seed}.json"
        if output_path.exists() and not args.force:
            payload = json.loads(output_path.read_text())
        else:
            source_x, source_y, _, _, _ = load_source_session(
                subject, args.cache_dir
            )
            subject_model = clone_model(model).to(device)
            finetune_metrics = optimize_model(
                subject_model,
                source_x,
                source_y,
                epochs=args.finetune_epochs,
                learning_rate=args.finetune_learning_rate,
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                warmup_epochs=min(args.warmup_epochs, args.finetune_epochs),
                seed=args.seed + subject,
                device=device,
            )
            target_accuracy, rows = evaluate_target_source(
                subject_model,
                subject,
                args.cache_dir,
                args.prefix,
                device,
                args.eval_batch_size,
            )
            state = state_dict_cpu(subject_model)
            digest = state_digest(state)
            payload = {
                "subject": subject,
                "seed": args.seed,
                "fold": args.fold_index,
                "training_subjects": training,
                "pretrain_metrics": pretrain_metrics,
                "finetune_metrics": finetune_metrics,
                "mean_target_source_accuracy": target_accuracy,
                "source_hash": digest,
                "rows": rows,
            }
            output_path.write_text(json.dumps(payload, indent=2))
            torch.save(
                {
                    "subject": subject,
                    "seed": args.seed,
                    "fold": args.fold_index,
                    "state_dict": state,
                    "source_hash": digest,
                    "config": vars(args),
                },
                checkpoint_dir / f"S{subject}_seed{args.seed}.pt",
            )
        completed.append(
            {
                "subject": subject,
                "finetune_accuracy": payload["finetune_metrics"]["accuracy"],
                "mean_target_source_accuracy": payload[
                    "mean_target_source_accuracy"
                ],
            }
        )
        print(
            f"S{subject}: fine={payload['finetune_metrics']['accuracy']:.1f}% "
            f"target={payload['mean_target_source_accuracy']:.2f}%",
            flush=True,
        )

    summary = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "resolved_device": str(device),
        },
        "training_subjects": training,
        "held_out_subjects": held_out,
        "pretrain_metrics": pretrain_metrics,
        "completed": completed,
    }
    (fold_dir / f"summary_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
