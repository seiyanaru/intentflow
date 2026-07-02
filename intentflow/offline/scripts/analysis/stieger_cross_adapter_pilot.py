"""Prospective Stieger EEGNet cross-adapter pilot.

Each subject is trained on session 1. For every later session, only the
first ``prefix`` unlabeled trials may modify the adapter state. Evaluation
uses the remaining trials with a frozen state, and every session resets to
the identical source checkpoint.
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
from braindecode.models import EEGNet as BraindecodeEEGNet
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
OFFLINE_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(OFFLINE_DIR))

from models.tcformer.eegnet import EEGNetModule

from cross_adapter_core import (
    adapt_tent_affine,
    apply_reference,
    clone_model,
    ea_reference,
    load_state_dict_copy,
    predict_labels,
    risk_utility_summary,
    seed_everything,
    state_dict_cpu,
    state_digest,
    update_adabn,
)


DEFAULT_CACHE = Path("/home/islabshi/workspace-local2/mne_data/stieger_epochs_cache")
DEFAULT_OUTPUT = (
    OFFLINE_DIR / "results" / "research_outputs" / "260622_cross_adapter_pilot"
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-16")
    parser.add_argument(
        "--backbone",
        choices=("local_eegnet", "braindecode_eegnet"),
        default="local_eegnet",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--adabn-mixing", type=float, default=0.5)
    parser.add_argument("--tent-learning-rate", type=float, default=1e-3)
    parser.add_argument("--tent-steps", type=int, default=1)
    parser.add_argument(
        "--adapters",
        default="ea,adabn,tent",
        help="Comma-separated subset of ea,adabn,tent.",
    )
    parser.add_argument(
        "--source-checkpoint-root",
        type=Path,
        help=(
            "Reuse source checkpoints found recursively below this directory. "
            "This skips source training but recomputes source alignment from cache."
        ),
    )
    parser.add_argument("--interaug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-target-sessions", type=int)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def interaug(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_segments: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-conditional segment recombination used by the existing repo."""
    n_channels, n_times = x.shape[1:]
    segment_length = n_times // n_segments
    augmented_x: list[np.ndarray] = []
    augmented_y: list[int] = []
    per_class = max(1, len(y) // (2 * len(np.unique(y))))
    for class_id in np.unique(y):
        class_trials = x[y == class_id]
        for _ in range(per_class):
            trial = np.empty((n_channels, n_times), dtype=np.float32)
            for segment in range(n_segments):
                source = class_trials[rng.integers(len(class_trials))]
                start = segment * segment_length
                stop = n_times if segment == n_segments - 1 else (segment + 1) * segment_length
                trial[:, start:stop] = source[:, start:stop]
            augmented_x.append(trial)
            augmented_y.append(int(class_id))
    return np.stack(augmented_x), np.asarray(augmented_y, dtype=np.int64)


def train_eegnet(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
    use_interaug: bool,
    backbone: str,
) -> tuple[nn.Module, dict[str, float]]:
    seed_everything(seed)
    model = build_eegnet(x, y, device, backbone)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    rng = np.random.default_rng(seed)
    last_loss = float("nan")
    start_time = time.time()
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            multiplier = (epoch + 1) / max(1, warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            multiplier = 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * multiplier

        if use_interaug:
            augmented_x, augmented_y = interaug(x, y, rng)
            epoch_x = np.concatenate([x, augmented_x])
            epoch_y = np.concatenate([y, augmented_y])
        else:
            epoch_x, epoch_y = x, y
        permutation = rng.permutation(len(epoch_y))
        model.train()
        losses: list[float] = []
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = torch.from_numpy(epoch_x[indices]).to(device)
            batch_y = torch.from_numpy(epoch_y[indices]).long().to(device)
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses))
    model.eval()
    train_predictions = predict_labels(model, x, device, batch_size)
    return model, {
        "train_loss": last_loss,
        "train_accuracy": float(np.mean(train_predictions == y) * 100),
        "train_seconds": time.time() - start_time,
    }


def build_eegnet(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    backbone: str,
) -> nn.Module:
    if backbone == "local_eegnet":
        model = EEGNetModule(
            n_channels=x.shape[1],
            n_classes=len(np.unique(y)),
            input_window_samples=x.shape[2],
        )
    elif backbone == "braindecode_eegnet":
        model = BraindecodeEEGNet(
            n_chans=x.shape[1],
            n_outputs=len(np.unique(y)),
            n_times=x.shape[2],
            final_conv_length="auto",
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model.to(device)


def load_source_checkpoint(
    root: Path,
    subject: int,
    seed: int,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    backbone: str,
) -> tuple[nn.Module, dict[str, float], dict[str, torch.Tensor], str]:
    matches = list(root.glob(f"**/S{subject}_seed{seed}.pt"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one source checkpoint for S{subject} seed {seed}, found {matches}"
        )
    checkpoint = torch.load(matches[0], map_location="cpu", weights_only=False)
    model = build_eegnet(x, y, device, backbone)
    source_state = {
        key: value.detach().cpu().clone()
        for key, value in checkpoint["state_dict"].items()
    }
    load_state_dict_copy(model, source_state)
    source_hash = state_digest(source_state)
    if source_hash != checkpoint["source_hash"]:
        raise RuntimeError(f"Checkpoint digest mismatch: {matches[0]}")
    return model, checkpoint["train_metrics"], source_state, source_hash


def evaluate_subject(
    subject: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    cache_path = args.cache_dir / f"S{subject}_epochs.npz"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    cache = np.load(cache_path)
    x = cache["X"].astype(np.float32)
    y = cache["y"].astype(np.int64)
    sessions = cache["sess"].astype(np.int64)
    session_ids = sorted(np.unique(sessions).tolist())
    source_session = session_ids[0]
    source_mask = sessions == source_session
    source_reference = ea_reference(x[source_mask])
    source_x = apply_reference(source_reference, x[source_mask])
    source_y = y[source_mask]

    if args.source_checkpoint_root is None:
        model, train_metrics = train_eegnet(
            source_x,
            source_y,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            use_interaug=args.interaug,
            backbone=args.backbone,
        )
        source_state = state_dict_cpu(model)
        source_hash = state_digest(source_state)
    else:
        model, train_metrics, source_state, source_hash = load_source_checkpoint(
            args.source_checkpoint_root,
            subject,
            args.seed,
            source_x,
            source_y,
            device,
            args.backbone,
        )
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "subject": subject,
            "seed": args.seed,
            "state_dict": source_state,
            "source_hash": source_hash,
            "train_metrics": train_metrics,
            "config": vars(args),
        },
        checkpoint_dir / f"S{subject}_seed{args.seed}.pt",
    )

    target_ids = session_ids[1:]
    if args.max_target_sessions is not None:
        target_ids = target_ids[: args.max_target_sessions]
    rows: list[dict] = []
    reset_hashes: list[str] = []
    adapters = {
        adapter.strip().lower()
        for adapter in args.adapters.split(",")
        if adapter.strip()
    }
    invalid_adapters = adapters - {"ea", "adabn", "tent"}
    if invalid_adapters:
        raise ValueError(f"Unknown adapters: {sorted(invalid_adapters)}")
    if not adapters:
        raise ValueError("At least one adapter must be selected")
    for session_id in target_ids:
        mask = sessions == session_id
        session_x = x[mask]
        session_y = y[mask]
        if len(session_y) <= args.prefix:
            continue
        prefix_raw = session_x[: args.prefix]
        evaluation_y = session_y[args.prefix :]
        source_aligned = apply_reference(source_reference, session_x)
        prefix_source = source_aligned[: args.prefix]
        evaluation_source = source_aligned[args.prefix :]

        load_state_dict_copy(model, source_state)
        reset_hash = state_digest(state_dict_cpu(model))
        reset_hashes.append(reset_hash)
        if reset_hash != source_hash:
            raise RuntimeError(f"Source reset mismatch for S{subject} session {session_id}")
        source_predictions = predict_labels(
            model, evaluation_source, device, args.eval_batch_size
        )
        source_accuracy = float(np.mean(source_predictions == evaluation_y) * 100)

        if "ea" in adapters:
            ea_model = clone_model(model).to(device)
            target_reference = ea_reference(prefix_raw)
            evaluation_ea = apply_reference(target_reference, session_x)[args.prefix :]
            ea_predictions = predict_labels(
                ea_model, evaluation_ea, device, args.eval_batch_size
            )
            ea_accuracy = float(np.mean(ea_predictions == evaluation_y) * 100)
            rows.append(
                {
                    "subject": subject,
                    "session": int(session_id),
                    "seed": args.seed,
                    "adapter": "ea",
                    "source_acc": source_accuracy,
                    "adapted_acc": ea_accuracy,
                    "delta_pp": ea_accuracy - source_accuracy,
                    "n_eval": len(evaluation_y),
                    "source_hash": source_hash,
                }
            )

        if "adabn" in adapters:
            adabn_model = clone_model(model).to(device)
            update_adabn(
                adabn_model,
                torch.from_numpy(prefix_source).to(device),
                mixing=args.adabn_mixing,
            )
            adabn_predictions = predict_labels(
                adabn_model, evaluation_source, device, args.eval_batch_size
            )
            adabn_accuracy = float(np.mean(adabn_predictions == evaluation_y) * 100)
            rows.append(
                {
                    "subject": subject,
                    "session": int(session_id),
                    "seed": args.seed,
                    "adapter": "adabn",
                    "source_acc": source_accuracy,
                    "adapted_acc": adabn_accuracy,
                    "delta_pp": adabn_accuracy - source_accuracy,
                    "n_eval": len(evaluation_y),
                    "source_hash": source_hash,
                }
            )

        if "tent" in adapters:
            tent_model = clone_model(model).to(device)
            tent_losses = adapt_tent_affine(
                tent_model,
                torch.from_numpy(prefix_source).to(device),
                learning_rate=args.tent_learning_rate,
                steps=args.tent_steps,
            )
            tent_predictions = predict_labels(
                tent_model, evaluation_source, device, args.eval_batch_size
            )
            tent_accuracy = float(np.mean(tent_predictions == evaluation_y) * 100)
            rows.append(
                {
                    "subject": subject,
                    "session": int(session_id),
                    "seed": args.seed,
                    "adapter": "tent",
                    "source_acc": source_accuracy,
                    "adapted_acc": tent_accuracy,
                    "delta_pp": tent_accuracy - source_accuracy,
                    "n_eval": len(evaluation_y),
                    "source_hash": source_hash,
                    "tent_prefix_entropy_last": tent_losses[-1],
                }
            )

    return {
        "subject": subject,
        "seed": args.seed,
        "source_session": int(source_session),
        "source_hash": source_hash,
        "all_resets_match": all(value == source_hash for value in reset_hashes),
        "train_metrics": train_metrics,
        "mean_target_source_accuracy": float(
            np.mean(
                [
                    row["source_acc"]
                    for row in rows
                    if row["adapter"] == next(iter(adapters))
                ]
            )
        ),
        "rows": rows,
    }


def json_ready_config(args: argparse.Namespace, device: torch.device) -> dict:
    config = vars(args).copy()
    config["cache_dir"] = str(config["cache_dir"])
    config["output_dir"] = str(config["output_dir"])
    if config["source_checkpoint_root"] is not None:
        config["source_checkpoint_root"] = str(config["source_checkpoint_root"])
    config["resolved_device"] = str(device)
    return config


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_dir = args.output_dir / "subjects"
    subject_dir.mkdir(exist_ok=True)
    print(
        f"device={device} subjects={parse_subjects(args.subjects)} "
        f"epochs={args.epochs} prefix={args.prefix}",
        flush=True,
    )
    all_rows: list[dict] = []
    completed: list[dict] = []
    failures: list[dict] = []
    requested_subjects = parse_subjects(args.subjects)
    for subject in requested_subjects:
        output_path = subject_dir / f"S{subject}_seed{args.seed}.json"
        if output_path.exists() and not args.force:
            payload = json.loads(output_path.read_text())
            print(f"S{subject}: resume ({len(payload['rows'])} rows)", flush=True)
        else:
            try:
                payload = evaluate_subject(subject, args, device)
                output_path.write_text(json.dumps(payload, indent=2))
                print(
                    f"S{subject}: train={payload['train_metrics']['train_accuracy']:.1f}% "
                    f"rows={len(payload['rows'])}",
                    flush=True,
                )
            except Exception as error:
                failures.append({"subject": subject, "error": repr(error)})
                print(f"S{subject}: FAIL {error!r}", flush=True)
                continue
        completed.append(
            {
                "subject": subject,
                "train_accuracy": payload["train_metrics"]["train_accuracy"],
                "mean_target_source_accuracy": payload[
                    "mean_target_source_accuracy"
                ],
                "all_resets_match": payload["all_resets_match"],
            }
        )
        all_rows.extend(payload["rows"])

    summaries = {
        adapter: risk_utility_summary(all_rows, adapter)
        for adapter in ("ea", "adabn", "tent")
        if any(row["adapter"] == adapter for row in all_rows)
    }
    target_source_accuracies = [
        row["mean_target_source_accuracy"] for row in completed
    ]
    required_completed = max(1, math.ceil(0.875 * len(requested_subjects)))
    report = {
        "config": json_ready_config(args, device),
        "completed": completed,
        "failures": failures,
        "summaries": summaries,
        "pilot_acceptance": {
            "requested_subjects": len(requested_subjects),
            "required_completed": required_completed,
            "completed_enough": len(completed) >= required_completed,
            "median_target_source_accuracy_ge_60": (
                float(np.median(target_source_accuracies)) >= 60
                if target_source_accuracies
                else False
            ),
            "all_resets_match": all(
                row["all_resets_match"] for row in completed
            ),
        },
    }
    (args.output_dir / f"summary_seed{args.seed}.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report["pilot_acceptance"], indent=2), flush=True)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
