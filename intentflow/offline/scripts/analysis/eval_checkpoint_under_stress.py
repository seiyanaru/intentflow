"""Evaluate a trained checkpoint under reproducible test-channel corruption.

This is a lightweight bypass around Lightning Trainer for stress studies. It
loads the repository datamodule, applies optional EA / RAA / artifact stress via
the normal preprocessing path, then runs plain torch inference on the test tensor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.get_datamodule_cls import get_datamodule_cls  # noqa: E402
from utils.get_model_cls import get_model_cls  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


RUNTIME_ADAPTER_PREFIXES = (
    "policy_otta.",
    "proto_otta.",
    "otta.",
    "replay_otta.",
    "dc_replay_otta.",
)


def strip_runtime_adapter_keys(state_dict):
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(RUNTIME_ADAPTER_PREFIXES)
    }


def load_preprocessing(cfg, dataset, args):
    preprocessing_cfg = cfg["preprocessing"]
    if dataset in preprocessing_cfg:
        preprocessing = dict(preprocessing_cfg[dataset])
        preprocessing["z_scale"] = cfg.get("z_scale", preprocessing.get("z_scale", True))
        if dataset == "bcic2a":
            preprocessing["data_path"] = cfg.get("data_path", preprocessing.get("data_path"))
            preprocessing["eval_label_path"] = cfg.get(
                "data_path_2a_eval_labels", preprocessing.get("eval_label_path")
            )
        elif dataset == "bcic2b":
            preprocessing["data_path"] = cfg.get(
                "data_path_2b", cfg.get("data_path", preprocessing.get("data_path"))
            )
    else:
        preprocessing = dict(preprocessing_cfg)

    preprocessing["interaug"] = False
    preprocessing["num_workers"] = 0

    if args.ea or args.ea_weight_mode is not None:
        ea_cfg = dict(preprocessing.get("ea", {}))
        if args.ea:
            ea_cfg["enabled"] = True
        if args.ea_shrinkage is not None:
            ea_cfg["shrinkage"] = args.ea_shrinkage
        if args.ea_power is not None:
            ea_cfg["power"] = args.ea_power
        if args.ea_weight_mode is not None:
            weight_cfg = dict(ea_cfg.get("channel_weighting", {}))
            weight_cfg["mode"] = args.ea_weight_mode
            weight_cfg["enabled"] = args.ea_weight_mode != "none"
            if args.ea_weight_strength is not None:
                weight_cfg["strength"] = args.ea_weight_strength
            if args.ea_weight_min is not None:
                weight_cfg["w_min"] = args.ea_weight_min
            if args.ea_weight_tau is not None:
                weight_cfg["tau"] = args.ea_weight_tau
            if args.ea_weight_margin is not None:
                weight_cfg["relative_margin"] = args.ea_weight_margin
            if args.ea_weight_post_power is not None:
                weight_cfg["post_weight_power"] = args.ea_weight_post_power
            if args.ea_weight_repair_strength is not None:
                weight_cfg["repair_strength"] = args.ea_weight_repair_strength
            if args.ea_weight_repair_threshold is not None:
                weight_cfg["repair_threshold"] = args.ea_weight_repair_threshold
            if args.ea_weight_repair_topk is not None:
                weight_cfg["repair_topk"] = args.ea_weight_repair_topk
            ea_cfg["channel_weighting"] = weight_cfg
        preprocessing["ea"] = ea_cfg

    if args.stress_mode is not None:
        stress = {
            "enabled": args.stress_mode != "none",
            "mode": args.stress_mode,
        }
        if args.stress_channels is not None:
            stress["channels"] = args.stress_channels
        if args.stress_n_channels is not None:
            stress["n_channels"] = args.stress_n_channels
        if args.stress_level is not None:
            stress["level"] = args.stress_level
        if args.stress_seed is not None:
            stress["seed"] = args.stress_seed
        if args.stress_p_trials is not None:
            stress["p_trials"] = args.stress_p_trials
        if args.stress_line_freq is not None:
            stress["line_freq"] = args.stress_line_freq
        preprocessing["artifact_stress"] = stress

    return preprocessing


def evaluate(model, x, y, batch_size):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            logits.append(model(x[start : start + batch_size]).detach().cpu())
    logits = torch.cat(logits, dim=0)
    preds = logits.argmax(dim=-1).numpy()
    labels = y.numpy()
    return {
        "test_acc": float((preds == labels).mean()),
        "test_kappa": float(cohen_kappa_score(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds).astype(int).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="bcic2a")
    parser.add_argument("--model", default=None)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ea", action="store_true")
    parser.add_argument("--ea_shrinkage", type=float, default=None)
    parser.add_argument("--ea_power", type=float, default=None)
    parser.add_argument("--ea_weight_mode", choices=["none", "var", "artifact", "full"], default=None)
    parser.add_argument("--ea_weight_strength", type=float, default=None)
    parser.add_argument("--ea_weight_min", type=float, default=None)
    parser.add_argument("--ea_weight_tau", type=float, default=None)
    parser.add_argument("--ea_weight_margin", type=float, default=None)
    parser.add_argument("--ea_weight_post_power", type=float, default=None)
    parser.add_argument("--ea_weight_repair_strength", type=float, default=None)
    parser.add_argument("--ea_weight_repair_threshold", type=float, default=None)
    parser.add_argument("--ea_weight_repair_topk", type=int, default=None)
    parser.add_argument(
        "--stress_mode",
        choices=[
            "none",
            "gaussian",
            "highvar",
            "flatline",
            "dropout",
            "line",
            "burst",
            "scale",
            "mixed",
        ],
        default=None,
    )
    parser.add_argument("--stress_channels", default=None)
    parser.add_argument("--stress_n_channels", type=int, default=None)
    parser.add_argument("--stress_level", type=float, default=None)
    parser.add_argument("--stress_seed", type=int, default=None)
    parser.add_argument("--stress_p_trials", type=float, default=None)
    parser.add_argument("--stress_line_freq", type=float, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = yaml.safe_load(open(args.config))
    datamodule_cls = get_datamodule_cls(args.dataset)
    preprocessing = load_preprocessing(cfg, args.dataset, args)
    dm = datamodule_cls(preprocessing, subject_id=args.subject)
    dm.setup("fit")

    model_name = args.model or cfg["model"]
    model_cls = get_model_cls(model_name)
    model_kwargs = dict(cfg["model_kwargs"])
    model_kwargs["n_channels"] = datamodule_cls.channels
    model_kwargs["n_classes"] = datamodule_cls.classes
    model = model_cls(
        **model_kwargs,
        max_epochs=cfg.get("max_epochs", 1000),
        subject_id=args.subject,
        model_name=model_name,
        results_dir=str(Path(args.output).parent),
    )

    ckpt_path = Path(args.checkpoint_dir) / f"checkpoints/subject_{args.subject}_model.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_runtime_adapter_keys(ckpt["state_dict"]))

    device = torch.device(args.device)
    model.to(device)
    x, y = dm.test_dataset.tensors
    x = x.to(device)
    metrics = evaluate(model, x, y, args.batch_size)
    metrics.update(
        {
            "subject": args.subject,
            "checkpoint_dir": args.checkpoint_dir,
            "preprocessing": {
                "ea": preprocessing.get("ea", {"enabled": False}),
                "artifact_stress": preprocessing.get("artifact_stress", {"enabled": False}),
            },
        }
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"S{args.subject}: acc={metrics['test_acc'] * 100:.2f}, "
        f"kappa={metrics['test_kappa']:.3f}, output={out}"
    )


if __name__ == "__main__":
    main()
