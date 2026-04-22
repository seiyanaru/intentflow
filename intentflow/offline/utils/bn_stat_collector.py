#!/usr/bin/env python3
"""
Phase-A proxy validation for train-time adaptation design.

Collect BN-input statistics for:
  - clean_T (source train split)
  - aug_T   (channel gain jitter on source train split)
  - eval_E  (evaluation split)

Then compute:
  A-1: marginal KL improvement
  A-2: drift direction cosine similarity
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Keep third-party libs from writing under read-only $HOME.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.seed import seed_everything

EPS = 1e-8


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def channel_gain_jitter(
    x: torch.Tensor,
    std: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply channel-wise multiplicative jitter.
    x: [B, C, T]
    """
    noise = torch.randn(
        (x.size(0), x.size(1), 1),
        dtype=x.dtype,
        device="cpu",
        generator=generator,
    ).to(x.device, non_blocking=True)
    gain = (1.0 + std * noise).clamp(min=0.1, max=10.0)
    return x * gain


@dataclass
class RunningChannelStats:
    sum_: torch.Tensor
    sq_sum: torch.Tensor
    count: int = 0

    @classmethod
    def create(cls, n_channels: int) -> "RunningChannelStats":
        return cls(
            sum_=torch.zeros(n_channels, dtype=torch.float64),
            sq_sum=torch.zeros(n_channels, dtype=torch.float64),
            count=0,
        )

    def update(self, x: torch.Tensor) -> None:
        # x is BN input: [B, C, ...]
        reduce_dims = (0,) + tuple(range(2, x.dim()))
        ch_sum = x.double().sum(dim=reduce_dims).cpu()
        ch_sq_sum = (x.double() ** 2).sum(dim=reduce_dims).cpu()
        n = x.size(0)
        for dim in x.shape[2:]:
            n *= dim

        self.sum_.add_(ch_sum)
        self.sq_sum.add_(ch_sq_sum)
        self.count += int(n)

    def finalize(self) -> Dict[str, List[float]]:
        denom = max(self.count, 1)
        mean = self.sum_ / denom
        var = (self.sq_sum / denom) - mean * mean
        var = var.clamp_min(EPS)
        logvar = var.log()
        return {
            "mu": mean.float().tolist(),
            "var": var.float().tolist(),
            "logvar": logvar.float().tolist(),
            "count": self.count,
        }


class BNInputCollector:
    def __init__(self, bn_layers: List[Tuple[str, nn.Module]]) -> None:
        self.bn_layers = bn_layers
        self.handles = []
        self.active_split: Optional[str] = None
        self.stats: Dict[str, Dict[str, RunningChannelStats]] = {}

        for name, module in self.bn_layers:
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, layer_name: str):
        def hook_fn(_module, inputs, _outputs):
            if self.active_split is None:
                return
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or x.dim() < 3:
                return

            split_stats = self.stats.setdefault(self.active_split, {})
            if layer_name not in split_stats:
                split_stats[layer_name] = RunningChannelStats.create(x.size(1))
            split_stats[layer_name].update(x.detach())

        return hook_fn

    def collect_from_loader(
        self,
        split_name: str,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        gain_std: float = 0.0,
        seed: int = 0,
        max_batches: Optional[int] = None,
    ) -> None:
        self.active_split = split_name
        jitter_gen = None
        if gain_std > 0:
            jitter_gen = torch.Generator(device="cpu")
            jitter_gen.manual_seed(seed)

        model.eval()
        with torch.no_grad():
            for batch_idx, (x, _y) in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                x = x.to(device, non_blocking=True)
                if gain_std > 0:
                    x = channel_gain_jitter(x, std=gain_std, generator=jitter_gen)
                _ = model(x)
        self.active_split = None

    def export(self) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for split, layer_stats in self.stats.items():
            out[split] = {}
            for layer_name, st in layer_stats.items():
                out[split][layer_name] = st.finalize()
        return out

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def as_tensor(values: List[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def gaussian_kl(
    mu_p: torch.Tensor,
    var_p: torch.Tensor,
    mu_q: torch.Tensor,
    var_q: torch.Tensor,
) -> torch.Tensor:
    # KL(N_p || N_q) per channel
    var_p = var_p.clamp_min(EPS)
    var_q = var_q.clamp_min(EPS)
    return 0.5 * (
        torch.log(var_q / var_p) + (var_p + (mu_p - mu_q) ** 2) / var_q - 1.0
    )


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.norm(a) * torch.norm(b)
    if float(denom) < EPS:
        return 0.0
    return float(torch.dot(a, b) / denom)


def build_runtime_config(config_path: Path, dataset: str, model_name: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    runtime = {
        "model": model_name,
        "model_kwargs": dict(cfg["model_kwargs"]),
        "max_epochs": cfg.get("max_epochs", 1000),
        "seed": cfg.get("seed", 0),
        "dataset_name": dataset,
    }

    preproc = dict(cfg["preprocessing"][dataset])
    preproc["z_scale"] = cfg.get("z_scale", True)

    if dataset == "bcic2b":
        preproc["data_path"] = cfg.get("data_path_2b", cfg.get("data_path"))
    else:
        preproc["data_path"] = cfg.get("data_path")
    if dataset == "bcic2a":
        preproc["eval_label_path"] = cfg.get("data_path_2a_eval_labels")

    # Force deterministic clean/eval collection (no interaug collate noise).
    preproc["interaug"] = False
    runtime["preprocessing"] = preproc
    return runtime


def get_backbone(model: nn.Module) -> nn.Module:
    # ClassificationModule wrapper keeps the actual network in `.model`.
    return model.model if hasattr(model, "model") else model


def layer_bucket(layer_index: int, total_layers: int) -> str:
    return "shallow" if layer_index < (total_layers // 2) else "deep"


def normalize_checkpoint_state_dict_keys(
    loaded_state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Make checkpoint key prefixes compatible with target model.
    Handles common nesting difference:
      - loaded:  model.model.*
      - target:  model.*
    """
    loaded_keys = set(loaded_state_dict.keys())
    target_keys = set(model_state_dict.keys())
    if loaded_keys == target_keys:
        return loaded_state_dict

    remapped = {
        (k.replace("model.model.", "model.", 1) if k.startswith("model.model.") else k): v
        for k, v in loaded_state_dict.items()
    }
    if set(remapped.keys()) == target_keys:
        return remapped

    # Fallback: leave unchanged; caller will raise with detailed mismatch.
    return loaded_state_dict


def run_subject_gain(
    subject_id: int,
    gain_std: float,
    runtime_cfg: Dict,
    checkpoint_dir: Path,
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: Optional[int],
    seed: int,
) -> Dict:
    datamodule_cls = get_datamodule_cls(dataset_name)
    model_cls = get_model_cls(runtime_cfg["model"])

    model_kwargs = dict(runtime_cfg["model_kwargs"])
    model_kwargs["n_channels"] = datamodule_cls.channels
    model_kwargs["n_classes"] = datamodule_cls.classes

    model = model_cls(
        **model_kwargs,
        max_epochs=runtime_cfg["max_epochs"],
        subject_id=subject_id,
        model_name=runtime_cfg["model"],
        results_dir=str(checkpoint_dir),
    )

    ckpt_path = checkpoint_dir / "checkpoints" / f"subject_{subject_id}_model.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    loaded_sd = state["state_dict"]
    norm_sd = normalize_checkpoint_state_dict_keys(loaded_sd, model.state_dict())
    model.load_state_dict(norm_sd, strict=True)
    model.to(device)
    model.eval()

    datamodule = datamodule_cls(runtime_cfg["preprocessing"], subject_id=subject_id)
    datamodule.setup(None)

    train_loader = DataLoader(
        datamodule.train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        datamodule.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    backbone = get_backbone(model)
    bn_layers = [
        (name, module)
        for name, module in backbone.named_modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    ]
    if not bn_layers:
        raise RuntimeError("No BN layers found in model.")

    collector = BNInputCollector(bn_layers)
    try:
        collector.collect_from_loader(
            split_name="clean",
            model=model,
            loader=train_loader,
            device=device,
            gain_std=0.0,
            seed=seed + subject_id * 100 + 1,
            max_batches=max_batches,
        )
        collector.collect_from_loader(
            split_name="aug",
            model=model,
            loader=train_loader,
            device=device,
            gain_std=gain_std,
            seed=seed + subject_id * 100 + 2,
            max_batches=max_batches,
        )
        collector.collect_from_loader(
            split_name="eval",
            model=model,
            loader=eval_loader,
            device=device,
            gain_std=0.0,
            seed=seed + subject_id * 100 + 3,
            max_batches=max_batches,
        )
        split_stats = collector.export()
    finally:
        collector.close()

    layer_rows: List[Dict] = []
    shallow_positive_improvement = 0
    shallow_positive_cos = 0
    shallow_total = 0

    for idx, (layer_name, _module) in enumerate(bn_layers):
        clean = split_stats["clean"][layer_name]
        aug = split_stats["aug"][layer_name]
        eval_ = split_stats["eval"][layer_name]

        mu_clean = as_tensor(clean["mu"])
        var_clean = as_tensor(clean["var"])
        lv_clean = as_tensor(clean["logvar"])

        mu_aug = as_tensor(aug["mu"])
        var_aug = as_tensor(aug["var"])
        lv_aug = as_tensor(aug["logvar"])

        mu_eval = as_tensor(eval_["mu"])
        var_eval = as_tensor(eval_["var"])
        lv_eval = as_tensor(eval_["logvar"])

        kl_clean_eval_ch = gaussian_kl(mu_clean, var_clean, mu_eval, var_eval)
        kl_aug_eval_ch = gaussian_kl(mu_aug, var_aug, mu_eval, var_eval)
        kl_clean_eval = float(kl_clean_eval_ch.mean())
        kl_aug_eval = float(kl_aug_eval_ch.mean())
        improvement = kl_clean_eval - kl_aug_eval

        drift_aug = torch.cat([mu_aug - mu_clean, lv_aug - lv_clean], dim=0)
        drift_eval = torch.cat([mu_eval - mu_clean, lv_eval - lv_clean], dim=0)
        cos = cosine_similarity(drift_aug, drift_eval)

        bucket = layer_bucket(idx, len(bn_layers))
        if bucket == "shallow":
            shallow_total += 1
            shallow_positive_improvement += int(improvement > 0.0)
            shallow_positive_cos += int(cos > 0.0)

        layer_rows.append(
            {
                "subject_id": subject_id,
                "gain_std": gain_std,
                "layer_index": idx,
                "layer_name": layer_name,
                "bucket": bucket,
                "kl_clean_eval": kl_clean_eval,
                "kl_aug_eval": kl_aug_eval,
                "improvement": improvement,
                "kl_clean_eval_median": float(kl_clean_eval_ch.median()),
                "kl_aug_eval_median": float(kl_aug_eval_ch.median()),
                "kl_clean_eval_max": float(kl_clean_eval_ch.max()),
                "kl_aug_eval_max": float(kl_aug_eval_ch.max()),
                "cos_sim": cos,
            }
        )

    a1_pass = shallow_positive_improvement > (shallow_total / 2.0)
    a2_pass = shallow_positive_cos > (shallow_total / 2.0)

    return {
        "subject_id": subject_id,
        "gain_std": gain_std,
        "n_bn_layers": len(bn_layers),
        "n_shallow_layers": shallow_total,
        "a1_positive_shallow": shallow_positive_improvement,
        "a2_positive_shallow": shallow_positive_cos,
        "a1_pass": a1_pass,
        "a2_pass": a2_pass,
        "both_pass": bool(a1_pass and a2_pass),
        "layers": layer_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-A BN proxy validator")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="bcic2a")
    parser.add_argument(
        "--model",
        type=str,
        default="tcformer",
        help="Backbone model for Phase-A stats collection (default: tcformer).",
    )
    parser.add_argument("--subject_ids", type=str, default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--gain_stds", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--allow_cpu_fallback",
        action="store_true",
        help="Allow CPU fallback even when gpu_id >= 0.",
    )
    args = parser.parse_args()

    subject_ids = parse_csv_ints(args.subject_ids)
    gain_stds = parse_csv_floats(args.gain_stds)

    runtime_cfg = build_runtime_config(Path(args.config), args.dataset, args.model)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()

    if args.output_dir is not None:
        out_dir = Path(args.output_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (
            Path(__file__).resolve().parents[1]
            / "results"
            / f"proxy_validation_phaseA_{ts}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu_id >= 0 and not torch.cuda.is_available() and not args.allow_cpu_fallback:
        raise RuntimeError(
            "gpu_id >= 0 was requested but CUDA is unavailable in this runtime. "
            "Use --allow_cpu_fallback to force CPU, or run on a GPU-enabled node."
        )

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )

    seed_everything(args.seed)
    torch.set_grad_enabled(False)

    print("=" * 72)
    print("Phase-A Proxy Validation")
    print(f"Config       : {args.config}")
    print(f"Model        : {args.model}")
    print(f"Checkpoints  : {checkpoint_dir}")
    print(f"Dataset      : {args.dataset}")
    print(f"Subjects     : {subject_ids}")
    print(f"Gain stds    : {gain_stds}")
    print(f"Device       : {device}")
    print(f"Output       : {out_dir}")
    print("=" * 72)

    subject_results: List[Dict] = []
    layer_rows: List[Dict] = []

    for gain_std in gain_stds:
        print(f"\n[Gain std {gain_std:.4f}]")
        for subject_id in subject_ids:
            print(f"  - Subject {subject_id}: collecting stats...")
            result = run_subject_gain(
                subject_id=subject_id,
                gain_std=gain_std,
                runtime_cfg=runtime_cfg,
                checkpoint_dir=checkpoint_dir,
                dataset_name=args.dataset,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_batches=args.max_batches,
                seed=args.seed,
            )
            subject_results.append(result)
            layer_rows.extend(result["layers"])
            print(
                "    "
                f"A1={result['a1_positive_shallow']}/{result['n_shallow_layers']} "
                f"A2={result['a2_positive_shallow']}/{result['n_shallow_layers']} "
                f"pass(A1,A2,both)=({result['a1_pass']},{result['a2_pass']},{result['both_pass']})"
            )

    # Aggregate summary by gain std
    summary_by_gain = {}
    for gain_std in gain_stds:
        rows = [r for r in subject_results if abs(r["gain_std"] - gain_std) < 1e-12]
        n = len(rows)
        summary_by_gain[f"{gain_std:.6f}"] = {
            "subjects": n,
            "a1_pass": int(sum(int(r["a1_pass"]) for r in rows)),
            "a2_pass": int(sum(int(r["a2_pass"]) for r in rows)),
            "both_pass": int(sum(int(r["both_pass"]) for r in rows)),
        }

    summary = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "config": str(Path(args.config).resolve()),
            "checkpoint_dir": str(checkpoint_dir),
            "dataset": args.dataset,
            "model": args.model,
            "subject_ids": subject_ids,
            "gain_stds": gain_stds,
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "max_batches": args.max_batches,
        },
        "summary_by_gain": summary_by_gain,
        "subject_results": subject_results,
    }

    summary_path = out_dir / "phaseA_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = out_dir / "phaseA_layer_metrics.csv"
    csv_fields = [
        "subject_id",
        "gain_std",
        "layer_index",
        "layer_name",
        "bucket",
        "kl_clean_eval",
        "kl_aug_eval",
        "improvement",
        "kl_clean_eval_median",
        "kl_aug_eval_median",
        "kl_clean_eval_max",
        "kl_aug_eval_max",
        "cos_sim",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in layer_rows:
            writer.writerow(row)

    print("\n=== Phase-A summary ===")
    for gain_key, row in summary_by_gain.items():
        n = row["subjects"]
        print(
            f"gain_std={gain_key}: "
            f"A1={row['a1_pass']}/{n}, "
            f"A2={row['a2_pass']}/{n}, "
            f"both={row['both_pass']}/{n}"
        )
    print(f"Saved: {summary_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
