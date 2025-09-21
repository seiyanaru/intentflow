"""Export trained models to ONNX format alongside normalization metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from intentflow.offline.models.eeg_transformer import EEGTransformerTiny
from intentflow.offline.models.eegnet_lawhern import EEGNet


def build_model(name: str, in_ch: int, time_steps: int, n_classes: int = 2) -> nn.Module:
  """Create a model instance for ONNX export."""
  model_name = name.lower()
  if model_name == "eegnet":
    return EEGNet(in_ch=in_ch, n_classes=n_classes, T=time_steps)
  if model_name == "transformer":
    return EEGTransformerTiny(in_ch=in_ch, n_classes=n_classes, T=time_steps)
  raise ValueError(f"Unknown model: {name}")


def export_model(ckpt: str, model_name: str, out_path: str, stats_out: Optional[str] = None) -> Dict[str, Any]:
  """Export the model to ONNX and persist normalization statistics."""
  checkpoint = torch.load(ckpt, map_location="cpu")
  meta = checkpoint.get("meta", {})
  in_ch = int(meta.get("C"))
  time_steps = int(meta.get("T"))
  model = build_model(model_name, in_ch=in_ch, time_steps=time_steps)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()

  dummy = torch.zeros(1, in_ch, time_steps, dtype=torch.float32)
  out_file = Path(out_path)
  out_file.parent.mkdir(parents=True, exist_ok=True)

  torch.onnx.export(
    model,
    dummy,
    out_file,
    input_names=["eeg"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes=None,
  )

  stats_path = Path(stats_out) if stats_out else out_file.with_suffix(".stats.json")
  stats_payload = {
    "model": model_name,
    "C": in_ch,
    "T": time_steps,
    "mean": [float(v) for v in checkpoint.get("mean", [])],
    "std": [float(v) for v in checkpoint.get("std", [])],
  }
  stats_path.parent.mkdir(parents=True, exist_ok=True)
  with stats_path.open("w", encoding="utf-8") as handle:
    json.dump(stats_payload, handle)
  return {"onnx": str(out_file), "stats": str(stats_path)}


def parse_args() -> argparse.Namespace:
  """Parse command line arguments for ONNX export."""
  parser = argparse.ArgumentParser(description="Export intentflow checkpoints to ONNX")
  parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint file path")
  parser.add_argument("--out", type=str, required=True, help="Destination ONNX path")
  parser.add_argument("--model", type=str, choices=["eegnet", "transformer"], help="Model name")
  parser.add_argument("--stats-out", type=str, default=None, help="Optional JSON path for normalization stats")
  return parser.parse_args()


def main() -> None:
  """CLI entry point for exporting ONNX models."""
  args = parse_args()
  try:
    export_model(args.ckpt, args.model, args.out, args.stats_out)
  except Exception as exc:  # noqa: BLE001
    print(f"ONNX export failed: {exc}")
    raise


if __name__ == "__main__":
  main()
