"""Export trained models to ONNX format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from intentflow.offline.models.eeg_transformer import EEGTransformerTiny
from intentflow.offline.models.eegnet_lawhern import EEGNet


def build_model(name: str, in_ch: int, time_steps: int, n_classes: int = 2) -> nn.Module:
  """Create a model instance for ONNX export."""
  name = name.lower()
  if name == "eegnet":
    return EEGNet(in_ch=in_ch, n_classes=n_classes, T=time_steps)
  if name == "transformer":
    return EEGTransformerTiny(in_ch=in_ch, n_classes=n_classes, T=time_steps)
  raise ValueError(f"Unknown model: {name}")


def export_model(ckpt: str, model_name: str, out_path: str) -> Path:
  """Load a checkpoint and export the model to ONNX."""
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
  return out_file


def parse_args() -> argparse.Namespace:
  """Parse command line arguments for ONNX export."""
  parser = argparse.ArgumentParser(description="Export intentflow checkpoints to ONNX")
  parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint file path")
  parser.add_argument("--out", type=str, required=True, help="Destination ONNX path")
  parser.add_argument("--model", type=str, choices=["eegnet", "transformer"], help="Model name")
  return parser.parse_args()


def main() -> None:
  """CLI entry point for exporting ONNX models."""
  args = parse_args()
  try:
    export_model(ckpt=args.ckpt, model_name=args.model, out_path=args.out)
  except Exception as exc:  # noqa: BLE001
    print(f"ONNX export failed: {exc}")
    raise


if __name__ == "__main__":
  main()
