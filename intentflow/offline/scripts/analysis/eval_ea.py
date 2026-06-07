"""Step1: incremental Euclidean Alignment (EA) on a frozen Base TCFormer.

Loads each subject's Base checkpoint (no re-training), runs the test set twice
(EA-off = source baseline, EA-on = incremental EA in the raw-EEG preprocessing
stage), and reports per-subject delta. This isolates the single effect that
TCFormer_Hybrid lacked (alignment), with the backbone fully frozen.

EA (He & Wu 2019), incremental/online form:
  per trial x (C,T): cov = x x^T / T ; R_t = running mean of cov ;
  x_aligned = R_t^{-1/2} x
R is reset per subject (test session). EA adds no parameters -> the Base
checkpoint loads unchanged.

GPU by default (CLAUDE.md). Read-only on checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from utils.get_datamodule_cls import get_datamodule_cls  # noqa: E402
from utils.get_model_cls import get_model_cls  # noqa: E402


class IncrementalEA:
    """Online Euclidean Alignment over the raw-EEG covariance (per subject)."""

    def __init__(self):
        self.R = None
        self.n = 0

    @torch.no_grad()
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T). EA whitens channels by the running mean covariance.
        if x.dim() != 3:
            raise ValueError(f"EA expects (B,C,T), got {tuple(x.shape)}")
        out = x.clone()
        for i in range(x.shape[0]):
            xi = x[i]  # (C, T)
            cov = (xi @ xi.transpose(-1, -2)) / xi.shape[-1]  # (C, C)
            self.n += 1
            if self.R is None:
                self.R = cov.clone()
            else:
                self.R = self.R + (cov - self.R) / self.n
            evals, evecs = torch.linalg.eigh(self.R)
            r_isqrt = evecs @ torch.diag(evals.clamp_min(1e-6).rsqrt()) @ evecs.transpose(-1, -2)
            out[i] = r_isqrt @ xi
        return out


@torch.no_grad()
def evaluate(model, loader, device, use_ea: bool) -> float:
    model.eval()
    ea = IncrementalEA() if use_ea else None
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        if ea is not None:
            x = ea.transform(x)
        logits = model(x)
        correct += int((logits.argmax(-1) == y).sum().item())
        total += int(y.numel())
    return 100.0 * correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources_dir", required=True, help="dir with s{id}/checkpoints/subject_{id}_model.ckpt")
    ap.add_argument("--config", required=True, help="representative config.yaml (model_kwargs + preprocessing)")
    ap.add_argument("--dataset", default="bcic2a")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--subjects", default="all")
    args = ap.parse_args()

    if args.gpu_id >= 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; rerun on a GPU node (CLAUDE.md GPU policy).")
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")

    cfg = yaml.safe_load(open(args.config))
    mk = dict(cfg["model_kwargs"])
    datamodule_cls = get_datamodule_cls(args.dataset)
    model_cls = get_model_cls("tcformer")
    mk["n_channels"] = datamodule_cls.channels
    mk["n_classes"] = datamodule_cls.classes

    if args.subjects == "all":
        subs = list(datamodule_cls.all_subject_ids)
    else:
        subs = [int(s) for s in args.subjects.split(",")]

    rows = []
    for sid in subs:
        ckpt_path = os.path.join(args.sources_dir, f"s{sid}", "checkpoints", f"subject_{sid}_model.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"s{sid}: checkpoint missing, skip ({ckpt_path})")
            continue
        dm = datamodule_cls(cfg["preprocessing"], subject_id=sid)
        dm.setup("fit")
        loader = dm.test_dataloader()

        model = model_cls(**mk, max_epochs=cfg.get("max_epochs", 1000),
                          subject_id=sid, model_name="tcformer", results_dir=".")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        # shape sanity once
        if not rows:
            sx, _ = dm.test_dataset[0]
            print(f"[shape] test sample x = {tuple(sx.shape)} (expect (C,T) -> batched (B,C,T))")

        acc_src = evaluate(model, loader, device, use_ea=False)
        acc_ea = evaluate(model, loader, device, use_ea=True)
        rows.append((sid, acc_src, acc_ea, acc_ea - acc_src))
        print(f"s{sid}: src={acc_src:6.2f}  ea={acc_ea:6.2f}  delta={acc_ea - acc_src:+6.2f}")

    if rows:
        src = np.mean([r[1] for r in rows])
        ea = np.mean([r[2] for r in rows])
        print("-" * 40)
        print(f"MEAN: src={src:6.2f}  ea={ea:6.2f}  delta={ea - src:+6.2f}")
        print("[check] src mean should match source_only (2a: ~82.72) to validate the loader.")


if __name__ == "__main__":
    main()
