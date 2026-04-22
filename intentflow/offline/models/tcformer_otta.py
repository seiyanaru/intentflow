"""
TCFormer with Pmax-SAL OTTA Integration

This module provides a TCFormer variant that uses Pmax-SAL gated
online test-time adaptation for improved cross-subject performance.
"""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.pmax_sal_otta import PmaxSAL_OTTA
from models.tcformer.classification_module import ClassificationModule
from models.tcformer.tcformer import TCFormer as TCFormerBase
from utils.montage_mapper import get_electrode_roles


class TCFormerOTTAModule(pl.LightningModule):
    """
    TCFormer with Pmax-SAL OTTA for calibration-free cross-subject MI-EEG.

    This module wraps TCFormer with the Pmax-SAL gated adaptation mechanism,
    enabling online adaptation during test time without requiring calibration.
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        # OTTA parameters
        pmax_threshold: float = 0.7,
        sal_threshold: float = 0.5,
        energy_threshold: Optional[float] = None,
        energy_quantile: float = 0.95,
        energy_temperature: float = 1.0,
        neuro_beta: float = 0.1,
        strict_tri_lock: bool = True,
        enable_otta: bool = True,
        # TCFormer parameters
        F1: int = 32,
        D: int = 2,
        temp_kernel_lengths: list = None,
        q_heads: int = 4,
        kv_heads: int = 2,
        trans_depth: int = 2,
        tcn_kernel_size: int = 3,
        tcn_depth: int = 2,
        tcn_drop: float = 0.2,
        # Training parameters
        lr: float = 0.0009,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        scheduler: bool = True,
        max_epochs: int = 1000,
        warmup_epochs: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if temp_kernel_lengths is None:
            temp_kernel_lengths = [20, 32, 64]

        self.tcformer = TCFormerBase(
            n_classes=n_classes,
            Chans=n_channels,
            F1=F1,
            D=D,
            temp_kernel_lengths=temp_kernel_lengths,
            q_heads=q_heads,
            kv_heads=kv_heads,
            trans_depth=trans_depth,
            tcn_kernel_size=tcn_kernel_size,
            tcn_depth=tcn_depth,
            tcn_drop=tcn_drop,
        )

        self.otta = None
        self.enable_otta = enable_otta
        self.pmax_threshold = pmax_threshold
        self.sal_threshold = sal_threshold
        self.energy_threshold = energy_threshold
        self.energy_quantile = energy_quantile
        self.energy_temperature = energy_temperature
        self.neuro_beta = neuro_beta
        self.strict_tri_lock = strict_tri_lock

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.use_scheduler = scheduler
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.n_classes = n_classes

        self.test_outputs = []
        self.train_dataloader_ref = None

    def forward(self, x):
        return self.tcformer(x)

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )

        if self.use_scheduler:
            from utils.lr_scheduler import linear_warmup_cosine_decay

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=linear_warmup_cosine_decay(
                        self.warmup_epochs,
                        self.max_epochs,
                    ),
                ),
                "interval": "epoch",
            }
            return [optimizer], [scheduler]

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)

        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)

        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def on_test_start(self):
        if self.enable_otta:
            print("[TCFormerOTTA] Initializing Pmax-SAL OTTA...")

            self.otta = PmaxSAL_OTTA(
                model=self.tcformer,
                n_classes=self.n_classes,
                pmax_threshold=self.pmax_threshold,
                sal_threshold=self.sal_threshold,
                energy_threshold=self.energy_threshold,
                energy_quantile=self.energy_quantile,
                energy_temperature=self.energy_temperature,
                neuro_beta=self.neuro_beta,
                strict_tri_lock=self.strict_tri_lock,
                enable_adaptation=True,
            )

            if self.train_dataloader_ref is not None:
                self.otta.compute_source_prototypes(
                    self.train_dataloader_ref,
                    device=self.device,
                )
            else:
                print("[TCFormerOTTA] Warning: No training data for prototypes")

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.otta is not None and self.enable_otta:
            result = self.otta(x, return_debug=True)
            logits = result["logits"]
            preds = result["pred"]
            pmax = result["pmax"]
            sal = result["sal"]
            adapted = result["adapted"]
        else:
            logits = self.forward(x)
            preds = logits.argmax(dim=-1)
            pmax = torch.softmax(logits, dim=-1).max(dim=-1)[0]
            sal = torch.ones_like(pmax)
            adapted = False

        correct = (preds == y).float()
        self.test_outputs.append(
            {
                "preds": preds.cpu(),
                "labels": y.cpu(),
                "correct": correct.cpu(),
                "pmax": pmax.cpu(),
                "sal": sal.cpu(),
                "adapted": adapted,
            }
        )
        return {"correct": correct.mean()}

    def on_test_epoch_end(self):
        all_preds = torch.cat([o["preds"] for o in self.test_outputs])
        all_labels = torch.cat([o["labels"] for o in self.test_outputs])
        all_correct = torch.cat([o["correct"] for o in self.test_outputs])
        all_pmax = torch.cat([o["pmax"] for o in self.test_outputs])
        all_sal = torch.cat([o["sal"] for o in self.test_outputs])

        kappa = self.test_kappa(all_preds, all_labels) if hasattr(self, "test_kappa") else 0.0
        acc = all_correct.mean().item()

        print("\n[TCFormerOTTA] Test Results:")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(f"  Kappa: {kappa:.4f}")
        print(f"  Pmax (mean): {all_pmax.mean():.3f}")
        print(f"  SAL (mean): {all_sal.mean():.3f}")

        if self.otta is not None:
            self.otta.print_stats()

        self.log("test_acc", acc)
        self.log("test_kappa", kappa)

        self.test_outputs = []
        return acc

    def set_train_dataloader(self, dataloader):
        self.train_dataloader_ref = dataloader


class TCFormerOTTA(ClassificationModule):
    """
    TCFormer with OTTA wrapper for use with existing training pipeline.
    """

    def __init__(self, n_classes, **kwargs):
        pmax_threshold = kwargs.pop("pmax_threshold", 0.7)
        sal_threshold = kwargs.pop("sal_threshold", 0.5)
        energy_threshold = kwargs.pop("energy_threshold", None)
        energy_quantile = kwargs.pop("energy_quantile", 0.95)
        energy_temperature = kwargs.pop("energy_temperature", 1.0)
        neuro_beta = kwargs.pop("neuro_beta", 0.1)
        strict_tri_lock = kwargs.pop("strict_tri_lock", True)
        enable_otta = kwargs.pop("enable_otta", True)
        adapt_mode = kwargs.pop("adapt_mode", "bn_stat")
        bn_momentum = kwargs.pop("bn_momentum", 0.1)
        bn_update_target = kwargs.pop("bn_update_target", "both")
        bn_shallow_mean_momentum = kwargs.pop("bn_shallow_mean_momentum", None)
        bn_shallow_var_momentum = kwargs.pop("bn_shallow_var_momentum", None)
        bn_deep_mean_momentum = kwargs.pop("bn_deep_mean_momentum", None)
        bn_deep_var_momentum = kwargs.pop("bn_deep_var_momentum", None)

        model = TCFormerBase(
            n_classes=n_classes,
            n_channels=kwargs.get("n_channels", 22),
            F1=kwargs.get("F1", 32),
            D=kwargs.get("D", 2),
            temp_kernel_lengths=kwargs.get("temp_kernel_lengths", [20, 32, 64]),
            q_heads=kwargs.get("q_heads", 4),
            kv_heads=kwargs.get("kv_heads", 2),
            trans_depth=kwargs.get("trans_depth", 2),
            tcn_kernel_size=kwargs.get("tcn_kernel_size", 3),
            tcn_depth=kwargs.get("tcn_depth", 2),
            tcn_drop=kwargs.get("tcn_drop", 0.2),
        )

        super().__init__(model=model, n_classes=n_classes, **kwargs)

        self.pmax_threshold = pmax_threshold
        self.sal_threshold = sal_threshold
        self.energy_threshold = energy_threshold
        self.energy_quantile = energy_quantile
        self.energy_temperature = energy_temperature
        self.neuro_beta = neuro_beta
        self.strict_tri_lock = strict_tri_lock
        self.enable_otta = enable_otta
        self.adapt_mode = adapt_mode
        self.bn_momentum = bn_momentum
        self.bn_update_target = bn_update_target
        self.bn_shallow_mean_momentum = bn_shallow_mean_momentum
        self.bn_shallow_var_momentum = bn_shallow_var_momentum
        self.bn_deep_mean_momentum = bn_deep_mean_momentum
        self.bn_deep_var_momentum = bn_deep_var_momentum
        self.otta = None
        self.train_dataloader_ref = None
        self.n_classes = n_classes

        self.test_otta_stats = []

    def on_test_start(self):
        if self.enable_otta:
            print("[TCFormerOTTA] Initializing Pmax-SAL OTTA...")

            self.otta = PmaxSAL_OTTA(
                model=self.model,
                n_classes=self.n_classes,
                pmax_threshold=self.pmax_threshold,
                sal_threshold=self.sal_threshold,
                energy_threshold=self.energy_threshold,
                energy_quantile=self.energy_quantile,
                energy_temperature=self.energy_temperature,
                neuro_beta=self.neuro_beta,
                strict_tri_lock=self.strict_tri_lock,
                bn_momentum=self.bn_momentum,
                bn_shallow_mean_momentum=self.bn_shallow_mean_momentum,
                bn_shallow_var_momentum=self.bn_shallow_var_momentum,
                bn_deep_mean_momentum=self.bn_deep_mean_momentum,
                bn_deep_var_momentum=self.bn_deep_var_momentum,
                enable_adaptation=True,
            )
            self.otta.adapt_mode = self.adapt_mode
            self.otta.bn_update_target = self.bn_update_target
            print(
                f"[TCFormerOTTA] adapt_mode={self.adapt_mode}, "
                f"bn_momentum={self.bn_momentum}, "
                f"bn_update_target={self.bn_update_target}"
            )
            if any(
                v is not None
                for v in (
                    self.bn_shallow_mean_momentum,
                    self.bn_shallow_var_momentum,
                    self.bn_deep_mean_momentum,
                    self.bn_deep_var_momentum,
                )
            ):
                print(
                    "[TCFormerOTTA] split momentum overrides: "
                    f"shallow_mean={self.bn_shallow_mean_momentum}, "
                    f"shallow_var={self.bn_shallow_var_momentum}, "
                    f"deep_mean={self.bn_deep_mean_momentum}, "
                    f"deep_var={self.bn_deep_var_momentum}"
                )

            try:
                datamodule = getattr(self.trainer, "datamodule", None)
                if datamodule:
                    ch_names = None
                    if hasattr(datamodule, "test_set") and hasattr(datamodule.test_set, "ch_names"):
                        ch_names = datamodule.test_set.ch_names
                    elif hasattr(datamodule, "dataset_test") and hasattr(datamodule.dataset_test, "ch_names"):
                        ch_names = datamodule.dataset_test.ch_names
                    elif hasattr(datamodule, "dataset") and hasattr(datamodule.dataset, "datasets"):
                        ds = datamodule.dataset.datasets[0]
                        if hasattr(ds, "ch_names"):
                            ch_names = ds.ch_names
                        elif hasattr(ds, "raw") and hasattr(ds.raw, "ch_names"):
                            ch_names = ds.raw.ch_names
                        elif hasattr(ds, "windows") and hasattr(ds.windows, "ch_names"):
                            ch_names = ds.windows.ch_names

                    if ch_names:
                        roles = get_electrode_roles(ch_names)
                        self.otta.set_channel_roles(roles)
                        print(
                            "[TCFormerOTTA] Neuro-Gating enabled. "
                            f"Motor channels: {len(roles['motor'])}, "
                            f"Noise channels: {len(roles['noise'])}"
                        )
                    else:
                        print("[TCFormerOTTA] Warning: No channel names found in datamodule. Neuro-Gating disabled.")
            except Exception as e:
                print(f"[TCFormerOTTA] Warning: Failed to setup Neuro-Gating roles: {e}")

            if self.train_dataloader_ref is not None:
                self.otta.compute_source_prototypes(
                    self.train_dataloader_ref,
                    device=self.device,
                )

    def set_train_dataloader(self, dataloader):
        self.train_dataloader_ref = dataloader

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.otta is not None and self.enable_otta:
            result = self.otta(x, return_debug=True)
            logits = result["logits"]
            preds = result["pred"]

            entry = {
                "pmax": result["pmax"].cpu(),
                "sal": result["sal"].cpu(),
                "adapted": result["adapted"],
                "adapt_weight": result["adapt_weight"].cpu(),
                "pred": result["pred"].cpu(),
                "original_pred": result["original_pred"].cpu(),
                "label": y.cpu(),
                "entropy": result["entropy"].cpu(),
                "logit_norm": result["logit_norm"].cpu(),
                "bn_drift_norm": result["bn_drift_norm"].cpu(),
                "bn_drift_layers": result["bn_drift_layers"].cpu(),
            }
            if "neuro_score" in result and result["neuro_score"] is not None:
                entry["neuro_score"] = result["neuro_score"].cpu()
            if "energy_score" in result and result["energy_score"] is not None:
                entry["energy_score"] = result["energy_score"].cpu()

            self.test_otta_stats.append(entry)
        else:
            logits = self.forward(x)
            preds = logits.argmax(dim=-1)

        loss = nn.functional.cross_entropy(logits, y)
        acc = (preds == y).float().mean()

        if hasattr(self, "test_kappa"):
            self.test_kappa.update(preds, y)
        if hasattr(self, "test_cm"):
            self.test_cm.update(preds, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        if hasattr(self, "test_kappa"):
            self.log("test_kappa", self.test_kappa, prog_bar=False, on_step=False, on_epoch=True)

        if self.test_logits is not None:
            self.test_logits.append(logits.detach().cpu())
        if self.test_labels is not None:
            self.test_labels.append(y.detach().cpu())

        return {"test_loss": loss, "test_acc": acc}

    def on_test_epoch_end(self):
        if self.test_otta_stats and hasattr(self, "subject_id") and self.subject_id != "unknown":
            os.makedirs(self.results_dir, exist_ok=True)
            stats_path = os.path.join(
                self.results_dir,
                f"otta_stats_s{self.subject_id}_{self.model_name}.npz",
            )

            pmax = torch.cat([x["pmax"] for x in self.test_otta_stats], dim=0).numpy()
            sal = torch.cat([x["sal"] for x in self.test_otta_stats], dim=0).numpy()
            adapted = torch.cat([x["adapted"] for x in self.test_otta_stats], dim=0).numpy()
            adapt_weight = torch.cat([x["adapt_weight"] for x in self.test_otta_stats], dim=0).numpy()
            pred = torch.cat([x["pred"] for x in self.test_otta_stats], dim=0).numpy()
            original_pred = torch.cat([x["original_pred"] for x in self.test_otta_stats], dim=0).numpy()
            label = torch.cat([x["label"] for x in self.test_otta_stats], dim=0).numpy()

            save_dict = {
                "pmax": pmax,
                "sal": sal,
                "adapted": adapted,
                "adapt_weight": adapt_weight,
                "pred": pred,
                "original_pred": original_pred,
                "label": label,
                "entropy": torch.cat([x["entropy"] for x in self.test_otta_stats], dim=0).numpy(),
                "logit_norm": torch.cat([x["logit_norm"] for x in self.test_otta_stats], dim=0).numpy(),
                "bn_drift_norm": torch.cat(
                    [x["bn_drift_norm"].expand(x["pmax"].shape[0]) for x in self.test_otta_stats],
                    dim=0,
                ).numpy(),
                "bn_drift_layers": torch.stack(
                    [
                        x["bn_drift_layers"] if x["bn_drift_layers"].numel() > 0
                        else torch.zeros(12)
                        for x in self.test_otta_stats
                    ],
                    dim=0,
                ).numpy(),
            }
            if "neuro_score" in self.test_otta_stats[0]:
                save_dict["neuro_score"] = torch.cat(
                    [x["neuro_score"] for x in self.test_otta_stats],
                    dim=0,
                ).numpy()
            if any("energy_score" in x for x in self.test_otta_stats):
                energy_parts = []
                for x in self.test_otta_stats:
                    if "energy_score" in x:
                        energy_parts.append(x["energy_score"])
                    else:
                        energy_parts.append(torch.full_like(x["pmax"], float("nan")))
                save_dict["energy_score"] = torch.cat(energy_parts, dim=0).numpy()

            np.savez(stats_path, **save_dict)
            print(f"[TCFormerOTTA] Saved OTTA stats to {stats_path}")
            self.test_otta_stats = []

        super().on_test_epoch_end()
