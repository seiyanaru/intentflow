"""
TCFormer + BN-frozen prototype EMA OTTA.

First-pass implementation of proposal E'':
deep prototype only, forward-only, all BN layers frozen at test time.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from models.prototype_ema_otta import PrototypeEMA_OTTA
from models.tcformer.classification_module import ClassificationModule
from models.tcformer.tcformer import TCFormerModule


class TCFormerProtoOTTA(ClassificationModule):
    """TCFormer wrapper with deep prototype EMA test-time adaptation."""

    def __init__(self, n_classes: int, **kwargs):
        pmax_threshold = kwargs.pop("pmax_threshold", 0.7)
        sal_threshold = kwargs.pop("sal_threshold", 0.5)
        energy_threshold = kwargs.pop("energy_threshold", None)
        energy_quantile = kwargs.pop("energy_quantile", 0.95)
        energy_temperature = kwargs.pop("energy_temperature", 1.0)
        proto_momentum = kwargs.pop("proto_momentum", 0.05)
        fusion_alpha = kwargs.pop("fusion_alpha", 0.3)
        prototype_logit_scale = kwargs.pop("prototype_logit_scale", 1.0)
        use_energy_gate = kwargs.pop("use_energy_gate", True)
        enable_otta = kwargs.pop("enable_otta", True)
        update_before_fusion = kwargs.pop("update_before_fusion", False)

        model = TCFormerModule(
            n_channels=kwargs.get("n_channels", 22),
            n_classes=n_classes,
            F1=kwargs.get("F1", 32),
            temp_kernel_lengths=kwargs.get("temp_kernel_lengths", [20, 32, 64]),
            pool_length_1=kwargs.get("pool_length_1", 8),
            pool_length_2=kwargs.get("pool_length_2", 7),
            D=kwargs.get("D", 2),
            dropout_conv=kwargs.get("dropout_conv", 0.4),
            d_group=kwargs.get("d_group", 16),
            tcn_depth=kwargs.get("tcn_depth", 2),
            kernel_length_tcn=kwargs.get("kernel_length_tcn", 4),
            dropout_tcn=kwargs.get("dropout_tcn", 0.3),
            use_group_attn=kwargs.get("use_group_attn", True),
            q_heads=kwargs.get("q_heads", 4),
            kv_heads=kwargs.get("kv_heads", 2),
            trans_depth=kwargs.get("trans_depth", 2),
            trans_dropout=kwargs.get("trans_dropout", 0.4),
            use_eca=kwargs.get("use_eca", True),
        )
        super().__init__(model=model, n_classes=n_classes, **kwargs)

        self.pmax_threshold = pmax_threshold
        self.sal_threshold = sal_threshold
        self.energy_threshold = energy_threshold
        self.energy_quantile = energy_quantile
        self.energy_temperature = energy_temperature
        self.proto_momentum = proto_momentum
        self.fusion_alpha = fusion_alpha
        self.prototype_logit_scale = prototype_logit_scale
        self.use_energy_gate = use_energy_gate
        self.enable_otta = enable_otta
        self.update_before_fusion = update_before_fusion
        self.n_classes = n_classes

        self.proto_otta = None
        self.train_dataloader_ref = None
        self.test_proto_stats = []

    def set_train_dataloader(self, dataloader):
        self.train_dataloader_ref = dataloader

    def on_test_start(self):
        if not self.enable_otta:
            return

        print("[TCFormerProtoOTTA] Initializing BN-frozen prototype EMA OTTA...")
        self.proto_otta = PrototypeEMA_OTTA(
            model=self.model,
            n_classes=self.n_classes,
            pmax_threshold=self.pmax_threshold,
            sal_threshold=self.sal_threshold,
            energy_threshold=self.energy_threshold,
            energy_quantile=self.energy_quantile,
            energy_temperature=self.energy_temperature,
            proto_momentum=self.proto_momentum,
            fusion_alpha=self.fusion_alpha,
            prototype_logit_scale=self.prototype_logit_scale,
            use_energy_gate=self.use_energy_gate,
            enable_adaptation=True,
            update_before_fusion=self.update_before_fusion,
        )
        print(
            "[TCFormerProtoOTTA] "
            f"pmax={self.pmax_threshold}, sal={self.sal_threshold}, "
            f"energy_gate={self.use_energy_gate}, q={self.energy_quantile}, "
            f"m={self.proto_momentum}, alpha={self.fusion_alpha}, "
            f"update_before_fusion={self.update_before_fusion}"
        )

        if self.train_dataloader_ref is not None:
            self.proto_otta.compute_source_prototypes(self.train_dataloader_ref, device=self.device)
        else:
            print("[TCFormerProtoOTTA] Warning: no train dataloader; prototypes unavailable.")

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.proto_otta is not None and self.enable_otta:
            result = self.proto_otta(x, return_debug=True)
            logits = result["logits"]
            preds = result["pred"]
            self.test_proto_stats.append(
                {
                    "pmax": result["pmax"].detach().cpu(),
                    "fused_pmax": result["fused_pmax"].detach().cpu(),
                    "sal": result["sal"].detach().cpu(),
                    "energy_score": result["energy_score"].detach().cpu(),
                    "adapted": result["adapted"].detach().cpu(),
                    "adapt_weight": result["adapt_weight"].detach().cpu(),
                    "abstained": result["abstained"].detach().cpu(),
                    "pred": result["pred"].detach().cpu(),
                    "original_pred": result["original_pred"].detach().cpu(),
                    "label": y.detach().cpu(),
                    "gate_high_pmax": result["gate_high_pmax"].detach().cpu(),
                    "gate_high_sal": result["gate_high_sal"].detach().cpu(),
                    "gate_safe_energy": result["gate_safe_energy"].detach().cpu(),
                    "energy_th": result["energy_th"].detach().cpu(),
                }
            )
        else:
            logits = self.forward(x)
            preds = logits.argmax(dim=-1)

        loss = nn.functional.cross_entropy(logits, y)
        acc = (preds == y).float().mean()

        self.test_kappa.update(preds, y)
        self.test_cm.update(preds, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_kappa", self.test_kappa, prog_bar=False, on_step=False, on_epoch=True)

        self.test_logits.append(logits.detach().cpu())
        self.test_labels.append(y.detach().cpu())
        return {"test_loss": loss, "test_acc": acc}

    def on_test_epoch_end(self):
        if self.test_proto_stats and self.subject_id != "unknown":
            os.makedirs(self.results_dir, exist_ok=True)
            stats_path = os.path.join(
                self.results_dir,
                f"proto_otta_stats_s{self.subject_id}_{self.model_name}.npz",
            )

            save_dict = {}
            for key in (
                "pmax",
                "fused_pmax",
                "sal",
                "energy_score",
                "adapted",
                "adapt_weight",
                "abstained",
                "pred",
                "original_pred",
                "label",
                "gate_high_pmax",
                "gate_high_sal",
                "gate_safe_energy",
            ):
                save_dict[key] = torch.cat([x[key] for x in self.test_proto_stats], dim=0).numpy()

            save_dict["energy_th"] = np.array(
                [
                    float(x["energy_th"].reshape(-1)[0].item())
                    for x in self.test_proto_stats
                ],
                dtype=np.float32,
            )
            np.savez(stats_path, **save_dict)
            print(f"[TCFormerProtoOTTA] Saved prototype OTTA stats to {stats_path}")

            if self.proto_otta is not None:
                self.proto_otta.print_stats()
            self.test_proto_stats = []

        super().on_test_epoch_end()
