"""TCFormer wrapper for Deferred-Commit Replay OTTA."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from models.deferred_commit_replay_otta import DeferredCommitReplayOTTA
from models.tcformer.classification_module import ClassificationModule
from models.tcformer.tcformer import TCFormerModule
from utils.montage_mapper import get_electrode_roles


class TCFormerDeferredCommitReplayOTTA(ClassificationModule):
    """TCFormer + DC-Replay OTTA control layer."""

    def __init__(self, n_classes: int, **kwargs):
        # Policy / SafeCommit kwargs.
        pmax_threshold = kwargs.pop("pmax_threshold", 0.7)
        sal_threshold = kwargs.pop("sal_threshold", 0.5)
        energy_threshold = kwargs.pop("energy_threshold", None)
        energy_quantile = kwargs.pop("energy_quantile", 0.95)
        energy_temperature = kwargs.pop("energy_temperature", 1.0)
        prototype_fusion_alpha = kwargs.pop("prototype_fusion_alpha", 0.2)
        prototype_logit_scale = kwargs.pop("prototype_logit_scale", 1.0)
        proto_momentum = kwargs.pop("proto_momentum", 0.05)
        bn_momentum = kwargs.pop("bn_momentum", 0.01)
        bn_shallow_mean_momentum = kwargs.pop("bn_shallow_mean_momentum", None)
        bn_shallow_var_momentum = kwargs.pop("bn_shallow_var_momentum", None)
        bn_deep_mean_momentum = kwargs.pop("bn_deep_mean_momentum", None)
        bn_deep_var_momentum = kwargs.pop("bn_deep_var_momentum", None)
        logit_bias_momentum = kwargs.pop("logit_bias_momentum", 0.02)
        max_logit_bias = kwargs.pop("max_logit_bias", 0.25)
        shallow_var_risk_threshold = kwargs.pop("shallow_var_risk_threshold", 0.15)
        margin_tolerance = kwargs.pop("margin_tolerance", 0.02)
        sal_tolerance = kwargs.pop("sal_tolerance", 0.05)
        prototype_margin_tolerance = kwargs.pop("prototype_margin_tolerance", 0.05)
        energy_tolerance = kwargs.pop("energy_tolerance", 0.0)
        max_bn_drift = kwargs.pop("max_bn_drift", 1.0)
        max_shallow_var_delta = kwargs.pop("max_shallow_var_delta", 0.25)
        min_neuro_score = kwargs.pop("min_neuro_score", None)
        abstain_on_ood = kwargs.pop("abstain_on_ood", True)
        energy_blocks_update = kwargs.pop("energy_blocks_update", True)
        energy_abstain_margin = kwargs.pop("energy_abstain_margin", 2.0)
        energy_abstain_z = kwargs.pop("energy_abstain_z", 6.0)
        allowed_operators = kwargs.pop("allowed_operators", None)
        history_window = kwargs.pop("history_window", 32)
        enable_otta = kwargs.pop("enable_otta", True)

        # Replay / external-memory kwargs.
        replay_capacity = kwargs.pop("replay_capacity", 32)
        replay_min_size = kwargs.pop("replay_min_size", 16)
        replay_per_class_min = kwargs.pop("replay_per_class_min", 2)
        replay_pmax_threshold = kwargs.pop("replay_pmax_threshold", 0.85)
        replay_sal_threshold = kwargs.pop("replay_sal_threshold", 0.6)
        replay_seed_per_class = kwargs.pop("replay_seed_per_class", 8)
        replay_warmup_source = kwargs.pop("replay_warmup_source", True)
        sim_score_tolerance = kwargs.pop("sim_score_tolerance", 0.0)
        replay_acc_weight = kwargs.pop("replay_acc_weight", 1.0)
        replay_margin_weight = kwargs.pop("replay_margin_weight", 0.3)
        replay_proto_weight = kwargs.pop("replay_proto_weight", 0.3)
        require_replay_ready = kwargs.pop("require_replay_ready", True)
        replay_weight_mode = kwargs.pop("replay_weight_mode", "uniform")
        select_best_candidate = kwargs.pop("select_best_candidate", False)

        # Deferred-Commit kwargs.
        dc_keys = {
            "dc_enable_correction": True,
            "dc_correction_mode": "memory_prior",
            "dc_prior_correction_strength": 0.2,
            "dc_max_prior_correction": 1.0,
            "dc_min_memory_for_correction": 8,
            "dc_enable_memory_update": True,
            "dc_memory_admission_threshold": 0.55,
            "dc_memory_confidence_weight": 0.35,
            "dc_memory_density_weight": 0.15,
            "dc_memory_temporal_weight": 0.15,
            "dc_memory_prototype_weight": 0.15,
            "dc_memory_balance_weight": 0.10,
            "dc_memory_uncertainty_weight": 0.15,
            "dc_memory_ood_weight": 0.20,
            "dc_memory_disagreement_weight": 0.10,
            "dc_commit_mode": "replay_gated",
            "dc_min_memory_for_commit": 16,
            "dc_commit_cooldown": 8,
            "dc_prior_drift_threshold": 0.08,
            "dc_proto_drift_threshold": 0.08,
            "dc_proto_drift_reference": 0.75,
            "dc_drift_score_threshold": 0.08,
            "dc_prior_drift_weight": 1.0,
            "dc_proto_drift_weight": 1.0,
            "dc_random_commit_prob": 0.05,
            "dc_allowed_commit_operators": None,
        }
        dc_kwargs: Dict[str, Any] = {
            key: kwargs.pop(key, default) for key, default in dc_keys.items()
        }

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
        )
        super().__init__(model=model, n_classes=n_classes, **kwargs)

        self.n_classes = n_classes
        self.enable_otta = enable_otta
        self.dc_kwargs = dc_kwargs
        self.policy_safe_kwargs: Dict[str, Any] = {
            "pmax_threshold": pmax_threshold,
            "sal_threshold": sal_threshold,
            "energy_threshold": energy_threshold,
            "energy_quantile": energy_quantile,
            "energy_temperature": energy_temperature,
            "prototype_fusion_alpha": prototype_fusion_alpha,
            "prototype_logit_scale": prototype_logit_scale,
            "proto_momentum": proto_momentum,
            "bn_momentum": bn_momentum,
            "bn_shallow_mean_momentum": bn_shallow_mean_momentum,
            "bn_shallow_var_momentum": bn_shallow_var_momentum,
            "bn_deep_mean_momentum": bn_deep_mean_momentum,
            "bn_deep_var_momentum": bn_deep_var_momentum,
            "logit_bias_momentum": logit_bias_momentum,
            "max_logit_bias": max_logit_bias,
            "shallow_var_risk_threshold": shallow_var_risk_threshold,
            "margin_tolerance": margin_tolerance,
            "sal_tolerance": sal_tolerance,
            "prototype_margin_tolerance": prototype_margin_tolerance,
            "energy_tolerance": energy_tolerance,
            "max_bn_drift": max_bn_drift,
            "max_shallow_var_delta": max_shallow_var_delta,
            "min_neuro_score": min_neuro_score,
            "abstain_on_ood": abstain_on_ood,
            "energy_blocks_update": energy_blocks_update,
            "energy_abstain_margin": energy_abstain_margin,
            "energy_abstain_z": energy_abstain_z,
            "allowed_operators": allowed_operators,
            "history_window": history_window,
            "enable_adaptation": enable_otta,
        }
        self.replay_kwargs: Dict[str, Any] = {
            "replay_capacity": replay_capacity,
            "replay_min_size": replay_min_size,
            "replay_per_class_min": replay_per_class_min,
            "replay_pmax_threshold": replay_pmax_threshold,
            "replay_sal_threshold": replay_sal_threshold,
            "replay_seed_per_class": replay_seed_per_class,
            "replay_warmup_source": replay_warmup_source,
            "sim_score_tolerance": sim_score_tolerance,
            "replay_acc_weight": replay_acc_weight,
            "replay_margin_weight": replay_margin_weight,
            "replay_proto_weight": replay_proto_weight,
            "require_replay_ready": require_replay_ready,
            "replay_weight_mode": replay_weight_mode,
            "select_best_candidate": select_best_candidate,
        }

        self.dc_replay_otta = None
        self.train_dataloader_ref = None
        self.test_dc_replay_records: List[Dict[str, Any]] = []

    def set_train_dataloader(self, dataloader):
        self.train_dataloader_ref = dataloader

    def on_test_start(self):
        if not self.enable_otta:
            return

        print("[TCFormerDCReplayOTTA] Initializing Deferred-Commit Replay OTTA...")
        self.dc_replay_otta = DeferredCommitReplayOTTA(
            model=self.model,
            n_classes=self.n_classes,
            **self.policy_safe_kwargs,
            **self.replay_kwargs,
            **self.dc_kwargs,
        )
        self.dc_replay_otta.to(self.device)

        try:
            datamodule = getattr(self.trainer, "datamodule", None)
            ch_names = self._find_channel_names(datamodule)
            if ch_names:
                self.dc_replay_otta.set_channel_roles(get_electrode_roles(ch_names))
            else:
                print("[TCFormerDCReplayOTTA] Warning: channel names unavailable; neuro state disabled.")
        except Exception as exc:
            print(f"[TCFormerDCReplayOTTA] Warning: failed to set channel roles: {exc}")

        if self.train_dataloader_ref is not None:
            self.dc_replay_otta.compute_source_statistics(self.train_dataloader_ref, device=self.device)
        else:
            print("[TCFormerDCReplayOTTA] Warning: no train dataloader; memory not seeded.")

    @staticmethod
    def _find_channel_names(datamodule) -> List[str]:
        if datamodule is None:
            return []
        for attr in ("test_set", "dataset_test"):
            dataset = getattr(datamodule, attr, None)
            if hasattr(dataset, "ch_names"):
                return dataset.ch_names
        dataset = getattr(datamodule, "dataset", None)
        if dataset is not None and hasattr(dataset, "datasets") and dataset.datasets:
            ds = dataset.datasets[0]
            if hasattr(ds, "ch_names"):
                return ds.ch_names
            if hasattr(ds, "raw") and hasattr(ds.raw, "ch_names"):
                return ds.raw.ch_names
            if hasattr(ds, "windows") and hasattr(ds.windows, "ch_names"):
                return ds.windows.ch_names
        return []

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.dc_replay_otta is not None and self.enable_otta:
            result = self.dc_replay_otta(x, return_debug=True)
            logits = result["logits"]
            preds = result["pred"]
            record = dict(result["log_record"])
            record["pred"] = int(preds.detach().cpu()[0].item()) if preds.numel() == 1 else -1
            record["original_pred"] = int(result["original_pred"].detach().cpu()[0].item()) if result["original_pred"].numel() == 1 else -1
            record["label"] = int(y.detach().cpu()[0].item()) if y.numel() == 1 else -1
            record["correct"] = float(record["pred"] == record["label"]) if record["label"] >= 0 else 0.0
            record["original_correct"] = (
                float(record["original_pred"] == record["label"]) if record["label"] >= 0 else 0.0
            )
            self.test_dc_replay_records.append(record)
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
        if self.test_dc_replay_records and self.subject_id != "unknown":
            os.makedirs(self.results_dir, exist_ok=True)
            stats_path = os.path.join(
                self.results_dir,
                f"dc_replay_otta_stats_s{self.subject_id}_{self.model_name}.npz",
            )
            all_keys: set = set()
            for row in self.test_dc_replay_records:
                all_keys.update(row.keys())
            keys = sorted(all_keys)
            save_dict = {}
            for key in keys:
                values = [row.get(key) for row in self.test_dc_replay_records]
                if any(isinstance(v, str) for v in values if v is not None):
                    save_dict[key] = np.asarray(
                        ["" if v is None else str(v) for v in values],
                        dtype="U128",
                    )
                else:
                    save_dict[key] = np.asarray(
                        [np.nan if v is None else v for v in values],
                        dtype=np.float32,
                    )
            np.savez(stats_path, **save_dict)
            print(f"[TCFormerDCReplayOTTA] Saved Deferred-Commit Replay OTTA stats to {stats_path}")

            if self.dc_replay_otta is not None:
                self.dc_replay_otta.print_stats()
            self.test_dc_replay_records = []

        super().on_test_epoch_end()
