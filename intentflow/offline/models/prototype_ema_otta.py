"""
Forward-only prototype EMA OTTA for TCFormer.

This implements the first-pass version of proposal E'':
- keep all model weights and BN layers frozen at test time
- compute source class prototypes from train-session features
- gate target trials with pmax x SAL x optional energy
- update only class prototypes with EMA for accepted target trials
- fuse classifier logits with prototype-cosine logits
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeEMA_OTTA(nn.Module):
    """Deep-feature prototype EMA adapter with no model parameter updates."""

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        pmax_threshold: float = 0.7,
        sal_threshold: float = 0.5,
        energy_threshold: Optional[float] = None,
        energy_quantile: float = 0.95,
        energy_temperature: float = 1.0,
        proto_momentum: float = 0.05,
        fusion_alpha: float = 0.3,
        prototype_logit_scale: float = 1.0,
        use_energy_gate: bool = True,
        enable_adaptation: bool = True,
        update_before_fusion: bool = False,
    ):
        super().__init__()
        self.model = model
        self.n_classes = int(n_classes)
        self.pmax_threshold = float(pmax_threshold)
        self.sal_threshold = float(sal_threshold)
        self.fixed_energy_threshold = None if energy_threshold is None else float(energy_threshold)
        self.energy_quantile = float(energy_quantile)
        self.energy_temperature = float(energy_temperature)
        self.proto_momentum = float(proto_momentum)
        self.fusion_alpha = float(fusion_alpha)
        self.prototype_logit_scale = float(prototype_logit_scale)
        self.use_energy_gate = bool(use_energy_gate)
        self.enable_adaptation = bool(enable_adaptation)
        self.update_before_fusion = bool(update_before_fusion)

        self.register_buffer("source_prototypes", None)
        self.register_buffer("target_prototypes", None)
        self.register_buffer("prototype_counts", torch.zeros(self.n_classes))
        self.register_buffer("source_energy_threshold", torch.tensor(float("nan")))

        self._features = None
        self._register_feature_hook()
        self._warned_no_features = False
        self._warned_no_prototypes = False
        self._warned_no_energy_threshold = False

        self.stats = {
            "total_samples": 0,
            "accepted_samples": 0,
            "skipped_pmax": 0,
            "skipped_sal": 0,
            "skipped_energy": 0,
        }

    def _register_feature_hook(self) -> None:
        """Capture classifier-input features from the wrapped TCFormer."""

        def hook_fn(module, inputs, output):
            if isinstance(inputs, tuple) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
                self._features = inputs[0]
            elif torch.is_tensor(inputs):
                self._features = inputs
            elif torch.is_tensor(output):
                self._features = output

        for name, module in self.model.named_modules():
            if "classifier" in name.lower() or "fc" in name.lower():
                module.register_forward_hook(hook_fn)
                return

    @staticmethod
    def _flatten_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim > 2:
            features = features.mean(dim=-1)
        if features.ndim == 3 and features.shape[-1] == 1:
            features = features.squeeze(-1)
        return features

    def _get_features_or_zeros(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if self._features is None:
            if not self._warned_no_features:
                print("[PrototypeEMA] Warning: classifier features unavailable. SAL gate will remain closed.")
                self._warned_no_features = True
            return None
        features = self._flatten_features(self._features)
        if features.shape[0] != batch_size:
            return None
        return features.to(device)

    def compute_pmax(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1)

    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        t = self.energy_temperature
        return -t * torch.logsumexp(logits / t, dim=1)

    def _resolve_energy_threshold(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.fixed_energy_threshold is not None:
            return torch.tensor(self.fixed_energy_threshold, device=device)
        if torch.isfinite(self.source_energy_threshold).item():
            return self.source_energy_threshold.to(device)
        return None

    def compute_sal(self, features: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        prototypes = self.target_prototypes
        if prototypes is None:
            if not self._warned_no_prototypes:
                print("[PrototypeEMA] Warning: prototypes unavailable. Forcing SAL=0.")
                self._warned_no_prototypes = True
            return torch.zeros(features.shape[0], device=features.device)

        features_n = F.normalize(features, p=2, dim=1)
        proto_n = F.normalize(prototypes.to(features.device), p=2, dim=1)
        return (features_n * proto_n[pred]).sum(dim=1)

    def compute_proto_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.target_prototypes is None:
            return torch.zeros(features.shape[0], self.n_classes, device=features.device)
        features_n = F.normalize(features, p=2, dim=1)
        proto_n = F.normalize(self.target_prototypes.to(features.device), p=2, dim=1)
        return self.prototype_logit_scale * (features_n @ proto_n.t())

    def compute_gate(
        self,
        pmax: torch.Tensor,
        sal: torch.Tensor,
        energy: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        high_pmax = pmax > self.pmax_threshold
        high_sal = sal > self.sal_threshold

        if not self.use_energy_gate:
            safe_energy = torch.ones_like(high_pmax, dtype=torch.bool)
            energy_th = None
        else:
            energy_th = self._resolve_energy_threshold(pmax.device)
            if energy is None or energy_th is None:
                safe_energy = torch.zeros_like(high_pmax, dtype=torch.bool)
                if not self._warned_no_energy_threshold:
                    print("[PrototypeEMA] Warning: energy threshold unavailable. Blocking adaptation.")
                    self._warned_no_energy_threshold = True
            else:
                safe_energy = energy <= energy_th

        should_update = high_pmax & high_sal & safe_energy
        gate = {
            "high_pmax": high_pmax,
            "high_sal": high_sal,
            "safe_energy": safe_energy,
            "energy_th": torch.tensor(float("nan"), device=pmax.device)
            if energy_th is None
            else energy_th.detach(),
        }
        return should_update, gate

    def compute_source_prototypes(self, dataloader, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        feature_sums = {}
        feature_counts = {}
        source_energies = []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = self.model(x)
                features = self._get_features_or_zeros(x.shape[0], device)
                if features is None:
                    continue

                source_energies.append(self.compute_energy(logits).detach().cpu())
                for i in range(y.shape[0]):
                    label = int(y[i].item())
                    feat = features[i].detach()
                    if feat.ndim > 1:
                        feat = feat.squeeze()
                    if label not in feature_sums:
                        feature_sums[label] = torch.zeros_like(feat, device=device)
                        feature_counts[label] = 0
                    feature_sums[label] += feat
                    feature_counts[label] += 1

        if source_energies:
            all_energies = torch.cat(source_energies).to(device)
            if self.fixed_energy_threshold is not None:
                th = torch.tensor(self.fixed_energy_threshold, device=device)
                src = "fixed"
            else:
                q = min(max(self.energy_quantile, 0.0), 1.0)
                th = torch.quantile(all_energies, q)
                src = f"quantile={q:.2f}"
            self.source_energy_threshold.copy_(th.detach())
            print(f"[PrototypeEMA] Energy threshold={th.item():.4f} ({src})")

        if not feature_sums:
            print("[PrototypeEMA] Warning: no source features captured.")
            self.source_prototypes = None
            self.target_prototypes = None
            return

        feature_dim = next(iter(feature_sums.values())).numel()
        prototypes = torch.zeros(self.n_classes, feature_dim, device=device)
        counts = torch.zeros(self.n_classes, device=device)

        for label in range(self.n_classes):
            if label in feature_sums and feature_counts[label] > 0:
                prototypes[label] = feature_sums[label] / feature_counts[label]
                counts[label] = feature_counts[label]

        prototypes = F.normalize(prototypes, p=2, dim=1)
        self.source_prototypes = prototypes.detach().clone()
        self.target_prototypes = prototypes.detach().clone()
        self.prototype_counts.copy_(counts.detach().to(self.prototype_counts.device))
        print(f"[PrototypeEMA] Source prototypes: shape={tuple(prototypes.shape)}")

    def _ema_update(self, features: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> None:
        if self.target_prototypes is None or not mask.any():
            return

        with torch.no_grad():
            features_n = F.normalize(features.detach(), p=2, dim=1)
            for feat, cls in zip(features_n[mask], pred[mask]):
                k = int(cls.item())
                old = self.target_prototypes[k].to(feat.device)
                new = F.normalize((1.0 - self.proto_momentum) * old + self.proto_momentum * feat, p=2, dim=0)
                self.target_prototypes[k].copy_(new.to(self.target_prototypes.device))
                self.prototype_counts[k] += 1

    def forward(self, x: torch.Tensor, return_debug: bool = False) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)

        pmax, pred = self.compute_pmax(logits)
        features = self._get_features_or_zeros(x.shape[0], x.device)
        if features is None:
            sal = torch.zeros_like(pmax)
            proto_logits = torch.zeros_like(logits)
        else:
            sal = self.compute_sal(features, pred)
            energy = self.compute_energy(logits)
            should_update, gate = self.compute_gate(pmax, sal, energy)

            if self.enable_adaptation and self.update_before_fusion:
                self._ema_update(features, pred, should_update)

            proto_logits = self.compute_proto_logits(features)

            if self.enable_adaptation and not self.update_before_fusion:
                self._ema_update(features, pred, should_update)

            self.stats["total_samples"] += int(x.shape[0])
            self.stats["accepted_samples"] += int(should_update.sum().item())
            self.stats["skipped_pmax"] += int((~gate["high_pmax"]).sum().item())
            self.stats["skipped_sal"] += int((gate["high_pmax"] & ~gate["high_sal"]).sum().item())
            self.stats["skipped_energy"] += int(
                (gate["high_pmax"] & gate["high_sal"] & ~gate["safe_energy"]).sum().item()
            )

            fused_logits = (1.0 - self.fusion_alpha) * logits + self.fusion_alpha * proto_logits
            fused_pmax, fused_pred = self.compute_pmax(fused_logits)

            result = {
                "logits": fused_logits,
                "classifier_logits": logits,
                "proto_logits": proto_logits,
                "pred": fused_pred,
                "original_pred": pred,
                "pmax": pmax,
                "fused_pmax": fused_pmax,
                "sal": sal,
                "energy_score": energy,
                "adapted": should_update.detach().cpu(),
                "adapt_weight": should_update.to(logits.dtype),
                "abstained": (~should_update).detach().cpu(),
                "gate_high_pmax": gate["high_pmax"].detach().cpu(),
                "gate_high_sal": gate["high_sal"].detach().cpu(),
                "gate_safe_energy": gate["safe_energy"].detach().cpu(),
                "energy_th": gate["energy_th"].detach().cpu().reshape(1),
            }
            if return_debug:
                result["features"] = features
                result["stats"] = self.stats.copy()
            return result

        energy = self.compute_energy(logits)
        should_update, gate = self.compute_gate(pmax, sal, energy)
        self.stats["total_samples"] += int(x.shape[0])

        return {
            "logits": logits,
            "classifier_logits": logits,
            "proto_logits": proto_logits,
            "pred": pred,
            "original_pred": pred,
            "pmax": pmax,
            "fused_pmax": pmax,
            "sal": sal,
            "energy_score": energy,
            "adapted": should_update.detach().cpu(),
            "adapt_weight": should_update.to(logits.dtype),
            "abstained": (~should_update).detach().cpu(),
            "gate_high_pmax": gate["high_pmax"].detach().cpu(),
            "gate_high_sal": gate["high_sal"].detach().cpu(),
            "gate_safe_energy": gate["safe_energy"].detach().cpu(),
            "energy_th": gate["energy_th"].detach().cpu().reshape(1),
        }

    def print_stats(self) -> None:
        total = self.stats["total_samples"]
        if total == 0:
            print("[PrototypeEMA] No samples processed")
            return
        accepted = self.stats["accepted_samples"]
        print("[PrototypeEMA] Adaptation Statistics:")
        print(f"  Total samples: {total}")
        print(f"  Prototype EMA updates: {accepted} ({100.0 * accepted / total:.1f}%)")
        print(f"  Skipped (pmax): {self.stats['skipped_pmax']} ({100.0 * self.stats['skipped_pmax'] / total:.1f}%)")
        print(f"  Skipped (SAL): {self.stats['skipped_sal']} ({100.0 * self.stats['skipped_sal'] / total:.1f}%)")
        print(
            f"  Skipped (energy): {self.stats['skipped_energy']} "
            f"({100.0 * self.stats['skipped_energy'] / total:.1f}%)"
        )
