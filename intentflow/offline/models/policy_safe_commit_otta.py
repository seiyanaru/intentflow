"""
Policy-SafeCommit online test-time adaptation for TCFormer.

The intent is to treat adaptation as a guarded action-selection problem:
StateExtractor reads the current trial state, RuleBasedPolicy proposes one or
more update operators, and SafeCommit accepts only operators that do not violate
source-alignment and stability checks.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)


def _mean_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    if torch.is_tensor(x):
        if x.numel() == 0:
            return default
        return float(x.detach().float().mean().item())
    return float(x)


def _bool_float(value: bool) -> float:
    return 1.0 if bool(value) else 0.0


@dataclass
class OperatorResult:
    """Diagnostics returned by an adaptation operator."""

    name: str
    bn_drift_norm: float = 0.0
    shallow_var_delta: float = 0.0
    layer_drifts: Optional[List[float]] = None


class OperatorBank:
    """Mutation operators used by Policy-SafeCommit OTTA."""

    def __init__(
        self,
        controller: "PolicySafeCommitOTTA",
        bn_momentum: float = 0.01,
        shallow_mean_momentum: Optional[float] = None,
        shallow_var_momentum: Optional[float] = None,
        deep_mean_momentum: Optional[float] = None,
        deep_var_momentum: Optional[float] = None,
        proto_momentum: float = 0.05,
        logit_bias_momentum: float = 0.02,
        max_logit_bias: float = 0.25,
    ):
        self.controller = controller
        self.bn_momentum = float(bn_momentum)
        self.shallow_mean_momentum = shallow_mean_momentum
        self.shallow_var_momentum = shallow_var_momentum
        self.deep_mean_momentum = deep_mean_momentum
        self.deep_var_momentum = deep_var_momentum
        self.proto_momentum = float(proto_momentum)
        self.logit_bias_momentum = float(logit_bias_momentum)
        self.max_logit_bias = float(max_logit_bias)
        self.source_bn_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.capture_source_bn()

    def bn_layers(self) -> List[Tuple[str, nn.Module]]:
        return [(n, m) for n, m in self.controller.model.named_modules() if isinstance(m, BN_TYPES)]

    def capture_source_bn(self) -> None:
        self.source_bn_stats = {}
        for name, module in self.bn_layers():
            self.source_bn_stats[name] = {
                "running_mean": module.running_mean.detach().clone(),
                "running_var": module.running_var.detach().clone(),
            }

    def snapshot(self) -> Dict[str, Any]:
        bn_state = {}
        bn_momentum = {}
        for name, module in self.bn_layers():
            bn_state[name] = (
                module.running_mean.detach().clone(),
                module.running_var.detach().clone(),
            )
            bn_momentum[name] = module.momentum

        return {
            "bn_state": bn_state,
            "bn_momentum": bn_momentum,
            "target_prototypes": None
            if self.controller.target_prototypes is None
            else self.controller.target_prototypes.detach().clone(),
            "prototype_counts": self.controller.prototype_counts.detach().clone(),
            "logit_bias": self.controller.logit_bias.detach().clone(),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        for name, module in self.bn_layers():
            if name in snapshot["bn_state"]:
                running_mean, running_var = snapshot["bn_state"][name]
                module.running_mean.copy_(running_mean)
                module.running_var.copy_(running_var)
                module.momentum = snapshot["bn_momentum"][name]

        target_prototypes = snapshot["target_prototypes"]
        if target_prototypes is None:
            self.controller.target_prototypes = None
        else:
            self.controller.target_prototypes = target_prototypes.to(self.controller.logit_bias.device)
        self.controller.prototype_counts.copy_(snapshot["prototype_counts"].to(self.controller.prototype_counts.device))
        self.controller.logit_bias.copy_(snapshot["logit_bias"].to(self.controller.logit_bias.device))

    def apply(self, operator: str, x: torch.Tensor, state: Dict[str, Any]) -> OperatorResult:
        if operator == "prototype_update":
            self.apply_prototype_update(state)
            return OperatorResult(name=operator)
        if operator == "logit_bias_update":
            self.apply_logit_bias_update(state)
            return OperatorResult(name=operator)
        if operator == "deep_BN_update":
            return self.apply_bn_update(x, "deep", operator)
        if operator == "hybrid_BN_update":
            return self.apply_bn_update(x, "shallow_mean_deep_both", operator)
        if operator == "shallow_var_update":
            return self.apply_bn_update(x, "shallow_var_only", operator)
        if operator in ("no_update", "abstain"):
            return OperatorResult(name=operator)
        raise ValueError(f"Unknown adaptation operator: {operator}")

    def apply_prototype_update(self, state: Dict[str, Any]) -> None:
        features = state.get("features")
        pred = state.get("pred")
        if features is None or pred is None or self.controller.target_prototypes is None:
            return

        features = F.normalize(features.detach(), p=2, dim=1)
        with torch.no_grad():
            for feat, cls in zip(features, pred.detach()):
                k = int(cls.item())
                old = self.controller.target_prototypes[k].to(feat.device)
                new = F.normalize(
                    (1.0 - self.proto_momentum) * old + self.proto_momentum * feat,
                    p=2,
                    dim=0,
                )
                self.controller.target_prototypes[k].copy_(new.to(self.controller.target_prototypes.device))
                self.controller.prototype_counts[k] += 1

    def apply_logit_bias_update(self, state: Dict[str, Any]) -> None:
        logits = state.get("classifier_logits")
        pred = state.get("pred")
        if logits is None or pred is None:
            return

        probs = F.softmax(logits.detach(), dim=-1)
        one_hot = F.one_hot(pred.detach(), num_classes=self.controller.n_classes).to(probs.dtype)
        delta = (one_hot - probs).mean(dim=0)
        with torch.no_grad():
            updated = (1.0 - self.logit_bias_momentum) * self.controller.logit_bias + self.logit_bias_momentum * delta
            updated = updated.clamp(-self.max_logit_bias, self.max_logit_bias)
            self.controller.logit_bias.copy_(updated)

    def apply_bn_update(self, x: torch.Tensor, update_target: str, operator_name: str) -> OperatorResult:
        bn_layers = self.bn_layers()
        if not bn_layers:
            return OperatorResult(name=operator_name)

        n_bn = len(bn_layers)
        shallow_names = {n for n, _ in bn_layers[: n_bn // 2]}
        deep_names = {n for n, _ in bn_layers[n_bn // 2 :]}
        all_names = [n for n, _ in bn_layers]

        desired = {name: {"mean": 0.0, "var": 0.0} for name in all_names}

        def set_momenta(names: Sequence[str], mean: Optional[float] = None, var: Optional[float] = None) -> None:
            for name in names:
                if mean is not None:
                    desired[name]["mean"] = float(mean)
                if var is not None:
                    desired[name]["var"] = float(var)

        if update_target == "deep":
            set_momenta(deep_names, mean=self.bn_momentum, var=self.bn_momentum)
        elif update_target == "shallow_mean_deep_both":
            set_momenta(shallow_names, mean=self.bn_momentum, var=0.0)
            set_momenta(deep_names, mean=self.bn_momentum, var=self.bn_momentum)
        elif update_target == "shallow_var_only":
            set_momenta(shallow_names, mean=0.0, var=self.bn_momentum)
        else:
            raise ValueError(f"Unsupported BN update target: {update_target}")

        override_map = {
            "shallow": {"mean": self.shallow_mean_momentum, "var": self.shallow_var_momentum},
            "deep": {"mean": self.deep_mean_momentum, "var": self.deep_var_momentum},
        }
        for name in all_names:
            scope = "shallow" if name in shallow_names else "deep"
            for stat_name in ("mean", "var"):
                override = override_map[scope][stat_name]
                if override is not None and desired[name][stat_name] > 0.0:
                    desired[name][stat_name] = float(override)

        active_momenta = {
            name: max(stats["mean"], stats["var"])
            for name, stats in desired.items()
        }

        before = {}
        original_momenta = {}
        for name, module in bn_layers:
            before[name] = (
                module.running_mean.detach().clone(),
                module.running_var.detach().clone(),
            )
            original_momenta[name] = module.momentum
            module.momentum = active_momenta[name]

        was_training = self.controller.model.training
        self.controller.model.eval()
        for _, module in bn_layers:
            module.train()
        with torch.no_grad():
            _ = self.controller.model(x)
        self.controller.model.train(was_training)
        if not was_training:
            self.controller.model.eval()

        for name, module in bn_layers:
            before_mean, before_var = before[name]
            desired_mean = desired[name]["mean"]
            desired_var = desired[name]["var"]
            base_momentum = active_momenta[name]
            updated_mean = module.running_mean.detach().clone()
            updated_var = module.running_var.detach().clone()

            if base_momentum <= 0.0:
                module.running_mean.copy_(before_mean)
                module.running_var.copy_(before_var)
                continue

            if desired_mean <= 0.0:
                module.running_mean.copy_(before_mean)
            elif desired_mean != base_momentum:
                scale = desired_mean / base_momentum
                module.running_mean.copy_(before_mean + (updated_mean - before_mean) * scale)

            if desired_var <= 0.0:
                module.running_var.copy_(before_var)
            elif desired_var != base_momentum:
                scale = desired_var / base_momentum
                module.running_var.copy_(before_var + (updated_var - before_var) * scale)

        layer_drifts = []
        shallow_var_delta = 0.0
        for name, module in bn_layers:
            before_mean, before_var = before[name]
            mean_d = (module.running_mean - before_mean).norm().item()
            var_d = (module.running_var - before_var).norm().item()
            layer_drifts.append(mean_d + var_d)
            if name in shallow_names:
                shallow_var_delta += var_d
            module.momentum = original_momenta[name]

        return OperatorResult(
            name=operator_name,
            bn_drift_norm=float(sum(layer_drifts)),
            shallow_var_delta=float(shallow_var_delta),
            layer_drifts=layer_drifts,
        )

    def shallow_var_drift_risk(self) -> float:
        bn_layers = self.bn_layers()
        if not bn_layers or not self.source_bn_stats:
            return 0.0

        n_bn = len(bn_layers)
        shallow_layers = bn_layers[: n_bn // 2]
        total = 0.0
        for name, module in shallow_layers:
            src = self.source_bn_stats.get(name)
            if src is None:
                continue
            denom = src["running_var"].norm().item() + 1e-6
            total += (module.running_var - src["running_var"].to(module.running_var.device)).norm().item() / denom
        return float(total)


class StateExtractor:
    """Builds per-trial state used by the adaptation policy and SafeCommit."""

    def __init__(self, controller: "PolicySafeCommitOTTA"):
        self.controller = controller

    def extract(
        self,
        x: torch.Tensor,
        forward_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        logits = forward_out["logits"]
        classifier_logits = forward_out["classifier_logits"]
        features = forward_out.get("features")

        probs = F.softmax(logits, dim=-1)
        pmax, pred = probs.max(dim=-1)
        topk = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
        margin = topk[:, 0] if topk.shape[-1] == 1 else topk[:, 0] - topk[:, 1]
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        energy = self.controller.compute_energy(classifier_logits)

        if features is None:
            sal = torch.zeros_like(pmax)
            target_sal = torch.zeros_like(pmax)
            proto_margin = torch.zeros_like(pmax)
        else:
            sal = self.controller.compute_sal(features, pred, use_source=True)
            target_sal = self.controller.compute_sal(features, pred, use_source=False)
            proto_margin = self.controller.compute_prototype_margin(features, use_source=False)

        neuro_score, eca_ratio = self.controller.compute_neuro_score()
        if neuro_score.ndim == 0:
            neuro_score = neuro_score.reshape(1).repeat(pmax.shape[0])
        if eca_ratio.ndim == 0:
            eca_ratio = eca_ratio.reshape(1).repeat(pmax.shape[0])

        energy_threshold = self.controller.resolve_energy_threshold(energy.device)
        energy_z = self.controller.compute_energy_z(energy)
        if energy_threshold is None:
            energy_ood = torch.zeros_like(pmax, dtype=torch.bool)
            energy_excess = torch.zeros_like(pmax)
        else:
            energy_excess = energy - energy_threshold
            energy_ood = energy > energy_threshold
        energy_severe_ood = self.controller.is_severe_energy_ood(energy, energy_threshold, energy_z)

        return {
            "logits": logits,
            "classifier_logits": classifier_logits,
            "proto_logits": forward_out["proto_logits"],
            "features": features,
            "pred": pred,
            "pmax": pmax,
            "entropy": entropy,
            "margin": margin,
            "sal": sal,
            "target_sal": target_sal,
            "energy": energy,
            "energy_z": energy_z,
            "energy_excess": energy_excess,
            "energy_ood": energy_ood,
            "energy_severe_ood": energy_severe_ood,
            "energy_threshold": energy_threshold,
            "neuro_score": neuro_score.to(pmax.device),
            "eca_motor_noise_ratio": eca_ratio.to(pmax.device),
            "prototype_margin": proto_margin,
            "shallow_var_drift_risk": torch.tensor(
                self.controller.operator_bank.shallow_var_drift_risk(),
                device=pmax.device,
            ),
            "recent_abstain_rate": torch.tensor(self.controller.recent_abstain_rate(), device=pmax.device),
            "recent_update_rate": torch.tensor(self.controller.recent_update_rate(), device=pmax.device),
        }


class RuleBasedPolicy:
    """First policy: deterministic, interpretable, and loggable."""

    def __init__(
        self,
        pmax_threshold: float = 0.7,
        sal_threshold: float = 0.5,
        shallow_var_risk_threshold: float = 0.15,
        very_safe_pmax_margin: float = 0.15,
        very_safe_sal_margin: float = 0.15,
        abstain_on_ood: bool = True,
        energy_blocks_update: bool = True,
        allowed_operators: Optional[Sequence[str]] = None,
    ):
        self.pmax_threshold = float(pmax_threshold)
        self.sal_threshold = float(sal_threshold)
        self.shallow_var_risk_threshold = float(shallow_var_risk_threshold)
        self.very_safe_pmax_margin = float(very_safe_pmax_margin)
        self.very_safe_sal_margin = float(very_safe_sal_margin)
        self.abstain_on_ood = bool(abstain_on_ood)
        self.energy_blocks_update = bool(energy_blocks_update)
        self.allowed_operators = set(allowed_operators or [
            "no_update",
            "prototype_update",
            "logit_bias_update",
            "deep_BN_update",
            "hybrid_BN_update",
            "shallow_var_update",
            "abstain",
        ])

    def propose(self, state: Dict[str, Any]) -> List[str]:
        pmax = _mean_float(state.get("pmax"))
        sal = _mean_float(state.get("sal"))
        energy_ood = bool(torch.is_tensor(state.get("energy_ood")) and state["energy_ood"].detach().bool().any().item())
        energy_severe_ood = bool(
            torch.is_tensor(state.get("energy_severe_ood"))
            and state["energy_severe_ood"].detach().bool().any().item()
        )
        shallow_risk = _mean_float(state.get("shallow_var_drift_risk"))

        if energy_severe_ood and self.abstain_on_ood:
            return self._filter(["abstain"])

        if energy_ood and self.energy_blocks_update:
            return self._filter(["no_update"])

        if pmax < self.pmax_threshold or sal < self.sal_threshold:
            return self._filter(["no_update"])

        very_safe = (
            pmax >= self.pmax_threshold + self.very_safe_pmax_margin
            and sal >= self.sal_threshold + self.very_safe_sal_margin
            and shallow_risk < self.shallow_var_risk_threshold
        )

        if very_safe:
            return self._filter([
                "shallow_var_update",
                "hybrid_BN_update",
                "deep_BN_update",
                "prototype_update",
                "logit_bias_update",
                "no_update",
            ])

        if shallow_risk >= self.shallow_var_risk_threshold:
            return self._filter([
                "prototype_update",
                "hybrid_BN_update",
                "logit_bias_update",
                "no_update",
            ])

        return self._filter([
            "prototype_update",
            "deep_BN_update",
            "logit_bias_update",
            "no_update",
        ])

    def _filter(self, operators: Sequence[str]) -> List[str]:
        filtered = [op for op in operators if op in self.allowed_operators]
        return filtered or ["no_update"]


class SafeCommit:
    """Try candidate operators and commit only if safety checks pass."""

    def __init__(
        self,
        controller: "PolicySafeCommitOTTA",
        margin_tolerance: float = 0.02,
        sal_tolerance: float = 0.05,
        prototype_margin_tolerance: float = 0.05,
        energy_tolerance: float = 0.0,
        max_bn_drift: float = 1.0,
        max_shallow_var_delta: float = 0.25,
        min_neuro_score: Optional[float] = None,
    ):
        self.controller = controller
        self.margin_tolerance = float(margin_tolerance)
        self.sal_tolerance = float(sal_tolerance)
        self.prototype_margin_tolerance = float(prototype_margin_tolerance)
        self.energy_tolerance = float(energy_tolerance)
        self.max_bn_drift = float(max_bn_drift)
        self.max_shallow_var_delta = float(max_shallow_var_delta)
        self.min_neuro_score = None if min_neuro_score is None else float(min_neuro_score)

    def run(
        self,
        x: torch.Tensor,
        before_state: Dict[str, Any],
        candidates: Sequence[str],
    ) -> Dict[str, Any]:
        rejected: List[str] = []
        reject_reasons: List[str] = []

        for operator in candidates:
            if operator == "abstain":
                return self._result(
                    before_state,
                    before_state,
                    operator="abstain",
                    committed=False,
                    abstained=True,
                    rejected=rejected,
                    reject_reasons=reject_reasons,
                    op_result=OperatorResult(name=operator),
                    reason="policy_abstain",
                )

            if operator == "no_update":
                return self._result(
                    before_state,
                    before_state,
                    operator="no_update",
                    committed=False,
                    abstained=False,
                    rejected=rejected,
                    reject_reasons=reject_reasons,
                    op_result=OperatorResult(name=operator),
                    reason="policy_no_update",
                )

            snapshot = self.controller.operator_bank.snapshot()
            op_result = self.controller.operator_bank.apply(operator, x, before_state)
            after_forward = self.controller.forward_once(x)
            after_state = self.controller.state_extractor.extract(x, after_forward)
            safe, reason = self.is_safe(before_state, after_state, op_result)
            if safe:
                return self._result(
                    before_state,
                    after_state,
                    operator=operator,
                    committed=True,
                    abstained=False,
                    rejected=rejected,
                    reject_reasons=reject_reasons,
                    op_result=op_result,
                    reason=reason,
                )

            self.controller.operator_bank.restore(snapshot)
            rejected.append(operator)
            reject_reasons.append(reason)

        return self._result(
            before_state,
            before_state,
            operator="no_update",
            committed=False,
            abstained=False,
            rejected=rejected,
            reject_reasons=reject_reasons,
            op_result=OperatorResult(name="no_update"),
            reason="all_candidates_rejected",
        )

    def is_safe(
        self,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        op_result: OperatorResult,
    ) -> Tuple[bool, str]:
        before_margin = _mean_float(before_state.get("margin"))
        after_margin = _mean_float(after_state.get("margin"))
        if after_margin + self.margin_tolerance < before_margin:
            return False, "margin_drop"

        before_sal = _mean_float(before_state.get("sal"))
        after_sal = _mean_float(after_state.get("sal"))
        if after_sal + self.sal_tolerance < before_sal:
            return False, "sal_drop"

        before_proto_margin = _mean_float(before_state.get("prototype_margin"))
        after_proto_margin = _mean_float(after_state.get("prototype_margin"))
        if after_proto_margin + self.prototype_margin_tolerance < before_proto_margin:
            return False, "prototype_margin_drop"

        energy_threshold = after_state.get("energy_threshold")
        if energy_threshold is not None:
            after_energy = _mean_float(after_state.get("energy"))
            if after_energy > float(energy_threshold.detach().item()) + self.energy_tolerance:
                return False, "energy_ood"

        if self.min_neuro_score is not None:
            after_neuro = _mean_float(after_state.get("neuro_score"))
            if after_neuro < self.min_neuro_score:
                return False, "neuro_low"

        if op_result.bn_drift_norm > self.max_bn_drift:
            return False, "bn_drift"
        if op_result.shallow_var_delta > self.max_shallow_var_delta:
            return False, "shallow_var_drift"

        return True, "safe"

    def _result(
        self,
        before_state: Dict[str, Any],
        final_state: Dict[str, Any],
        operator: str,
        committed: bool,
        abstained: bool,
        rejected: Sequence[str],
        reject_reasons: Sequence[str],
        op_result: OperatorResult,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "before_state": before_state,
            "final_state": final_state,
            "operator": operator,
            "committed": bool(committed),
            "abstained": bool(abstained),
            "rejected": list(rejected),
            "reject_reasons": list(reject_reasons),
            "op_result": op_result,
            "reason": reason,
        }


class PolicySafeCommitOTTA(nn.Module):
    """TCFormer-sidecar controller for safe online test-time adaptation."""

    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        pmax_threshold: float = 0.7,
        sal_threshold: float = 0.5,
        energy_threshold: Optional[float] = None,
        energy_quantile: float = 0.95,
        energy_temperature: float = 1.0,
        prototype_fusion_alpha: float = 0.2,
        prototype_logit_scale: float = 1.0,
        proto_momentum: float = 0.05,
        bn_momentum: float = 0.01,
        bn_shallow_mean_momentum: Optional[float] = None,
        bn_shallow_var_momentum: Optional[float] = None,
        bn_deep_mean_momentum: Optional[float] = None,
        bn_deep_var_momentum: Optional[float] = None,
        logit_bias_momentum: float = 0.02,
        max_logit_bias: float = 0.25,
        shallow_var_risk_threshold: float = 0.15,
        margin_tolerance: float = 0.02,
        sal_tolerance: float = 0.05,
        prototype_margin_tolerance: float = 0.05,
        energy_tolerance: float = 0.0,
        max_bn_drift: float = 1.0,
        max_shallow_var_delta: float = 0.25,
        min_neuro_score: Optional[float] = None,
        abstain_on_ood: bool = True,
        energy_blocks_update: bool = True,
        energy_abstain_margin: float = 2.0,
        energy_abstain_z: float = 6.0,
        allowed_operators: Optional[Sequence[str]] = None,
        history_window: int = 32,
        enable_adaptation: bool = True,
    ):
        super().__init__()
        self.model = model
        self.n_classes = int(n_classes)
        self.pmax_threshold = float(pmax_threshold)
        self.sal_threshold = float(sal_threshold)
        self.fixed_energy_threshold = None if energy_threshold is None else float(energy_threshold)
        self.energy_quantile = float(energy_quantile)
        self.energy_temperature = float(energy_temperature)
        self.energy_abstain_margin = float(energy_abstain_margin)
        self.energy_abstain_z = float(energy_abstain_z)
        self.prototype_fusion_alpha = float(prototype_fusion_alpha)
        self.prototype_logit_scale = float(prototype_logit_scale)
        self.enable_adaptation = bool(enable_adaptation)

        self.register_buffer("source_energy_threshold", torch.tensor(float("nan")))
        self.register_buffer("source_energy_mean", torch.tensor(float("nan")))
        self.register_buffer("source_energy_std", torch.tensor(float("nan")))
        self.register_buffer("prototype_counts", torch.zeros(self.n_classes))
        self.register_buffer("logit_bias", torch.zeros(self.n_classes))
        self.source_prototypes: Optional[torch.Tensor] = None
        self.target_prototypes: Optional[torch.Tensor] = None

        self._features: Optional[torch.Tensor] = None
        self._feature_hook_handle = None
        self._register_feature_hook()
        self._warned_no_features = False
        self._warned_no_prototypes = False

        self.channel_roles: Optional[Dict[str, List[int]]] = None
        self.recent_updates: Deque[bool] = deque(maxlen=int(history_window))
        self.recent_abstains: Deque[bool] = deque(maxlen=int(history_window))

        self.operator_bank = OperatorBank(
            self,
            bn_momentum=bn_momentum,
            shallow_mean_momentum=bn_shallow_mean_momentum,
            shallow_var_momentum=bn_shallow_var_momentum,
            deep_mean_momentum=bn_deep_mean_momentum,
            deep_var_momentum=bn_deep_var_momentum,
            proto_momentum=proto_momentum,
            logit_bias_momentum=logit_bias_momentum,
            max_logit_bias=max_logit_bias,
        )
        self.state_extractor = StateExtractor(self)
        self.policy = RuleBasedPolicy(
            pmax_threshold=pmax_threshold,
            sal_threshold=sal_threshold,
            shallow_var_risk_threshold=shallow_var_risk_threshold,
            abstain_on_ood=abstain_on_ood,
            energy_blocks_update=energy_blocks_update,
            allowed_operators=allowed_operators,
        )
        self.safe_commit = SafeCommit(
            self,
            margin_tolerance=margin_tolerance,
            sal_tolerance=sal_tolerance,
            prototype_margin_tolerance=prototype_margin_tolerance,
            energy_tolerance=energy_tolerance,
            max_bn_drift=max_bn_drift,
            max_shallow_var_delta=max_shallow_var_delta,
            min_neuro_score=min_neuro_score,
        )

        self.stats: Counter = Counter()
        self.trial_logs: List[Dict[str, Any]] = []

    def _register_feature_hook(self) -> None:
        def hook_fn(module, inputs, output):
            if isinstance(inputs, tuple) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
                self._features = inputs[0]
            elif torch.is_tensor(inputs):
                self._features = inputs
            elif torch.is_tensor(output):
                self._features = output

        for name, module in self.model.named_modules():
            if "classifier" in name.lower() or "fc" in name.lower():
                self._feature_hook_handle = module.register_forward_hook(hook_fn)
                return

    @staticmethod
    def flatten_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim > 2:
            features = features.mean(dim=-1)
        if features.ndim == 3 and features.shape[-1] == 1:
            features = features.squeeze(-1)
        return features

    def get_features(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if self._features is None:
            if not self._warned_no_features:
                print("[PolicySafeCommit] Warning: classifier features unavailable. Prototype/SAL gates are closed.")
                self._warned_no_features = True
            return None
        features = self.flatten_features(self._features)
        if features.shape[0] != batch_size:
            return None
        return features.to(device)

    def set_channel_roles(self, roles: Dict[str, List[int]]) -> None:
        self.channel_roles = roles
        print(
            "[PolicySafeCommit] Neuro state enabled: "
            f"{len(roles.get('motor', []))} motor, {len(roles.get('noise', []))} noise channels."
        )

    def compute_source_statistics(self, dataloader, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        self.operator_bank.capture_source_bn()
        feature_sums: Dict[int, torch.Tensor] = {}
        feature_counts: Dict[int, int] = {}
        source_energies: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = self.model(x)
                features = self.get_features(x.shape[0], device)
                source_energies.append(self.compute_energy(logits).detach().cpu())
                if features is None:
                    continue
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
            energy_mean = all_energies.mean()
            energy_std = all_energies.std(unbiased=False).clamp_min(1e-6)
            if self.fixed_energy_threshold is not None:
                threshold = torch.tensor(self.fixed_energy_threshold, device=device)
                src = "fixed"
            else:
                q = min(max(self.energy_quantile, 0.0), 1.0)
                threshold = torch.quantile(all_energies, q)
                src = f"quantile={q:.2f}"
            self.source_energy_threshold.copy_(threshold.detach().to(self.source_energy_threshold.device))
            self.source_energy_mean.copy_(energy_mean.detach().to(self.source_energy_mean.device))
            self.source_energy_std.copy_(energy_std.detach().to(self.source_energy_std.device))
            print(
                "[PolicySafeCommit] Energy "
                f"mean={energy_mean.item():.4f}, std={energy_std.item():.4f}, "
                f"threshold={threshold.item():.4f} ({src})"
            )

        if not feature_sums:
            self.source_prototypes = None
            self.target_prototypes = None
            print("[PolicySafeCommit] Warning: no source features captured; prototype operators disabled.")
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
        print(f"[PolicySafeCommit] Source prototypes: shape={tuple(prototypes.shape)}")

    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        t = self.energy_temperature
        return -t * torch.logsumexp(logits / t, dim=1)

    def resolve_energy_threshold(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.fixed_energy_threshold is not None:
            return torch.tensor(self.fixed_energy_threshold, device=device)
        if torch.isfinite(self.source_energy_threshold).item():
            return self.source_energy_threshold.to(device)
        return None

    def compute_energy_z(self, energy: torch.Tensor) -> torch.Tensor:
        if torch.isfinite(self.source_energy_mean).item() and torch.isfinite(self.source_energy_std).item():
            return (energy - self.source_energy_mean.to(energy.device)) / self.source_energy_std.to(energy.device).clamp_min(1e-6)
        return torch.zeros_like(energy)

    def is_severe_energy_ood(
        self,
        energy: torch.Tensor,
        energy_threshold: Optional[torch.Tensor],
        energy_z: torch.Tensor,
    ) -> torch.Tensor:
        severe_by_z = energy_z > self.energy_abstain_z
        if energy_threshold is None:
            return severe_by_z
        severe_by_margin = energy > (energy_threshold.to(energy.device) + self.energy_abstain_margin)
        return severe_by_z | severe_by_margin

    def compute_sal(self, features: torch.Tensor, pred: torch.Tensor, use_source: bool = True) -> torch.Tensor:
        prototypes = self.source_prototypes if use_source else self.target_prototypes
        if prototypes is None:
            if not self._warned_no_prototypes:
                print("[PolicySafeCommit] Warning: prototypes unavailable. Forcing SAL=0.")
                self._warned_no_prototypes = True
            return torch.zeros(features.shape[0], device=features.device)
        features_n = F.normalize(features, p=2, dim=1)
        proto_n = F.normalize(prototypes.to(features.device), p=2, dim=1)
        return (features_n * proto_n[pred]).sum(dim=1)

    def compute_proto_logits(self, features: Optional[torch.Tensor]) -> torch.Tensor:
        if features is None or self.target_prototypes is None:
            device = self.logit_bias.device
            batch = 1 if features is None else features.shape[0]
            return torch.zeros(batch, self.n_classes, device=device)
        features_n = F.normalize(features, p=2, dim=1)
        proto_n = F.normalize(self.target_prototypes.to(features.device), p=2, dim=1)
        return self.prototype_logit_scale * (features_n @ proto_n.t())

    def compute_prototype_margin(self, features: torch.Tensor, use_source: bool = False) -> torch.Tensor:
        prototypes = self.source_prototypes if use_source else self.target_prototypes
        if prototypes is None:
            return torch.zeros(features.shape[0], device=features.device)
        features_n = F.normalize(features, p=2, dim=1)
        proto_n = F.normalize(prototypes.to(features.device), p=2, dim=1)
        sims = features_n @ proto_n.t()
        topk = sims.topk(k=min(2, sims.shape[-1]), dim=-1).values
        return topk[:, 0] if topk.shape[-1] == 1 else topk[:, 0] - topk[:, 1]

    def compute_neuro_score(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.logit_bias.device
        target_model = self.model.model if hasattr(self.model, "model") else self.model
        weights = None
        if hasattr(target_model, "conv_block") and hasattr(target_model.conv_block, "last_eca_weights"):
            weights = target_model.conv_block.last_eca_weights
        if weights is None or self.channel_roles is None:
            return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)

        motor_idx = self.channel_roles.get("motor", [])
        noise_idx = self.channel_roles.get("noise", [])
        if not motor_idx:
            return torch.zeros(weights.shape[0], device=weights.device), torch.ones(weights.shape[0], device=weights.device)
        motor_w = weights[:, motor_idx].mean(dim=1)
        if noise_idx:
            noise_w = weights[:, noise_idx].mean(dim=1)
        else:
            noise_w = torch.zeros_like(motor_w)
        score = (motor_w - noise_w) / (motor_w + noise_w + 1e-6)
        ratio = motor_w / (noise_w + 1e-6)
        return score, ratio

    def compose_logits(
        self,
        classifier_logits: torch.Tensor,
        proto_logits: torch.Tensor,
    ) -> torch.Tensor:
        bias = self.logit_bias.to(classifier_logits.device).reshape(1, -1)
        base_logits = classifier_logits + bias
        alpha = self.prototype_fusion_alpha if self.target_prototypes is not None else 0.0
        if alpha <= 0.0:
            return base_logits
        return (1.0 - alpha) * base_logits + alpha * proto_logits.to(base_logits.device)

    def forward_once(self, x: torch.Tensor) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            classifier_logits = self.model(x)
        features = self.get_features(x.shape[0], x.device)
        if features is None:
            proto_logits = torch.zeros_like(classifier_logits)
        else:
            proto_logits = self.compute_proto_logits(features).to(classifier_logits.device)
        logits = self.compose_logits(classifier_logits, proto_logits)
        return {
            "logits": logits,
            "classifier_logits": classifier_logits,
            "proto_logits": proto_logits,
            "features": features,
        }

    def recent_update_rate(self) -> float:
        if not self.recent_updates:
            return 0.0
        return float(sum(self.recent_updates) / len(self.recent_updates))

    def recent_abstain_rate(self) -> float:
        if not self.recent_abstains:
            return 0.0
        return float(sum(self.recent_abstains) / len(self.recent_abstains))

    def forward(self, x: torch.Tensor, return_debug: bool = False) -> Dict[str, Any]:
        before_forward = self.forward_once(x)
        before_state = self.state_extractor.extract(x, before_forward)

        if self.enable_adaptation:
            candidates = self.policy.propose(before_state)
            commit = self.safe_commit.run(x, before_state, candidates)
        else:
            candidates = ["no_update"]
            commit = self.safe_commit._result(
                before_state,
                before_state,
                operator="no_update",
                committed=False,
                abstained=False,
                rejected=[],
                reject_reasons=[],
                op_result=OperatorResult(name="no_update"),
                reason="adaptation_disabled",
            )

        final_state = commit["final_state"]
        logits = final_state["logits"]
        probs = F.softmax(logits, dim=-1)
        pmax, pred = probs.max(dim=-1)
        original_pred = before_state["pred"]

        adapted = bool(commit["committed"] and commit["operator"] not in ("no_update", "abstain"))
        abstained = bool(commit["abstained"])
        self.recent_updates.append(adapted)
        self.recent_abstains.append(abstained)

        self.stats["total_samples"] += int(x.shape[0])
        self.stats[f"operator/{commit['operator']}"] += int(x.shape[0])
        self.stats["committed_updates"] += int(adapted) * int(x.shape[0])
        self.stats["abstained"] += int(abstained) * int(x.shape[0])
        self.stats["rejected_candidates"] += len(commit["rejected"])

        record = self.make_record(before_state, final_state, candidates, commit)
        self.trial_logs.append(record)

        result = {
            "logits": logits,
            "classifier_logits": final_state["classifier_logits"],
            "proto_logits": final_state["proto_logits"],
            "pred": pred,
            "original_pred": original_pred,
            "pmax": pmax,
            "entropy": final_state["entropy"],
            "margin": final_state["margin"],
            "sal": final_state["sal"],
            "target_sal": final_state["target_sal"],
            "energy_score": final_state["energy"],
            "energy_z": final_state["energy_z"],
            "energy_ood": final_state["energy_ood"],
            "energy_severe_ood": final_state["energy_severe_ood"],
            "neuro_score": final_state["neuro_score"],
            "eca_motor_noise_ratio": final_state["eca_motor_noise_ratio"],
            "prototype_margin": final_state["prototype_margin"],
            "shallow_var_drift_risk": final_state["shallow_var_drift_risk"],
            "adapted": torch.full((x.shape[0],), adapted, dtype=torch.bool, device=x.device),
            "abstained": torch.full((x.shape[0],), abstained, dtype=torch.bool, device=x.device),
            "committed": torch.full((x.shape[0],), bool(commit["committed"]), dtype=torch.bool, device=x.device),
            "operator": commit["operator"],
            "safecommit_reason": commit["reason"],
            "rejected": commit["rejected"],
            "bn_drift_norm": torch.tensor([commit["op_result"].bn_drift_norm], dtype=torch.float32, device=x.device),
            "shallow_var_delta": torch.tensor([commit["op_result"].shallow_var_delta], dtype=torch.float32, device=x.device),
            "log_record": record,
        }
        if return_debug:
            result["features"] = final_state.get("features")
            result["stats"] = dict(self.stats)
        return result

    def make_record(
        self,
        before_state: Dict[str, Any],
        final_state: Dict[str, Any],
        candidates: Sequence[str],
        commit: Dict[str, Any],
    ) -> Dict[str, Any]:
        op_result: OperatorResult = commit["op_result"]
        before_pred = before_state.get("pred")
        final_pred = final_state.get("pred")
        pred_changed = 0.0
        if torch.is_tensor(before_pred) and torch.is_tensor(final_pred):
            pred_changed = float((before_pred.detach().cpu() != final_pred.detach().cpu()).float().mean().item())

        def _delta_norm(name: str) -> float:
            before = before_state.get(name)
            after = final_state.get(name)
            if torch.is_tensor(before) and torch.is_tensor(after):
                return float((after.detach().float().cpu() - before.detach().float().cpu()).norm().item())
            return 0.0

        return {
            "operator": commit["operator"],
            "candidates": "|".join(candidates),
            "rejected": "|".join(commit["rejected"]),
            "reject_reasons": "|".join(commit["reject_reasons"]),
            "safecommit_reason": commit["reason"],
            "committed": _bool_float(commit["committed"]),
            "abstained": _bool_float(commit["abstained"]),
            "pmax_before": _mean_float(before_state.get("pmax")),
            "pmax_after": _mean_float(final_state.get("pmax")),
            "entropy_before": _mean_float(before_state.get("entropy")),
            "entropy_after": _mean_float(final_state.get("entropy")),
            "margin_before": _mean_float(before_state.get("margin")),
            "margin_after": _mean_float(final_state.get("margin")),
            "sal_before": _mean_float(before_state.get("sal")),
            "sal_after": _mean_float(final_state.get("sal")),
            "target_sal_before": _mean_float(before_state.get("target_sal")),
            "target_sal_after": _mean_float(final_state.get("target_sal")),
            "energy_before": _mean_float(before_state.get("energy")),
            "energy_after": _mean_float(final_state.get("energy")),
            "energy_z_before": _mean_float(before_state.get("energy_z")),
            "energy_z_after": _mean_float(final_state.get("energy_z")),
            "energy_excess_before": _mean_float(before_state.get("energy_excess")),
            "energy_excess_after": _mean_float(final_state.get("energy_excess")),
            "energy_ood": _bool_float(
                torch.is_tensor(final_state.get("energy_ood"))
                and final_state["energy_ood"].detach().bool().any().item()
            ),
            "energy_severe_ood": _bool_float(
                torch.is_tensor(final_state.get("energy_severe_ood"))
                and final_state["energy_severe_ood"].detach().bool().any().item()
            ),
            "neuro_score": _mean_float(final_state.get("neuro_score")),
            "eca_motor_noise_ratio": _mean_float(final_state.get("eca_motor_noise_ratio"), default=1.0),
            "prototype_margin_before": _mean_float(before_state.get("prototype_margin")),
            "prototype_margin_after": _mean_float(final_state.get("prototype_margin")),
            "shallow_var_drift_risk": _mean_float(final_state.get("shallow_var_drift_risk")),
            "recent_update_rate": _mean_float(final_state.get("recent_update_rate")),
            "recent_abstain_rate": _mean_float(final_state.get("recent_abstain_rate")),
            "current_trial_pred_changed": pred_changed,
            "final_logits_delta_norm": _delta_norm("logits"),
            "classifier_logits_delta_norm": _delta_norm("classifier_logits"),
            "proto_logits_delta_norm": _delta_norm("proto_logits"),
            "logit_bias_norm": float(self.logit_bias.detach().float().norm().item()),
            "prototype_source_drift_norm": self.prototype_source_drift_norm(),
            "bn_drift_norm": float(op_result.bn_drift_norm),
            "shallow_var_delta": float(op_result.shallow_var_delta),
        }

    def prototype_source_drift_norm(self) -> float:
        if self.source_prototypes is None or self.target_prototypes is None:
            return 0.0
        return float(
            (
                self.target_prototypes.detach().float().cpu()
                - self.source_prototypes.detach().float().cpu()
            ).norm().item()
        )

    def print_stats(self) -> None:
        total = int(self.stats.get("total_samples", 0))
        if total == 0:
            print("[PolicySafeCommit] No samples processed")
            return
        print("[PolicySafeCommit] Adaptation Statistics:")
        print(f"  Total samples: {total}")
        print(f"  Committed updates: {self.stats.get('committed_updates', 0)} ({100.0 * self.stats.get('committed_updates', 0) / total:.1f}%)")
        print(f"  Abstained: {self.stats.get('abstained', 0)} ({100.0 * self.stats.get('abstained', 0) / total:.1f}%)")
        print(f"  Rejected candidates: {self.stats.get('rejected_candidates', 0)}")
        for key, value in sorted(self.stats.items()):
            if key.startswith("operator/"):
                print(f"  {key.replace('operator/', '')}: {value} ({100.0 * value / total:.1f}%)")
