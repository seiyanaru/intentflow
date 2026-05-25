"""Deferred-Commit Replay OTTA.

DC-Replay OTTA reframes online adaptation as a three-level policy:

L1. Prediction correction: adjust the current prediction without mutating model
    state.
L2. External memory update: admit reliable target evidence to a replay/memory
    buffer.
L3. Model-state commit: only when persistent drift is supported by memory,
    run a small set of commit candidates through SafeCommit.

The initial implementation keeps L3 deliberately small: prior drift can trigger
`logit_bias_update`, prototype drift can trigger `prototype_update`, and all
other operators remain out of scope until this hierarchy is validated.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.dc_replay_memory import ExternalMemoryBuffer
from models.policy_safe_commit_otta import (
    OperatorResult,
    SafeCommit,
    StateExtractor,
    _bool_float,
    _mean_float,
)
from models.replay_safe_commit_otta import ReplayPolicySafeCommitOTTA


def _as_float_tensor(value: float, batch: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch,), float(value), dtype=torch.float32, device=device)


class DeferredCommitStateExtractor(StateExtractor):
    """StateExtractor that carries L1 correction diagnostics into records."""

    EXTRA_KEYS = (
        "raw_logits",
        "corrected_logits",
        "correction_active",
        "correction_strength",
        "correction_confidence",
        "raw_corrected_disagreement",
        "raw_pmax",
        "corrected_pmax",
        "correction_prior_kl",
        "correction_mode_id",
    )

    def extract(self, x: torch.Tensor, forward_out: Dict[str, Any]) -> Dict[str, Any]:
        state = super().extract(x, forward_out)
        for key in self.EXTRA_KEYS:
            if key in forward_out:
                state[key] = forward_out[key]
        return state


class DeferredCommitReplayOTTA(ReplayPolicySafeCommitOTTA):
    """ReplayPolicySafeCommitOTTA with deferred model-state commit policy."""

    def __init__(
        self,
        *args: Any,
        dc_enable_correction: bool = True,
        dc_correction_mode: str = "memory_prior",
        dc_prior_correction_strength: float = 0.2,
        dc_max_prior_correction: float = 1.0,
        dc_min_memory_for_correction: int = 8,
        dc_enable_memory_update: bool = True,
        dc_memory_admission_threshold: float = 0.55,
        dc_memory_confidence_weight: float = 0.35,
        dc_memory_density_weight: float = 0.15,
        dc_memory_temporal_weight: float = 0.15,
        dc_memory_prototype_weight: float = 0.15,
        dc_memory_balance_weight: float = 0.10,
        dc_memory_uncertainty_weight: float = 0.15,
        dc_memory_ood_weight: float = 0.20,
        dc_memory_disagreement_weight: float = 0.10,
        dc_commit_mode: str = "replay_gated",
        dc_min_memory_for_commit: int = 16,
        dc_commit_cooldown: int = 8,
        dc_prior_drift_threshold: float = 0.08,
        dc_proto_drift_threshold: float = 0.08,
        dc_proto_drift_reference: float = 0.75,
        dc_drift_score_threshold: float = 0.08,
        dc_prior_drift_weight: float = 1.0,
        dc_proto_drift_weight: float = 1.0,
        dc_random_commit_prob: float = 0.05,
        dc_allowed_commit_operators: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if dc_correction_mode not in ("none", "static_prior", "memory_prior"):
            raise ValueError(
                "dc_correction_mode must be one of "
                f"'none', 'static_prior', 'memory_prior', got {dc_correction_mode!r}"
            )
        if dc_commit_mode not in ("none", "replay_gated", "no_replay_gate", "random_sparse"):
            raise ValueError(
                "dc_commit_mode must be one of 'none', 'replay_gated', "
                f"'no_replay_gate', 'random_sparse', got {dc_commit_mode!r}"
            )

        self.dc_enable_correction = bool(dc_enable_correction)
        self.dc_correction_mode = dc_correction_mode
        self.dc_prior_correction_strength = float(dc_prior_correction_strength)
        self.dc_max_prior_correction = float(dc_max_prior_correction)
        self.dc_min_memory_for_correction = int(dc_min_memory_for_correction)

        self.dc_enable_memory_update = bool(dc_enable_memory_update)
        self.dc_memory_admission_threshold = float(dc_memory_admission_threshold)
        self.dc_memory_confidence_weight = float(dc_memory_confidence_weight)
        self.dc_memory_density_weight = float(dc_memory_density_weight)
        self.dc_memory_temporal_weight = float(dc_memory_temporal_weight)
        self.dc_memory_prototype_weight = float(dc_memory_prototype_weight)
        self.dc_memory_balance_weight = float(dc_memory_balance_weight)
        self.dc_memory_uncertainty_weight = float(dc_memory_uncertainty_weight)
        self.dc_memory_ood_weight = float(dc_memory_ood_weight)
        self.dc_memory_disagreement_weight = float(dc_memory_disagreement_weight)

        self.dc_commit_mode = dc_commit_mode
        self.dc_min_memory_for_commit = int(dc_min_memory_for_commit)
        self.dc_commit_cooldown = int(dc_commit_cooldown)
        self.dc_prior_drift_threshold = float(dc_prior_drift_threshold)
        self.dc_proto_drift_threshold = float(dc_proto_drift_threshold)
        self.dc_proto_drift_reference = float(dc_proto_drift_reference)
        self.dc_drift_score_threshold = float(dc_drift_score_threshold)
        self.dc_prior_drift_weight = float(dc_prior_drift_weight)
        self.dc_proto_drift_weight = float(dc_proto_drift_weight)
        self.dc_random_commit_prob = float(dc_random_commit_prob)
        self.dc_allowed_commit_operators = list(
            dc_allowed_commit_operators or ["logit_bias_update", "prototype_update"]
        )

        # Replace the simple replay buffer with richer external memory. The
        # existing ReplaySafeCommit instance reads controller.replay_buffer
        # dynamically, so it will use this buffer without replacement.
        self.replay_buffer = ExternalMemoryBuffer(
            capacity=self.replay_buffer.capacity,
            min_size_for_use=self.replay_buffer.min_size_for_use,
            per_class_min=self.replay_buffer.per_class_min,
            n_classes=self.n_classes,
        )

        self.state_extractor = DeferredCommitStateExtractor(self)
        self.register_buffer("dc_source_class_prior", torch.full((self.n_classes,), 1.0 / self.n_classes))
        self.register_buffer("dc_source_class_counts", torch.zeros(self.n_classes))

        parent_sc = self.safe_commit
        self.tier1_safe_commit = SafeCommit(
            self,
            margin_tolerance=parent_sc.margin_tolerance,
            sal_tolerance=parent_sc.sal_tolerance,
            prototype_margin_tolerance=parent_sc.prototype_margin_tolerance,
            energy_tolerance=parent_sc.energy_tolerance,
            max_bn_drift=parent_sc.max_bn_drift,
            max_shallow_var_delta=parent_sc.max_shallow_var_delta,
            min_neuro_score=parent_sc.min_neuro_score,
        )

        self._dc_trial_index = 0
        self._last_commit_trial = -10**9
        self._last_drift_diagnostics: Dict[str, float] = {}

    def compute_source_statistics(self, dataloader, device: Optional[torch.device] = None) -> None:
        super().compute_source_statistics(dataloader, device=device)
        counts = self.prototype_counts.detach().float().clone().to(self.dc_source_class_counts.device)
        self.dc_source_class_counts.copy_(counts)
        if counts.sum().item() > 0.0:
            prior = counts / counts.sum().clamp_min(1e-6)
        else:
            prior = torch.full_like(counts, 1.0 / self.n_classes)
        self.dc_source_class_prior.copy_(prior.to(self.dc_source_class_prior.device))
        print(
            "[DC-Replay] Source prior: "
            + ", ".join(f"{float(v):.3f}" for v in self.dc_source_class_prior.detach().cpu())
        )

    def _prior_from_counts(self, counts: Sequence[int], device: torch.device) -> torch.Tensor:
        c = torch.tensor(list(counts), dtype=torch.float32, device=device)
        if c.sum().item() <= 0.0:
            return torch.full((self.n_classes,), 1.0 / self.n_classes, device=device)
        return c / c.sum().clamp_min(1e-6)

    def _memory_prior(self, device: torch.device) -> Tuple[torch.Tensor, int]:
        target_counts = self.replay_buffer.target_class_counts()
        target_n = int(sum(target_counts))
        if target_n > 0:
            return self._prior_from_counts(target_counts, device=device), target_n
        return self.dc_source_class_prior.to(device), 0

    def _apply_prediction_correction(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = logits.shape[0]
        device = logits.device
        raw_probs = F.softmax(logits, dim=-1)
        raw_pmax, raw_pred = raw_probs.max(dim=-1)

        correction_vec = torch.zeros(self.n_classes, device=device)
        correction_active = False
        correction_mode_id = 0.0

        if self.dc_enable_correction and self.dc_correction_mode != "none":
            source_prior = self.dc_source_class_prior.to(device).clamp_min(1e-4)
            if self.dc_correction_mode == "static_prior":
                reference = torch.full_like(source_prior, 1.0 / self.n_classes)
                correction_vec = (source_prior.log() - reference.log())
                correction_active = True
                correction_mode_id = 1.0
            else:
                memory_prior, target_n = self._memory_prior(device=device)
                if target_n >= self.dc_min_memory_for_correction:
                    correction_vec = source_prior.log() - memory_prior.clamp_min(1e-4).log()
                    correction_active = True
                    correction_mode_id = 2.0

        if correction_active:
            correction_vec = correction_vec - correction_vec.mean()
            correction_vec = correction_vec.clamp(
                -self.dc_max_prior_correction,
                self.dc_max_prior_correction,
            )
            corrected_logits = logits + self.dc_prior_correction_strength * correction_vec.reshape(1, -1)
        else:
            corrected_logits = logits

        corrected_probs = F.softmax(corrected_logits, dim=-1)
        corrected_pmax, corrected_pred = corrected_probs.max(dim=-1)
        disagreement = (raw_pred != corrected_pred).float()
        correction_strength = (
            corrected_probs.clamp_min(1e-12)
            * (corrected_probs.clamp_min(1e-12).log() - raw_probs.clamp_min(1e-12).log())
        ).sum(dim=-1)
        prior_kl = correction_strength.detach().clone()

        return corrected_logits, {
            "raw_logits": logits,
            "corrected_logits": corrected_logits,
            "correction_active": _as_float_tensor(correction_active, batch, device),
            "correction_strength": correction_strength.detach(),
            "correction_confidence": corrected_pmax.detach(),
            "raw_corrected_disagreement": disagreement.detach(),
            "raw_pmax": raw_pmax.detach(),
            "corrected_pmax": corrected_pmax.detach(),
            "correction_prior_kl": prior_kl,
            "correction_mode_id": _as_float_tensor(correction_mode_id, batch, device),
        }

    def forward_once(self, x: torch.Tensor) -> Dict[str, Any]:
        forward_out = super().forward_once(x)
        base_logits = forward_out["logits"]
        corrected_logits, correction_diag = self._apply_prediction_correction(base_logits)
        forward_out.update(correction_diag)
        forward_out["logits"] = corrected_logits
        return forward_out

    def _class_balance_bonus(self, pred_int: int) -> float:
        before = self.replay_buffer.class_entropy(source_tag="target")
        counts = self.replay_buffer.target_class_counts()
        if 0 <= pred_int < len(counts):
            counts[pred_int] += 1
        total = float(sum(counts))
        if total <= 0.0:
            return 1.0
        entropy = 0.0
        for count in counts:
            if count <= 0:
                continue
            p = float(count) / total
            entropy -= p * math.log(p + 1e-12)
        entropy = entropy / math.log(float(self.n_classes)) if self.n_classes > 1 else 0.0
        # Positive if the new sample improves balance; still keep a small
        # floor so rare classes are encouraged without forcing low-confidence
        # samples into memory.
        return float(max(0.0, min(1.0, 0.5 + entropy - before)))

    def _temporal_consistency(self, pred_int: int) -> float:
        target_counts = self.replay_buffer.target_class_counts()
        target_n = sum(target_counts)
        if target_n <= 0:
            return 0.5
        return float(target_counts[pred_int] / max(target_n, 1))

    def _memory_admission_features(self, x: torch.Tensor, state: Dict[str, Any]) -> Dict[str, float]:
        del x
        pmax = _mean_float(state.get("pmax"))
        entropy = _mean_float(state.get("entropy"))
        max_entropy = math.log(float(self.n_classes)) if self.n_classes > 1 else 1.0
        uncertainty = max(0.0, min(1.0, entropy / max(max_entropy, 1e-6)))
        proto_support = max(0.0, min(1.0, (_mean_float(state.get("target_sal")) + 1.0) * 0.5))
        density_support = proto_support
        disagreement = max(0.0, min(1.0, _mean_float(state.get("raw_corrected_disagreement"))))
        energy_ood = bool(
            torch.is_tensor(state.get("energy_ood"))
            and state["energy_ood"].detach().bool().any().item()
        )
        energy_severe = bool(
            torch.is_tensor(state.get("energy_severe_ood"))
            and state["energy_severe_ood"].detach().bool().any().item()
        )
        ood_risk = 1.0 if energy_severe else (0.5 if energy_ood else 0.0)
        pred = state.get("pred")
        pred_int = int(pred.detach().reshape(-1)[0].item()) if torch.is_tensor(pred) else -1
        temporal = self._temporal_consistency(pred_int) if pred_int >= 0 else 0.0
        balance = self._class_balance_bonus(pred_int) if pred_int >= 0 else 0.0

        score = (
            self.dc_memory_confidence_weight * pmax
            + self.dc_memory_density_weight * density_support
            + self.dc_memory_temporal_weight * temporal
            + self.dc_memory_prototype_weight * proto_support
            + self.dc_memory_balance_weight * balance
            - self.dc_memory_uncertainty_weight * uncertainty
            - self.dc_memory_ood_weight * ood_risk
            - self.dc_memory_disagreement_weight * disagreement
        )
        return {
            "memory_add_score": float(score),
            "memory_confidence": float(pmax),
            "memory_density_support": float(density_support),
            "memory_temporal_consistency": float(temporal),
            "memory_prototype_support": float(proto_support),
            "memory_class_balance_bonus": float(balance),
            "memory_uncertainty": float(uncertainty),
            "memory_ood_risk": float(ood_risk),
            "memory_raw_corrected_disagreement": float(disagreement),
        }

    def _maybe_admit_to_replay(
        self,
        x: torch.Tensor,
        final_state: Dict[str, Any],
    ) -> bool:
        if not self.dc_enable_memory_update or x.shape[0] != 1:
            self._last_memory_admission = {
                "memory_add_score": 0.0,
                "memory_admitted": 0.0,
                "memory_reject_reason": "memory_disabled",
            }
            return False

        features = self._memory_admission_features(x, final_state)
        pred = final_state.get("pred")
        pmax = final_state.get("pmax")
        if not torch.is_tensor(pred) or not torch.is_tensor(pmax):
            self._last_memory_admission = {
                **features,
                "memory_admitted": 0.0,
                "memory_reject_reason": "missing_state",
            }
            return False

        if features["memory_add_score"] < self.dc_memory_admission_threshold:
            self._last_memory_admission = {
                **features,
                "memory_admitted": 0.0,
                "memory_reject_reason": "score_below_threshold",
            }
            return False

        pred_int = int(pred.detach().reshape(-1)[0].item())
        pmax_v = float(pmax.detach().mean().item())
        disagreement = features["memory_raw_corrected_disagreement"]
        source_tag = "target_corrected_only" if disagreement > 0.5 else "target_pseudo"
        metadata = {
            "memory_reliability_score": features["memory_add_score"],
            "raw_corrected_disagreement": disagreement,
            "corrected_only": 1.0 if source_tag == "target_corrected_only" else 0.0,
            "prototype_support": features["memory_prototype_support"],
            "density_support": features["memory_density_support"],
            "temporal_consistency": features["memory_temporal_consistency"],
            "correction_strength": _mean_float(final_state.get("correction_strength")),
            "correction_confidence": _mean_float(final_state.get("correction_confidence")),
        }
        self.replay_buffer.append(
            x.detach(),
            pseudo_label=pred_int,
            source_tag=source_tag,
            pmax=pmax_v,
            metadata=metadata,
            step=self._dc_trial_index,
        )
        self._last_memory_admission = {
            **features,
            "memory_admitted": 1.0,
            "memory_reject_reason": "admitted",
        }
        return True

    def _compute_drift_diagnostics(self) -> Dict[str, float]:
        device = self.logit_bias.device
        target_counts = self.replay_buffer.target_class_counts()
        target_n = int(sum(target_counts))
        source_prior = self.dc_source_class_prior.to(device).clamp_min(1e-6)
        if target_n >= self.dc_min_memory_for_commit:
            target_prior = self._prior_from_counts(target_counts, device=device).clamp_min(1e-6)
            prior_drift = float((target_prior * (target_prior.log() - source_prior.log())).sum().item())
        else:
            prior_drift = 0.0

        proto_support = self.replay_buffer.metadata_mean(
            "prototype_support",
            source_tag="target",
            default=float("nan"),
        )
        if math.isnan(proto_support) or target_n < self.dc_min_memory_for_commit:
            proto_drift = 0.0
        else:
            proto_drift = max(0.0, self.dc_proto_drift_reference - float(proto_support))

        drift_score = (
            self.dc_prior_drift_weight * prior_drift
            + self.dc_proto_drift_weight * proto_drift
        )
        cooldown_remaining = max(
            0,
            self.dc_commit_cooldown - (self._dc_trial_index - self._last_commit_trial),
        )
        return {
            "dc_target_memory_size": float(target_n),
            "dc_prior_drift_score": float(prior_drift),
            "dc_proto_drift_score": float(proto_drift),
            "dc_persistent_drift_score": float(drift_score),
            "dc_commit_cooldown_remaining": float(cooldown_remaining),
        }

    def _propose_dc_commit_candidates(
        self,
        before_state: Dict[str, Any],
        drift: Dict[str, float],
    ) -> Tuple[List[str], str]:
        energy_severe = bool(
            torch.is_tensor(before_state.get("energy_severe_ood"))
            and before_state["energy_severe_ood"].detach().bool().any().item()
        )
        if energy_severe and self.policy.abstain_on_ood:
            return ["abstain"], "energy_severe_ood"

        if self.dc_commit_mode == "none" or not self.enable_adaptation:
            return ["no_update"], "commit_disabled"

        target_n = drift["dc_target_memory_size"]
        if target_n < self.dc_min_memory_for_commit:
            return ["no_update"], "memory_not_ready"

        if drift["dc_commit_cooldown_remaining"] > 0:
            return ["no_update"], "cooldown"

        candidates: List[str] = []
        if (
            drift["dc_prior_drift_score"] >= self.dc_prior_drift_threshold
            and "logit_bias_update" in self.dc_allowed_commit_operators
        ):
            candidates.append("logit_bias_update")
        if (
            drift["dc_proto_drift_score"] >= self.dc_proto_drift_threshold
            and "prototype_update" in self.dc_allowed_commit_operators
        ):
            candidates.append("prototype_update")

        if drift["dc_persistent_drift_score"] < self.dc_drift_score_threshold:
            candidates = []

        if self.dc_commit_mode == "random_sparse":
            draw = float(torch.rand((), device=self.logit_bias.device).item())
            if draw >= self.dc_random_commit_prob:
                return ["no_update"], "random_skip"
            random_candidates = [
                op for op in self.dc_allowed_commit_operators if op in ("logit_bias_update", "prototype_update")
            ]
            return (random_candidates + ["no_update"]) if random_candidates else ["no_update"], "random_sparse"

        if not candidates:
            return ["no_update"], "no_persistent_drift"
        return candidates + ["no_update"], "persistent_drift"

    def _run_commit_policy(
        self,
        x: torch.Tensor,
        before_state: Dict[str, Any],
        candidates: Sequence[str],
    ) -> Dict[str, Any]:
        if self.dc_commit_mode == "replay_gated":
            return self.safe_commit.run(x, before_state, candidates)
        if self.dc_commit_mode in ("no_replay_gate", "random_sparse"):
            return self.tier1_safe_commit.run(x, before_state, candidates)
        return self.safe_commit._result(
            before_state,
            before_state,
            operator="no_update",
            committed=False,
            abstained=False,
            rejected=[],
            reject_reasons=[],
            op_result=OperatorResult(name="no_update"),
            reason="commit_disabled",
        )

    def forward(self, x: torch.Tensor, return_debug: bool = False) -> Dict[str, Any]:
        self._last_memory_admission = {}
        before_forward = self.forward_once(x)
        before_state = self.state_extractor.extract(x, before_forward)

        drift = self._compute_drift_diagnostics()
        candidates, dc_commit_reason = self._propose_dc_commit_candidates(before_state, drift)
        commit = self._run_commit_policy(x, before_state, candidates)

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
        self.stats[f"dc_commit_reason/{dc_commit_reason}"] += int(x.shape[0])

        if adapted:
            self._last_commit_trial = self._dc_trial_index

        admitted = self._maybe_admit_to_replay(x, final_state)
        record = self.make_record(before_state, final_state, candidates, commit)

        record.update(drift)
        record["dc_commit_mode"] = self.dc_commit_mode
        record["dc_commit_reason"] = dc_commit_reason
        record["dc_level_reached"] = 3.0 if adapted else (2.0 if admitted else 1.0)
        record["dc_model_state_committed"] = _bool_float(adapted)
        for k, v in self._last_memory_admission.items():
            record[k] = v
        for k, v in self.replay_buffer.summary().items():
            record[k] = v

        for k, v in (commit.get("replay_diagnostics") or {}).items():
            record[k] = float(v)

        cand_diag = commit.get("replay_candidate_diagnostics") or []
        record["candidate_ops"] = "|".join(str(d.get("op", "")) for d in cand_diag)
        record["candidate_reasons"] = "|".join(str(d.get("reason", "")) for d in cand_diag)
        record["candidate_tiers"] = "|".join(str(d.get("tier", "")) for d in cand_diag)
        record["candidate_sim_scores"] = "|".join(
            f"{d.get('sim_score', float('nan')):.6e}" for d in cand_diag
        )
        record["candidate_passes"] = "|".join(
            f"{int(d.get('replay_safe_pass', 0))}" for d in cand_diag
        )
        record["candidate_selected"] = "|".join(
            f"{int(d.get('selected', 0))}" for d in cand_diag
        )
        record["candidate_count"] = float(len(cand_diag) if cand_diag else len([op for op in candidates if op != "no_update"]))
        record["replay_admitted"] = float(admitted)
        record["replay_buffer_size"] = float(len(self.replay_buffer))

        self.trial_logs.append(record)
        self._dc_trial_index += int(x.shape[0])

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
            "replay_admitted": admitted,
            "replay_buffer_size": len(self.replay_buffer),
        }
        if return_debug:
            result["features"] = final_state.get("features")
            result["stats"] = dict(self.stats)
            result["replay_diagnostics"] = commit.get("replay_diagnostics") or {}
            result["dc_drift_diagnostics"] = dict(drift)
        return result

    def make_record(
        self,
        before_state: Dict[str, Any],
        final_state: Dict[str, Any],
        candidates: Sequence[str],
        commit: Dict[str, Any],
    ) -> Dict[str, Any]:
        record = super().make_record(before_state, final_state, candidates, commit)
        record.update(
            {
                "raw_pmax_before": _mean_float(before_state.get("raw_pmax")),
                "corrected_pmax_before": _mean_float(before_state.get("corrected_pmax")),
                "correction_active": _mean_float(before_state.get("correction_active")),
                "correction_strength": _mean_float(before_state.get("correction_strength")),
                "correction_confidence": _mean_float(before_state.get("correction_confidence")),
                "raw_corrected_disagreement": _mean_float(before_state.get("raw_corrected_disagreement")),
                "correction_prior_kl": _mean_float(before_state.get("correction_prior_kl")),
                "correction_mode_id": _mean_float(before_state.get("correction_mode_id")),
            }
        )
        return record
