"""Replay-Validation Policy-SafeCommit OTTA.

Replaces the immediate-margin SafeCommit gate of `policy_safe_commit_otta`
with a *replay-based* gate: each candidate operator is provisionally
applied, then evaluated on a buffer of recent high-confidence trials.
Commit only if the simulated reward (Δreplay_acc + Δreplay_margin +
Δreplay_proto_cos) is **strictly greater than `sim_score_tolerance`** AND
the existing margin/sal/proto guards pass.

Why: in `policy_safe_commit_otta` we measured pred_changed=0 on every
trial — the immediate `is_safe` checks compare model output on the same
x before vs after the update, which is empty by construction in
forward-only OTTA. The real effect of an update is on *future* trials.
A small replay buffer of high-confidence target-session trials (with
pmax+SAL double-gated pseudo-labels) lets us sample that future effect
on the spot.

Inheritance: reuses `PolicySafeCommitOTTA`'s state extraction, policy,
operator bank, and bookkeeping. Only the SafeCommit class is replaced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.policy_safe_commit_otta import (
    OperatorResult,
    PolicySafeCommitOTTA,
    SafeCommit,
)
from models.replay_buffer import ReplayBuffer


class ReplaySafeCommit(SafeCommit):
    """SafeCommit variant that evaluates each operator on a replay buffer.

    The simulated reward combines three deltas (after − before) on the buffer:
      - replay_acc:     fraction of trials whose argmax matches pseudo-label
      - replay_margin:  mean (top1 - top2) of softmax outputs
      - replay_proto:   mean cosine similarity to target prototype of pseudo-label

    sim_score = w_acc·Δacc + w_margin·Δmargin + w_proto·Δproto

    A candidate is committed iff sim_score is **strictly greater than**
    `sim_score_tolerance` (default 0.0) AND the inherited immediate guards
    (margin/sal/proto/energy/bn_drift) all pass. sim_score == 0 (no
    measurable improvement) does not count as safe.
    """

    def __init__(
        self,
        controller: "ReplayPolicySafeCommitOTTA",
        sim_score_tolerance: float = 0.0,
        replay_acc_weight: float = 1.0,
        replay_margin_weight: float = 0.3,
        replay_proto_weight: float = 0.3,
        require_replay_ready: bool = True,
        log_replay_diagnostics: bool = True,
        replay_weight_mode: str = "uniform",
        select_best_candidate: bool = False,
        **kwargs: Any,
    ):
        super().__init__(controller, **kwargs)
        self.replay_controller = controller
        self.sim_score_tolerance = float(sim_score_tolerance)
        self.replay_acc_weight = float(replay_acc_weight)
        self.replay_margin_weight = float(replay_margin_weight)
        self.replay_proto_weight = float(replay_proto_weight)
        self.require_replay_ready = bool(require_replay_ready)
        self.log_replay_diagnostics = bool(log_replay_diagnostics)
        self.select_best_candidate = bool(select_best_candidate)
        if replay_weight_mode not in ("uniform", "pmax_class_inv"):
            raise ValueError(
                f"replay_weight_mode must be 'uniform' or 'pmax_class_inv', got {replay_weight_mode!r}"
            )
        # H6: pmax_class_inv weights each buffer trial by
        #   w_i = pmax_i * inv_freq(pseudo_label_i)
        # so rare-class outcomes dominate sim_score; suppresses majority-class drift.
        self.replay_weight_mode = replay_weight_mode
        self._last_replay_diagnostics: Dict[str, float] = {}
        # Per-candidate trace within a single trial. Each entry is a dict
        # {op, reason, sim_score, replay_safe_pass, ...}. Cleared on run()
        # so we can prove which rejected candidates were stopped by replay.
        self._candidate_diagnostics: List[Dict[str, Any]] = []

    def reset_diagnostics(self) -> None:
        self._last_replay_diagnostics = {}
        self._candidate_diagnostics = []

    def get_candidate_diagnostics(self) -> List[Dict[str, Any]]:
        return list(self._candidate_diagnostics)

    def get_last_replay_diagnostics(self) -> Dict[str, float]:
        return dict(self._last_replay_diagnostics)

    def _mark_selected_candidate(self, operator: str) -> None:
        """Mark which passing candidate was actually committed.

        In first-safe mode this is the first passing candidate. In best-safe
        mode several candidates may pass; the selected one is the max-sim_score
        candidate. The flattened diagnostics use this for candidate_selected.
        """
        for diag in self._candidate_diagnostics:
            diag["selected"] = 0.0
        for diag in reversed(self._candidate_diagnostics):
            if diag.get("op") == operator and float(diag.get("replay_safe_pass", 0.0)) > 0.5:
                diag["selected"] = 1.0
                selected_diag = {
                    k: v
                    for k, v in diag.items()
                    if k not in ("op", "reason", "tier", "selected")
                }
                if self.log_replay_diagnostics:
                    self._last_replay_diagnostics = selected_diag
                return

    def _compute_buffer_weights(self, ys: torch.Tensor, pmax: torch.Tensor) -> torch.Tensor:
        """Per-trial weights used to aggregate replay metrics.

        - 'uniform': all trials weighted 1/N (existing behaviour).
        - 'pmax_class_inv': w_i = pmax_i * inv_freq(class_i), normalised.
          inv_freq(c) = N / (n_c + 1)  where n_c = trials of class c in buffer.
          Adding 1 keeps zero-count classes from blowing up.

        Returns a tensor of shape (N,) on the same device as `ys`.
        """
        n = ys.shape[0]
        if n == 0:
            return torch.empty(0, device=ys.device)
        if self.replay_weight_mode == "uniform":
            return torch.full((n,), 1.0 / n, device=ys.device, dtype=torch.float32)

        # pmax_class_inv
        ctrl = self.replay_controller
        n_classes = ctrl.n_classes
        # class-frequency in buffer
        counts = torch.bincount(ys.clamp(min=0, max=n_classes - 1), minlength=n_classes).to(torch.float32)
        inv_freq = float(n) / (counts + 1.0)  # (n_classes,)
        w = pmax * inv_freq[ys]
        w_sum = w.sum()
        if w_sum.item() <= 0.0:
            return torch.full((n,), 1.0 / n, device=ys.device, dtype=torch.float32)
        return w / w_sum

    def _evaluate_replay(self) -> Optional[Dict[str, float]]:
        """Forward the replay buffer through the *fused* output path.

        We call `controller.forward_once`, which applies `compose_logits`
        (logit_bias + prototype fusion) on top of the classifier output,
        so logit_bias_update / prototype_update / α-fusion all show up
        in replay_acc and replay_margin. Without this, the SafeCommit
        signal would be blind to those operators.

        Aggregation uses `replay_weight_mode`: 'uniform' (default) or
        'pmax_class_inv' (H6: pmax × inv-frequency weighting).

        Returns None if buffer is not ready.
        """
        ctrl = self.replay_controller
        buf = ctrl.replay_buffer
        if buf is None or not buf.is_ready():
            return None

        device = next(ctrl.model.parameters()).device
        xs, ys, pmax_buf = buf.materialize(device=device)
        with torch.no_grad():
            forward_out = ctrl.forward_once(xs)
        logits = forward_out["logits"]
        features = forward_out.get("features")

        probs = F.softmax(logits, dim=-1)
        topk = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
        margin = (
            topk[:, 0] - topk[:, 1]
            if topk.shape[-1] >= 2
            else topk[:, 0]
        )
        correct = (logits.argmax(dim=-1) == ys).float()  # (N,)

        weights = self._compute_buffer_weights(ys, pmax_buf)
        replay_acc = float((weights * correct).sum().item())
        replay_margin = float((weights * margin).sum().item())

        replay_proto_cos = 0.0
        if ctrl.target_prototypes is not None and features is not None:
            feats_n = F.normalize(features, p=2, dim=1)
            proto_n = F.normalize(ctrl.target_prototypes.to(feats_n.device), p=2, dim=1)
            cos_full = feats_n @ proto_n.t()  # (N, C)
            gather_idx = ys.clamp(min=0, max=cos_full.shape[1] - 1).unsqueeze(-1)
            per_trial_cos = cos_full.gather(1, gather_idx).squeeze(-1)
            replay_proto_cos = float((weights * per_trial_cos).sum().item())

        return {
            "replay_acc": replay_acc,
            "replay_margin": replay_margin,
            "replay_proto_cos": replay_proto_cos,
            "replay_buffer_size": float(len(buf)),
        }

    def is_safe(
        self,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        op_result: OperatorResult,
    ) -> Tuple[bool, str]:
        """Override: tier-1 immediate guards from parent + tier-2 replay.

        Each call appends one entry to `_candidate_diagnostics`, so the
        controller can prove later which rejected candidates were stopped
        at which tier and with what sim_score.
        """
        op_name = op_result.name

        # Tier 1: parent's immediate margin/sal/proto/energy/bn checks.
        ok, reason = super().is_safe(before_state, after_state, op_result)
        if not ok:
            tier1_diag = {
                "replay_status": 0.0,  # tier-1 reject; replay was not consulted
                "replay_safe_pass": 0.0,
            }
            if self.log_replay_diagnostics:
                self._last_replay_diagnostics = tier1_diag
            self._candidate_diagnostics.append({
                "op": op_name,
                "reason": reason,
                "tier": "tier1",
                **tier1_diag,
            })
            return False, reason

        # Tier 2: replay-based simulated reward.
        before_replay = self._before_replay
        after_replay = self._evaluate_replay()
        if before_replay is None or after_replay is None:
            diag = {
                "replay_status": -1.0,
                "replay_safe_pass": 0.0,
            }
            if self.require_replay_ready:
                if self.log_replay_diagnostics:
                    self._last_replay_diagnostics = diag
                self._candidate_diagnostics.append({
                    "op": op_name,
                    "reason": "replay_not_ready",
                    "tier": "tier2",
                    **diag,
                })
                return False, "replay_not_ready"
            else:
                if self.log_replay_diagnostics:
                    self._last_replay_diagnostics = diag
                self._candidate_diagnostics.append({
                    "op": op_name,
                    "reason": "safe_no_replay",
                    "tier": "tier2",
                    **diag,
                })
                return True, "safe_no_replay"

        d_acc = after_replay["replay_acc"] - before_replay["replay_acc"]
        d_mar = after_replay["replay_margin"] - before_replay["replay_margin"]
        d_pro = after_replay["replay_proto_cos"] - before_replay["replay_proto_cos"]
        sim_score = (
            self.replay_acc_weight * d_acc
            + self.replay_margin_weight * d_mar
            + self.replay_proto_weight * d_pro
        )

        # Strict-positive gate: sim_score must STRICTLY EXCEED tolerance.
        # The tolerance defaults to 0.0; sim_score == 0 (no measurable
        # improvement on the replay buffer) does NOT count as safe.
        passed = sim_score > self.sim_score_tolerance

        diag = {
            "replay_status": 1.0,
            "replay_buffer_size": after_replay["replay_buffer_size"],
            "replay_acc_before": before_replay["replay_acc"],
            "replay_acc_after": after_replay["replay_acc"],
            "replay_margin_before": before_replay["replay_margin"],
            "replay_margin_after": after_replay["replay_margin"],
            "replay_proto_cos_before": before_replay["replay_proto_cos"],
            "replay_proto_cos_after": after_replay["replay_proto_cos"],
            "replay_d_acc": d_acc,
            "replay_d_margin": d_mar,
            "replay_d_proto_cos": d_pro,
            "sim_score": sim_score,
            "replay_safe_pass": 1.0 if passed else 0.0,
        }
        if self.log_replay_diagnostics:
            self._last_replay_diagnostics = diag

        final_reason = "safe_replay" if passed else "replay_sim_score_drop"
        self._candidate_diagnostics.append({
            "op": op_name,
            "reason": final_reason,
            "tier": "tier2",
            **diag,
        })
        if not passed:
            return False, "replay_sim_score_drop"
        return True, "safe_replay"

    def run(
        self,
        x: torch.Tensor,
        before_state: Dict[str, Any],
        candidates: Sequence[str],
    ) -> Dict[str, Any]:
        """Same loop as parent but precomputes the replay baseline once.

        We reset _last_replay_diagnostics at the start so that no_update /
        abstain / all-rejected paths do not inherit a previous trial's or
        a stale candidate's replay metrics.
        """
        self.reset_diagnostics()
        # Pre-commit replay baseline (model state before any operator apply).
        self._before_replay = self._evaluate_replay()
        if self.select_best_candidate:
            result = self._run_best_candidate(x, before_state, candidates)
        else:
            result = super().run(x, before_state, candidates)
            if result.get("committed"):
                self._mark_selected_candidate(str(result.get("operator", "")))
        # The diagnostics dict reflects the LAST is_safe() call within the
        # parent's candidate loop. For abstain/no_update/all-rejected outcomes
        # this stays empty (or holds the last attempted candidate's tier-1
        # reject diag); for committed outcomes it holds that operator's
        # full replay metrics.
        result["replay_diagnostics"] = self.get_last_replay_diagnostics()
        # Per-candidate trace covering every is_safe() call this trial.
        # Used by the controller to flatten into the trial record so we can
        # prove "replay rejected operator X with sim_score Y before
        # accepting operator Z" in P1 analysis.
        result["replay_candidate_diagnostics"] = self.get_candidate_diagnostics()
        return result

    def _run_best_candidate(
        self,
        x: torch.Tensor,
        before_state: Dict[str, Any],
        candidates: Sequence[str],
    ) -> Dict[str, Any]:
        """Evaluate every update candidate and commit the max-sim_score pass.

        The parent SafeCommit is first-safe: once an operator passes, it
        commits immediately. Replay validation gives us a scalar simulated
        reward per candidate, so best-safe mode uses that score for action
        selection while preserving the same tier-1/tier-2 safety criteria.
        """
        rejected: List[str] = []
        reject_reasons: List[str] = []
        saw_no_update = False
        best: Optional[Dict[str, Any]] = None

        for operator in candidates:
            if operator == "abstain":
                # Severe OOD policy paths should remain immediate.
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
                saw_no_update = True
                continue

            before_snapshot = self.controller.operator_bank.snapshot()
            op_result = self.controller.operator_bank.apply(operator, x, before_state)
            after_forward = self.controller.forward_once(x)
            after_state = self.controller.state_extractor.extract(x, after_forward)
            safe, reason = self.is_safe(before_state, after_state, op_result)
            diag_index = len(self._candidate_diagnostics) - 1
            diag = self._candidate_diagnostics[diag_index] if diag_index >= 0 else {}

            if safe:
                selected_snapshot = self.controller.operator_bank.snapshot()
                sim_score = float(diag.get("sim_score", 0.0))
                if best is None or sim_score > float(best["sim_score"]):
                    best = {
                        "operator": operator,
                        "after_state": after_state,
                        "op_result": op_result,
                        "reason": reason,
                        "snapshot": selected_snapshot,
                        "sim_score": sim_score,
                        "diag_index": diag_index,
                    }
                self.controller.operator_bank.restore(before_snapshot)
                continue

            self.controller.operator_bank.restore(before_snapshot)
            rejected.append(operator)
            reject_reasons.append(reason)

        if best is not None:
            self.controller.operator_bank.restore(best["snapshot"])
            for diag in self._candidate_diagnostics:
                diag["selected"] = 0.0
            diag_index = int(best["diag_index"])
            if 0 <= diag_index < len(self._candidate_diagnostics):
                self._candidate_diagnostics[diag_index]["selected"] = 1.0
                selected_diag = {
                    k: v
                    for k, v in self._candidate_diagnostics[diag_index].items()
                    if k not in ("op", "reason", "tier", "selected")
                }
                if self.log_replay_diagnostics:
                    self._last_replay_diagnostics = selected_diag
            return self._result(
                before_state,
                best["after_state"],
                operator=str(best["operator"]),
                committed=True,
                abstained=False,
                rejected=rejected,
                reject_reasons=reject_reasons,
                op_result=best["op_result"],
                reason=str(best["reason"]),
            )

        reason = "policy_no_update" if saw_no_update else "all_candidates_rejected"
        return self._result(
            before_state,
            before_state,
            operator="no_update",
            committed=False,
            abstained=False,
            rejected=rejected,
            reject_reasons=reject_reasons,
            op_result=OperatorResult(name="no_update"),
            reason=reason,
        )


class ReplayPolicySafeCommitOTTA(PolicySafeCommitOTTA):
    """PolicySafeCommitOTTA + replay buffer + ReplaySafeCommit gating."""

    def __init__(
        self,
        *args: Any,
        replay_capacity: int = 32,
        replay_min_size: int = 16,
        replay_per_class_min: int = 2,
        replay_pmax_threshold: float = 0.85,
        replay_sal_threshold: float = 0.6,
        replay_seed_per_class: int = 8,
        replay_warmup_source: bool = True,
        sim_score_tolerance: float = 0.0,
        replay_acc_weight: float = 1.0,
        replay_margin_weight: float = 0.3,
        replay_proto_weight: float = 0.3,
        require_replay_ready: bool = True,
        replay_weight_mode: str = "uniform",
        select_best_candidate: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.replay_pmax_threshold = float(replay_pmax_threshold)
        self.replay_sal_threshold = float(replay_sal_threshold)
        self.replay_seed_per_class = int(replay_seed_per_class)
        self.replay_warmup_source = bool(replay_warmup_source)

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            min_size_for_use=replay_min_size,
            per_class_min=replay_per_class_min,
            n_classes=self.n_classes,
        )

        # Replace SafeCommit with the replay-aware variant.
        # Keep the parent's tolerances for tier-1 guards.
        parent_sc = self.safe_commit
        self.safe_commit = ReplaySafeCommit(
            controller=self,
            margin_tolerance=parent_sc.margin_tolerance,
            sal_tolerance=parent_sc.sal_tolerance,
            prototype_margin_tolerance=parent_sc.prototype_margin_tolerance,
            energy_tolerance=parent_sc.energy_tolerance,
            max_bn_drift=parent_sc.max_bn_drift,
            max_shallow_var_delta=parent_sc.max_shallow_var_delta,
            min_neuro_score=parent_sc.min_neuro_score,
            sim_score_tolerance=sim_score_tolerance,
            replay_acc_weight=replay_acc_weight,
            replay_margin_weight=replay_margin_weight,
            replay_proto_weight=replay_proto_weight,
            require_replay_ready=require_replay_ready,
            replay_weight_mode=replay_weight_mode,
            select_best_candidate=select_best_candidate,
        )

    def compute_source_statistics(self, dataloader, device: Optional[torch.device] = None) -> None:
        super().compute_source_statistics(dataloader, device=device)
        if self.replay_warmup_source:
            target_device = device if device is not None else next(self.model.parameters()).device
            self.replay_buffer.seed_from_loader(
                dataloader,
                max_per_class=self.replay_seed_per_class,
                device=target_device,
            )
            print(
                f"[ReplaySafeCommit] Replay buffer seeded from source. "
                f"size={len(self.replay_buffer)}, class_counts={self.replay_buffer.class_counts()}"
            )

    def _maybe_admit_to_replay(
        self,
        x: torch.Tensor,
        final_state: Dict[str, Any],
    ) -> bool:
        """Admit current trial to replay buffer if pmax & SAL pass thresholds.

        Uses post-commit (final) pmax / sal (the values the controller will
        report). pseudo_label is the committed pred.
        """
        pmax = final_state.get("pmax")
        sal = final_state.get("sal")
        pred = final_state.get("pred")
        if pmax is None or sal is None or pred is None:
            return False
        # Single-trial scalars expected.
        pmax_v = float(pmax.detach().mean().item())
        sal_v = float(sal.detach().mean().item())
        if pmax_v < self.replay_pmax_threshold or sal_v < self.replay_sal_threshold:
            return False
        pred_int = int(pred.detach().reshape(-1)[0].item())
        if x.shape[0] != 1:
            return False
        self.replay_buffer.append(
            x.detach(),
            pseudo_label=pred_int,
            source_tag="target_pseudo",
            pmax=pmax_v,
        )
        return True

    def forward(self, x: torch.Tensor, return_debug: bool = False) -> Dict[str, Any]:
        """Same control flow as parent, plus replay buffer admission and
        injection of replay diagnostics into the trial record."""
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
        # Attach replay diagnostics for the COMMITTED candidate (or last
        # tier-1-rejected candidate for all-rejected outcomes).
        for k, v in (commit.get("replay_diagnostics") or {}).items():
            record[k] = float(v)

        # Attach per-candidate trace as pipe-separated columns. Each list
        # has one entry per is_safe() call this trial, in candidate order.
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
        record["candidate_count"] = float(len(cand_diag))

        # Attempt to admit current trial to replay buffer (post-commit).
        admitted = self._maybe_admit_to_replay(x, final_state)
        record["replay_admitted"] = float(admitted)
        # Use the contracted column name `replay_buffer_size` so analysis
        # scripts and pre-registered thresholds can rely on a single key.
        record["replay_buffer_size"] = float(len(self.replay_buffer))

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
            "replay_admitted": admitted,
            "replay_buffer_size": len(self.replay_buffer),
        }
        if return_debug:
            result["features"] = final_state.get("features")
            result["stats"] = dict(self.stats)
            result["replay_diagnostics"] = commit.get("replay_diagnostics") or {}
        return result
