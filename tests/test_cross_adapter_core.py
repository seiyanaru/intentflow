"""Unit tests for the prospective cross-adapter utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "intentflow" / "offline" / "scripts" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from cross_adapter_core import (
    adapt_tent_affine,
    apply_reference,
    ea_reference,
    lower_tail_cvar,
    risk_utility_summary,
    state_dict_cpu,
    state_digest,
    update_adabn,
)


class TinyBNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.classifier = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.bn(x))


def test_prefix_reference_is_independent_of_future_trials() -> None:
    rng = np.random.default_rng(0)
    prefix = rng.normal(size=(8, 3, 20))
    future_a = rng.normal(size=(10, 3, 20))
    future_b = rng.normal(size=(10, 3, 20)) * 100
    reference_a = ea_reference(np.concatenate([prefix, future_a])[: len(prefix)])
    reference_b = ea_reference(np.concatenate([prefix, future_b])[: len(prefix)])
    np.testing.assert_allclose(reference_a, reference_b)
    assert apply_reference(reference_a, prefix).shape == prefix.shape


def test_adabn_changes_running_stats_but_not_weights() -> None:
    model = TinyBNNet()
    before_weight = model.classifier.weight.detach().clone()
    before_mean = model.bn.running_mean.detach().clone()
    update_adabn(model, torch.randn(16, 3) + 4, mixing=0.5)
    assert not torch.equal(before_mean, model.bn.running_mean)
    assert torch.equal(before_weight, model.classifier.weight)


def test_prefix_tent_changes_only_bn_affine_and_is_future_independent() -> None:
    torch.manual_seed(0)
    prefix = torch.randn(16, 3)
    future_a = torch.randn(16, 3)
    future_b = torch.randn(16, 3) * 100
    model_a = TinyBNNet()
    model_b = TinyBNNet()
    model_b.load_state_dict(model_a.state_dict())
    classifier_before = model_a.classifier.weight.detach().clone()
    adapt_tent_affine(model_a, prefix, learning_rate=1e-2, steps=2)
    adapt_tent_affine(model_b, prefix, learning_rate=1e-2, steps=2)
    assert state_digest(state_dict_cpu(model_a)) == state_digest(state_dict_cpu(model_b))
    assert torch.equal(classifier_before, model_a.classifier.weight)
    with torch.no_grad():
        model_a(future_a)
        model_b(future_b)
    assert state_digest(state_dict_cpu(model_a)) == state_digest(state_dict_cpu(model_b))


def test_weighted_lower_tail_and_subject_balanced_summary() -> None:
    values = np.asarray([-10.0, 0.0, 10.0])
    weights = np.asarray([0.2, 0.3, 0.5])
    assert lower_tail_cvar(values, weights, alpha=0.25) == -8.0
    rows = [
        {
            "subject": 1,
            "adapter": "ea",
            "delta_pp": -10.0,
            "source_acc": 80.0,
            "adapted_acc": 70.0,
        },
        {
            "subject": 1,
            "adapter": "ea",
            "delta_pp": 10.0,
            "source_acc": 80.0,
            "adapted_acc": 90.0,
        },
        {
            "subject": 2,
            "adapter": "ea",
            "delta_pp": 4.0,
            "source_acc": 60.0,
            "adapted_acc": 64.0,
        },
    ]
    summary = risk_utility_summary(rows, "ea")
    assert summary["utility_mean_delta_pp"] == 2.0
    assert summary["n_subjects"] == 2
