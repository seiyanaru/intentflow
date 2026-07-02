"""Shared prospective utilities for the Stieger cross-adapter experiment."""

from __future__ import annotations

import copy
import hashlib
import random
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.linalg import eigh
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def trial_covariances(x: np.ndarray) -> np.ndarray:
    """Return per-trial covariance matrices for (trial, channel, time)."""
    return np.einsum("nct,ndt->ncd", x, x, optimize=True) / x.shape[-1]


def invsqrt_spd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values, vectors = eigh(matrix)
    values = np.clip(values, eps, None)
    return (vectors * values**-0.5) @ vectors.T


def ea_reference(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Estimate an EA whitening reference using only the supplied trials."""
    return invsqrt_spd(trial_covariances(x.astype(np.float64)).mean(axis=0), eps=eps)


def apply_reference(reference: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njt->nit", reference, x, optimize=True).astype(np.float32)


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_state_dict_copy(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict({key: value.clone() for key, value in state.items()}, strict=True)


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _batch_norms(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


def update_adabn(
    model: nn.Module,
    prefix: torch.Tensor,
    mixing: float,
) -> None:
    """Update BN running statistics once from the prospective prefix."""
    if not 0 < mixing <= 1:
        raise ValueError(f"AdaBN mixing must be in (0, 1], got {mixing}")
    model.eval()
    batch_norms = _batch_norms(model)
    original_momenta = [module.momentum for module in batch_norms]
    for module in batch_norms:
        module.train()
        module.momentum = mixing
    with torch.no_grad():
        model(prefix)
    for module, momentum in zip(batch_norms, original_momenta):
        module.momentum = momentum
    model.eval()


def adapt_tent_affine(
    model: nn.Module,
    prefix: torch.Tensor,
    learning_rate: float,
    steps: int,
) -> list[float]:
    """Prefix-only entropy minimization on BN affine parameters.

    BN layers stay in eval mode, so normalization uses source running
    statistics. Only gamma/beta are changed and evaluation is state-frozen.
    """
    if steps < 1:
        raise ValueError(f"Tent steps must be positive, got {steps}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    affine_parameters: list[nn.Parameter] = []
    for module in _batch_norms(model):
        if module.affine:
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            affine_parameters.extend([module.weight, module.bias])
    if not affine_parameters:
        raise RuntimeError("prefix-Tent requires at least one affine BatchNorm layer")

    optimizer = torch.optim.Adam(affine_parameters, lr=learning_rate)
    losses: list[float] = []
    for _ in range(steps):
        logits = model(prefix)
        probabilities = torch.softmax(logits, dim=1).clamp_min(1e-8)
        loss = -(probabilities * probabilities.log()).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return losses


@torch.no_grad()
def predict_labels(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = quantile * sorted_weights.sum()
    return float(sorted_values[np.searchsorted(cumulative, cutoff, side="left")])


def lower_tail_cvar(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Weighted mean of the lower alpha mass, including a partial boundary."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    target_mass = alpha * sorted_weights.sum()
    used_mass = 0.0
    total = 0.0
    for value, weight in zip(sorted_values, sorted_weights):
        take = min(float(weight), target_mass - used_mass)
        if take <= 0:
            break
        total += float(value) * take
        used_mass += take
    if used_mass <= 0:
        raise ValueError("weights must contain positive mass")
    return total / used_mass


def subject_balanced_weights(subjects: Sequence[int]) -> np.ndarray:
    counts = Counter(int(subject) for subject in subjects)
    n_subjects = len(counts)
    return np.asarray(
        [1.0 / (n_subjects * counts[int(subject)]) for subject in subjects],
        dtype=np.float64,
    )


def risk_utility_summary(
    rows: Iterable[Mapping[str, float | int | str]],
    adapter: str,
    cvar_alpha: float = 0.1,
) -> dict[str, float | int | None | str]:
    selected = [row for row in rows if row["adapter"] == adapter]
    if not selected:
        raise ValueError(f"No rows found for adapter={adapter}")
    subjects = [int(row["subject"]) for row in selected]
    deltas = np.asarray([float(row["delta_pp"]) for row in selected])
    source_acc = np.asarray([float(row["source_acc"]) for row in selected])
    adapted_acc = np.asarray([float(row["adapted_acc"]) for row in selected])
    weights = subject_balanced_weights(subjects)
    weights /= weights.sum()
    lower_cvar = lower_tail_cvar(deltas, weights, alpha=cvar_alpha)
    eligible_70 = source_acc >= 70.0
    crossing_70 = eligible_70 & (adapted_acc < 70.0)
    crossing_probability: float | None = (
        float(np.sum(weights[crossing_70]) / np.sum(weights[eligible_70]))
        if np.any(eligible_70)
        else None
    )
    return {
        "adapter": adapter,
        "n_subjects": len(set(subjects)),
        "n_sessions": len(selected),
        "utility_mean_delta_pp": float(np.sum(weights * deltas)),
        "lower_cvar10_delta_pp": float(lower_cvar),
        "risk_r10": float(-lower_cvar),
        "weighted_q05_delta_pp": weighted_quantile(deltas, weights, 0.05),
        "p_delta_lt_0": float(np.sum(weights[deltas < 0])),
        "p_delta_lt_minus3": float(np.sum(weights[deltas < -3])),
        "p_delta_lt_minus5": float(np.sum(weights[deltas < -5])),
        "p_delta_lt_minus10": float(np.sum(weights[deltas < -10])),
        "p_cross_below_70_given_source_ge_70": crossing_probability,
    }


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)
