"""Replay buffer for trial-replay validation in OTTA.

Holds a small FIFO of recent high-confidence trials with pseudo-labels.
The buffer is used by ReplaySafeCommit to evaluate candidate adaptation
operators against a fixed validation set drawn from the target session.

The buffer is intentionally small (default K=32) and pseudo-labelled —
we accept that pseudo-labels are noisy. The empirical justification is
that pmax+SAL double-gated trials have ~97% true accuracy on S2
(measured on the 260429 policy_safe_otta run, non-abstain trials).

Shape contract: each stored x has shape (1, n_channels, n_samples) on
the device of the model. The buffer stacks them lazily for forward.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List, Optional, Tuple

import torch


class ReplayBuffer:
    """FIFO buffer of (x, pseudo_label, source_tag, pmax) tuples.

    source_tag in {"warmup_source", "target_pseudo"}; carried for diagnostics
    only, not used by selection logic. pmax is admission-time confidence;
    used by H6 (pmax × class-inverse-frequency) weighting.
    """

    def __init__(
        self,
        capacity: int = 32,
        min_size_for_use: int = 16,
        per_class_min: int = 2,
        n_classes: int = 4,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if min_size_for_use > capacity:
            raise ValueError(
                f"min_size_for_use={min_size_for_use} exceeds capacity={capacity}"
            )
        self.capacity = int(capacity)
        self.min_size_for_use = int(min_size_for_use)
        self.per_class_min = int(per_class_min)
        self.n_classes = int(n_classes)
        # (x, pseudo_label, source_tag, pmax)
        self._items: Deque[Tuple[torch.Tensor, int, str, float]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def is_ready(self) -> bool:
        return len(self._items) >= self.min_size_for_use

    def class_counts(self) -> List[int]:
        counts = [0] * self.n_classes
        for _, y, _, _ in self._items:
            if 0 <= y < self.n_classes:
                counts[y] += 1
        return counts

    def append(
        self,
        x: torch.Tensor,
        pseudo_label: int,
        source_tag: str,
        pmax: float = 1.0,
    ) -> None:
        """Append one trial. x is expected to be (1, C, T) or (C, T)."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] != 1:
            raise ValueError(f"replay buffer expects single-trial x, got shape {tuple(x.shape)}")
        self._items.append(
            (x.detach().cpu().clone(), int(pseudo_label), str(source_tag), float(pmax))
        )

    def seed_from_loader(
        self,
        loader: Iterable,
        max_per_class: int,
        device: torch.device,
    ) -> None:
        """Seed the buffer with class-balanced source train trials.

        Source-seeded trials are anchored at pmax=1.0 (we trust true labels).
        """
        per_class_added = [0] * self.n_classes
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0], batch[1]
                for i in range(x.shape[0]):
                    label = int(y[i].item())
                    if label < 0 or label >= self.n_classes:
                        continue
                    if per_class_added[label] >= max_per_class:
                        continue
                    if len(self._items) >= self.capacity:
                        return
                    self.append(x[i:i + 1], pseudo_label=label, source_tag="warmup_source", pmax=1.0)
                    per_class_added[label] += 1
                if all(c >= max_per_class for c in per_class_added):
                    return

    def materialize(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stacked tensors (X: (N, C, T), y: (N,), pmax: (N,)) on `device`."""
        if not self._items:
            raise RuntimeError("replay buffer empty; cannot materialize")
        xs = torch.cat([item[0] for item in self._items], dim=0).to(device)
        ys = torch.tensor([item[1] for item in self._items], dtype=torch.long, device=device)
        pmax = torch.tensor([item[3] for item in self._items], dtype=torch.float32, device=device)
        return xs, ys, pmax

    def source_tags(self) -> List[str]:
        return [tag for _, _, tag, _ in self._items]
