"""External memory used by Deferred-Commit Replay OTTA.

This buffer intentionally has the same minimal replay interface as
`ReplayBuffer` (`materialize`, `is_ready`, `class_counts`) so it can be used by
ReplaySafeCommit, while carrying richer metadata for memory admission and
reliability diagnostics.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import torch


class ExternalMemoryBuffer:
    """FIFO external memory of trials, pseudo-labels, and admission metadata."""

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
        # Each item is a dict with x, pseudo_label, source_tag, pmax, step, metadata.
        self._items: Deque[Dict[str, Any]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def is_ready(self) -> bool:
        return len(self._items) >= self.min_size_for_use

    @staticmethod
    def _is_target_tag(tag: str) -> bool:
        return str(tag).startswith("target")

    def _iter_items(self, source_tag: Optional[str] = None):
        for item in self._items:
            tag = str(item.get("source_tag", ""))
            if source_tag is None:
                yield item
            elif source_tag == "target":
                if self._is_target_tag(tag):
                    yield item
            elif tag == source_tag:
                yield item

    def class_counts(self, source_tag: Optional[str] = None) -> List[int]:
        counts = [0] * self.n_classes
        for item in self._iter_items(source_tag=source_tag):
            y = int(item.get("pseudo_label", -1))
            if 0 <= y < self.n_classes:
                counts[y] += 1
        return counts

    def target_class_counts(self) -> List[int]:
        return self.class_counts(source_tag="target")

    def target_size(self) -> int:
        return sum(self.target_class_counts())

    def append(
        self,
        x: torch.Tensor,
        pseudo_label: int,
        source_tag: str,
        pmax: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> None:
        """Append one trial. x is expected to be (1, C, T) or (C, T)."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] != 1:
            raise ValueError(f"external memory expects single-trial x, got shape {tuple(x.shape)}")
        self._items.append(
            {
                "x": x.detach().cpu().clone(),
                "pseudo_label": int(pseudo_label),
                "source_tag": str(source_tag),
                "pmax": float(pmax),
                "metadata": dict(metadata or {}),
                "step": int(step),
            }
        )

    def seed_from_loader(
        self,
        loader: Iterable,
        max_per_class: int,
        device: torch.device,
    ) -> None:
        """Seed memory with class-balanced source train trials."""
        del device  # Stored on CPU; kept for API compatibility with ReplayBuffer.
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
                    self.append(
                        x[i:i + 1],
                        pseudo_label=label,
                        source_tag="warmup_source",
                        pmax=1.0,
                        metadata={
                            "memory_reliability_score": 1.0,
                            "raw_corrected_disagreement": 0.0,
                            "corrected_only": 0.0,
                            "prototype_support": 1.0,
                            "density_support": 1.0,
                            "temporal_consistency": 1.0,
                        },
                    )
                    per_class_added[label] += 1
                if all(c >= max_per_class for c in per_class_added):
                    return

    def materialize(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stacked tensors (X: (N, C, T), y: (N,), pmax: (N,)) on device."""
        if not self._items:
            raise RuntimeError("external memory empty; cannot materialize")
        xs = torch.cat([item["x"] for item in self._items], dim=0).to(device)
        ys = torch.tensor([item["pseudo_label"] for item in self._items], dtype=torch.long, device=device)
        pmax = torch.tensor([item["pmax"] for item in self._items], dtype=torch.float32, device=device)
        return xs, ys, pmax

    def source_tags(self) -> List[str]:
        return [str(item.get("source_tag", "")) for item in self._items]

    def metadata_mean(
        self,
        key: str,
        source_tag: Optional[str] = None,
        default: float = 0.0,
    ) -> float:
        values: List[float] = []
        for item in self._iter_items(source_tag=source_tag):
            meta = item.get("metadata", {})
            value = meta.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            return float(default)
        return float(sum(values) / len(values))

    def class_entropy(self, source_tag: Optional[str] = None, normalized: bool = True) -> float:
        counts = self.class_counts(source_tag=source_tag)
        total = float(sum(counts))
        if total <= 0.0:
            return 0.0
        entropy = 0.0
        for count in counts:
            if count <= 0:
                continue
            p = float(count) / total
            entropy -= p * math.log(p + 1e-12)
        if normalized and self.n_classes > 1:
            entropy /= math.log(float(self.n_classes))
        return float(entropy)

    def summary(self) -> Dict[str, float]:
        """Observable memory reliability diagnostics; no labels required."""
        target_n = float(self.target_size())
        total_n = float(len(self._items))
        return {
            "memory_buffer_size": total_n,
            "memory_target_size": target_n,
            "memory_class_entropy": self.class_entropy(source_tag=None),
            "memory_target_class_entropy": self.class_entropy(source_tag="target"),
            "memory_reliability_proxy": self.metadata_mean(
                "memory_reliability_score",
                source_tag="target",
                default=float("nan"),
            ),
            "memory_raw_corrected_disagreement_rate": self.metadata_mean(
                "raw_corrected_disagreement",
                source_tag="target",
                default=0.0,
            ),
            "memory_corrected_only_rate": self.metadata_mean(
                "corrected_only",
                source_tag="target",
                default=0.0,
            ),
            "memory_proto_support_mean": self.metadata_mean(
                "prototype_support",
                source_tag="target",
                default=float("nan"),
            ),
            "memory_density_support_mean": self.metadata_mean(
                "density_support",
                source_tag="target",
                default=float("nan"),
            ),
            "memory_temporal_consistency_mean": self.metadata_mean(
                "temporal_consistency",
                source_tag="target",
                default=float("nan"),
            ),
        }
