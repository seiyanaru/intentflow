"""Simple JSONL logger for recording streaming samples."""
# TODO(P1): Extend replay to feed logged data back through /ingest for validation.

from __future__ import annotations

import json
import time
from typing import IO, Optional, Sequence


class Logger:
  """Write streaming samples to a JSON Lines file."""

  def __init__(self, path: str) -> None:
    """Open the target file for logging."""
    self.path = path
    self._fh: Optional[IO[str]] = open(path, "w", encoding="utf-8")

  def write_sample(self, row: Sequence[float]) -> None:
    """Append a single sample row to the log file."""
    if self._fh is None:
      raise RuntimeError("Logger is not open")
    record = {"t": time.time(), "x": list(row)}
    self._fh.write(json.dumps(record) + "\n")
    self._fh.flush()

  def close(self) -> None:
    """Close the underlying file handle."""
    if self._fh is not None:
      self._fh.close()
      self._fh = None

  def __enter__(self) -> "Logger":
    """Enter the logging context and return self."""
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    """Ensure the log file is closed on context exit."""
    self.close()
