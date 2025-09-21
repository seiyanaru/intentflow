"""Replay NPZ dataset samples to the ingest WebSocket endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import List

import numpy as np
import websockets


def parse_args() -> argparse.Namespace:
  """Parse command line arguments for the NPZ replay script."""
  parser = argparse.ArgumentParser(description="Replay NPZ samples to /ingest")
  parser.add_argument("--npz", type=str, default="data/processed/mi_dummy.npz", help="NPZ file containing X")
  parser.add_argument("--ws", type=str, default="ws://127.0.0.1:8000/ingest", help="Ingest WebSocket URL")
  parser.add_argument("--fs", type=int, default=250, help="Sampling rate for replay")
  parser.add_argument("--realtime", action="store_true", help="Sleep between chunks to emulate realtime")
  return parser.parse_args()


def load_stream(npz_path: str) -> np.ndarray:
  """Load NPZ data and concatenate trials into a continuous [N_total, C] stream."""
  data = np.load(npz_path)
  X = data["X"].astype(np.float32)
  segments: List[np.ndarray] = [segment.transpose(1, 0) for segment in X]
  return np.concatenate(segments, axis=0)


async def send_stream(stream: np.ndarray, ws_url: str, fs: int, realtime: bool) -> None:
  """Send the stream in fixed-size chunks to the ingest endpoint."""
  chunk_size = max(1, int(0.2 * fs))
  ch = stream.shape[1]
  async with websockets.connect(ws_url) as websocket:
    for start in range(0, stream.shape[0], chunk_size):
      chunk = stream[start : start + chunk_size]
      payload = {
        "fs": fs,
        "ch": ch,
        "samples": chunk.tolist(),
      }
      await websocket.send(json.dumps(payload))
      if realtime:
        await asyncio.sleep(len(chunk) / fs)


def main() -> None:
  """Entry point for replaying NPZ data to /ingest."""
  args = parse_args()
  stream = load_stream(args.npz)
  asyncio.run(send_stream(stream, args.ws, args.fs, args.realtime))


if __name__ == "__main__":
  main()
