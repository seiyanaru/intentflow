"""Replay intent CSV files to the control WebSocket endpoint."""
# TODO(P1): Stream recorded runs directly through /ingest for closed-loop validation.

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import websockets
from websockets import ConnectionClosed, WebSocketException


def parse_bool(value: str) -> bool:
  """Parse a truthy/falsy string into a boolean value."""
  truthy = {"true", "1", "yes", "y"}
  falsy = {"false", "0", "no", "n"}
  lowered = value.lower()
  if lowered in truthy:
    return True
  if lowered in falsy:
    return False
  raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments for the intent replay utility."""
  parser = argparse.ArgumentParser(description="Replay intents from CSV")
  parser.add_argument("path", type=Path, help="CSV file with ts,intent,conf columns")
  parser.add_argument("--ws", type=str, default="ws://127.0.0.1:8000/control", help="Control WebSocket URL")
  parser.add_argument("--realtime", type=parse_bool, default=True, help="Replay using timestamp gaps (true/false)")
  return parser.parse_args()


def load_rows(path: Path) -> List[Tuple[float, str, float]]:
  """Load CSV rows as (timestamp, intent, confidence) tuples."""
  rows: List[Tuple[float, str, float]] = []
  with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for line in reader:
      try:
        ts = float(line["ts"])
        intent = str(line["intent"]).strip()
        conf = float(line["conf"])
      except (KeyError, TypeError, ValueError) as exc:
        print(f"Skipping malformed row {line}: {exc}")
        continue
      rows.append((ts, intent, conf))
  rows.sort(key=lambda item: item[0])
  return rows


async def send_rows(rows: Sequence[Tuple[float, str, float]], ws_url: str, realtime: bool) -> None:
  """Send rows over websockets, respecting timing when requested."""
  index = 0
  prev_ts: float | None = None
  delay_default = 0.05
  while index < len(rows):
    try:
      async with websockets.connect(ws_url) as websocket:
        print(f"Connected to {ws_url}")
        prev_ts = None
        while index < len(rows):
          ts_value, intent, conf = rows[index]
          if realtime and prev_ts is not None:
            wait_time = max(0.0, ts_value - prev_ts)
            if wait_time:
              await asyncio.sleep(wait_time)
          elif not realtime:
            await asyncio.sleep(delay_default)
          prev_ts = ts_value
          payload = json.dumps({
            "type": "intent",
            "intent": intent,
            "conf": conf,
            "ts": time.time(),
            "protocol_version": 1,
          })
          try:
            await websocket.send(payload)
          except (ConnectionClosed, WebSocketException, OSError) as exc:
            print(f"Send error: {exc}")
            break
          index += 1
        else:
          return
    except asyncio.CancelledError:
      raise
    except (ConnectionClosed, WebSocketException, OSError) as exc:
      print(f"Connection error: {exc}. Retrying in 3s")
      await asyncio.sleep(3)
    except Exception as exc:  # noqa: BLE001
      print(f"Unexpected error: {exc}. Retrying in 3s")
      await asyncio.sleep(3)


def main() -> None:
  """Entrypoint for replaying intents to the control channel."""
  args = parse_args()
  rows = load_rows(args.path)
  if not rows:
    print("No valid rows found in CSV; exiting.")
    return
  asyncio.run(send_rows(rows, args.ws, args.realtime))


if __name__ == "__main__":
  main()
