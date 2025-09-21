"""Send alternating intent messages to the control WebSocket endpoint."""
# TODO(P1): Allow configuring payloads and metadata from the command line.

from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets
from websockets import ConnectionClosed, WebSocketException


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments for the demo generator."""
  parser = argparse.ArgumentParser(description="Generate alternating intents")
  parser.add_argument("--ws", type=str, default="ws://127.0.0.1:8000/control", help="Control WebSocket URL")
  parser.add_argument("--hz", type=int, default=20, help="Send rate in Hertz")
  return parser.parse_args()


async def send_intents(ws_url: str, hz: int) -> None:
  """Connect to the control channel and stream alternating intents."""
  delay = 1.0 / max(1, hz)
  intents = ["left", "right"]
  index = 0
  while True:
    try:
      async with websockets.connect(ws_url) as websocket:
        print(f"Connected to {ws_url}")
        while True:
          intent = intents[index % len(intents)]
          index += 1
          payload = json.dumps({
            "type": "intent",
            "intent": intent,
            "conf": 0.9,
            "ts": time.time(),
            "protocol_version": 1,
          })
          try:
            await websocket.send(payload)
          except (ConnectionClosed, WebSocketException, OSError) as exc:
            print(f"Send error: {exc}")
            break
          await asyncio.sleep(delay)
    except asyncio.CancelledError:
      raise
    except (ConnectionClosed, WebSocketException, OSError) as exc:
      print(f"Connection error: {exc}. Retrying in 3s")
      await asyncio.sleep(3)
    except Exception as exc:  # noqa: BLE001
      print(f"Unexpected error: {exc}. Retrying in 3s")
      await asyncio.sleep(3)


def main() -> None:
  """Run the asynchronous intent generator."""
  args = parse_args()
  asyncio.run(send_intents(args.ws, args.hz))


if __name__ == "__main__":
  main()
