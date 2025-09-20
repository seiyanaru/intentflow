"""Simulate intent events via WebSocket for integration testing."""
# TODO: Make the payload and timing configurable via CLI options.

import asyncio
import json
import time
from typing import List, Dict, Any

import websockets

async def send_events(uri: str, payloads: List[Dict[str, Any]]) -> None:
  async with websockets.connect(uri) as websocket:
    for payload in payloads:
      payload["ts"] = time.time()
      await websocket.send(json.dumps(payload))
      await asyncio.sleep(0.2)

async def main() -> None:
  uri = "ws://localhost:8000/control"
  events = [
    {"type": "intent", "intent": "left", "conf": 0.9, "meta": {"source": "simulate.py"}, "protocol_version": 1},
    {"type": "intent", "intent": "left", "conf": 0.92, "meta": {"source": "simulate.py"}, "protocol_version": 1},
    {"type": "intent", "intent": "left", "conf": 0.94, "meta": {"source": "simulate.py"}, "protocol_version": 1},
  ]
  await send_events(uri, events)

if __name__ == "__main__":
  asyncio.run(main())
