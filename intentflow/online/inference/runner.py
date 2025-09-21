"""Utilities for streaming intents to the control channel."""
# TODO(P1): Replace DummyModel with ONNX Runtime-backed inference.

from __future__ import annotations

import asyncio
import json
import time
from typing import Tuple

import websockets
from websockets import ConnectionClosed, WebSocketException

from intentflow.online.inference.stabilizer import Stabilizer


class DummyModel:
  """Dummy model that alternates intents for demonstration purposes."""

  def __init__(self) -> None:
    """Initialize the alternating state."""
    self._last_intent = "right"

  def predict(self) -> Tuple[str, float]:
    """Return the next alternating intent with a fixed confidence."""
    self._last_intent = "left" if self._last_intent == "right" else "right"
    return self._last_intent, 0.9


async def stream_intent(ws_url: str, hz: int = 20, conf_th: float = 0.7) -> None:
  """Continuously send stabilized intents to the control WebSocket."""
  delay = 1.0 / max(1, hz)
  model = DummyModel()
  stabilizer = Stabilizer(conf_th=conf_th, ema_ms=150, require_consec=3, min_interval_ms=200)

  while True:
    try:
      async with websockets.connect(ws_url) as websocket:
        print(f"Connected to {ws_url}")
        while True:
          intent, conf = model.predict()
          result = stabilizer.apply(intent, conf)
          stable_intent = result["intent"]
          if stable_intent != "none":
            message = json.dumps({
              "type": "intent",
              "intent": stable_intent,
              "conf": float(result["conf_ema"]),
              "ts": time.time(),
              "protocol_version": 1,
            })
            try:
              await websocket.send(message)
            except (ConnectionClosed, WebSocketException, OSError) as exc:
              print(f"stream_intent send error: {exc}")
              break
          await asyncio.sleep(delay)
    except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
      raise
    except (ConnectionClosed, WebSocketException, OSError) as exc:
      print(f"stream_intent connection issue: {exc}. Retrying in 3s")
      await asyncio.sleep(3)
    except Exception as exc:  # noqa: BLE001
      print(f"stream_intent unexpected error: {exc}. Retrying in 3s")
      await asyncio.sleep(3)
