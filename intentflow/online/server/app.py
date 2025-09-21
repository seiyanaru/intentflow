"""FastAPI application exposing health, control, and ingest interfaces."""
# TODO(P1): /ingest should buffer raw samples, window them, and feed the ONNX pipeline.

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Set, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError


class IntentMessage(BaseModel):
  """Schema for intent messages exchanged over the control channel."""

  type: str = "intent"
  intent: str
  conf: float
  ts: float
  protocol_version: int = 1


def _validate_intent_message(payload: Union[str, Dict[str, Any]]) -> None:
  """Validate an intent payload raising if the schema does not match."""
  if isinstance(payload, str):
    data: Dict[str, Any] = json.loads(payload)
  else:
    data = payload
  IntentMessage(**data)


app = FastAPI(title="intentflow-bridge")
INTENT_BUS: asyncio.Queue[str] = asyncio.Queue()
INGEST_COUNTER = {"messages": 0, "samples": 0}
ACTIVE_SOCKETS: Set[WebSocket] = set()


async def broadcast_message(message: str) -> None:
  """Broadcast a message to all active sockets, pruning failed connections."""
  dead_sockets: list[WebSocket] = []
  for socket in list(ACTIVE_SOCKETS):
    try:
      await socket.send_text(message)
    except Exception as exc:  # noqa: BLE001
      print(f"broadcast_message error: {exc}")
      dead_sockets.append(socket)
  for socket in dead_sockets:
    ACTIVE_SOCKETS.discard(socket)


async def _control_echo_loop(websocket: WebSocket, stop_event: asyncio.Event) -> None:
  """Echo messages from a single websocket back to its sender."""
  while not stop_event.is_set():
    try:
      message = await websocket.receive_text()
    except WebSocketDisconnect:
      stop_event.set()
      return
    except Exception as exc:  # noqa: BLE001
      print(f"control receive error: {exc}")
      continue
    try:
      await websocket.send_text(message)
    except Exception as exc:  # noqa: BLE001
      print(f"control send error: {exc}")
      stop_event.set()
      return


async def _control_bus_loop(stop_event: asyncio.Event) -> None:
  """Dispatch bus messages to all active sockets until stopped."""
  while not stop_event.is_set():
    try:
      message = await asyncio.wait_for(INTENT_BUS.get(), timeout=0.5)
    except asyncio.TimeoutError:
      continue
    except Exception as exc:  # noqa: BLE001
      print(f"intent bus error: {exc}")
      continue
    await broadcast_message(message)


@app.get("/health")
async def health() -> Dict[str, Any]:
  """Return service status information."""
  return {"ok": True, "clients": len(ACTIVE_SOCKETS)}


@app.websocket("/control")
async def control(websocket: WebSocket) -> None:
  """Handle bidirectional control channel connections."""
  await websocket.accept()
  ACTIVE_SOCKETS.add(websocket)
  stop_event = asyncio.Event()
  echo_task = asyncio.create_task(_control_echo_loop(websocket, stop_event))
  bus_task = asyncio.create_task(_control_bus_loop(stop_event))
  try:
    done, pending = await asyncio.wait({echo_task, bus_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
      try:
        task.result()
      except Exception as exc:  # noqa: BLE001
        print(f"control task error: {exc}")
    stop_event.set()
    for task in pending:
      task.cancel()
      try:
        await task
      except asyncio.CancelledError:
        continue
      except Exception as exc:  # noqa: BLE001
        print(f"pending task error: {exc}")
  finally:
    ACTIVE_SOCKETS.discard(websocket)
    try:
      await websocket.close()
    except Exception as exc:  # noqa: BLE001
      print(f"websocket close error: {exc}")


@app.websocket("/ingest")
async def ingest(websocket: WebSocket) -> None:
  """Accept data ingestion messages and update counters."""
  await websocket.accept()
  try:
    while True:
      try:
        message = await websocket.receive_text()
      except WebSocketDisconnect:
        return
      except Exception as exc:  # noqa: BLE001
        print(f"ingest receive error: {exc}")
        continue
      try:
        payload = json.loads(message)
      except json.JSONDecodeError as exc:
        print(f"ingest parse error: {exc}")
        continue
      try:
        _validate_intent_message(payload)
      except (ValidationError, ValueError):
        pass
      INGEST_COUNTER["messages"] += 1
      samples = payload.get("samples", [])
      if isinstance(samples, list):
        INGEST_COUNTER["samples"] += len(samples)
  finally:
    try:
      await websocket.close()
    except Exception as exc:  # noqa: BLE001
      print(f"ingest close error: {exc}")


async def enqueue_intent(message: IntentMessage) -> None:
  """Place an intent message onto the broadcast bus."""
  await INTENT_BUS.put(message.model_dump_json())


# TODO(P1): Accept JSON from the Windows .NET bridge through /ingest.
