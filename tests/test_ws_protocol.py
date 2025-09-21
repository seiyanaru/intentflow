"""WebSocket protocol integration tests for the online bridge."""

from __future__ import annotations

import asyncio

from starlette.testclient import TestClient

from intentflow.online.server.app import INTENT_BUS, app


def _drain_bus() -> None:
  """Ensure the intent bus queue is empty before tests."""
  while True:
    try:
      INTENT_BUS.get_nowait()
    except asyncio.QueueEmpty:
      break


def test_health_endpoint() -> None:
  """Health endpoint reports OK and zero clients before connections."""
  _drain_bus()
  with TestClient(app) as client:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["clients"] == 0


def test_control_echo() -> None:
  """Control websocket echoes text messages back to the sender."""
  _drain_bus()
  with TestClient(app) as client:
    with client.websocket_connect("/control") as websocket:
      websocket.send_text("ping")
      assert websocket.receive_text() == "ping"


def test_control_bus_broadcast() -> None:
  """Messages enqueued on the intent bus are broadcast to clients."""
  _drain_bus()
  with TestClient(app) as client:
    with client.websocket_connect("/control") as websocket:
      message = "{\"type\":\"intent\",\"intent\":\"left\",\"conf\":0.9,\"ts\":0,\"protocol_version\":1}"
      INTENT_BUS.put_nowait(message)
      assert websocket.receive_text() == message
