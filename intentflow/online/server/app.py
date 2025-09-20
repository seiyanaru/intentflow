"""FastAPI application exposing health, metrics, and control streams."""
# TODO: Attach real inference pipeline and recorder integration.

from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

def create_app() -> FastAPI:
  app = FastAPI(title="Intentflow Online API", version="0.1.0")
  register_routes(app)
  return app

def register_routes(app: FastAPI) -> None:
  @app.get("/health")
  async def health() -> dict[str, str]:
    return {"status": "ok"}

  @app.get("/metrics")
  async def metrics() -> dict[str, float]:
    return {"latency_ms": 0.0, "throughput_hz": 0.0}

  @app.websocket("/control")
  async def control(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
      while True:
        message = await websocket.receive_text()
        # TODO: Route messages through stabilizer and adaptation layers.
        await websocket.send_text(message)
    except WebSocketDisconnect:
      # TODO: Track disconnections and cleanup resources.
      return

app = create_app()
