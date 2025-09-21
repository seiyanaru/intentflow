"""Command-line interface for launching the intentflow online server."""
# TODO(P1): Add extended CLI options for adapters, logging sinks, and dashboards.

from __future__ import annotations

import argparse

import uvicorn


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments for the online server."""
  parser = argparse.ArgumentParser(description="Start the intentflow FastAPI bridge")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Hostname or IP to bind the server")
  parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
  return parser.parse_args()


def main() -> None:
  """Entry point that delegates to uvicorn for serving FastAPI."""
  args = parse_args()
  try:
    uvicorn.run("intentflow.online.server.app:app", host=args.host, port=args.port, reload=False)
  except Exception as exc:  # noqa: BLE001
    print(f"Server failed to start: {exc}")


if __name__ == "__main__":
  main()
