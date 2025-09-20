"""Command-line entry point for the intentflow online stack."""
# TODO: Extend CLI options for adapters, logging sinks, and dashboards.

from __future__ import annotations

import argparse

import uvicorn

def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Intentflow online inference server")
  parser.add_argument("--config", type=str, default="configs/online.yaml", help="Path to online inference config file")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface for FastAPI server")
  parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
  parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
  return parser

def run_server(host: str, port: int, reload: bool) -> None:
  """Launch the uvicorn server with the FastAPI application."""
  # TODO: Load config and initialize pipeline before serving.
  from intentflow.online.server.app import create_app
  uvicorn.run(create_app(), host=host, port=port, reload=reload)

def main() -> None:
  parser = build_parser()
  args = parser.parse_args()
  run_server(args.host, args.port, args.reload)

if __name__ == "__main__":
  main()
