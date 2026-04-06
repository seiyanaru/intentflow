#!/usr/bin/env python3
"""Web-based real-time EEG waveform viewer for Unicorn UDP stream.

This script receives UDP packets from Unicorn and serves a web page
with live scrolling waveforms using Plotly.

Usage:
    python unicorn_web_viewer.py --udp-port 11002 --web-port 8080
    Then open: http://localhost:8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import time
from collections import deque
from typing import Optional

import numpy as np
from aiohttp import web


class UnicornUDPSource:
    """UDP receiver for Unicorn EEG data."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 11002,
        n_channels: int = 8,
        window_sec: float = 5.0,
        sfreq: int = 250,
        packet_format: str = "auto",
        delimiter: str = ",",
    ):
        self.host = host
        self.port = port
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.window_sec = window_sec
        self.packet_format = packet_format
        self.delimiter = delimiter

        self.max_samples = int(window_sec * sfreq)
        self.buffer = deque(maxlen=self.max_samples)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.setblocking(False)

        self._received_packets = 0
        self._dropped_packets = 0

        print(f"[UDPSource] Listening on {self.host}:{self.port}")

    def _parse_csv(self, data: bytes) -> Optional[np.ndarray]:
        try:
            text = data.decode("utf-8", errors="ignore").strip()
            if not text:
                return None
            rows = []
            for line in text.splitlines():
                cols = [c for c in line.strip().split(self.delimiter) if c != ""]
                if not cols:
                    continue
                vals = [float(c) for c in cols]
                rows.append(vals)
            if not rows:
                return None
            flat = np.asarray(rows, dtype=np.float32).reshape(-1)
            n_total = flat.size
            n_valid = (n_total // self.n_channels) * self.n_channels
            if n_valid == 0:
                return None
            arr = flat[:n_valid].reshape(-1, self.n_channels)
            return arr.T.astype(np.float32, copy=False)
        except Exception:
            return None

    def _parse_f32le(self, data: bytes) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(data, dtype="<f4")
            n_total = arr.size
            n_valid = (n_total // self.n_channels) * self.n_channels
            if n_valid == 0:
                return None
            arr = arr[:n_valid].reshape(-1, self.n_channels)
            return arr.T.astype(np.float32, copy=False)
        except Exception:
            return None

    def _parse_unicorn17f(self, data: bytes) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(data, dtype="<f4")
            if arr.size != 17:
                return None
            if self.n_channels > 17:
                return None
            eeg = arr[: self.n_channels].astype(np.float32, copy=False)
            return eeg.reshape(self.n_channels, 1)
        except Exception:
            return None

    def _parse_packet(self, data: bytes) -> Optional[np.ndarray]:
        if self.packet_format == "csv":
            return self._parse_csv(data)
        if self.packet_format == "f32le":
            return self._parse_f32le(data)
        if self.packet_format == "unicorn17f":
            return self._parse_unicorn17f(data)
        if self.packet_format == "auto":
            result = self._parse_csv(data)
            if result is not None:
                return result
            result = self._parse_unicorn17f(data)
            if result is not None:
                return result
            return self._parse_f32le(data)
        return None

    def poll_data(self) -> None:
        """Poll UDP socket and append to buffer."""
        while True:
            try:
                data, _ = self._sock.recvfrom(65535)
            except BlockingIOError:
                break

            self._received_packets += 1
            parsed = self._parse_packet(data)
            if parsed is None:
                self._dropped_packets += 1
                continue

            for i in range(parsed.shape[1]):
                self.buffer.append(parsed[:, i])

    def get_data_dict(self) -> dict:
        """Get current buffer as JSON-serializable dict."""
        if len(self.buffer) == 0:
            data = np.zeros((self.n_channels, 1), dtype=np.float32)
        else:
            data = np.array(self.buffer).T  # [C, T]

        n_samples = data.shape[1]
        time_axis = np.linspace(0, self.window_sec, n_samples).tolist()

        return {
            "time": time_axis,
            "channels": [data[i, :].tolist() for i in range(self.n_channels)],
            "n_samples": n_samples,
            "received": self._received_packets,
            "dropped": self._dropped_packets,
        }

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


# Global UDP source instance
udp_source: Optional[UnicornUDPSource] = None


async def handle_index(request):
    """Serve the HTML page."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Unicorn EEG Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        #status {
            background: #fff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        #plot {
            background: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat {
            display: inline-block;
            margin-right: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Unicorn UDP Real-Time EEG Viewer</h1>
    <div id="status">
        <span class="stat">Samples: <span id="samples">0</span></span>
        <span class="stat">Received: <span id="received">0</span></span>
        <span class="stat">Dropped: <span id="dropped">0</span></span>
        <span class="stat">Update Rate: <span id="rate">0</span> Hz</span>
    </div>
    <div id="plot"></div>

    <script>
        const channelNames = Array.from({length: 8}, (_, i) => `Ch${i+1}`);
        const colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ];
        const scale = 50;

        let traces = channelNames.map((name, i) => ({
            x: [],
            y: [],
            mode: 'lines',
            name: name,
            line: {color: colors[i], width: 1.5},
        }));

        let layout = {
            title: 'EEG Waveforms (stacked)',
            xaxis: {title: 'Time (s)'},
            yaxis: {
                title: 'Channel (offset)',
                tickmode: 'array',
                tickvals: channelNames.map((_, i) => i * scale),
                ticktext: channelNames,
            },
            showlegend: true,
            height: 600,
            margin: {l: 80, r: 50, t: 50, b: 50},
        };

        Plotly.newPlot('plot', traces, layout);

        let lastUpdate = Date.now();
        let updateCount = 0;

        async function updatePlot() {
            try {
                const response = await fetch('/data');
                const data = await response.json();

                for (let i = 0; i < data.channels.length; i++) {
                    traces[i].x = data.time;
                    traces[i].y = data.channels[i].map(v => v + i * scale);
                }

                Plotly.react('plot', traces, layout);

                document.getElementById('samples').textContent = data.n_samples;
                document.getElementById('received').textContent = data.received;
                document.getElementById('dropped').textContent = data.dropped;

                updateCount++;
                const now = Date.now();
                if (now - lastUpdate > 1000) {
                    const rate = (updateCount / ((now - lastUpdate) / 1000)).toFixed(1);
                    document.getElementById('rate').textContent = rate;
                    lastUpdate = now;
                    updateCount = 0;
                }
            } catch (error) {
                console.error('Update failed:', error);
            }
        }

        setInterval(updatePlot, 100);  // Update every 100ms
    </script>
</body>
</html>
    """
    return web.Response(text=html, content_type="text/html")


async def handle_data(request):
    """Serve current EEG data as JSON."""
    global udp_source
    if udp_source is None:
        return web.json_response({"error": "UDP source not initialized"}, status=500)

    udp_source.poll_data()
    data = udp_source.get_data_dict()
    return web.json_response(data)


async def udp_poll_task(app):
    """Background task to poll UDP."""
    global udp_source
    while True:
        if udp_source is not None:
            udp_source.poll_data()
        await asyncio.sleep(0.01)


async def start_background_tasks(app):
    app['udp_poll'] = asyncio.create_task(udp_poll_task(app))


async def cleanup_background_tasks(app):
    app['udp_poll'].cancel()
    await app['udp_poll']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web-based real-time Unicorn UDP EEG viewer"
    )
    parser.add_argument("--udp-host", type=str, default="0.0.0.0", help="UDP listen host")
    parser.add_argument("--udp-port", type=int, default=11002, help="UDP listen port")
    parser.add_argument("--web-host", type=str, default="0.0.0.0", help="Web server host")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port")
    parser.add_argument("--channels", type=int, default=8, help="Number of EEG channels")
    parser.add_argument("--window", type=float, default=5.0, help="Display window (seconds)")
    parser.add_argument("--sfreq", type=int, default=250, help="Sampling frequency (Hz)")
    parser.add_argument("--packet-format", type=str, default="auto", help="UDP packet format")
    return parser.parse_args()


def main():
    args = parse_args()

    global udp_source
    udp_source = UnicornUDPSource(
        host=args.udp_host,
        port=args.udp_port,
        n_channels=args.channels,
        window_sec=args.window,
        sfreq=args.sfreq,
        packet_format=args.packet_format,
    )

    app = web.Application()
    app.router.add_get('/', handle_index)
    app.router.add_get('/data', handle_data)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    print("=" * 60)
    print("Unicorn Web-Based EEG Viewer")
    print("=" * 60)
    print(f"UDP:  {args.udp_host}:{args.udp_port}")
    print(f"Web:  http://{args.web_host}:{args.web_port}")
    print(f"Channels: {args.channels}, Window: {args.window}s @ {args.sfreq}Hz")
    print("=" * 60)
    print(f"\nOpen your browser to: http://localhost:{args.web_port}")
    print("Press Ctrl+C to stop.\n")

    try:
        web.run_app(app, host=args.web_host, port=args.web_port, print=None)
    except KeyboardInterrupt:
        print("\n[Viewer] Stopped by user")
    finally:
        if udp_source:
            udp_source.close()


if __name__ == "__main__":
    main()
