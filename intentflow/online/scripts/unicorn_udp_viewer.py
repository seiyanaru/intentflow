#!/usr/bin/env python3
"""Real-time EEG waveform viewer for Unicorn UDP stream.

This script receives UDP packets from Unicorn (same format as the inference server)
and displays live scrolling waveforms for all channels.

Usage:
    python unicorn_udp_viewer.py --port 11002 --channels 8

Requirements:
    pip install matplotlib numpy
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class UnicornUDPViewer:
    """Real-time UDP EEG viewer with matplotlib."""

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

        # Ring buffer for visualization
        self.max_samples = int(window_sec * sfreq)
        self.buffer = deque(maxlen=self.max_samples)

        # UDP socket setup
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.setblocking(False)
        self._sock.settimeout(0.001)

        # Statistics
        self._received_packets = 0
        self._dropped_packets = 0
        self._last_update = time.time()
        self._update_rate = 0.0

        print(f"[Viewer] Listening on {self.host}:{self.port}")
        print(f"[Viewer] Channels: {self.n_channels}, Window: {self.window_sec}s @ {self.sfreq}Hz")
        print(f"[Viewer] Packet format: {self.packet_format}")

    def _parse_csv(self, data: bytes) -> Optional[np.ndarray]:
        """Parse CSV bytes into [C, T] float32."""
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
            arr = flat[:n_valid].reshape(-1, self.n_channels)  # [T, C]
            return arr.T.astype(np.float32, copy=False)  # [C, T]
        except Exception:
            return None

    def _parse_f32le(self, data: bytes) -> Optional[np.ndarray]:
        """Parse float32 little-endian bytes into [C, T]."""
        try:
            arr = np.frombuffer(data, dtype="<f4")
            n_total = arr.size
            n_valid = (n_total // self.n_channels) * self.n_channels
            if n_valid == 0:
                return None
            arr = arr[:n_valid].reshape(-1, self.n_channels)  # [T, C]
            return arr.T.astype(np.float32, copy=False)
        except Exception:
            return None

    def _parse_unicorn17f(self, data: bytes) -> Optional[np.ndarray]:
        """Parse Unicorn UDP 17-float32 packet."""
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
        """Parse packet based on format setting."""
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
            except (BlockingIOError, socket.timeout):
                break

            self._received_packets += 1
            parsed = self._parse_packet(data)
            if parsed is None:
                self._dropped_packets += 1
                continue

            # Append samples: parsed is [C, T]
            for i in range(parsed.shape[1]):
                self.buffer.append(parsed[:, i])

        # Update rate calculation
        now = time.time()
        if now - self._last_update > 1.0:
            self._update_rate = self._received_packets / (now - self._last_update)
            self._last_update = now
            self._received_packets = 0

    def get_display_data(self) -> np.ndarray:
        """Get current buffer as [C, T] array."""
        if len(self.buffer) == 0:
            return np.zeros((self.n_channels, 1), dtype=np.float32)
        return np.array(self.buffer).T  # [T, C] -> [C, T]

    def close(self) -> None:
        """Close UDP socket."""
        try:
            self._sock.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time Unicorn UDP EEG waveform viewer"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="UDP listen host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11002,
        help="UDP listen port",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=8,
        help="Number of EEG channels",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Display window duration in seconds",
    )
    parser.add_argument(
        "--sfreq",
        type=int,
        default=250,
        help="Sampling frequency in Hz",
    )
    parser.add_argument(
        "--packet-format",
        type=str,
        default="auto",
        choices=["csv", "f32le", "unicorn17f", "auto"],
        help="Expected UDP packet format",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=50.0,
        help="Y-axis scale (microvolts spacing between channels)",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=50,
        help="Plot update interval in milliseconds",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    viewer = UnicornUDPViewer(
        host=args.host,
        port=args.port,
        n_channels=args.channels,
        window_sec=args.window,
        sfreq=args.sfreq,
        packet_format=args.packet_format,
    )

    # Setup matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    lines = []
    channel_labels = [f"Ch{i+1}" for i in range(args.channels)]
    colors = plt.cm.tab10(np.linspace(0, 1, args.channels))

    for i in range(args.channels):
        (line,) = ax.plot([], [], label=channel_labels[i], color=colors[i], linewidth=0.8)
        lines.append(line)

    ax.set_xlim(0, args.window)
    ax.set_ylim(-args.scale, args.channels * args.scale)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel (offset for clarity)")
    ax.set_yticks([i * args.scale for i in range(args.channels)])
    ax.set_yticklabels(channel_labels)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Unicorn UDP Real-Time EEG")

    # Status text
    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    def update_plot(frame):
        viewer.poll_data()
        data = viewer.get_display_data()  # [C, T]

        n_samples = data.shape[1]
        time_axis = np.linspace(0, args.window, n_samples)

        for i, line in enumerate(lines):
            # Offset each channel for visibility
            y_data = data[i, :] + i * args.scale
            line.set_data(time_axis, y_data)

        # Update status
        status_text.set_text(
            f"Packets: {viewer._received_packets + viewer._dropped_packets} "
            f"| Dropped: {viewer._dropped_packets} "
            f"| Rate: {viewer._update_rate:.1f} pkt/s "
            f"| Samples: {n_samples}"
        )

        return lines + [status_text]

    print("[Viewer] Starting animation... Close window to stop.")
    ani = animation.FuncAnimation(
        fig,
        update_plot,
        interval=args.update_interval,
        blit=True,
        cache_frame_data=False,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n[Viewer] Stopped by user")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
