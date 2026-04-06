"""UDP-based Unicorn stream reader with online windowing."""

from __future__ import annotations

import socket
from typing import Optional

import numpy as np


class UnicornUDPReader:
    """
    Read Unicorn EEG samples from UDP and emit sliding windows.

    Expected packet payload (configurable):
    - csv: "v1,v2,...,vN" or multiple lines per packet
    - f32le: raw float32 little-endian bytes
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 1001,
        n_channels: int = 8,
        sfreq: int = 250,
        window_sec: float = 4.0,
        hop_sec: float = 0.25,
        packet_format: str = "csv",
        delimiter: str = ",",
        recv_buf_bytes: int = 65535,
    ) -> None:
        self.host = host
        self.port = port
        self.n_channels = int(n_channels)
        self.sfreq = int(sfreq)
        self.packet_format = packet_format
        self.delimiter = delimiter

        self.window_samples = max(1, int(window_sec * self.sfreq))
        self.hop_samples = max(1, int(hop_sec * self.sfreq))
        self._buffer = np.zeros((self.n_channels, self.window_samples), dtype=np.float32)
        self._total_samples = 0
        self._last_emit_samples = 0

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.setblocking(False)
        self._recv_buf_bytes = recv_buf_bytes

        self._dropped_packets = 0
        self._received_packets = 0

    def close(self) -> None:
        """Close UDP socket."""
        try:
            self._sock.close()
        except Exception:
            pass

    def _append_samples(self, samples: np.ndarray) -> None:
        """Append samples [C, T] into ring buffer."""
        n = samples.shape[1]
        self._total_samples += n

        if n >= self.window_samples:
            self._buffer[:, :] = samples[:, -self.window_samples :]
            return

        self._buffer = np.roll(self._buffer, -n, axis=1)
        self._buffer[:, -n:] = samples.astype(np.float32, copy=False)

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

    def _parse_packet(self, data: bytes) -> Optional[np.ndarray]:
        if self.packet_format == "csv":
            return self._parse_csv(data)
        if self.packet_format == "f32le":
            return self._parse_f32le(data)
        raise ValueError(f"Unsupported packet_format: {self.packet_format}")

    def poll_window(self) -> Optional[np.ndarray]:
        """
        Poll available UDP packets and emit one window if hop condition is met.

        Returns:
            Window [C, T] or None when not enough new data.
        """
        while True:
            try:
                data, _ = self._sock.recvfrom(self._recv_buf_bytes)
            except BlockingIOError:
                break

            self._received_packets += 1
            parsed = self._parse_packet(data)
            if parsed is None:
                self._dropped_packets += 1
                continue
            self._append_samples(parsed)

        enough_history = self._total_samples >= self.window_samples
        enough_hop = (self._total_samples - self._last_emit_samples) >= self.hop_samples
        if enough_history and enough_hop:
            self._last_emit_samples = self._total_samples
            return self._buffer.copy()
        return None

    def stats(self) -> dict:
        """Return reader statistics for debugging."""
        return {
            "received_packets": self._received_packets,
            "dropped_packets": self._dropped_packets,
            "total_samples": self._total_samples,
            "window_samples": self.window_samples,
            "hop_samples": self.hop_samples,
        }

