#!/usr/bin/env python3
"""Windows-side bridge from Unicorn/LSL to UDP CSV packets.

This script is intended to run on the Windows laptop that receives Unicorn EEG.
It forwards 8-channel samples to the lab PC in the CSV format expected by
``intentflow.online.recorder.unicorn_udp_reader.UnicornUDPReader``.

Source modes:
- ``lsl``: subscribe to an LSL EEG stream and forward samples
- ``mock``: generate synthetic 8ch waveforms for transport testing
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from typing import Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge Unicorn/LSL samples to UDP CSV packets"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="lsl",
        choices=["lsl", "mock"],
        help="Input source mode",
    )
    parser.add_argument(
        "--target-host",
        type=str,
        required=True,
        help="Destination host IP for UDP packets",
    )
    parser.add_argument(
        "--target-port",
        type=int,
        default=11001,
        help="Destination UDP port",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=8,
        help="Number of EEG channels to forward",
    )
    parser.add_argument(
        "--samples-per-packet",
        type=int,
        default=1,
        help="Number of samples to bundle into one UDP packet",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="CSV delimiter",
    )
    parser.add_argument(
        "--stream-name",
        type=str,
        default="",
        help="LSL stream name filter",
    )
    parser.add_argument(
        "--stream-type",
        type=str,
        default="EEG",
        help="LSL stream type filter",
    )
    parser.add_argument(
        "--resolve-timeout",
        type=float,
        default=8.0,
        help="LSL stream resolve timeout in seconds",
    )
    parser.add_argument(
        "--lsl-timeout",
        type=float,
        default=0.5,
        help="Timeout for each LSL pull_chunk call",
    )
    parser.add_argument(
        "--mock-rate",
        type=float,
        default=250.0,
        help="Sampling rate for mock mode",
    )
    parser.add_argument(
        "--mock-amplitude",
        type=float,
        default=20.0,
        help="Signal amplitude for mock mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print periodic forwarding stats",
    )
    return parser.parse_args()


def resolve_lsl_stream(args: argparse.Namespace):
    try:
        from pylsl import StreamInlet, resolve_byprop
    except ImportError as exc:  # pragma: no cover - runtime guidance
        raise SystemExit(
            "pylsl is required for --source lsl. Install with: pip install pylsl"
        ) from exc

    streams = []
    if args.stream_name:
        streams = resolve_byprop(
            "name",
            args.stream_name,
            timeout=args.resolve_timeout,
        )
    if not streams:
        streams = resolve_byprop(
            "type",
            args.stream_type,
            timeout=args.resolve_timeout,
        )
    if not streams:
        raise SystemExit(
            "No LSL stream found. Check Unicorn stream publication and stream filters."
        )

    stream = streams[0]
    inlet = StreamInlet(stream, max_buflen=4, max_chunklen=max(1, args.samples_per_packet))
    print(
        "[Bridge] Connected to LSL stream: "
        f"name={stream.name()}, type={stream.type()}, channels={stream.channel_count()}, "
        f"sfreq={stream.nominal_srate()}"
    )
    return inlet


def iter_lsl_samples(args: argparse.Namespace) -> Iterable[List[float]]:
    inlet = resolve_lsl_stream(args)
    while True:
        chunk, _timestamps = inlet.pull_chunk(
            timeout=args.lsl_timeout,
            max_samples=max(1, args.samples_per_packet * 8),
        )
        if not chunk:
            continue
        for sample in chunk:
            if len(sample) < args.channels:
                continue
            yield [float(v) for v in sample[: args.channels]]


def iter_mock_samples(args: argparse.Namespace) -> Iterable[List[float]]:
    dt = 1.0 / max(1e-6, args.mock_rate)
    t = 0
    while True:
        base = 2.0 * math.pi * (t / max(1.0, args.mock_rate))
        sample = []
        for ch in range(args.channels):
            phase = ch * 0.35
            freq = 8.0 + (ch % 3) * 2.0
            value = args.mock_amplitude * math.sin(base * freq + phase)
            sample.append(value)
        yield sample
        t += 1
        time.sleep(dt)


def build_payload(
    rows: Sequence[Sequence[float]],
    delimiter: str,
) -> bytes:
    lines = []
    for row in rows:
        lines.append(delimiter.join(f"{float(v):.6f}" for v in row))
    return ("\n".join(lines) + "\n").encode("utf-8")


def main() -> None:
    args = parse_args()

    if args.samples_per_packet <= 0:
        raise SystemExit("--samples-per-packet must be positive")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.target_host, args.target_port)

    if args.source == "lsl":
        sample_iter = iter_lsl_samples(args)
    else:
        sample_iter = iter_mock_samples(args)

    print(
        "[Bridge] Forwarding started: "
        f"source={args.source}, dest={args.target_host}:{args.target_port}, "
        f"channels={args.channels}, spp={args.samples_per_packet}"
    )

    sent_packets = 0
    sent_samples = 0
    last_log = time.time()
    batch: List[List[float]] = []

    try:
        for sample in sample_iter:
            batch.append(sample)
            if len(batch) < args.samples_per_packet:
                continue

            payload = build_payload(batch, args.delimiter)
            sock.sendto(payload, dest)
            sent_packets += 1
            sent_samples += len(batch)
            batch.clear()

            now = time.time()
            if args.verbose and (now - last_log) >= 1.0:
                print(
                    "[Bridge] Sent "
                    f"{sent_packets} packets / {sent_samples} samples "
                    f"to {args.target_host}:{args.target_port}"
                )
                last_log = now
    except KeyboardInterrupt:
        print("\n[Bridge] Stopped by user")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
