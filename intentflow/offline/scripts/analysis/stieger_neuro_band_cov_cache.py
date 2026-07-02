"""Build a task-aware band-wise covariance cache for Neuro-LEMA E1.

This script mirrors ``stieger_task_cov_cache.py`` but stores one covariance
tensor per neurophysiology-informed band:

* broad_8_30
* mu_8_13
* low_beta_13_20
* high_beta_20_30

The cache keeps Stieger task/target metadata so LR, UD, and 2D can be evaluated
as native tasks without silently mixing control contexts.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np
from scipy.io import loadmat

from stieger_task_cov_cache import (
    DEFAULT_RAW_ROOT,
    EEG60,
    EVENT_NAMES,
    normalized_channel_names,
    parse_range,
    retained_trial,
)


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

DEFAULT_OUTPUT = Path(
    "/home/islabshi/workspace-local2/mne_data/stieger_neuro_band_cov_cache"
)
BANDS = {
    "broad_8_30": (8.0, 30.0),
    "mu_8_13": (8.0, 13.0),
    "low_beta_13_20": (13.0, 20.0),
    "high_beta_20_30": (20.0, 30.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--sessions", default="1-11")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def continuous_raw_and_metadata(raw_path: Path) -> tuple[mne.io.RawArray, np.ndarray]:
    """Load one Stieger MAT file and return continuous raw + retained metadata."""

    container = loadmat(
        raw_path,
        squeeze_me=True,
        struct_as_record=False,
        verify_compressed_data_integrity=False,
    )["BCI"]
    sampling_rate = float(container.SRATE)
    channel_names = normalized_channel_names(container.chaninfo.label)
    channel_indices = [channel_names.index(channel) for channel in EEG60]

    eeg_parts: list[np.ndarray] = []
    stim_parts: list[np.ndarray] = []
    retained_metadata: list[tuple[int, int, int, int, int]] = []
    for data, times, trial in zip(
        container.data, container.time, container.TrialData
    ):
        eeg_parts.append(data[channel_indices])
        stim = np.zeros_like(times)
        if retained_trial(trial):
            cue_sample = int(2 * sampling_rate)
            if times[cue_sample] != 0:
                raise RuntimeError(f"{raw_path}: cue sample is not time zero")
            event_code = int(trial.tasknumber) * 10 + int(trial.targetnumber)
            stim[cue_sample] = event_code
            retained_metadata.append(
                (
                    event_code,
                    int(trial.tasknumber),
                    int(trial.targetnumber),
                    int(trial.runnumber),
                    int(trial.trialnumber),
                )
            )
        stim_parts.append(stim[None])

    eeg = np.concatenate(eeg_parts, axis=1) * 1e-6
    stim = np.concatenate(stim_parts, axis=1)
    data = np.concatenate([eeg, stim], axis=0)
    info = mne.create_info(
        EEG60 + ["stim"],
        sampling_rate,
        ["eeg"] * len(EEG60) + ["stim"],
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    return raw, np.asarray(retained_metadata, dtype=np.int16)


def band_covariances(
    raw: mne.io.RawArray,
    metadata: np.ndarray,
    low_hz: float,
    high_hz: float,
    raw_path: Path,
) -> np.ndarray:
    """Return per-trial covariances after band-pass filtering and epoching."""

    band_raw = raw.copy()
    band_raw.filter(
        l_freq=low_hz,
        h_freq=high_hz,
        method="iir",
        picks="eeg",
        verbose=False,
    )
    events = mne.find_events(band_raw, shortest_event=0, verbose=False)
    event_id = {name: code for code, name in EVENT_NAMES.items()}
    epochs = mne.Epochs(
        band_raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=3.0,
        proj=False,
        baseline=None,
        preload=True,
        picks=np.arange(len(EEG60)),
        event_repeated="drop",
        on_missing="ignore",
        verbose=False,
    )
    epochs.resample(250, verbose=False)
    x = epochs.get_data() * 1e6
    observed_codes = epochs.events[:, -1].astype(np.int16)
    expected_codes = metadata[:, 0].astype(np.int16)
    if not np.array_equal(observed_codes, expected_codes):
        raise RuntimeError(
            f"{raw_path}: event/metadata mismatch for band {low_hz}-{high_hz}Hz "
            f"({len(observed_codes)} vs {len(expected_codes)})"
        )
    return (
        np.einsum("nct,ndt->ncd", x, x, optimize=True) / x.shape[-1]
    ).astype(np.float32)


def session_payload(raw_path: Path) -> dict[str, np.ndarray]:
    raw, metadata = continuous_raw_and_metadata(raw_path)
    payload: dict[str, np.ndarray] = {
        "event_code": metadata[:, 0],
        "task": metadata[:, 1].astype(np.int8),
        "target": metadata[:, 2].astype(np.int8),
        "run": metadata[:, 3].astype(np.int8),
        "trial": metadata[:, 4],
        "channels": np.asarray(EEG60),
        "bands": np.asarray(list(BANDS)),
    }
    for band_name, (low_hz, high_hz) in BANDS.items():
        payload[f"cov_{band_name}"] = band_covariances(
            raw,
            metadata,
            low_hz,
            high_hz,
            raw_path,
        )
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_sessions = parse_range(args.sessions)
    for subject in parse_range(args.subjects):
        subject_dir = args.output_dir / f"S{subject}"
        subject_dir.mkdir(exist_ok=True)
        raw_paths = sorted(
            args.raw_root.glob(f"S{subject}_Session_*.mat"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        for raw_path in raw_paths:
            session = int(raw_path.stem.split("_")[-1])
            if session not in requested_sessions:
                continue
            output = subject_dir / f"session_{session}.npz"
            if output.exists() and not args.force:
                print(f"S{subject}/{session}: resume", flush=True)
                continue
            payload = session_payload(raw_path)
            payload["subject"] = np.asarray(subject, dtype=np.int16)
            payload["session"] = np.asarray(session, dtype=np.int16)
            np.savez_compressed(output, **payload)
            counts = {
                task: int(np.sum(payload["task"] == task))
                for task in np.unique(payload["task"])
            }
            print(
                f"S{subject}/{session}: n={len(payload['task'])} "
                f"task_counts={counts}",
                flush=True,
            )


if __name__ == "__main__":
    main()
