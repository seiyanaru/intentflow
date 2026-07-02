"""Build an exact task-aware covariance cache for all Stieger paradigms.

The implementation mirrors the MOABB preprocessing used by the legacy cache:
continuous per-session 8--30 Hz IIR filtering, epochs from 0--3 seconds,
resampling to 250 Hz, and the same ordered 60 EEG channels.  Event codes retain
both ``tasknumber`` and ``targetnumber`` so LR, UD, and 2D contexts cannot be
silently mixed.

One compressed covariance file is written per subject/session, making the
process resumable without storing another full epoch cache.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

EEG60 = [
    "AF3", "AF4", "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2",
    "CP3", "CP4", "CP5", "CP6", "CPz", "Cz", "F1", "F2", "F3", "F4",
    "F5", "F6", "F7", "F8", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "FCz", "FT7", "FT8", "Fp1", "Fp2", "Fpz", "Fz", "O1", "O2", "Oz",
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PO3", "PO4",
    "PO5", "PO6", "PO7", "PO8", "POz", "Pz", "T7", "T8", "TP7", "TP8",
]
EVENT_NAMES = {
    11: "lr_right",
    12: "lr_left",
    23: "ud_both",
    24: "ud_rest",
    31: "2d_right",
    32: "2d_left",
    33: "2d_both",
    34: "2d_rest",
}
DEFAULT_RAW_ROOT = Path(
    "/home/islabshi/workspace-local2/mne_data/MNE-Stieger2021-data"
)
DEFAULT_OUTPUT = Path(
    "/home/islabshi/workspace-local2/mne_data/stieger_task_cov_cache"
)


def parse_range(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = map(int, part.split("-", maxsplit=1))
            values.extend(range(start, stop + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--sessions", default="1-11")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalized_channel_names(labels) -> list[str]:
    return [
        str(channel).replace("Z", "z").replace("FP", "Fp")
        for channel in labels.tolist()
    ]


def retained_trial(trial) -> bool:
    code = int(trial.tasknumber) * 10 + int(trial.targetnumber)
    return (
        code in EVENT_NAMES
        and int(trial.artifact) == 0
        and float(trial.triallength) + 2 > 3.0
    )


def session_covariances(raw_path: Path) -> dict[str, np.ndarray]:
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
    raw.filter(
        l_freq=8,
        h_freq=30,
        method="iir",
        picks="eeg",
        verbose=False,
    )
    events = mne.find_events(raw, shortest_event=0, verbose=False)
    event_id = {name: code for code, name in EVENT_NAMES.items()}
    epochs = mne.Epochs(
        raw,
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
    expected_codes = np.asarray(
        [row[0] for row in retained_metadata], dtype=np.int16
    )
    if not np.array_equal(observed_codes, expected_codes):
        raise RuntimeError(
            f"{raw_path}: event/metadata mismatch "
            f"({len(observed_codes)} vs {len(expected_codes)})"
        )
    covariances = (
        np.einsum("nct,ndt->ncd", x, x, optimize=True) / x.shape[-1]
    ).astype(np.float32)
    metadata = np.asarray(retained_metadata, dtype=np.int16)
    return {
        "covariances": covariances,
        "event_code": metadata[:, 0],
        "task": metadata[:, 1].astype(np.int8),
        "target": metadata[:, 2].astype(np.int8),
        "run": metadata[:, 3].astype(np.int8),
        "trial": metadata[:, 4],
        "channels": np.asarray(EEG60),
        "n_times": np.asarray(x.shape[-1], dtype=np.int16),
    }


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
            payload = session_covariances(raw_path)
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
