"""Attach Stieger task/run/trial context to the existing horizontal cache.

The existing ``stieger_epochs_cache`` was produced with MOABB's
``LeftRightImagery`` paradigm.  It therefore contains target numbers 1/2 from
both the one-dimensional LR task (tasknumber=1) and the two-dimensional task
(tasknumber=3).  This script reconstructs the retained-trial metadata from the
raw MAT files and verifies exact label/order agreement before saving a compact
sidecar file per subject.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DEFAULT_EPOCH_CACHE = Path(
    "/home/islabshi/workspace-local2/mne_data/stieger_epochs_cache"
)
DEFAULT_RAW_ROOT = Path(
    "/home/islabshi/workspace-local2/mne_data/MNE-Stieger2021-data"
)
DEFAULT_OUTPUT = Path(
    "/home/islabshi/workspace-local2/mne_data/stieger_task_metadata"
)


def parse_subjects(value: str) -> list[int]:
    subjects: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, stop = map(int, part.split("-", maxsplit=1))
            subjects.extend(range(start, stop + 1))
        elif part.strip():
            subjects.append(int(part))
    return sorted(set(subjects))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-62")
    parser.add_argument("--epoch-cache", type=Path, default=DEFAULT_EPOCH_CACHE)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def retained_horizontal_trials(container) -> list:
    return [
        trial
        for trial in container.TrialData
        if int(trial.targetnumber) in (1, 2)
        and int(trial.artifact) == 0
        and float(trial.triallength) + 2 > 3.0
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subject in parse_subjects(args.subjects):
        output = args.output_dir / f"S{subject}_taskmeta.npz"
        if output.exists() and not args.force:
            print(f"S{subject}: resume", flush=True)
            continue
        cache = np.load(args.epoch_cache / f"S{subject}_epochs.npz")
        cached_y = cache["y"].astype(np.int8)
        cached_session = cache["sess"].astype(np.int16)
        task: list[int] = []
        run: list[int] = []
        trial_number: list[int] = []
        target: list[int] = []
        session_values: list[int] = []
        for session in sorted(np.unique(cached_session).tolist()):
            raw_path = args.raw_root / f"S{subject}_Session_{session}.mat"
            container = loadmat(
                raw_path,
                squeeze_me=True,
                struct_as_record=False,
                verify_compressed_data_integrity=False,
            )["BCI"]
            retained = retained_horizontal_trials(container)
            raw_binary_y = np.asarray(
                [int(item.targetnumber) == 1 for item in retained],
                dtype=np.int8,
            )
            session_cached_y = cached_y[cached_session == session]
            if not np.array_equal(raw_binary_y, session_cached_y):
                raise RuntimeError(
                    f"S{subject} session {session}: raw/cache order mismatch "
                    f"({len(raw_binary_y)} vs {len(session_cached_y)})"
                )
            task.extend(int(item.tasknumber) for item in retained)
            run.extend(int(item.runnumber) for item in retained)
            trial_number.extend(int(item.trialnumber) for item in retained)
            target.extend(int(item.targetnumber) for item in retained)
            session_values.extend([int(session)] * len(retained))
            del container
        if len(task) != len(cached_y):
            raise RuntimeError(
                f"S{subject}: metadata/cache length mismatch {len(task)} vs {len(cached_y)}"
            )
        np.savez_compressed(
            output,
            task=np.asarray(task, dtype=np.int8),
            run=np.asarray(run, dtype=np.int8),
            trial=np.asarray(trial_number, dtype=np.int16),
            target=np.asarray(target, dtype=np.int8),
            session=np.asarray(session_values, dtype=np.int16),
        )
        counts = {
            value: int(np.sum(np.asarray(task) == value))
            for value in sorted(set(task))
        }
        print(f"S{subject}: n={len(task)} task_counts={counts}", flush=True)


if __name__ == "__main__":
    main()
