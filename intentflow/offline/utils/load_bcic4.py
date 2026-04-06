import os
import pandas as pd
import mne
import numpy as np
from typing import Dict, Optional, List
from scipy.io import loadmat

from braindecode.datasets import BaseConcatDataset, BaseDataset, MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)

def scale(data, factor):
    return data * factor


def _load_bcic2a_eval_labels(
    eval_label_path: Optional[str],
    subject_id: int,
) -> Optional[np.ndarray]:
    """Load BCIC2a evaluation labels (A0xE.mat) and return class indices in [0, 3]."""
    if not eval_label_path:
        return None

    subj = f"{subject_id:02d}"
    candidates: List[str] = []
    if os.path.isdir(eval_label_path):
        candidates.extend([
            os.path.join(eval_label_path, f"A{subj}E.mat"),
            os.path.join(eval_label_path, f"A{subj}E.gdf.mat"),
            os.path.join(eval_label_path, f"true_labels_A{subj}E.mat"),
        ])
    elif os.path.isfile(eval_label_path):
        candidates.append(eval_label_path)

    mat_path = next((p for p in candidates if os.path.exists(p)), None)
    if mat_path is None:
        return None

    mat = loadmat(mat_path)
    known_keys = ["classlabel", "true_y", "labels", "y_test", "label"]

    label_arr = None
    for key in known_keys:
        if key in mat:
            label_arr = np.asarray(mat[key]).squeeze()
            break

    if label_arr is None:
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value).squeeze()
            if arr.ndim == 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                uniq = set(np.unique(arr).astype(int).tolist())
                if uniq.issubset({0, 1, 2, 3, 4}) and len(uniq) > 1:
                    label_arr = arr
                    break

    if label_arr is None:
        return None

    labels = label_arr.astype(int).reshape(-1)
    if labels.min() >= 1 and labels.max() <= 4:
        labels = labels - 1
    elif labels.min() < 0 or labels.max() > 3:
        return None

    return labels


def _inject_bcic2a_eval_labels_into_raw(
    raw: mne.io.BaseRaw,
    labels_0to3: np.ndarray,
    filepath: str,
) -> bool:
    """Replace 783 cue markers in E session with class markers 769-772."""
    descriptions = np.array(raw.annotations.description, dtype=object)
    cue_idx = np.where(descriptions == "783")[0]
    if cue_idx.size == 0:
        return False
    if cue_idx.size != labels_0to3.size:
        print(
            f"Warning: Label count mismatch in {filepath}: "
            f"found {cue_idx.size} cues(783), labels={labels_0to3.size}. Skipping session_E."
        )
        return False

    event_codes = np.array(["769", "770", "771", "772"], dtype=object)
    descriptions[cue_idx] = event_codes[labels_0to3]
    raw.set_annotations(
        mne.Annotations(
            onset=raw.annotations.onset.copy(),
            duration=raw.annotations.duration.copy(),
            description=descriptions.tolist(),
            orig_time=raw.annotations.orig_time,
        )
    )
    return True


def load_bcic4(subject_ids: list, dataset: str = "2a", preprocessing_dict: Dict = None,
              verbose: str = "WARNING", data_path: Optional[str] = None):
    
    # NOTE:
    # `dataset` is the dataset code string ("2a" or "2b"). Do NOT overwrite it with a dataset object.
    dataset_code = dataset

    if data_path:
        # Local GDF loading
        datasets = []
        for subject_id in subject_ids:
            subj_str = f"{subject_id:02d}"
            
            if dataset_code == "2a":
                eval_label_path = preprocessing_dict.get("eval_label_path")
                eval_labels = _load_bcic2a_eval_labels(eval_label_path, subject_id)
                # BCIC IV 2a filename convention: A01T.gdf, A01E.gdf, etc.
                for session_code, suffix in [("session_T", "T"), ("session_E", "E")]:
                    filename = f"A{subj_str}{suffix}.gdf"
                    filepath = os.path.join(data_path, filename)
                    
                    if not os.path.exists(filepath):
                        print(f"Warning: File {filepath} not found. Skipping.")
                        continue
                        
                    # Load raw data
                    try:
                        raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=verbose)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                        continue

                    # Check if raw contains target events (2a: 769-772)
                    target_events = {'769', '770', '771', '772'}
                    if not any(event in raw.annotations.description for event in target_events):
                        # Evaluation session in BCIC2a often has only cue marker "783".
                        # If true labels are provided (A0xE.mat), inject pseudo labels into cues.
                        if suffix == "E" and eval_labels is not None:
                            injected = _inject_bcic2a_eval_labels_into_raw(raw, eval_labels, filepath)
                            if injected and verbose:
                                print(f"Injected evaluation labels into {filepath} from {eval_label_path}.")
                        if verbose:
                            if not any(event in raw.annotations.description for event in target_events):
                                print(
                                    f"Skipping {filepath}: No target events found. "
                                    f"For BCIC2a session_E, provide eval labels via preprocessing.eval_label_path."
                                )
                        if not any(event in raw.annotations.description for event in target_events):
                            continue

                    description = pd.Series({
                        "subject": subject_id,
                        "session": session_code,
                        "run": session_code
                    })
                    datasets.append(BaseDataset(raw, description))
                    
            elif dataset_code == "2b":
                # BCIC IV 2b filename convention: B0101T.gdf, B0102T.gdf, ..., B0104E.gdf, B0105E.gdf
                # Sessions 0,1,2 are Training (T), Sessions 3,4 are Evaluation (E)
                for session_idx in range(5):  # 0-4
                    if session_idx < 3:
                        suffix = "T"
                    else:
                        suffix = "E"
                    
                    filename = f"B{subj_str}{session_idx+1:02d}{suffix}.gdf"
                    filepath = os.path.join(data_path, filename)
                    
                    if not os.path.exists(filepath):
                        print(f"Warning: File {filepath} not found. Skipping.")
                        continue
                        
                    # Load raw data
                    try:
                        raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=verbose)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                        continue

                    # Check if raw contains target events (2b: 769=left, 770=right)
                    target_events = {'769', '770'}
                    if not any(event in raw.annotations.description for event in target_events):
                        if verbose:
                            print(f"Skipping {filepath}: No target events found.")
                        continue

                    description = pd.Series({
                        "subject": subject_id,
                        "session": f"session_{session_idx}",
                        "run": f"session_{session_idx}"
                    })
                    datasets.append(BaseDataset(raw, description))
        
        if not datasets:
            raise FileNotFoundError(f"No valid GDF files found in {data_path} for subjects {subject_ids}")
            
        dataset_obj = BaseConcatDataset(datasets)
        
    else:
        # Use MOABB if data_path is not provided
        dataset_name = "BNCI2014001" if dataset_code == "2a" else "BNCI2014004"
        dataset_obj = MOABBDataset(dataset_name, subject_ids=subject_ids)

    # Preprocessing pipeline
    n_channels = 22 if dataset_code == "2a" else 3
    preprocessors = [
        # Pick EEG channels based on dataset
        # For BCIC IV 2a: first 22 are EEG
        # For BCIC IV 2b: first 3 are EEG
        Preprocessor(lambda raw: raw.pick(range(n_channels)), apply_on_array=False),
        Preprocessor(scale, factor=1e6, apply_on_array=True),
        Preprocessor("resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose)
    ]

    l_freq, h_freq = preprocessing_dict.get("low_cut"), preprocessing_dict.get("high_cut")
    if l_freq is not None or h_freq is not None:
        preprocessors.append(Preprocessor("filter", l_freq=l_freq, h_freq=h_freq,
                                          verbose=verbose))

    preprocess(dataset_obj, preprocessors)

    # create windows
    # BaseConcatDataset access
    sfreq = dataset_obj.datasets[0].raw.info["sfreq"]
    
    # Define window relative to event
    # 4.0s duration -> 1000 samples at 250Hz
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    
    # Event mapping
    # BCIC IV 2a: 769=Left Hand, 770=Right Hand, 771=Foot, 772=Tongue (4 classes)
    # BCIC IV 2b: 769=Left Hand, 770=Right Hand (2 classes)
    if verbose:
        descriptions = set()
        for ds in dataset_obj.datasets:
            descriptions.update(ds.raw.annotations.description)
        print("Unique descriptions in annotations:", descriptions)

    if dataset_code == "2a":
        mapping = {
            '769': 0, '770': 1, '771': 2, '772': 3,
        }
    else:  # 2b
        mapping = {
            '769': 0, '770': 1,
        }

    # Apply mapping for GDF files to ensure 0-3 labels
    # Braindecode will create windows from trial_start_offset to trial_stop_offset
    # resulting in fixed length windows if offsets are fixed.
    windows_dataset = create_windows_from_events(
        dataset_obj,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples, 
        mapping=mapping if data_path else None, 
        preload=True,
        drop_bad_windows=False 
    )

    return windows_dataset
