import os
import pandas as pd
import mne
from typing import Dict, Optional, List

from braindecode.datasets import BaseConcatDataset, BaseDataset, MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
    # scale, # Removed, define locally or use lambda
)

def scale(data, factor):
    return data * factor



def load_bcic4(subject_ids: list, dataset: str = "2a", preprocessing_dict: Dict = None,
              verbose: str = "WARNING", data_path: Optional[str] = None):
    
    if data_path:
        # Local GDF loading
        datasets = []
        for subject_id in subject_ids:
            subj_str = f"{subject_id:02d}"
            
            if dataset == "2a":
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
                        if verbose:
                            print(f"Skipping {filepath}: No target events found.")
                        continue

                    description = pd.Series({
                        "subject": subject_id,
                        "session": session_code,
                        "run": session_code
                    })
                    
                    datasets.append(BaseDataset(raw, description))
                    
            elif dataset == "2b":
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
            
        dataset = BaseConcatDataset(datasets)
        
    else:
        # Use MOABB if data_path is not provided
        dataset_name = "BNCI2014001" if dataset == "2a" else "BNCI2014004"
        dataset = MOABBDataset(dataset_name, subject_ids=subject_ids)

    # Preprocessing pipeline
    n_channels = 22 if dataset == "2a" else 3
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

    preprocess(dataset, preprocessors)

    # create windows
    # BaseConcatDataset access
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    
    # Define window relative to event
    # 4.0s duration -> 1000 samples at 250Hz
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    
    # Event mapping
    # BCIC IV 2a: 769=Left Hand, 770=Right Hand, 771=Foot, 772=Tongue (4 classes)
    # BCIC IV 2b: 769=Left Hand, 770=Right Hand (2 classes)
    if verbose:
        descriptions = set()
        for ds in dataset.datasets:
             descriptions.update(ds.raw.annotations.description)
        print("Unique descriptions in annotations:", descriptions)

    if dataset == "2a":
        mapping = {
            '769': 0, '770': 1, '771': 2, '772': 3,
        }
    else:  # 2b
        mapping = {
            '769': 0, '770': 1,
        }

    
    # If using MOABB, mapping might differ or be handled internally, 
    # but providing mapping is safer for consistency.
    # MOABBDataset usually returns annotations with string descriptions, 
    # but create_windows_from_events can handle mapping from original event codes if present,
    # or from string descriptions if we map strings.
    # For local GDF, we have raw event codes (769..).
    # For MOABB, we might have 'left_hand', etc.
    
    if data_path:
        # Local loading uses raw event codes
        pass 
    else:
        # MOABB dataset might have different event handling. 
        # Standard BNCI2014001 has events mapped.
        # However, to match local loader, we might need to check.
        # Usually BNCI2014001 (2a) in braindecode/moabb works with standard mapping?
        # Let's assume the mapping argument works for GDF codes.
        # If MOABB converts to strings, we might need string keys.
        # BNCI2014001 events: 'left_hand', 'right_hand', 'feet', 'tongue'
        # If dataset comes from MOABB, it might be safer to let it use default or map strings.
        
        # Checking if we should provide mapping for MOABB path
        # Existing code didn't provide mapping -> it relied on braindecode defaults or MOABB
        # Let's inspect original code behavior.
        # Original: create_windows_from_events(..., mapping=None)
        # Braindecode infers mapping from events. 
        # For GDF, events are 769..772. We MUST map them to 0..3 for the model.
        pass

    # Apply mapping for GDF files to ensure 0-3 labels
    # Braindecode will create windows from trial_start_offset to trial_stop_offset
    # resulting in fixed length windows if offsets are fixed.
    windows_dataset = create_windows_from_events(
        dataset, 
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples, 
        mapping=mapping if data_path else None, 
        preload=True,
        drop_bad_windows=False 
    )

    return windows_dataset
