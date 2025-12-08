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
            # BCIC IV 2a filename convention
            # A01T.gdf, A01E.gdf, etc.
            subj_str = f"{subject_id:02d}"
            
            # Load both Training (T) and Evaluation (E) sessions
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

                # Check if raw contains target events
                target_events = {'769', '770', '771', '772'}
                # annotations.description is a numpy array of strings
                if not any(event in raw.annotations.description for event in target_events):
                    if verbose:
                        print(f"Skipping {filepath}: No target events found.")
                    continue

                # Prepare description

                description = pd.Series({
                    "subject": subject_id,
                    "session": session_code,
                    "run": session_code
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
    preprocessors = [
        # Pick EEG channels. MNE usually detects them from GDF, but sometimes type setting is needed.
        # For BCIC IV 2a, first 22 are EEG.
        # Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        # Explicitly pick first 22 channels to avoid EOG
        Preprocessor(lambda raw: raw.pick(range(22)), apply_on_array=False),
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
    
    # Event mapping for BCIC IV 2a
    # 769: Left Hand, 770: Right Hand, 771: Foot, 772: Tongue
    if verbose:
        descriptions = set()
        for ds in dataset.datasets:
             descriptions.update(ds.raw.annotations.description)
        print("Unique descriptions in annotations:", descriptions)

    mapping = {
        '769': 0, '770': 1, '771': 2, '772': 3,
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
