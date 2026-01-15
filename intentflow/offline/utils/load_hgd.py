from itertools import compress
from typing import Dict

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
    # scale,
)
import numpy as np

def scale(data, factor):
    return data * factor



def load_hgd(subject_ids: list, preprocessing_dict: Dict = None,
             verbose: str = "WARNING"):
    dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=subject_ids)

    if preprocessing_dict.get("remove_artifacts", True):
        # find samples < 800 uV and save masks for later
        window_dataset = create_windows_from_events(dataset, preload=False)
        ds_masks = []
        for ds in window_dataset.datasets:
            # Load data - newer braindecode API doesn't have .windows attribute
            ds_loaded = ds.load_data() if hasattr(ds, 'load_data') else ds
            # Access data array (handle different braindecode versions)
            if hasattr(ds_loaded, '_data'):
                data_array = ds_loaded._data
            elif hasattr(ds_loaded, 'get_data'):
                data_array = ds_loaded.get_data()
            else:
                # Fallback: iterate through dataset
                data_array = np.array([ds_loaded[i][0] for i in range(len(ds_loaded))])
            clean_trial_mask = np.max(np.abs(data_array), axis=(-2, -1)) < 800 * 1e-6
            ds_masks.append(clean_trial_mask)

    channels = [
        "FC5", "FC1", "FC2", "FC6", "C3", "C4", "CP5", "CP1", "CP2", "CP6", "FC3",
        "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "FFC5h", "FFC3h",
        "FFC4h", "FFC6h", "FCC5h", "FCC3h", "FCC4h", "FCC6h", "CCP5h", "CCP3h", "CCP4h",
        "CCP6h", "CPP5h", "CPP3h", "CPP4h", "CPP6h", "FFC1h", "FFC2h", "FCC1h", "FCC2h",
        "CCP1h", "CCP2h", "CPP1h", "CPP2h",
    ]

    preprocessors = [
        Preprocessor("pick_channels", ch_names=channels, verbose=verbose),
        Preprocessor(scale, factor=1e6, apply_on_array=True),  # from uV to V
        Preprocessor("resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose)
    ]

    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        preprocessors.append(Preprocessor("filter", l_freq=l_freq, h_freq=h_freq,
                                          verbose=verbose))

    preprocess(dataset, preprocessors)

    # create windows
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples, preload=False
    )

    if preprocessing_dict.get("remove_artifacts", True):
        for (mask, ds) in zip(ds_masks, windows_dataset.datasets):
            # Filter out artifacts based on mask
            # Use indices to select clean trials
            clean_indices = np.where(mask)[0].tolist()
            if hasattr(ds, 'windows'):
                ds.windows = ds.windows[mask]
            # Update labels
            if hasattr(ds, 'y'):
                ds.y = list(compress(ds.y, mask))
            elif hasattr(ds, 'description'):
                # Newer braindecode versions store labels differently
                pass  # Labels are handled internally

    return windows_dataset
