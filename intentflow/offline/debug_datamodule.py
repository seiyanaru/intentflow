
import os
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
sys.path.append(os.getcwd())

from intentflow.offline.utils.get_datamodule_cls import get_datamodule_cls

def print_structure(obj, depth=0, max_depth=3):
    indent = "  " * depth
    if depth > max_depth:
        return
    
    # Check for channel info
    if hasattr(obj, 'ch_names'):
        print(f"{indent}[FOUND ch_names]: {obj.ch_names}")
    
    # Check identifying attrs
    if hasattr(obj, 'datasets'):
        print(f"{indent}Has datasets list (len={len(obj.datasets)})")
        if len(obj.datasets) > 0:
            print_structure(obj.datasets[0], depth+1, max_depth)
    
    if hasattr(obj, 'windows'):
        print(f"{indent}Has windows attribute")
        print_structure(obj.windows, depth+1, max_depth)

def debug_datamodule():
    dataset_name = "bcic2a"
    subject_id = 1
    
    # Minimal config
    preprocessing_dict = {
        "data_path": "/workspace-cloud/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/",
        "z_scale": True,
        "batch_size": 16,
        "sfreq": 250,
        "low_cut": None, 
        "high_cut": None,
        "start": 0.0,
        "stop": 4.0,
    }
    
    print(f"Instantiating {dataset_name} for Subject {subject_id}...")
    cls = get_datamodule_cls(dataset_name)
    dm = cls(preprocessing_dict, subject_id)
    
    print("Running prepare_data()...")
    dm.prepare_data()
    
    print("Running setup()...")
    dm.setup()
    
    ds = dm.dataset.datasets[0]
    print(f"Type of ds: {type(ds)}")
    print(f"ds keys: {ds.__dict__.keys()}")
    
    if hasattr(ds, 'raw'):
        print(f"Found .raw attribute (Type: {type(ds.raw)})")
        if hasattr(ds.raw, 'ch_names'):
            print(f"FOUND ch_names in ds.raw: {ds.raw.ch_names}")
    
    if hasattr(ds, 'windows'):
        print(f"Found .windows attribute (Type: {type(ds.windows)})")
        try:
            # Check if it has info directly
            if hasattr(ds.windows, 'info'):
                    print(f"ds.windows.info keys: {ds.windows.info.keys()}")
                    if 'ch_names' in ds.windows.info:
                        print(f"FOUND ch_names in ds.windows.info: {ds.windows.info['ch_names']}")
            # Check if it is a BaseConcatDataset or similar
            if hasattr(ds.windows, 'datasets'):
                    print(f"ds.windows has datasets len={len(ds.windows.datasets)}")
        except Exception as e:
            print(f"Error inspecting windows: {e}")

    # Check description
    if hasattr(ds, 'description'):
        print(f"ds.description: {ds.description}")

    print("\n--- Searching for Montage ---")
    # Simulate logic in TCFormerOTTA we want to fix
    found = False
    if hasattr(dm, 'dataset') and hasattr(dm.dataset, 'datasets'):
         ds = dm.dataset.datasets[0]
         if hasattr(ds, 'ch_names'):
             print(f"Found in dm.dataset.datasets[0].ch_names: {ds.ch_names}")
             found = True
         elif hasattr(ds, 'raw') and hasattr(ds.raw, 'ch_names'):
             print(f"Found in dm.dataset.datasets[0].raw.ch_names: {ds.raw.ch_names}")
             found = True
         elif hasattr(ds, 'windows') and hasattr(ds.windows, 'ch_names'):
             print(f"Found in dm.dataset.datasets[0].windows.ch_names: {ds.windows.ch_names}")
             found = True
             
    if not found:
        print("FAILED to find channel names with current logic.")
    else:
        # Verify Montage Mapping
        from intentflow.offline.utils.montage_mapper import get_electrode_roles
        
        roles = get_electrode_roles(ds.raw.ch_names) # We now know where they are
        print("\n--- Montage Verification ---")
        print(f"Motor Channels: {len(roles['motor'])} -> {[ds.raw.ch_names[i] for i in roles['motor']]}")
        print(f"Noise Channels: {len(roles['noise'])} -> {[ds.raw.ch_names[i] for i in roles['noise']]}")

if __name__ == "__main__":
    debug_datamodule()
