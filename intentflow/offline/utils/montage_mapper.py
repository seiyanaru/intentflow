
"""
Automated Montage Mapping for Neuro-Gated OTTA.

This module provides functionality to automatically assign roles (Motor, Noise, Neutral)
to EEG channels based on their names, enabling universal dataset compatibility.
"""

import re

# Standard 10-20 System Role Definitions
# Based on literature (Pfurtscheller et al., etc.) for Motor Imagery
STANDARD_ROLES = {
    # Signals of Interest (Motor Cortex)
    "MOTOR": [
        "C3", "C4", "Cz",       # Primary
        "FC3", "FC4", "FCz",    # Pre-motor
        "CP3", "CP4", "CPz",    # Somatosensory
        "C1", "C2", "C5", "C6", # High-density Motor
        "FC1", "FC2", "FC5", "FC6",
        "CP1", "CP2", "CP5", "CP6"
    ],
    
    # Artifact Sources (To be monitored/gated)
    "NOISE": [
        "Fp1", "Fp2", "FpZ",    # Eye blinks (Vertical EOG)
        "F7", "F8",             # Eye movement (Horizontal EOG)
        "AF3", "AF4", "AF7", "AF8", "AFz", 
        "T7", "T8",             # Temporal muscle (Jaw clenching)
        "T3", "T4",             # Old nomenclature for T7/T8
        "FT7", "FT8", "TP7", "TP8",
        "M1", "M2", "A1", "A2", # Mastoids (often reference, but if active -> noise)
        "EOG", "EOG-L", "EOG-R", "EOG-H", "EOG-V", # Explicit EOG channels
        "Fz", "Pz", "POz", "P1", "P2" # Non-motor/irrelevant channels for MI
    ]
}

def normalize_name(name: str) -> str:
    """
    Normalize channel name to standard 10-20 format.
    Examples: "EEG-C3" -> "C3", "C3." -> "C3", "Channel 1" -> "Channel 1"
    """
    # Remove common prefixes/suffixes
    name = name.upper()
    name = re.sub(r'EEG[-_:]?', '', name)  # Remove EEG- prefix
    name = re.sub(r'[._].*$', '', name)    # Remove suffixes like .1 or _ref
    name = name.strip()
    return name

def get_electrode_roles(ch_names: list) -> dict:
    """
    Map channel names to roles based on standard dictionary.
    
    Args:
        ch_names: List of channel names from the dataset.
        
    Returns:
        Dictionary with 'motor' and 'noise' keys containing lists of indices.
    """
    roles = {'motor': [], 'noise': []}
    
    print("[MontageMapper] Auto-configuring channel roles...")
    
    for idx, raw_name in enumerate(ch_names):
        name = normalize_name(raw_name)
        
        if name in [n.upper() for n in STANDARD_ROLES['MOTOR']]:
            roles['motor'].append(idx)
            # print(f"  - Ch {idx} ({raw_name}) -> MOTOR")
            
        elif name in [n.upper() for n in STANDARD_ROLES['NOISE']]:
            roles['noise'].append(idx)
            # print(f"  - Ch {idx} ({raw_name}) -> NOISE")
            
    # Summary
    print(f"[MontageMapper] Identified {len(roles['motor'])} Motor channels and {len(roles['noise'])} Noise channels.")
    
    # Fallback / Sanity Check
    if not roles['motor']:
        print("[MontageMapper] WARNING: No motor channels identified! Check channel nomenclature.")
        
    return roles

if __name__ == "__main__":
    # Test case (BCIC 2a style)
    test_chans = ['EEG-Fz', 'EEG-0', 'EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Fp1', 'EOG-left']
    roles = get_electrode_roles(test_chans)
    print("Result:", roles)
