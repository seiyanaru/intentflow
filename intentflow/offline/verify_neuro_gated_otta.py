
"""
Verification script for Neuro-Gated OTTA.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.tcformer_otta import TCFormerOTTA

class MockDataset:
    def __init__(self):
        # Mixed Motor and Noise channels
        self.ch_names = ['C3', 'C4', 'Cz', 'Fp1', 'EOG', 'Pz', 'Fz'] 

class MockDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.test_set = MockDataset()
        self.dataset_test = self.test_set # Fallback check

def verify_neuro_gating():
    print("--- Verifying Neuro-Gated OTTA Implementation ---")
    
    # 1. Instantiate Model
    model = TCFormerOTTA(
        n_classes=4,
        n_channels=7,
        enable_otta=True,
        pmax_threshold=0.7,
        sal_threshold=0.5
    )
    
    # Mock log method to avoid MisconfigurationException
    model.log = lambda *args, **kwargs: None
    model.log_dict = lambda *args, **kwargs: None
    
    # Mock Trainer
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1)
    trainer.datamodule = MockDataModule()
    model.trainer = trainer
    
    # 2. Trigger on_test_start (where Neuro-Gating init happens)
    print("\n[Step 1] Triggering on_test_start...")
    
    print(f"Debug: model.trainer exists? {model.trainer is not None}")
    if model.trainer:
        print(f"Debug: model.trainer.datamodule exists? {getattr(model.trainer, 'datamodule', None) is not None}")
    
    model.on_test_start()
    
    # Check if roles were assigned
    if model.otta.channel_roles:

        print("SUCCESS: Channel roles assigned.")
        print("Roles:", model.otta.channel_roles)
        
        # Verify indices
        # C3(0), C4(1), Cz(2) -> Motor
        # Fp1(3), EOG(4) -> Noise
        motor = model.otta.channel_roles['motor']
        noise = model.otta.channel_roles['noise']
        assert 0 in motor and 1 in motor and 2 in motor
        assert 3 in noise and 4 in noise
        print("Indices verification passed.")
    else:
        print("FAILURE: Channel roles not assigned.")
        return

    # 3. Run Test Step
    print("\n[Step 2] Running Test Step...")
    # Mock batch
    B, C, T = 2, 7, 1000
    x = torch.randn(B, C, T)
    y = torch.randint(0, 4, (B,))
    batch = (x, y)
    
    # Run step
    out = model.test_step(batch, 0)
    
    # 4. Check Stats
    print("\n[Step 3] Checking OTTA Stats...")
    stats = model.test_otta_stats
    if len(stats) > 0:
        entry = stats[0]
        if 'neuro_score' in entry:
            print("SUCCESS: neuro_score found in stats.")
            print("Scores:", entry['neuro_score'])
            
            # Check range
            score = entry['neuro_score']
            if (score >= -1).all() and (score <= 1).all():
                 print("Range check passed [-1, 1].")
            else:
                 print("Range check FAILED.")
        else:
             print("FAILURE: neuro_score MISSING from stats.")
    else:
        print("FAILURE: No stats recorded.")

if __name__ == "__main__":
    verify_neuro_gating()
