"""
TCFormer with Pmax-SAL OTTA Integration

This module provides a TCFormer variant that uses Pmax-SAL gated
online test-time adaptation for improved cross-subject performance.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, Any
import json
import os
import numpy as np

from models.tcformer.tcformer import TCFormer as TCFormerBase
from models.tcformer.classification_module import ClassificationModule
from models.pmax_sal_otta import PmaxSAL_OTTA


class TCFormerOTTAModule(pl.LightningModule):
    """
    TCFormer with Pmax-SAL OTTA for calibration-free cross-subject MI-EEG.
    
    This module wraps TCFormer with the Pmax-SAL gated adaptation mechanism,
    enabling online adaptation during test time without requiring calibration.
    """
    
    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        # OTTA parameters
        pmax_threshold: float = 0.7,
        sal_threshold: float = 0.5,
        enable_otta: bool = True,
        # TCFormer parameters  
        F1: int = 32,
        D: int = 2,
        temp_kernel_lengths: list = None,
        q_heads: int = 4,
        kv_heads: int = 2,
        trans_depth: int = 2,
        tcn_kernel_size: int = 3,
        tcn_depth: int = 2,
        tcn_drop: float = 0.2,
        # Training parameters
        lr: float = 0.0009,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        scheduler: bool = True,
        max_epochs: int = 1000,
        warmup_epochs: int = 20,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Default kernel lengths
        if temp_kernel_lengths is None:
            temp_kernel_lengths = [20, 32, 64]
        
        # Create base TCFormer model
        self.tcformer = TCFormerBase(
            n_classes=n_classes,
            Chans=n_channels,
            F1=F1,
            D=D,
            temp_kernel_lengths=temp_kernel_lengths,
            q_heads=q_heads,
            kv_heads=kv_heads,
            trans_depth=trans_depth,
            tcn_kernel_size=tcn_kernel_size,
            tcn_depth=tcn_depth,
            tcn_drop=tcn_drop,
        )
        
        # OTTA wrapper (initialized after training)
        self.otta = None
        self.enable_otta = enable_otta
        self.pmax_threshold = pmax_threshold
        self.sal_threshold = sal_threshold
        
        # Training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.use_scheduler = scheduler
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.n_classes = n_classes
        
        # Metrics
        self.test_outputs = []
        self.train_dataloader_ref = None
        
    def forward(self, x):
        """Forward pass."""
        return self.tcformer(x)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        
        if self.use_scheduler:
            from utils.lr_scheduler import linear_warmup_cosine_decay
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=linear_warmup_cosine_decay(
                        self.warmup_epochs,
                        self.max_epochs
                    )
                ),
                'interval': 'epoch',
            }
            return [optimizer], [scheduler]
        
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        # Compute accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def on_test_start(self):
        """Initialize OTTA before testing."""
        if self.enable_otta:
            print("[TCFormerOTTA] Initializing Pmax-SAL OTTA...")
            
            self.otta = PmaxSAL_OTTA(
                model=self.tcformer,
                n_classes=self.n_classes,
                pmax_threshold=self.pmax_threshold,
                sal_threshold=self.sal_threshold,
                enable_adaptation=True,
            )
            
            # Compute source prototypes if training data is available
            if self.train_dataloader_ref is not None:
                self.otta.compute_source_prototypes(
                    self.train_dataloader_ref,
                    device=self.device
                )
            else:
                print("[TCFormerOTTA] Warning: No training data for prototypes")
    
    def test_step(self, batch, batch_idx):
        """Test step with OTTA."""
        x, y = batch
        
        if self.otta is not None and self.enable_otta:
            # Use OTTA for prediction
            result = self.otta(x, return_debug=True)
            logits = result['logits']
            preds = result['pred']
            pmax = result['pmax']
            sal = result['sal']
            adapted = result['adapted']
        else:
            # Standard prediction
            logits = self.forward(x)
            preds = logits.argmax(dim=-1)
            pmax = torch.softmax(logits, dim=-1).max(dim=-1)[0]
            sal = torch.ones_like(pmax)
            adapted = False
        
        # Compute metrics
        correct = (preds == y).float()
        
        self.test_outputs.append({
            'preds': preds.cpu(),
            'labels': y.cpu(),
            'correct': correct.cpu(),
            'pmax': pmax.cpu(),
            'sal': sal.cpu(),
            'adapted': adapted,
        })
        
        return {'correct': correct.mean()}
    
    def on_test_epoch_end(self):
        """Aggregate test results."""
        all_preds = torch.cat([o['preds'] for o in self.test_outputs])
        all_labels = torch.cat([o['labels'] for o in self.test_outputs])
        all_correct = torch.cat([o['correct'] for o in self.test_outputs])
        all_pmax = torch.cat([o['pmax'] for o in self.test_outputs])
        all_sal = torch.cat([o['sal'] for o in self.test_outputs])
        
        
        # Calculate Kappa
        kappa = self.test_kappa(all_preds, all_labels)
        
        acc = all_correct.mean().item()
        
        print(f"\n[TCFormerOTTA] Test Results:")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(f"  Kappa: {kappa:.4f}")
        print(f"  Pmax (mean): {all_pmax.mean():.3f}")
        print(f"  SAL (mean): {all_sal.mean():.3f}")
        
        if self.otta is not None:
            self.otta.print_stats()
        
        self.log('test_acc', acc)
        self.log('test_kappa', kappa)
        
        # Clear outputs
        self.test_outputs = []
        
        return acc
    
    def set_train_dataloader(self, dataloader):
        """Set training dataloader for prototype computation."""
        self.train_dataloader_ref = dataloader


class TCFormerOTTA(ClassificationModule):
    """
    TCFormer with OTTA wrapper for use with existing training pipeline.
    """
    
    def __init__(self, n_classes, **kwargs):
        # Extract OTTA-specific args
        pmax_threshold = kwargs.pop('pmax_threshold', 0.7)
        sal_threshold = kwargs.pop('sal_threshold', 0.5)
        enable_otta = kwargs.pop('enable_otta', True)
        
        # Create base TCFormer
        model = TCFormerBase(
            n_classes=n_classes,
            n_channels=kwargs.get('n_channels', 22),
            F1=kwargs.get('F1', 32),
            D=kwargs.get('D', 2),
            temp_kernel_lengths=kwargs.get('temp_kernel_lengths', [20, 32, 64]),
            q_heads=kwargs.get('q_heads', 4),
            kv_heads=kwargs.get('kv_heads', 2),
            trans_depth=kwargs.get('trans_depth', 2),
            tcn_kernel_size=kwargs.get('tcn_kernel_size', 3),
            tcn_depth=kwargs.get('tcn_depth', 2),
            tcn_drop=kwargs.get('tcn_drop', 0.2),
        )
        
        # Initialize parent
        super().__init__(model=model, n_classes=n_classes, **kwargs)
        
        # Store OTTA config
        self.pmax_threshold = pmax_threshold
        self.sal_threshold = sal_threshold
        self.enable_otta = enable_otta
        self.otta = None
        self.train_dataloader_ref = None
        self.n_classes = n_classes
        
        # Storage for OTTA statistics
        self.test_otta_stats = []
    
    def on_test_start(self):
        """Initialize OTTA before testing."""
        if self.enable_otta:
            print("[TCFormerOTTA] Initializing Pmax-SAL OTTA...")
            
            self.otta = PmaxSAL_OTTA(
                model=self.model,
                n_classes=self.n_classes,
                pmax_threshold=self.pmax_threshold,
                sal_threshold=self.sal_threshold,
                enable_adaptation=True,
            )
            
            if self.train_dataloader_ref is not None:
                self.otta.compute_source_prototypes(
                    self.train_dataloader_ref,
                    device=self.device
                )
    
    def test_step(self, batch, batch_idx):
        """Override test step to use OTTA."""
        x, y = batch
        
        if self.otta is not None and self.enable_otta:
            result = self.otta(x, return_debug=True)
            logits = result['logits']
            preds = result['pred']
            
            # Store OTTA stats for analysis
            self.test_otta_stats.append({
                'pmax': result['pmax'].cpu(),
                'sal': result['sal'].cpu(),
                'adapted': torch.tensor(result['adapted']).repeat(len(x)), # Broadcast boolean
                'adapt_weight': result['adapt_weight'].cpu(),
                'pred': result['pred'].cpu(),
                'label': y.cpu(),
            })
        else:
            logits = self.forward(x)
            preds = logits.argmax(dim=-1)
        
        # Use parent's test logic
        loss = nn.functional.cross_entropy(logits, y)
        acc = (preds == y).float().mean()
        
        # IMPORTANT: Update metric objects
        self.test_kappa.update(preds, y)
        self.test_cm.update(preds, y)
        
        # Log metrics (on_epoch=True adds them to valid/test results dict)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_kappa', self.test_kappa, prog_bar=False, on_step=False, on_epoch=True)
        
        # Store for analysis (needed by ClassificationModule.on_test_epoch_end)
        if self.test_logits:
             self.test_logits.append(logits.detach().cpu())
        if self.test_labels:
             self.test_labels.append(y.detach().cpu())
        
        return {'test_loss': loss, 'test_acc': acc}

    def on_test_epoch_end(self):
        """Save OTTA statistics."""
        # 1. Save detailed OTTA stats
        if self.test_otta_stats and self.subject_id != "unknown":
            import os
            import numpy as np
            os.makedirs(self.results_dir, exist_ok=True)
            stats_path = os.path.join(self.results_dir, f"otta_stats_s{self.subject_id}_{self.model_name}.npz")
            
            # Aggregate stats
            pmax = torch.cat([x['pmax'] for x in self.test_otta_stats], dim=0).numpy()
            sal = torch.cat([x['sal'] for x in self.test_otta_stats], dim=0).numpy()
            adapted = torch.cat([x['adapted'] for x in self.test_otta_stats], dim=0).numpy()
            adapt_weight = torch.cat([x['adapt_weight'] for x in self.test_otta_stats], dim=0).numpy()
            pred = torch.cat([x['pred'] for x in self.test_otta_stats], dim=0).numpy()
            label = torch.cat([x['label'] for x in self.test_otta_stats], dim=0).numpy()
            
            np.savez(
                stats_path,
                pmax=pmax,
                sal=sal,
                adapted=adapted,
                adapt_weight=adapt_weight,
                pred=pred,
                label=label
            )
            print(f"[TCFormerOTTA] Saved OTTA stats to {stats_path}")
            self.test_otta_stats = []
            
        # 2. Call parent's logic
        super().on_test_epoch_end()
        

