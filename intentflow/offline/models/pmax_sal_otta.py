"""
Pmax-SAL Gated Online Test-Time Adaptation (OTTA) for MI-EEG

This module implements a novel TTA approach combining:
1. Pmax: Maximum prediction probability as overconfidence indicator
2. SAL (Source Alignment Level): Cosine similarity with source prototypes

Reference:
- EATA: Efficient Test-Time Model Adaptation without Forgetting
- MI-IASW: Test-Time Adaptation for Cross-Subject Motor Imagery EEG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import copy


class PmaxSAL_OTTA(nn.Module):
    """
    Pmax-SAL Gated Online Test-Time Adaptation Module
    
    Gating Logic:
    - High pmax + High SAL → Trust, adapt BN
    - High pmax + Low SAL  → Overconfidence, skip adaptation
    - Low pmax + High SAL  → Uncertain but aligned, cautious adapt
    - Low pmax + Low SAL   → Unreliable, skip adaptation
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_classes: int = 4,
        pmax_threshold: float = 0.7,
        sal_threshold: float = 0.5,
        alpha: float = 0.5,  # Weight between pmax and SAL
        bn_momentum: float = 0.1,  # BN stats update momentum
        enable_adaptation: bool = True,
    ):
        """
        Args:
            model: Pretrained model (TCFormer)
            n_classes: Number of classes
            pmax_threshold: Threshold for pmax confidence
            sal_threshold: Threshold for Source Alignment Level
            alpha: Weight for combining pmax and SAL in gate
            bn_momentum: Momentum for BN stats update
            enable_adaptation: Whether to enable adaptation
        """
        super().__init__()
        
        self.model = model
        self.n_classes = n_classes
        self.pmax_threshold = pmax_threshold
        self.sal_threshold = sal_threshold
        self.alpha = alpha
        self.bn_momentum = bn_momentum
        self.enable_adaptation = enable_adaptation
        
        # Source prototypes: [n_classes, feature_dim]
        # Will be computed during training
        self.register_buffer('source_prototypes', None)
        self.register_buffer('prototype_counts', torch.zeros(n_classes))
        
        # Store source BN statistics
        self.source_bn_stats = {}
        self._save_source_bn_stats()
        
        # Feature extraction hook
        self._features = None
        self._register_feature_hook()
        
        # Statistics for logging
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'skipped_overconfident': 0,
            'skipped_unreliable': 0,
        }
    
    def _register_feature_hook(self):
        """Register hook to capture features before final classifier."""
        # Find the last layer before classifier
        # For TCFormer, this is typically the pooling layer output
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._features = output[0]
            else:
                self._features = output
        
        # Try to find classifier layer and hook its input
        for name, module in self.model.named_modules():
            if 'classifier' in name.lower() or 'fc' in name.lower():
                if hasattr(module, 'register_forward_hook'):
                    # Hook on the input to classifier
                    module.register_forward_hook(hook_fn)
                    break
    
    def _save_source_bn_stats(self):
        """Save BN statistics from source-trained model."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.source_bn_stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                }
    
    def _restore_source_bn_stats(self):
        """Restore BN statistics to source values."""
        for name, module in self.model.named_modules():
            if name in self.source_bn_stats:
                module.running_mean.copy_(self.source_bn_stats[name]['running_mean'])
                module.running_var.copy_(self.source_bn_stats[name]['running_var'])
    
    def _update_bn_stats(self, x: torch.Tensor):
        """Update BN running statistics with current batch."""
        # Set model to training mode temporarily to update BN stats
        was_training = self.model.training
        self.model.train()
        
        with torch.no_grad():
            # Forward pass to update BN stats
            _ = self.model(x)
        
        if not was_training:
            self.model.eval()
    
    def compute_source_prototypes(
        self,
        dataloader,
        device: torch.device = None
    ):
        """
        Compute source prototypes from training data.
        Should be called after training, before test-time adaptation.
        
        Args:
            dataloader: DataLoader with source data
            device: Device to use
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        self.model.eval()
        
        # Accumulate features per class
        feature_sums = {}
        feature_counts = {}
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(device), batch[1].to(device)
                
                # Forward pass
                logits = self.model(x)
                
                if self._features is not None:
                    features = self._features
                    # Flatten if needed (HANDLE [B, D, 1] case)
                    if len(features.shape) > 2:
                        features = features.mean(dim=-1)
                    
                    # Ensure features are [B, D]
                    if len(features.shape) == 3 and features.shape[2] == 1:
                         features = features.squeeze(2)
                    
                    # Accumulate per class
                    for i in range(len(y)):
                        label = y[i].item()
                        feat_vec = features[i]
                        # Handle potential singleton dim in vec [D, 1] -> [D]
                        if feat_vec.dim() > 1:
                            feat_vec = feat_vec.squeeze()
                            
                        if label not in feature_sums:
                            feature_sums[label] = torch.zeros_like(feat_vec)
                            feature_counts[label] = 0
                        feature_sums[label] += feat_vec
                        feature_counts[label] += 1
        
        # Compute mean prototypes
        feature_dim = next(iter(feature_sums.values())).shape[0]
        prototypes = torch.zeros(self.n_classes, feature_dim, device=device)
        
        for label in range(self.n_classes):
            if label in feature_sums and feature_counts[label] > 0:
                prototypes[label] = feature_sums[label] / feature_counts[label]
                self.prototype_counts[label] = feature_counts[label]
        
        # Normalize prototypes
        prototypes = F.normalize(prototypes, p=2, dim=1)
        self.source_prototypes = prototypes
        
        print(f"[PmaxSAL] Computed source prototypes: shape={prototypes.shape}")
    
    def compute_pmax(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute maximum prediction probability.
        
        Returns:
            pmax: Maximum softmax probability [B]
            pred: Predicted class [B]
        """
        probs = F.softmax(logits, dim=-1)
        pmax, pred = probs.max(dim=-1)
        return pmax, pred
    
    def compute_sal(
        self,
        features: torch.Tensor,
        pred_classes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Source Alignment Level (SAL).
        
        SAL = cosine_similarity(feature, prototype[predicted_class])
        
        Args:
            features: Feature vectors [B, D]
            pred_classes: Predicted class indices [B]
        
        Returns:
            sal: Source Alignment Level [B]
        """
        if self.source_prototypes is None:
            # No prototypes available, return high SAL to enable adaptation
            return torch.ones(features.shape[0], device=features.device)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Get prototype for each predicted class
        batch_prototypes = self.source_prototypes[pred_classes]  # [B, D]
        
        # Compute cosine similarity
        sal = (features * batch_prototypes).sum(dim=1)  # [B]
        
        return sal
    
    def compute_gate(
        self,
        pmax: torch.Tensor,
        sal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating decision based on pmax and SAL.
        
        Gating Logic:
        - High pmax + High SAL → adapt_weight = 1.0
        - High pmax + Low SAL  → adapt_weight = 0.0 (overconfident)
        - Low pmax + High SAL  → adapt_weight = 0.5 (cautious)
        - Low pmax + Low SAL   → adapt_weight = 0.0 (unreliable)
        
        Returns:
            should_adapt: Boolean mask [B]
            adapt_weight: Weight for adaptation [B]
        """
        high_pmax = pmax > self.pmax_threshold
        high_sal = sal > self.sal_threshold
        
        # Initialize weights
        adapt_weight = torch.zeros_like(pmax)
        
        # Case 1: High pmax + High SAL → Full adaptation
        mask1 = high_pmax & high_sal
        adapt_weight[mask1] = 1.0
        
        # Case 2: High pmax + Low SAL → Skip (overconfident)
        mask2 = high_pmax & ~high_sal
        adapt_weight[mask2] = 0.0
        
        # Case 3: Low pmax + High SAL → Cautious adaptation
        mask3 = ~high_pmax & high_sal
        adapt_weight[mask3] = 0.5
        
        # Case 4: Low pmax + Low SAL → Skip (unreliable)
        mask4 = ~high_pmax & ~high_sal
        adapt_weight[mask4] = 0.0
        
        should_adapt = adapt_weight > 0
        
        return should_adapt, adapt_weight
    
    def forward(
        self,
        x: torch.Tensor,
        return_debug: bool = False
    ) -> Dict:
        """
        Forward pass with Pmax-SAL gated adaptation.
        
        Args:
            x: Input EEG tensor [B, C, T]
            return_debug: Whether to return debug information
        
        Returns:
            Dictionary containing:
            - logits: Model outputs
            - pred: Predictions
            - pmax: Maximum probabilities
            - sal: Source Alignment Levels
            - adapted: Whether adaptation was applied
        """
        # Step 1: Forward pass through frozen model
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        
        # Step 2: Compute pmax
        pmax, pred = self.compute_pmax(logits)
        
        # Step 3: Get features and compute SAL
        features = self._features
        
        # Apply same flattening logic as in compute_source_prototypes
        if features is not None:
             if len(features.shape) > 2:
                 features = features.mean(dim=-1)
             if len(features.shape) == 3 and features.shape[2] == 1:
                 features = features.squeeze(2)
        
        if features is not None:
             # Debug prints
             # print(f"DEBUG: features shape={features.shape}, prototypes shape={self.source_prototypes.shape}")
             pass
        
        sal = self.compute_sal(features, pred) if features is not None else pmax.clone()
        
        # Step 4: Gating decision
        should_adapt, adapt_weight = self.compute_gate(pmax, sal)
        
        # Step 5: Adaptation (if enabled)
        adapted = False
        if self.enable_adaptation and should_adapt.any():
            # Get samples to adapt
            adapt_mask = should_adapt
            
            if adapt_mask.sum() > 0:
                # Update BN stats with adapted samples
                self._update_bn_stats(x[adapt_mask])
                adapted = True
                self.adaptation_stats['adapted_samples'] += adapt_mask.sum().item()
            
            # Track skipped samples
            overconfident_mask = (pmax > self.pmax_threshold) & (sal < self.sal_threshold)
            self.adaptation_stats['skipped_overconfident'] += overconfident_mask.sum().item()
            
            unreliable_mask = (pmax < self.pmax_threshold) & (sal < self.sal_threshold)
            self.adaptation_stats['skipped_unreliable'] += unreliable_mask.sum().item()
        
        self.adaptation_stats['total_samples'] += x.shape[0]
        
        # Step 6: Final forward with potentially updated BN
        if adapted:
            with torch.no_grad():
                logits = self.model(x)
                pmax, pred = self.compute_pmax(logits)
        
        result = {
            'logits': logits,
            'pred': pred,
            'pmax': pmax,
            'sal': sal,
            'adapted': adapted,
            'adapt_weight': adapt_weight,
        }
        
        if return_debug:
            result['features'] = features
            result['stats'] = self.adaptation_stats.copy()
        
        return result
    
    def reset_stats(self):
        """Reset adaptation statistics."""
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'skipped_overconfident': 0,
            'skipped_unreliable': 0,
        }
    
    def get_adaptation_rate(self) -> float:
        """Get the rate of samples that were adapted."""
        if self.adaptation_stats['total_samples'] == 0:
            return 0.0
        return self.adaptation_stats['adapted_samples'] / self.adaptation_stats['total_samples']
    
    def print_stats(self):
        """Print adaptation statistics."""
        total = self.adaptation_stats['total_samples']
        if total == 0:
            print("[PmaxSAL] No samples processed")
            return
        
        adapted = self.adaptation_stats['adapted_samples']
        overconf = self.adaptation_stats['skipped_overconfident']
        unreliable = self.adaptation_stats['skipped_unreliable']
        
        print(f"[PmaxSAL] Adaptation Statistics:")
        print(f"  Total samples: {total}")
        print(f"  Adapted: {adapted} ({100*adapted/total:.1f}%)")
        print(f"  Skipped (overconfident): {overconf} ({100*overconf/total:.1f}%)")
        print(f"  Skipped (unreliable): {unreliable} ({100*unreliable/total:.1f}%)")
