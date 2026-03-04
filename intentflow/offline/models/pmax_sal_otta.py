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


class OnlineNormalizer(nn.Module):
    """
    Robust Online Normalizer with 3-Lock Strategy.
    1. Prior Init
    2. Warm-up
    3. Safety Clamp
    """
    def __init__(self, momentum: float = 0.1, warmup_steps: int = 20):
        super().__init__()
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        
        # 1. Prior Initialization (from S1 stats: mean~0.0, std~0.01)
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(0.0001)) # std=0.01 -> var=0.0001
        self.register_buffer('count', torch.tensor(0))
        
        # Logging stats
        self.last_z = 0.0
        self.last_mean = 0.0
        self.last_std = 0.0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Update stats and normalize input.
        Args:
            x: Input tensor (batch of scores) [B]
            
        Returns:
            z: Normalized scores
            is_warmed_up: Boolean indicating if warm-up is complete
        """
        # Batch stats
        if x.numel() > 1:
            batch_mean = x.mean()
            batch_var = x.var(unbiased=False)
        else:
            batch_mean = x.mean()
            batch_var = torch.tensor(0.0, device=x.device)
            
        # Update running stats (Exponential Moving Average)
        # Handle first update differently if needed, but Prior Init makes it easier.
        if self.training or True: # Always update in TTA
             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
             # Correct variance update usually involves mean shift, but simple EMA on var is approx OK for streaming
             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

        self.count += 1
        is_warmed_up = self.count > self.warmup_steps
        
        # 2. Warm-up (Force stats update but invalid z output? Or just return z but flag as not ready?)
        # We will return z, but the caller should check is_warmed_up.
        
        # Compute Z-score
        # 3. Safety Clamp (Safe division)
        std = torch.sqrt(self.running_var + 1e-6)
        z = (x - self.running_mean) / std
        
        # 3. Safety Clamp (Clip Z)
        z = torch.clamp(z, -3.0, 3.0)
        
        # Update logs
        self.last_z = z.mean().item()
        self.last_mean = self.running_mean.item()
        self.last_std = std.item()

        return z, is_warmed_up


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
        
        self.class_probs = []
        
        # Neuro-Gated OTTA config
        self.channel_roles = None
        self.neuro_factor = 50.0 # Deprecated in Phase 7, kept for compatibility if needed
        self.neuro_beta = 5.0   # Strong Conservative Gating (Was 0.1)
        
        # Phase 7: Online Normalizer
        # Warmup steps set to 0 to allow adaptation from Batch 1
        self.normalizer = OnlineNormalizer(momentum=0.1, warmup_steps=0)
        
        # Statistics for logging
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'skipped_overconfident': 0,
            'skipped_unreliable': 0,
        }

    def set_channel_roles(self, roles: Dict[str, list]):
        """Set channel roles (motor/noise indices) for Neuro-Gating."""
        self.channel_roles = roles
        print(f"[PmaxSAL] Neuro-Gating enabled: {len(roles['motor'])} Motor, {len(roles['noise'])} Noise channels.")

    def compute_neuro_score(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute Neuro-Score based on attention weights of Motor vs Noise channels.
        
        Args:
            weights: Channel attention weights [B, C]
            
        Returns:
            neuro_score: Score in [-1, 1] range. [B]
                         Positive -> Motor focus (Trustworthy)
                         Negative -> Noise focus (Artifact)
        """
        if self.channel_roles is None or weights is None:
            # Return scalar 0.0 tensor, correct device handling tricky if weights=None
            device = self.prototype_counts.device
            return torch.tensor(0.0, device=device)

        motor_idx = self.channel_roles.get('motor', [])
        noise_idx = self.channel_roles.get('noise', [])
        
        if not motor_idx:
            return torch.zeros(weights.shape[0], device=weights.device)
            
        # Extract weights
        motor_w = weights[:, motor_idx].mean(dim=1) if motor_idx else torch.zeros(weights.shape[0], device=weights.device)
        noise_w = weights[:, noise_idx].mean(dim=1) if noise_idx else torch.zeros(weights.shape[0], device=weights.device)
        
        # Calculate score: (Motor - Noise) / (Motor + Noise + epsilon)
        # This normalizes the score to [-1, 1] relative to the total attention on these groups
        total = motor_w + noise_w + 1e-6
        score = (motor_w - noise_w) / total
        
        return score
    
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
        sal: torch.Tensor,
        neuro_score: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating decision based on pmax and SAL, optionally modulated by Neuro-Score.
        
        Gating Logic:
        - High pmax + High SAL → adapt_weight = 1.0
        - High pmax + Low SAL  → adapt_weight = 0.0 (overconfident)
        - Low pmax + High SAL  → adapt_weight = 0.5 (cautious)
        - Low pmax + Low SAL   → adapt_weight = 0.0 (unreliable)
        
        Neuro-Gating Modifier:
        - If Neuro-Score is High (Motor focus): Lower thresholds (encourage adapt)
        - If Neuro-Score is Low (Noise focus): Raise thresholds (discourage adapt)
        
        Returns:
            should_adapt: Boolean mask [B]
            adapt_weight: Weight for adaptation [B]
        """
        
        # Dynamic Thresholding
        pmax_th = self.pmax_threshold
        sal_th = self.sal_threshold
        
        # Phase 7: Online Z-score Normalization + Conservative Gating
        
        # Ensure pmax_th/sal_th are tensors on correct device
        if not torch.is_tensor(self.pmax_threshold):
                base_pmax = torch.tensor(self.pmax_threshold, device=pmax.device)
        else:
                base_pmax = self.pmax_threshold.to(pmax.device)
        
        if not torch.is_tensor(self.sal_threshold):
                base_sal = torch.tensor(self.sal_threshold, device=sal.device)
        else:
                base_sal = self.sal_threshold.to(sal.device)
        
        # Normalize Score
        z_score, is_warmed_up = self.normalizer(neuro_score)
        
        # Conservative Gating Logic: 
        # Only INCREASE threshold if Z is negative (bad state).
        # Modifier = beta * ReLU(-z)
        # If Z > 0 (Good), Modifier = 0 -> Threshold stays at base (Optimum)
        # If Z < 0 (Bad), Modifier > 0 -> Threshold increases -> Block adaptation
        
        negative_z = F.relu(-z_score) # Positive only if Z is negative
        modifier = self.neuro_beta * negative_z
        
        pmax_th = base_pmax + modifier
        sal_th = base_sal + modifier
        
        # Safety Clamp for Thresholds explanation:
        # We allow thresholds to go up to 1.0 (impossible to pass), but base shouldn't go below initial.
        # So clamp min is base (implied by logic) or explicit.
        # Max clamp 1.1 ensures we can block completely.
        # DEBUG
        if torch.rand(1).item() < 0.05: # Print occasionally
            print(f"[DEBUG-Cons] Beta={self.neuro_beta}, Z_mean={z_score.mean().item():.2f}, Mod_mean={modifier.mean().item():.4f}")
            
        # Handle Warm-up: If not warmed up, force thresholds to be very high
        if not is_warmed_up:
            pmax_th = torch.ones_like(pmax_th) * 1.1
            sal_th = torch.ones_like(sal_th) * 1.1
            
        else:
            pmax_th = torch.tensor(pmax_th, device=pmax.device)
            sal_th = torch.tensor(sal_th, device=sal.device)

        high_pmax = pmax > pmax_th
        high_sal = sal > sal_th
        
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
            - adapted: Whether adaptation was applied (Sample-wise Bool Tensor)
        """
        # Step 1: Forward pass through frozen model
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        
        # Try to retrieve channel attention weights (Neuro-Gating)
        att_weights = None
        
        # Handle wrapper (TCFormerBase -> TCFormerModule)
        target_model = self.model
        if hasattr(target_model, 'model'):
            target_model = target_model.model
            
        if hasattr(target_model, 'conv_block') and hasattr(target_model.conv_block, 'last_eca_weights'):
            att_weights = target_model.conv_block.last_eca_weights # [B, C]
        else:
             # Debugging why it failed
             if not hasattr(target_model, 'conv_block'):
                 pass # print("[PmaxSAL-Debug] target_model has no conv_block")
             elif not hasattr(target_model.conv_block, 'last_eca_weights'):
                 pass # print("[PmaxSAL-Debug] conv_block has no last_eca_weights")
        
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
        
        sal = self.compute_sal(features, pred) if features is not None else pmax.clone()
        
        # Step 3.5: Compute Neuro-Score
        neuro_score = self.compute_neuro_score(att_weights) # [B]
        
        # If neuro_score is scalar, expand to [B]
        if neuro_score.ndim == 0:
             neuro_score = neuro_score.unsqueeze(0).repeat(pmax.shape[0])
        
        # Step 4: Gating decision (with Neuro-Score)
        should_adapt, adapt_weight = self.compute_gate(pmax, sal, neuro_score)
        
        # Step 5: Adaptation (if enabled)
        adapted_flag = False # Internal Batch Flag
        original_pred = pred.clone() # Save original prediction
        
        if self.enable_adaptation and should_adapt.any():
            # Get samples to adapt
            adapt_mask = should_adapt
            
            if adapt_mask.sum() > 0:
                # Update BN stats with adapted samples
                self._update_bn_stats(x[adapt_mask])
                adapted_flag = True
                self.adaptation_stats['adapted_samples'] += adapt_mask.sum().item()
            
            # Track skipped samples
            # Using dynamic thresholds for counting logic is tricky, 
            # for now we just count based on adapt_weight being 0.
            pass
        
        self.adaptation_stats['total_samples'] += x.shape[0]
        
        # Step 6: Final forward with potentially updated BN
        if adapted_flag:
            with torch.no_grad():
                logits = self.model(x)
                pmax, pred = self.compute_pmax(logits)
        
        result = {
            'logits': logits,
            'pred': pred, # Final prediction
            'original_pred': original_pred, # Pre-adaptation prediction
            'pmax': pmax,
            'sal': sal,
            'neuro_score': neuro_score, # Log this!
            'adapted': should_adapt.detach().cpu(), # Return mask on CPU
            'adapt_weight': adapt_weight,
        }
        
        if return_debug:
            result['features'] = features
            result['stats'] = self.adaptation_stats.copy()
            if att_weights is not None:
                result['att_weights'] = att_weights
        
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
