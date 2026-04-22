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
        energy_threshold: Optional[float] = None,
        energy_quantile: float = 0.95,
        energy_temperature: float = 1.0,
        neuro_beta: float = 0.1,
        strict_tri_lock: bool = True,
        alpha: float = 0.5,  # Weight between pmax and SAL
        bn_momentum: float = 0.1,  # BN stats update momentum
        bn_shallow_mean_momentum: Optional[float] = None,
        bn_shallow_var_momentum: Optional[float] = None,
        bn_deep_mean_momentum: Optional[float] = None,
        bn_deep_var_momentum: Optional[float] = None,
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
        self.bn_shallow_mean_momentum = bn_shallow_mean_momentum
        self.bn_shallow_var_momentum = bn_shallow_var_momentum
        self.bn_deep_mean_momentum = bn_deep_mean_momentum
        self.bn_deep_var_momentum = bn_deep_var_momentum
        self.enable_adaptation = enable_adaptation
        self.fixed_energy_threshold = None if energy_threshold is None else float(energy_threshold)
        self.energy_quantile = float(energy_quantile)
        self.energy_temperature = float(energy_temperature)
        self.strict_tri_lock = bool(strict_tri_lock)
        
        # Source prototypes: [n_classes, feature_dim]
        # Will be computed during training
        self.register_buffer('source_prototypes', None)
        self.register_buffer('prototype_counts', torch.zeros(n_classes))
        self.register_buffer('source_energy_threshold', torch.tensor(float("nan")))
        
        # Store source BN statistics
        self.source_bn_stats = {}
        self._save_source_bn_stats()
        
        # Feature extraction hook
        self._features = None
        self._register_feature_hook()
        
        self.class_probs = []
        
        # Neuro-Gated OTTA config
        self.channel_roles = None
        self.neuro_factor = 50.0 # Legacy parameter, kept for compatibility
        self.neuro_beta = float(neuro_beta)
        
        # Online Neuro-Score normalizer
        # Warmup steps set to 0 to allow adaptation from Batch 1
        self.normalizer = OnlineNormalizer(momentum=0.1, warmup_steps=0)
        
        # Statistics for logging
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'skipped_overconfident': 0,
            'skipped_unreliable': 0,
            'skipped_energy': 0,
        }
        self._last_gate_info = {}
        self._warned_no_prototypes = False
        self._warned_no_features = False
        self._warned_no_energy_threshold = False
        self._warned_missing_energy = False

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
            # Prefer classifier input (penultimate feature) over output logits.
            if isinstance(input, tuple) and len(input) > 0 and torch.is_tensor(input[0]):
                self._features = input[0]
            elif torch.is_tensor(input):
                self._features = input
            elif isinstance(output, tuple):
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
    
    def _update_bn_stats(self, x: torch.Tensor) -> tuple:
        """Update BN running statistics with current batch.

        Respects self.adapt_mode and self.bn_update_target:
          adapt_mode:
            'bn_stat'       – model.train() + forward (Dropout also enabled)
            'bn_stat_clean' – only BN layers in train mode
            'tent_bn/ln'    – gradient-based affine update (no running stats change)
            'source_only'   – no update
          bn_update_target (controls *what* gets updated):
            'both'                  – mean and var, all layers (default)
            'mean_only'             – only running_mean, all layers
            'var_only'              – only running_var, all layers
            'shallow'               – mean and var, first half (conv_block)
            'deep'                  – mean and var, second half (mix/reduce/tcn)
            'shallow_mean_only'     – only mean, shallow layers
            'shallow_var_only'      – only var,  shallow layers
            'deep_mean_only'        – only mean, deep layers
            'deep_var_only'         – only var,  deep layers
            'shallow_mean_deep_both'    – shallow: mean only (var frozen); deep: mean+var
                                          causal minimum intervention (current best)
            'shallow_mean_deep_mean_only'– shallow: mean only; deep: mean only
                                          conservative: all var frozen
            'shallow_mean_deep_var_only' – shallow: mean only; deep: var only
                                          isolates deep var contribution

        Returns:
            (total_drift: float, layer_drifts: list[float])
            layer_drifts[i] = drift of i-th BN layer in model order
        """
        mode = getattr(self, 'adapt_mode', 'bn_stat')

        if mode == 'source_only':
            return 0.0, []

        bn_momentum = float(getattr(self, 'bn_momentum', 0.1))
        update_target = getattr(self, 'bn_update_target', 'both')
        shallow_mean_override = getattr(self, 'bn_shallow_mean_momentum', None)
        shallow_var_override = getattr(self, 'bn_shallow_var_momentum', None)
        deep_mean_override = getattr(self, 'bn_deep_mean_momentum', None)
        deep_var_override = getattr(self, 'bn_deep_var_momentum', None)

        if mode in ('bn_stat', 'bn_stat_clean'):
            # Ordered list of (name, module) for all BN layers
            bn_layers = [(n, m) for n, m in self.model.named_modules()
                         if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
            n_bn = len(bn_layers)
            shallow_names = {n for n, _ in bn_layers[:n_bn // 2]}
            deep_names = {n for n, _ in bn_layers[n_bn // 2:]}
            all_names = [n for n, _ in bn_layers]

            desired_momenta = {
                name: {'mean': 0.0, 'var': 0.0}
                for name in all_names
            }

            def set_momenta(names, *, mean=None, var=None):
                for name in names:
                    if mean is not None:
                        desired_momenta[name]['mean'] = float(mean)
                    if var is not None:
                        desired_momenta[name]['var'] = float(var)

            if update_target == 'both':
                set_momenta(all_names, mean=bn_momentum, var=bn_momentum)
            elif update_target == 'mean_only':
                set_momenta(all_names, mean=bn_momentum, var=0.0)
            elif update_target == 'var_only':
                set_momenta(all_names, mean=0.0, var=bn_momentum)
            elif update_target == 'shallow':
                set_momenta(shallow_names, mean=bn_momentum, var=bn_momentum)
            elif update_target == 'deep':
                set_momenta(deep_names, mean=bn_momentum, var=bn_momentum)
            elif update_target == 'shallow_mean_only':
                set_momenta(shallow_names, mean=bn_momentum, var=0.0)
            elif update_target == 'shallow_var_only':
                set_momenta(shallow_names, mean=0.0, var=bn_momentum)
            elif update_target == 'deep_mean_only':
                set_momenta(deep_names, mean=bn_momentum, var=0.0)
            elif update_target == 'deep_var_only':
                set_momenta(deep_names, mean=0.0, var=bn_momentum)
            elif update_target == 'shallow_mean_deep_both':
                set_momenta(shallow_names, mean=bn_momentum, var=0.0)
                set_momenta(deep_names, mean=bn_momentum, var=bn_momentum)
            elif update_target == 'shallow_mean_deep_mean_only':
                set_momenta(shallow_names, mean=bn_momentum, var=0.0)
                set_momenta(deep_names, mean=bn_momentum, var=0.0)
            elif update_target == 'shallow_mean_deep_var_only':
                set_momenta(shallow_names, mean=bn_momentum, var=0.0)
                set_momenta(deep_names, mean=0.0, var=bn_momentum)
            else:
                raise ValueError(f"Unsupported bn_update_target: {update_target}")

            override_map = {
                'shallow': {'mean': shallow_mean_override, 'var': shallow_var_override},
                'deep': {'mean': deep_mean_override, 'var': deep_var_override},
            }
            for name in all_names:
                scope = 'shallow' if name in shallow_names else 'deep'
                for stat_name in ('mean', 'var'):
                    override = override_map[scope][stat_name]
                    if override is not None and desired_momenta[name][stat_name] > 0.0:
                        desired_momenta[name][stat_name] = float(override)

            active_momenta = {
                name: max(stats['mean'], stats['var'])
                for name, stats in desired_momenta.items()
            }

            # Snapshot all BN states; each layer forwards with the max momentum needed
            # for that layer, then each statistic is scaled back to its desired momentum.
            bn_before, orig_momentum = {}, {}
            for name, m in bn_layers:
                bn_before[name] = (m.running_mean.clone(), m.running_var.clone())
                orig_momentum[name] = m.momentum
                m.momentum = active_momenta[name]

            if mode == 'bn_stat':
                was_training = self.model.training
                self.model.train()
                with torch.no_grad():
                    _ = self.model(x)
                if not was_training:
                    self.model.eval()
            else:
                self.model.eval()
                for _, m in bn_layers:
                    m.train()
                with torch.no_grad():
                    _ = self.model(x)
                self.model.eval()

            # Selective mean/var update with per-stat momentum support.
            for name, m in bn_layers:
                before_mean, before_var = bn_before[name]
                desired_mean = desired_momenta[name]['mean']
                desired_var = desired_momenta[name]['var']
                base_momentum = active_momenta[name]
                updated_mean = m.running_mean.clone()
                updated_var = m.running_var.clone()

                if base_momentum <= 0.0:
                    m.running_mean.copy_(before_mean)
                    m.running_var.copy_(before_var)
                    continue

                if desired_mean <= 0.0:
                    m.running_mean.copy_(before_mean)
                elif desired_mean != base_momentum:
                    scale = desired_mean / base_momentum
                    m.running_mean.copy_(before_mean + (updated_mean - before_mean) * scale)

                if desired_var <= 0.0:
                    m.running_var.copy_(before_var)
                elif desired_var != base_momentum:
                    scale = desired_var / base_momentum
                    m.running_var.copy_(before_var + (updated_var - before_var) * scale)

            # Compute per-layer drift and restore original momentum
            layer_drifts = []
            for name, m in bn_layers:
                d = (m.running_mean - bn_before[name][0]).norm().item()
                d += (m.running_var  - bn_before[name][1]).norm().item()
                layer_drifts.append(d)
                m.momentum = orig_momentum[name]

            return sum(layer_drifts), layer_drifts

        elif mode in ('tent_bn', 'tent_bn_ln'):
            # Gradient-based entropy minimization on normalization affine params
            # PL 2.1.3 wraps test_step in torch.inference_mode(), which blocks
            # gradient computation. We must exit inference_mode AND enter
            # enable_grad: inference_mode(False) alone does not re-enable autograd.
            with torch.inference_mode(False), torch.enable_grad():
                self.model.eval()
                for module in self.model.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        module.train()

                params = []
                for module in self.model.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        if module.weight is not None:
                            params.append(module.weight)
                        if module.bias is not None:
                            params.append(module.bias)
                    if mode == 'tent_bn_ln' and isinstance(module, nn.LayerNorm):
                        if module.weight is not None:
                            params.append(module.weight)
                        if module.bias is not None:
                            params.append(module.bias)

                if params:
                    for p in params:
                        p.requires_grad_(True)

                    optimizer = torch.optim.SGD(params, lr=getattr(self, 'tent_lr', 0.001))
                    optimizer.zero_grad()

                    # x is an inference tensor (created by PL's test_step context).
                    # clone() propagates the inference flag, so we must create a
                    # genuinely new tensor inside this inference_mode(False) context.
                    x_tent = torch.empty_like(x).copy_(x)

                    # TCFormer lazily computes and caches RoPE _cos/_sin buffers on
                    # the first forward call (inside PL's inference_mode), making them
                    # inference tensors. Reset them so they are recomputed here in the
                    # non-inference context and can be saved for backward.
                    for _m in self.model.modules():
                        if hasattr(_m, '_cos'):
                            _m._cos = None
                        if hasattr(_m, '_sin'):
                            _m._sin = None

                    logits = self.model(x_tent)
                    probs = torch.softmax(logits, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                    entropy.backward()

                    # one-time diagnostic: verify gradient is actually flowing
                    if not getattr(self, '_tent_grad_debug_done', False):
                        self._tent_grad_debug_done = True
                        grad_norms = [
                            p.grad.norm().item() if p.grad is not None else float('nan')
                            for p in params
                        ]
                        _snap = [p.data.clone() for p in params]
                        optimizer.step()
                        delta_norms = [(p.data - b).norm().item() for p, b in zip(params, _snap)]
                        print(
                            f"[TENT-DIAG] mode={mode}, "
                            f"grad_enabled={torch.is_grad_enabled()}, "
                            f"n_params={len(params)}, "
                            f"grad_norms(first5)={[f'{g:.3e}' for g in grad_norms[:5]]}, "
                            f"delta_norms(first5)={[f'{d:.3e}' for d in delta_norms[:5]]}"
                        )
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    for p in params:
                        p.requires_grad_(False)

                self.model.eval()
            return 0.0, []  # drift_norm, layer_drifts: affine params change, not running stats

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
        
        # Accumulate features per class and energy scores
        feature_sums = {}
        feature_counts = {}
        source_energies = []
        
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
                        
                # Compute and store energy for the batch
                batch_energy = self.compute_energy(logits, self.energy_temperature)
                source_energies.append(batch_energy.cpu())
        
        # Compute statistical energy threshold from source domain.
        if source_energies:
            all_energies = torch.cat(source_energies)
            if self.fixed_energy_threshold is not None:
                th = torch.tensor(self.fixed_energy_threshold, device=device)
                src = "fixed"
            else:
                q = min(max(self.energy_quantile, 0.0), 1.0)
                th = torch.quantile(all_energies.to(device), q)
                src = f"quantile={q:.2f}"
            self.source_energy_threshold.copy_(th.detach())
            print(
                "[PmaxSAL] Computed Energy Threshold: "
                f"{self.source_energy_threshold.item():.4f} ({src}, T={self.energy_temperature:.2f})"
            )
        
        if not feature_sums:
            self.source_prototypes = None
            print("[PmaxSAL] Warning: no source features captured. SAL gate will remain closed.")
            return

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
    
    def compute_energy(self, logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """
        Compute Energy Score for Statistical Gating (In-Distribution check).
        Lower energy = In-Distribution. Higher = Out-of-Distribution.
        """
        energy = -T * torch.logsumexp(logits / T, dim=1)
        return energy

    def _resolve_energy_threshold(self, device: torch.device) -> Optional[torch.Tensor]:
        """Resolve runtime energy threshold with priority: fixed > calibrated."""
        if self.fixed_energy_threshold is not None:
            return torch.tensor(self.fixed_energy_threshold, device=device)
        if torch.isfinite(self.source_energy_threshold).item():
            return self.source_energy_threshold.to(device)
        return None
        
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
            # Fail-safe: No prototypes -> block adaptation path.
            if not self._warned_no_prototypes:
                print("[PmaxSAL] Warning: source prototypes unavailable. Forcing SAL=0.")
                self._warned_no_prototypes = True
            return torch.zeros(features.shape[0], device=features.device)
        
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
        neuro_score: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating decision based on confidence, physiological, and statistical gates.
        
        Tri-lock (strict mode):
        - pmax > dynamic pmax_th
        - sal > dynamic sal_th
        - energy <= energy_th
        
        Neuro-Gating Modifier:
        - If Neuro-Score is High (Motor focus): Lower thresholds (encourage adapt)
        - If Neuro-Score is Low (Noise focus): Raise thresholds (discourage adapt)
        
        Returns:
            should_adapt: Boolean mask [B]
            adapt_weight: Weight for adaptation [B]
        """
        
        base_pmax = torch.tensor(self.pmax_threshold, device=pmax.device)
        base_sal = torch.tensor(self.sal_threshold, device=sal.device)

        # Conservative neuro gating: raise thresholds only on negative Z.
        if neuro_score is None:
            neuro_score = torch.zeros_like(pmax)
        z_score, is_warmed_up = self.normalizer(neuro_score)
        negative_z = F.relu(-z_score)
        modifier = self.neuro_beta * negative_z
        pmax_th = base_pmax + modifier
        sal_th = base_sal + modifier

        if torch.rand(1).item() < 0.05:
            print(
                "[DEBUG-Cons] "
                f"Beta={self.neuro_beta}, Z_mean={z_score.mean().item():.2f}, "
                f"Mod_mean={modifier.mean().item():.4f}"
            )

        # Optional warm-up lock.
        if not is_warmed_up:
            pmax_th = torch.ones_like(pmax_th) * 1.1
            sal_th = torch.ones_like(sal_th) * 1.1

        high_pmax = pmax > pmax_th
        high_sal = sal > sal_th

        # Statistical gate.
        energy_th = self._resolve_energy_threshold(pmax.device)
        if energy is None:
            safe_energy = torch.zeros_like(high_pmax, dtype=torch.bool)
            if not self._warned_missing_energy:
                print("[PmaxSAL] Warning: energy score missing. Blocking adaptation.")
                self._warned_missing_energy = True
        elif energy_th is None:
            safe_energy = torch.zeros_like(high_pmax, dtype=torch.bool)
            if not self._warned_no_energy_threshold:
                print("[PmaxSAL] Warning: energy threshold unavailable. Blocking adaptation.")
                self._warned_no_energy_threshold = True
        else:
            safe_energy = energy <= energy_th

        # Tri-lock by default.
        if self.strict_tri_lock:
            should_adapt = high_pmax & high_sal & safe_energy
            adapt_weight = should_adapt.to(pmax.dtype)
        else:
            adapt_weight = torch.zeros_like(pmax)
            mask1 = high_pmax & high_sal & safe_energy
            mask3 = ~high_pmax & high_sal & safe_energy
            adapt_weight[mask1] = 1.0
            adapt_weight[mask3] = 0.5
            should_adapt = adapt_weight > 0

        self._last_gate_info = {
            "high_pmax": high_pmax,
            "high_sal": high_sal,
            "safe_energy": safe_energy,
            "energy_th": energy_th,
        }
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
        
        if features is not None:
            sal = self.compute_sal(features, pred)
        else:
            if not self._warned_no_features:
                print("[PmaxSAL] Warning: features unavailable. Forcing SAL=0.")
                self._warned_no_features = True
            sal = torch.zeros_like(pmax)
        
        # Step 3.5: Compute Neuro-Score
        neuro_score = self.compute_neuro_score(att_weights) # [B]
        
        # If neuro_score is scalar, expand to [B]
        if neuro_score.ndim == 0:
             neuro_score = neuro_score.unsqueeze(0).repeat(pmax.shape[0])
        
        # Step 3.8: Compute Energy Score (Statistical Gate)
        energy = self.compute_energy(logits, self.energy_temperature)
        
        # Step 4: Gating decision (with Neuro-Score and Energy)
        should_adapt, adapt_weight = self.compute_gate(pmax, sal, neuro_score, energy)
        
        # Step 5: Adaptation (if enabled)
        adapted_flag = False # Internal Batch Flag
        original_pred = pred.clone() # Save original prediction
        
        bn_drift_norm = 0.0
        bn_drift_layers: list = []
        if self.enable_adaptation and should_adapt.any():
            adapt_mask = should_adapt
            if adapt_mask.sum() > 0:
                bn_drift_norm, bn_drift_layers = self._update_bn_stats(x[adapt_mask])
                adapted_flag = True
                self.adaptation_stats['adapted_samples'] += adapt_mask.sum().item()

        self.adaptation_stats['total_samples'] += x.shape[0]
        gate = self._last_gate_info
        if gate:
            self.adaptation_stats['skipped_overconfident'] += (gate["high_pmax"] & ~gate["high_sal"]).sum().item()
            self.adaptation_stats['skipped_unreliable'] += (~gate["high_pmax"]).sum().item()
            self.adaptation_stats['skipped_energy'] += (gate["high_pmax"] & gate["high_sal"] & ~gate["safe_energy"]).sum().item()

        # Step 6: Final forward with potentially updated BN
        if adapted_flag:
            with torch.no_grad():
                logits = self.model(x)
                pmax, pred = self.compute_pmax(logits)

        # Per-trial diagnostics (entropy, logit_norm) from final logits
        with torch.no_grad():
            probs_final = torch.softmax(logits, dim=-1)
            entropy_per_sample = -(probs_final * torch.log(probs_final + 1e-8)).sum(dim=-1)
            logit_norm_per_sample = logits.norm(dim=-1)

        result = {
            'logits': logits,
            'pred': pred,
            'original_pred': original_pred,
            'pmax': pmax,
            'sal': sal,
            'neuro_score': neuro_score,
            'energy_score': energy,
            'entropy': entropy_per_sample,
            'logit_norm': logit_norm_per_sample,
            'bn_drift_norm': torch.tensor([bn_drift_norm], dtype=torch.float32),
            'bn_drift_layers': torch.tensor(bn_drift_layers, dtype=torch.float32),  # [n_bn_layers] or []
            'adapted': should_adapt.detach().cpu(),
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
            'skipped_energy': 0,
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
        skipped_energy = self.adaptation_stats['skipped_energy']
        
        print(f"[PmaxSAL] Adaptation Statistics:")
        print(f"  Total samples: {total}")
        print(f"  Adapted: {adapted} ({100*adapted/total:.1f}%)")
        print(f"  Skipped (overconfident): {overconf} ({100*overconf/total:.1f}%)")
        print(f"  Skipped (unreliable): {unreliable} ({100*unreliable/total:.1f}%)")
        print(f"  Skipped (energy gate): {skipped_energy} ({100*skipped_energy/total:.1f}%)")
