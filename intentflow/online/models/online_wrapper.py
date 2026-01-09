"""Online wrapper for TCFormer Hybrid with TTT state management.

This module provides a wrapper class that manages the stateful execution
of the TCFormer Hybrid model for online/streaming inference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import yaml
import numpy as np

# Add offline models to path for imports
_offline_path = Path(__file__).resolve().parent.parent.parent / "offline"
if str(_offline_path) not in sys.path:
    sys.path.insert(0, str(_offline_path))

from models.tcformer_ttt.tcformer_hybrid import TCFormerHybridModule
from models.tcformer_ttt.ttt_layer import TTTConfig, TTTCache


class OnlineTCFormerWrapper:
    """
    Wrapper for TCFormerHybridModule that manages TTT state across inference steps.
    
    This class is designed for online/streaming inference where the model's
    internal TTT state should persist across multiple prediction steps.
    
    Attributes:
        model: The underlying TCFormerHybridModule.
        device: The device (CPU/GPU) where the model runs.
        cache: The TTTCache holding the current state.
        n_classes: Number of output classes.
        class_labels: List of class label names (e.g., ["left", "right", "feet", "tongue"]).
    """
    
    # BCIC IV 2a class labels
    BCIC2A_LABELS = ["left_hand", "right_hand", "feet", "tongue"]
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda:0",
        reset_state_each_trial: bool = False,
    ):
        """
        Initialize the OnlineTCFormerWrapper.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint (.ckpt).
            config_path: Optional path to config.yaml. If None, will look in the
                         same directory as checkpoint or its parent.
            device: Device to run inference on (e.g., "cuda:0" or "cpu").
            reset_state_each_trial: If True, reset TTT state before each prediction.
                                    If False (default), state persists across predictions.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.reset_state_each_trial = reset_state_each_trial
        
        # Load config
        if config_path is None:
            config_path = self._find_config()
        self.config = self._load_config(config_path)
        
        # Build and load model
        self.model = self._build_model()
        self._load_checkpoint()
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize TTT cache (state)
        self.cache: Optional[TTTCache] = None
        self._batch_size = 1  # Default batch size for online inference
        
        # Class info - infer from config or default to 4 for BCIC2a
        dataset_name = self.config.get("dataset_name", "bcic2a")
        if dataset_name in ["bcic2a", "BNCI2014001"]:
            self.n_classes = 4
        elif dataset_name in ["bcic2b", "BNCI2014004"]:
            self.n_classes = 2
        else:
            self.n_classes = self.config.get("model_kwargs", {}).get("n_classes", 4)
        self.class_labels = self.BCIC2A_LABELS[:self.n_classes]
        
        print(f"[OnlineTCFormerWrapper] Loaded model from {checkpoint_path}")
        print(f"[OnlineTCFormerWrapper] Device: {self.device}")
        print(f"[OnlineTCFormerWrapper] Classes: {self.class_labels}")
        
    def _find_config(self) -> Path:
        """Find config.yaml in checkpoint directory or parent."""
        # Try checkpoint directory
        ckpt_dir = self.checkpoint_path.parent
        if (ckpt_dir / "config.yaml").exists():
            return ckpt_dir / "config.yaml"
        
        # Try parent (for structure like data/checkpoints/model.ckpt)
        parent = ckpt_dir.parent
        if (parent / "config.yaml").exists():
            return parent / "config.yaml"
        
        # Try grandparent
        grandparent = parent.parent
        if (grandparent / "config.yaml").exists():
            return grandparent / "config.yaml"
        
        raise FileNotFoundError(
            f"Could not find config.yaml near checkpoint: {self.checkpoint_path}"
        )
    
    def _load_config(self, config_path: str | Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    
    def _build_model(self) -> TCFormerHybridModule:
        """Build the TCFormerHybridModule from config."""
        model_kwargs = self.config.get("model_kwargs", {})
        ttt_config = model_kwargs.get("ttt_config", {})
        
        # Infer n_classes from dataset
        dataset_name = self.config.get("dataset_name", "bcic2a")
        if dataset_name in ["bcic2a", "BNCI2014001"]:
            n_classes = 4
        elif dataset_name in ["bcic2b", "BNCI2014004"]:
            n_classes = 2
        else:
            n_classes = model_kwargs.get("n_classes", 4)
        
        # Infer n_channels from dataset
        if dataset_name in ["bcic2a", "BNCI2014001"]:
            n_channels = 22
        elif dataset_name in ["bcic2b", "BNCI2014004"]:
            n_channels = 3
        else:
            n_channels = model_kwargs.get("n_channels", 22)
        
        model = TCFormerHybridModule(
            n_classes=n_classes,
            Chans=n_channels,
            F1=model_kwargs.get("F1", 32),
            D=model_kwargs.get("D", 2),
            d_group=model_kwargs.get("d_group", 16),
            temp_kernel_lengths=model_kwargs.get("temp_kernel_lengths", [20, 32, 64]),
            pool_length_1=model_kwargs.get("pool_length_1", 8),
            pool_length_2=model_kwargs.get("pool_length_2", 7),
            dropout_conv=model_kwargs.get("dropout_conv", 0.4),
            ttt_config=ttt_config,
            q_heads=model_kwargs.get("q_heads", 4),
            kv_heads=model_kwargs.get("kv_heads", 4),
            trans_depth=model_kwargs.get("trans_depth", 2),
            adapter_ratio=model_kwargs.get("adapter_ratio", 0.25),
            use_dynamic_gating=model_kwargs.get("use_dynamic_gating", False),
            use_group_attn=model_kwargs.get("use_group_attn", True),
            gating_mode=model_kwargs.get("gating_mode", "feature_stats"),
            entropy_threshold=model_kwargs.get("entropy_threshold", 0.95),
            alpha_max=model_kwargs.get("alpha_max", 0.5),
            lr_scale_max=model_kwargs.get("lr_scale_max", 0.5),
            entropy_gating_in_train=model_kwargs.get("entropy_gating_in_train", False),
        )
        return model
    
    def _load_checkpoint(self) -> None:
        """Load model weights from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove "model." prefix if present (from Lightning wrapper)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned_state_dict[k[6:]] = v  # Remove "model." prefix
            else:
                cleaned_state_dict[k] = v
        
        # Load with strict=False to handle any minor mismatches
        missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"[OnlineTCFormerWrapper] Warning: Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"[OnlineTCFormerWrapper] Warning: Unexpected keys: {unexpected[:5]}...")
    
    def reset_state(self) -> None:
        """Reset the TTT cache (internal state) to initial values."""
        if self._batch_size > 0:
            self.cache = self.model.create_cache(batch_size=self._batch_size)
        else:
            self.cache = None
        # print("[OnlineTCFormerWrapper] State reset.")
    
    def predict_step(
        self,
        window: np.ndarray,
        return_probs: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference on a single window of EEG data.
        
        This method performs:
        1. Preprocessing (tensor conversion)
        2. Forward pass through the model (with TTT adaptation)
        3. State update (TTT weights are updated internally)
        
        Args:
            window: Input EEG data of shape [C, T] (channels x time samples).
                    Expected to be already normalized (e.g., via StreamNormalizer).
            return_probs: If True, return softmax probabilities. Otherwise, raw logits.
            
        Returns:
            Dictionary containing:
                - "pred_idx": Predicted class index (int).
                - "pred_label": Predicted class label (str).
                - "confidence": Confidence score (max probability).
                - "probs" or "logits": Full output vector.
                - "entropy": Prediction entropy (uncertainty measure).
        """
        # Reset state if configured to do so
        if self.reset_state_each_trial:
            self.reset_state()
        
        # Initialize cache on first call
        if self.cache is None:
            self._batch_size = 1
            self.reset_state()
        
        # Convert to tensor: [C, T] -> [1, C, T]
        if isinstance(window, np.ndarray):
            x = torch.from_numpy(window).float()
        else:
            x = window.float()
        
        if x.ndim == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        x = x.to(self.device)
        
        # Forward pass (TTT adaptation happens internally if model is in eval mode)
        # Note: We don't use torch.no_grad() because TTT requires gradients for internal update
        with torch.inference_mode(mode=False):
            # Enable gradients for TTT internal update
            x.requires_grad_(False)  # Input doesn't need grad
            logits = self.model(x, cache_params=self.cache)
        
        # Post-process
        logits_np = logits.detach().cpu().numpy()[0]  # [n_classes]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        
        result = {
            "pred_idx": pred_idx,
            "pred_label": self.class_labels[pred_idx] if pred_idx < len(self.class_labels) else f"class_{pred_idx}",
            "confidence": confidence,
            "entropy": entropy,
        }
        
        if return_probs:
            result["probs"] = probs
        else:
            result["logits"] = logits_np
        
        return result
    
    def predict_batch(
        self,
        windows: np.ndarray,
        return_probs: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference on a batch of windows.
        
        Args:
            windows: Input EEG data of shape [B, C, T].
            return_probs: If True, return softmax probabilities.
            
        Returns:
            Dictionary with batched results.
        """
        batch_size = windows.shape[0]
        
        # Update cache if batch size changed
        if self._batch_size != batch_size:
            self._batch_size = batch_size
            self.reset_state()
        
        # Convert to tensor
        if isinstance(windows, np.ndarray):
            x = torch.from_numpy(windows).float()
        else:
            x = windows.float()
        
        x = x.to(self.device)
        
        with torch.inference_mode(mode=False):
            logits = self.model(x, cache_params=self.cache)
        
        logits_np = logits.detach().cpu().numpy()
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        entropies = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        
        result = {
            "pred_indices": pred_indices,
            "pred_labels": [self.class_labels[i] if i < len(self.class_labels) else f"class_{i}" for i in pred_indices],
            "confidences": confidences,
            "entropies": entropies,
        }
        
        if return_probs:
            result["probs"] = probs
        else:
            result["logits"] = logits_np
        
        return result
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information from the last forward pass."""
        if hasattr(self.model, "get_debug_batch"):
            dbg = self.model.get_debug_batch()
            # Convert tensors to numpy/float for serialization
            result = {}
            for k, v in dbg.items():
                if v is None:
                    result[k] = None
                elif isinstance(v, torch.Tensor):
                    result[k] = v.cpu().numpy().tolist()
                else:
                    result[k] = v
            return result
        return {}
    
    def __repr__(self) -> str:
        return (
            f"OnlineTCFormerWrapper("
            f"checkpoint={self.checkpoint_path.name}, "
            f"device={self.device}, "
            f"n_classes={self.n_classes})"
        )

