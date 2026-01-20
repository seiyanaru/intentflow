import torch
import numpy as np
import os
import json
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
from torchmetrics.classification import (
    MulticlassCohenKappa, MulticlassConfusionMatrix
)

import pytorch_lightning as pl
from utils.lr_scheduler import linear_warmup_cosine_decay
import random

# Helper: Randomly selects a subset of EEG channels (augmentations)
def select_random_channels(x, keep_ratio=0.9):
    """
    Select a subset of EEG channels.
    Args:
        x: Tensor of shape [B, C, T]
        keep_ratio: fraction of channels to keep
    Returns:
        Tensor of shape [B, C_selected, T]
    """
    B, C, T = x.shape
    keep_chs = int(C * keep_ratio)
    keep_indices = sorted(random.sample(range(C), keep_chs))
    return x[:, keep_indices, :], keep_indices

# Helper: Randomly masks EEG channels (augmentations)
def random_channel_mask(x, keep_ratio=0.9):
    """
    Randomly keeps a subset of EEG channels.
    Args:
        x: Tensor of shape [B, C, T]
        keep_ratio: Float, ratio of channels to keep (e.g., 0.9 to keep 90%).
    Returns:
        Augmented tensor with masked channels set to 0.
    """
    B, C, T = x.shape
    keep_chs = int(C * keep_ratio)
    keep_indices = sorted(random.sample(range(C), keep_chs))
    mask = torch.zeros_like(x)
    mask[:, keep_indices, :] = 1
    return x * mask

# Lightning module
class ClassificationModule(pl.LightningModule):
    def __init__(
            self,
            model,
            n_classes,
            lr=0.001,
            weight_decay=0.0,
            optimizer="adam",
            scheduler=False,
            max_epochs=1000,
            warmup_epochs=20,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        
        # Identification info passed via kwargs, default to empty/unknown if not provided
        self.subject_id = kwargs.get("subject_id", "unknown")
        self.model_name = kwargs.get("model_name", "model")
        self.results_dir = kwargs.get("results_dir", "results")

        # ── metrics ───────────────────────────────────────
        self.test_kappa = MulticlassCohenKappa(num_classes=n_classes)        
        self.test_cm = MulticlassConfusionMatrix(num_classes=n_classes)  
        # will hold the final cm on CPU after test
        self.test_confmat = None
        
        # ── storage for analysis ──────────────────────────
        self.train_history = []
        self.test_features = []
        self.test_labels = []
        self.test_logits = []
        # ── debug stats (optional; populated when model exposes them) ──
        self.test_alpha = []
        self.test_entropy = []
        self.test_clip_ratio = []
        self.test_group_attn_weights = []
        self.test_conv_norm_pre = []
        self.test_conv_norm_post = []
        self.test_group_attn_gamma = []
        self.test_gate_entropy = []
        self.test_ttt_lr_scale = []
        self.test_ttt_effective_base_lr = []
        self.test_group_attn_temperature = []
        
        # ── 2-pass debug stats (Pass 1 = attention-only, Pass 2 = with TTT) ──
        self.test_preds_pass1 = []       # Pass 1 predictions
        self.test_probs_pass1 = []       # Pass 1 probabilities
        self.test_entropy_pass1 = []     # Pass 1 entropy (raw)
        self.test_logits_pass1 = []      # Pass 1 logits
        self.test_pmax_pass1 = []        # Pass 1 max probability
        self.test_preds_pass2 = []       # Pass 2 predictions (final)
        self.test_entropy_pass2 = []     # Pass 2 entropy
        self.test_pmax_pass2 = []        # Pass 2 max probability
        self.test_correct_pass1 = []     # Whether Pass 1 was correct
        self.test_correct_pass2 = []     # Whether Pass 2 was correct
        self.test_delta_logits = []      # ||logits_pass2 - logits_pass1||
        self.test_delta_kl = []          # KL(p_pass1 || p_pass2)
        self.test_update_on = []         # Whether TTT was applied (alpha > 0)
        
        # Register hook to capture features from the layer before classification
        if hasattr(self.model, 'tcn_head'):
            if hasattr(self.model.tcn_head, 'classifier'):
                # Attach hook to the input of the classifier
                self.model.tcn_head.classifier.register_forward_hook(self._feature_hook)
            else:
                 pass
                 
        self._captured_features = None

    def _feature_hook(self, module, input, output):
        # Input to the classifier is the feature vector
        # input is a tuple (tensor,)
        if isinstance(input, tuple):
             self._captured_features = input[0].detach().cpu()
        else:
             self._captured_features = input.detach().cpu()

    # forward
    def forward(self, x):
        return self.model(x)

    # optimiser / scheduler
    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                         betas=betas,
                                         weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                          betas=betas,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError
        if self.hparams.scheduler:
            scheduler = LambdaLR(optimizer,
                                 linear_warmup_cosine_decay(self.hparams.warmup_epochs,
                                                            self.hparams.max_epochs))
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # steps
    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="val")
        return {"val_loss": loss, "val_acc": acc}

    def on_train_epoch_end(self):
        # Collect metrics from trainer.logged_metrics
        # Note: logged_metrics contains tensors on device
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                  for k, v in self.trainer.logged_metrics.items()}
        
        # Filter for current epoch metrics relevant to history
        epoch_metrics = {
            "epoch": self.current_epoch,
            "train_loss": metrics.get("train_loss", 0),
            "train_acc": metrics.get("train_acc", 0),
            "val_loss": metrics.get("val_loss", 0),
            "val_acc": metrics.get("val_acc", 0)
        }
        self.train_history.append(epoch_metrics)

    def test_step(self, batch, batch_idx):
        """
        IMPORTANT: run forward EXACTLY ONCE per batch.
        TTT-based models may update internal state on forward; calling forward twice
        contaminates metrics and debug signals.
        """
        x, y = batch

        with torch.no_grad():
            y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)

        # log scalar metrics
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        preds = torch.argmax(y_hat, dim=-1)
        self.test_kappa.update(preds, y)
        self.test_cm.update(preds, y)
        self.log("test_kappa", self.test_kappa, prog_bar=False, on_step=False, on_epoch=True)

        # Collect features from hook (captured during the SAME forward call)
        if self._captured_features is not None:
            self.test_features.append(self._captured_features)
            self._captured_features = None

        # Store logits/labels for existing analysis pipeline
        self.test_logits.append(y_hat.detach().cpu())
        self.test_labels.append(y.detach().cpu())

        # ── Debug stats (alpha/entropy/clip/group weights/norms) ──
        # Entropy from FINAL logits (user-selected)
        p = torch.softmax(y_hat, dim=-1).clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=-1)  # [B]
        self.test_entropy.append(entropy.detach().cpu())

        # Model-provided debug batch stats (optional)
        get_dbg = getattr(self.model, "get_debug_batch", None)
        if callable(get_dbg):
            dbg = get_dbg()
            if dbg.get("alpha") is not None:
                self.test_alpha.append(dbg["alpha"].detach().cpu())
            if dbg.get("gate_entropy") is not None:
                self.test_gate_entropy.append(dbg["gate_entropy"].detach().cpu())
            if dbg.get("ttt_lr_scale") is not None:
                self.test_ttt_lr_scale.append(dbg["ttt_lr_scale"].detach().cpu())
            if dbg.get("ttt_effective_base_lr") is not None:
                self.test_ttt_effective_base_lr.append(dbg["ttt_effective_base_lr"].detach().cpu())
            if dbg.get("clip_ratio") is not None:
                self.test_clip_ratio.append(dbg["clip_ratio"].detach().cpu())
            if dbg.get("group_attn_weights") is not None:
                self.test_group_attn_weights.append(dbg["group_attn_weights"].detach().cpu())
            if dbg.get("conv_norm_pre") is not None:
                self.test_conv_norm_pre.append(dbg["conv_norm_pre"].detach().cpu())
            if dbg.get("conv_norm_post") is not None:
                self.test_conv_norm_post.append(dbg["conv_norm_post"].detach().cpu())
            if dbg.get("group_attn_gamma") is not None:
                # scalar
                self.test_group_attn_gamma.append(float(dbg["group_attn_gamma"]))
            if dbg.get("group_attn_temperature") is not None:
                self.test_group_attn_temperature.append(float(dbg["group_attn_temperature"]))
            
            # ── 2-pass debug info (Pass 1 = attention-only, Pass 2 = with TTT) ──
            if dbg.get("preds_pass1") is not None:
                preds_pass1 = dbg["preds_pass1"].detach().cpu()
                self.test_preds_pass1.append(preds_pass1)
                self.test_preds_pass2.append(preds)  # Pass 2 is the final prediction
                
                # Compute correctness for flip analysis
                correct_pass1 = (preds_pass1 == y.cpu())
                correct_pass2 = (preds == y)
                self.test_correct_pass1.append(correct_pass1)
                self.test_correct_pass2.append(correct_pass2.cpu())
                
            if dbg.get("probs_pass1") is not None:
                self.test_probs_pass1.append(dbg["probs_pass1"].detach().cpu())
            if dbg.get("entropy_pass1") is not None:
                self.test_entropy_pass1.append(dbg["entropy_pass1"].detach().cpu())
            if dbg.get("logits_pass1") is not None:
                self.test_logits_pass1.append(dbg["logits_pass1"].detach().cpu())
            if dbg.get("pmax_pass1") is not None:
                self.test_pmax_pass1.append(dbg["pmax_pass1"].detach().cpu())
            
            # Pass 2 additional info
            if dbg.get("entropy_pass2") is not None:
                self.test_entropy_pass2.append(dbg["entropy_pass2"].detach().cpu())
            if dbg.get("pmax_pass2") is not None:
                self.test_pmax_pass2.append(dbg["pmax_pass2"].detach().cpu())
            
            # Delta metrics
            if dbg.get("delta_logits") is not None:
                self.test_delta_logits.append(dbg["delta_logits"].detach().cpu())
            if dbg.get("delta_kl") is not None:
                self.test_delta_kl.append(dbg["delta_kl"].detach().cpu())
            
            # Update info
            if dbg.get("update_on") is not None:
                self.test_update_on.append(dbg["update_on"].detach().cpu())

        return {"test_loss": loss, "test_acc": acc}

    # common logic
    def shared_step(self, batch, batch_idx, mode: str = "train"):
        x, y = batch
        if mode == "train":
            if self.hparams.get("random_channel_masking", False):
                # Add random EEG channel masking augmentation (did not improve the training)
                x = random_channel_mask(x, self.hparams.get("keep_ratio",0.9))
            if self.hparams.get("random_channel_selection", False):
                # Randomly select a subset of EEG channels (did not improve the training)
                x, _ = select_random_channels(x, self.hparams.get("keep_ratio",0.9))

        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        # Optional: Group-attention sparsity regularization (minimize entropy of group weights)
        # This encourages "select one band/group" behavior rather than uniform 1/G.
        lam_ga = float(self.hparams.get("group_attn_entropy_reg", 0.0) or 0.0)
        if lam_ga > 0.0:
            get_dbg = getattr(self.model, "get_debug_batch", None)
            if callable(get_dbg):
                dbg = get_dbg()
                w = dbg.get("group_attn_weights", None)  # [B, G]
                if w is not None and isinstance(w, torch.Tensor) and w.ndim == 2:
                    w = w.clamp_min(1e-12)
                    ga_entropy = -(w * w.log()).sum(dim=-1).mean()
                    loss = loss + lam_ga * ga_entropy

        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)
        # log scalar metrics
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss, acc
    
    # grab confusion matrix once per test epoch
    def on_test_epoch_end(self):
        # 1) raw counts  ───────────────────────────────────────────
        cm_counts = self.test_cm.compute()   # shape [C, C]
        self.test_cm.reset()

        # 2) row-normalise → %  (handle rows with 0 samples safely)
        with torch.no_grad():
            row_sums = cm_counts.sum(dim=1, keepdim=True).clamp_min(1)
            cm_percent = cm_counts.float() / row_sums * 100.0

        self.test_confmat = cm_percent.cpu()        # stash for plotting

        # ── Save Analysis Data ───────────────────────────
        if self.subject_id != "unknown":
            os.makedirs(self.results_dir, exist_ok=True)
            
            # 1. Save History
            history_path = os.path.join(self.results_dir, f"history_s{self.subject_id}_{self.model_name}.json")
            with open(history_path, 'w') as f:
                json.dump(self.train_history, f, indent=4)
            
            # 2. Save Features and Labels
            if self.test_features:
                features = torch.cat(self.test_features, dim=0).numpy()
                labels = torch.cat(self.test_labels, dim=0).numpy()
                features_path = os.path.join(self.results_dir, f"features_s{self.subject_id}_{self.model_name}.npz")
                np.savez(features_path, features=features, labels=labels)
                self.test_features = [] # clear
                self.test_labels = []

            # 3. Save Logits
            if self.test_logits:
                logits = torch.cat(self.test_logits, dim=0).numpy()
                logits_path = os.path.join(self.results_dir, f"logits_s{self.subject_id}_{self.model_name}.npy")
                np.save(logits_path, logits)
                self.test_logits = [] # clear

            # 4. Save Debug Stats (small json)
            debug = {}

            def _summarize_1d(name, arr):
                if not arr:
                    return
                x = torch.cat(arr, dim=0).numpy()
                debug[name] = {
                    "mean": float(np.mean(x)),
                    "p50": float(np.percentile(x, 50)),
                    "p90": float(np.percentile(x, 90)),
                }

            _summarize_1d("alpha", self.test_alpha)
            _summarize_1d("entropy", self.test_entropy)
            _summarize_1d("gate_entropy", self.test_gate_entropy)
            _summarize_1d("ttt_lr_scale", self.test_ttt_lr_scale)
            _summarize_1d("ttt_effective_base_lr", self.test_ttt_effective_base_lr)
            _summarize_1d("clip_ratio", self.test_clip_ratio)
            _summarize_1d("conv_norm_pre", self.test_conv_norm_pre)
            _summarize_1d("conv_norm_post", self.test_conv_norm_post)

            # alpha-entropy correlation (Pearson r)
            if self.test_alpha and self.test_entropy:
                a = torch.cat(self.test_alpha, dim=0).numpy()
                e = torch.cat(self.test_entropy, dim=0).numpy()
                if len(a) >= 2 and np.std(a) > 1e-12 and np.std(e) > 1e-12:
                    r = float(np.corrcoef(a, e)[0, 1])
                else:
                    r = None
                debug["alpha_entropy_pearson_r"] = r

            # group attn weights stats: [N, G]
            if self.test_group_attn_weights:
                w = torch.cat(self.test_group_attn_weights, dim=0).numpy()
                debug["group_attn_weights"] = {
                    "mean": [float(v) for v in np.mean(w, axis=0)],
                    "std": [float(v) for v in np.std(w, axis=0)],
                }

            # gamma (scalar, last seen)
            if self.test_group_attn_gamma:
                debug["group_attn_gamma_last"] = float(self.test_group_attn_gamma[-1])
            if self.test_group_attn_temperature:
                debug["group_attn_temperature_last"] = float(self.test_group_attn_temperature[-1])

            # ── 2-pass flip analysis (comprehensive) ──────────────────────────
            if self.test_correct_pass1 and self.test_correct_pass2:
                correct_p1 = torch.cat(self.test_correct_pass1, dim=0).numpy()
                correct_p2 = torch.cat(self.test_correct_pass2, dim=0).numpy()
                n_samples = len(correct_p1)
                
                # Flip analysis
                flip_to_correct = (~correct_p1) & correct_p2  # was wrong, now correct
                flip_to_wrong = correct_p1 & (~correct_p2)    # was correct, now wrong
                stayed_correct = correct_p1 & correct_p2
                stayed_wrong = (~correct_p1) & (~correct_p2)
                
                tpa = {
                    "n_samples": n_samples,
                    "acc_pass1": float(correct_p1.mean()),
                    "acc_pass2": float(correct_p2.mean()),
                    "acc_change": float(correct_p2.mean() - correct_p1.mean()),
                    "n_flip_to_correct": int(flip_to_correct.sum()),
                    "n_flip_to_wrong": int(flip_to_wrong.sum()),
                    "n_stayed_correct": int(stayed_correct.sum()),
                    "n_stayed_wrong": int(stayed_wrong.sum()),
                    "flip_to_correct_ratio": float(flip_to_correct.mean()),
                    "flip_to_wrong_ratio": float(flip_to_wrong.mean()),
                    "net_flip": int(flip_to_correct.sum() - flip_to_wrong.sum()),
                }
                
                # Entropy analysis by flip status
                if self.test_entropy_pass1:
                    entropy_p1 = torch.cat(self.test_entropy_pass1, dim=0).numpy()
                    tpa["entropy_pass1_mean"] = float(entropy_p1.mean())
                    tpa["entropy_pass1_std"] = float(entropy_p1.std())
                    
                    # Entropy by flip status
                    if flip_to_correct.sum() > 0:
                        tpa["entropy_p1_flip_to_correct"] = float(entropy_p1[flip_to_correct].mean())
                    if flip_to_wrong.sum() > 0:
                        tpa["entropy_p1_flip_to_wrong"] = float(entropy_p1[flip_to_wrong].mean())
                    if stayed_correct.sum() > 0:
                        tpa["entropy_p1_stayed_correct"] = float(entropy_p1[stayed_correct].mean())
                    if stayed_wrong.sum() > 0:
                        tpa["entropy_p1_stayed_wrong"] = float(entropy_p1[stayed_wrong].mean())
                
                if self.test_entropy_pass2:
                    entropy_p2 = torch.cat(self.test_entropy_pass2, dim=0).numpy()
                    tpa["entropy_pass2_mean"] = float(entropy_p2.mean())
                    tpa["entropy_pass2_std"] = float(entropy_p2.std())
                    
                    # Delta entropy
                    if self.test_entropy_pass1:
                        delta_entropy = entropy_p2 - entropy_p1
                        tpa["delta_entropy_mean"] = float(delta_entropy.mean())
                        if flip_to_correct.sum() > 0:
                            tpa["delta_entropy_flip_to_correct"] = float(delta_entropy[flip_to_correct].mean())
                        if flip_to_wrong.sum() > 0:
                            tpa["delta_entropy_flip_to_wrong"] = float(delta_entropy[flip_to_wrong].mean())
                
                # pmax analysis
                if self.test_pmax_pass1:
                    pmax_p1 = torch.cat(self.test_pmax_pass1, dim=0).numpy()
                    tpa["pmax_pass1_mean"] = float(pmax_p1.mean())
                    if flip_to_correct.sum() > 0:
                        tpa["pmax_p1_flip_to_correct"] = float(pmax_p1[flip_to_correct].mean())
                    if flip_to_wrong.sum() > 0:
                        tpa["pmax_p1_flip_to_wrong"] = float(pmax_p1[flip_to_wrong].mean())
                
                if self.test_pmax_pass2:
                    pmax_p2 = torch.cat(self.test_pmax_pass2, dim=0).numpy()
                    tpa["pmax_pass2_mean"] = float(pmax_p2.mean())
                
                # Alpha/lr_scale analysis by flip status
                if self.test_alpha:
                    alpha_all = torch.cat(self.test_alpha, dim=0).numpy()
                    tpa["alpha_mean"] = float(alpha_all.mean())
                    if flip_to_correct.sum() > 0:
                        tpa["alpha_flip_to_correct"] = float(alpha_all[flip_to_correct].mean())
                    if flip_to_wrong.sum() > 0:
                        tpa["alpha_flip_to_wrong"] = float(alpha_all[flip_to_wrong].mean())
                    if stayed_correct.sum() > 0:
                        tpa["alpha_stayed_correct"] = float(alpha_all[stayed_correct].mean())
                    if stayed_wrong.sum() > 0:
                        tpa["alpha_stayed_wrong"] = float(alpha_all[stayed_wrong].mean())
                
                # Delta metrics (how much the output changed)
                if self.test_delta_logits:
                    delta_logits = torch.cat(self.test_delta_logits, dim=0).numpy()
                    tpa["delta_logits_mean"] = float(delta_logits.mean())
                    tpa["delta_logits_p50"] = float(np.percentile(delta_logits, 50))
                    tpa["delta_logits_p90"] = float(np.percentile(delta_logits, 90))
                    if flip_to_correct.sum() > 0:
                        tpa["delta_logits_flip_to_correct"] = float(delta_logits[flip_to_correct].mean())
                    if flip_to_wrong.sum() > 0:
                        tpa["delta_logits_flip_to_wrong"] = float(delta_logits[flip_to_wrong].mean())
                    if stayed_correct.sum() > 0:
                        tpa["delta_logits_stayed_correct"] = float(delta_logits[stayed_correct].mean())
                    if stayed_wrong.sum() > 0:
                        tpa["delta_logits_stayed_wrong"] = float(delta_logits[stayed_wrong].mean())
                
                if self.test_delta_kl:
                    delta_kl = torch.cat(self.test_delta_kl, dim=0).numpy()
                    tpa["delta_kl_mean"] = float(delta_kl.mean())
                    tpa["delta_kl_p50"] = float(np.percentile(delta_kl, 50))
                    tpa["delta_kl_p90"] = float(np.percentile(delta_kl, 90))
                    if flip_to_correct.sum() > 0:
                        tpa["delta_kl_flip_to_correct"] = float(delta_kl[flip_to_correct].mean())
                    if flip_to_wrong.sum() > 0:
                        tpa["delta_kl_flip_to_wrong"] = float(delta_kl[flip_to_wrong].mean())
                
                # Update analysis (what % of samples actually had TTT applied)
                if self.test_update_on:
                    update_on = torch.cat(self.test_update_on, dim=0).numpy()
                    tpa["update_ratio"] = float(update_on.mean())
                    tpa["n_updated"] = int(update_on.sum())
                    tpa["n_not_updated"] = int((~update_on).sum())
                    
                    # Accuracy split by update status
                    if update_on.sum() > 0:
                        tpa["acc_pass1_when_updated"] = float(correct_p1[update_on].mean())
                        tpa["acc_pass2_when_updated"] = float(correct_p2[update_on].mean())
                        tpa["acc_change_when_updated"] = float(correct_p2[update_on].mean() - correct_p1[update_on].mean())
                    if (~update_on).sum() > 0:
                        tpa["acc_pass1_when_not_updated"] = float(correct_p1[~update_on].mean())
                        tpa["acc_pass2_when_not_updated"] = float(correct_p2[~update_on].mean())
                
                debug["two_pass_analysis"] = tpa

            # ── Save 2-pass sample-level data (comprehensive) ──────────────────
            if self.test_preds_pass1 and self.test_preds_pass2:
                twopass_samples = []
                preds_p1 = torch.cat(self.test_preds_pass1, dim=0).numpy()
                preds_p2 = torch.cat(self.test_preds_pass2, dim=0).numpy()
                labels = torch.cat(self.test_labels, dim=0).numpy() if self.test_labels else None
                
                # Optional arrays
                probs_p1 = torch.cat(self.test_probs_pass1, dim=0).numpy() if self.test_probs_pass1 else None
                entropy_p1 = torch.cat(self.test_entropy_pass1, dim=0).numpy() if self.test_entropy_pass1 else None
                entropy_p2 = torch.cat(self.test_entropy_pass2, dim=0).numpy() if self.test_entropy_pass2 else None
                pmax_p1 = torch.cat(self.test_pmax_pass1, dim=0).numpy() if self.test_pmax_pass1 else None
                pmax_p2 = torch.cat(self.test_pmax_pass2, dim=0).numpy() if self.test_pmax_pass2 else None
                alpha_all = torch.cat(self.test_alpha, dim=0).numpy() if self.test_alpha else None
                lr_scale_all = torch.cat(self.test_ttt_lr_scale, dim=0).numpy() if self.test_ttt_lr_scale else None
                delta_logits_all = torch.cat(self.test_delta_logits, dim=0).numpy() if self.test_delta_logits else None
                delta_kl_all = torch.cat(self.test_delta_kl, dim=0).numpy() if self.test_delta_kl else None
                update_on_all = torch.cat(self.test_update_on, dim=0).numpy() if self.test_update_on else None
                
                for i in range(len(preds_p1)):
                    sample = {
                        "idx": i,
                        "pred_pass1": int(preds_p1[i]),
                        "pred_pass2": int(preds_p2[i]),
                    }
                    if labels is not None:
                        sample["true_label"] = int(labels[i])
                        sample["correct_pass1"] = bool(preds_p1[i] == labels[i])
                        sample["correct_pass2"] = bool(preds_p2[i] == labels[i])
                        # Determine flip status
                        c1, c2 = sample["correct_pass1"], sample["correct_pass2"]
                        if not c1 and c2:
                            sample["flip_status"] = "flip_to_correct"
                        elif c1 and not c2:
                            sample["flip_status"] = "flip_to_wrong"
                        elif c1 and c2:
                            sample["flip_status"] = "stayed_correct"
                        else:
                            sample["flip_status"] = "stayed_wrong"
                    
                    # Pass 1 info
                    if entropy_p1 is not None:
                        sample["entropy_pass1"] = float(entropy_p1[i])
                    if pmax_p1 is not None:
                        sample["pmax_pass1"] = float(pmax_p1[i])
                    if probs_p1 is not None:
                        sample["probs_pass1"] = [float(p) for p in probs_p1[i]]
                    
                    # Pass 2 info
                    if entropy_p2 is not None:
                        sample["entropy_pass2"] = float(entropy_p2[i])
                    if pmax_p2 is not None:
                        sample["pmax_pass2"] = float(pmax_p2[i])
                    
                    # Gating info
                    if alpha_all is not None:
                        sample["alpha"] = float(alpha_all[i])
                    if lr_scale_all is not None:
                        sample["lr_scale"] = float(lr_scale_all[i])
                    if update_on_all is not None:
                        sample["update_on"] = bool(update_on_all[i])
                    
                    # Delta metrics
                    if delta_logits_all is not None:
                        sample["delta_logits"] = float(delta_logits_all[i])
                    if delta_kl_all is not None:
                        sample["delta_kl"] = float(delta_kl_all[i])
                    if entropy_p1 is not None and entropy_p2 is not None:
                        sample["delta_entropy"] = float(entropy_p2[i] - entropy_p1[i])
                    if pmax_p1 is not None and pmax_p2 is not None:
                        sample["delta_pmax"] = float(pmax_p2[i] - pmax_p1[i])
                    
                    twopass_samples.append(sample)
                
                twopass_path = os.path.join(self.results_dir, f"twopass_s{self.subject_id}_{self.model_name}.json")
                with open(twopass_path, "w") as f:
                    json.dump(twopass_samples, f, indent=2)

            if debug:
                debug_path = os.path.join(self.results_dir, f"debug_s{self.subject_id}_{self.model_name}.json")
                with open(debug_path, "w") as f:
                    json.dump(debug, f, indent=4)

            # clear debug buffers
            self.test_alpha = []
            self.test_entropy = []
            self.test_gate_entropy = []
            self.test_ttt_lr_scale = []
            self.test_ttt_effective_base_lr = []
            self.test_clip_ratio = []
            self.test_group_attn_weights = []
            self.test_conv_norm_pre = []
            self.test_conv_norm_post = []
            self.test_group_attn_gamma = []
            self.test_group_attn_temperature = []
            # 2-pass buffers
            self.test_preds_pass1 = []
            self.test_probs_pass1 = []
            self.test_entropy_pass1 = []
            self.test_logits_pass1 = []
            self.test_pmax_pass1 = []
            self.test_preds_pass2 = []
            self.test_entropy_pass2 = []
            self.test_pmax_pass2 = []
            self.test_correct_pass1 = []
            self.test_correct_pass2 = []
            self.test_delta_logits = []
            self.test_delta_kl = []
            self.test_update_on = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return torch.argmax(self.forward(x), dim=-1)
    