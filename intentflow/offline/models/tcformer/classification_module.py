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
        loss, acc = self.shared_step(batch, batch_idx, mode="test")
        
        # Collect logits and features
        x, y = batch
        
        if self._captured_features is not None:
             self.test_features.append(self._captured_features)
             self._captured_features = None # Reset
        
        with torch.no_grad():
             logits = self.forward(x)
        
        self.test_logits.append(logits.detach().cpu())
        self.test_labels.append(y.detach().cpu())

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
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)
        # log scalar metrics
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        if mode == "test":
            preds = torch.argmax(y_hat, dim=-1)
            # ── update epoch-level metrics ───────────────────────────
            self.test_kappa.update(preds, y)                       # accumulate
            self.test_cm.update(preds, y)
            
            self.log("test_kappa", self.test_kappa,                # Lightning will call .compute()
                    prog_bar=False, on_step=False, on_epoch=True)

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return torch.argmax(self.forward(x), dim=-1)
    