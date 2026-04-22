import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from models.tcformer.classification_module import (
    ClassificationModule,
    random_channel_mask,
    select_random_channels,
)
from models.tcformer.tcformer import TCFormer as TCFormerBase


def channel_gain_jitter(x: torch.Tensor, std: float) -> torch.Tensor:
    """Per-channel multiplicative gain jitter used as a mild session-shift proxy."""
    if std <= 0.0:
        return x
    noise = torch.randn((x.size(0), x.size(1), 1), device=x.device, dtype=x.dtype)
    gain = (1.0 + std * noise).clamp(min=0.1, max=10.0)
    return x * gain


class TCFormerAugShInv(ClassificationModule):
    """
    Phase-B train-time robustness wrapper.

    Keeps the plain TCFormer backbone and adds:
      1. mild channel gain jitter during training
      2. optional shallow-late invariance regularization

    The inner model is the existing TCFormer wrapper so checkpoints remain
    compatible with the current tcformer_otta evaluation path.
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int = 22,
        F1: int = 32,
        temp_kernel_lengths=(20, 32, 64),
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        D: int = 2,
        dropout_conv: float = 0.4,
        d_group: int = 16,
        tcn_depth: int = 2,
        kernel_length_tcn: int = 4,
        dropout_tcn: float = 0.3,
        use_group_attn: bool = True,
        q_heads: int = 4,
        kv_heads: int = 2,
        trans_depth: int = 2,
        trans_dropout: float = 0.4,
        session_shift_aug: bool = False,
        gain_jitter_std: float = 0.05,
        lambda_aug: float = 0.5,
        lambda_shallow_inv: float = 0.0,
        shallow_inv_mode: str = "mean_logvar",
        **kwargs,
    ):
        model = TCFormerBase(
            n_classes=n_classes,
            n_channels=n_channels,
            F1=F1,
            temp_kernel_lengths=temp_kernel_lengths,
            pool_length_1=pool_length_1,
            pool_length_2=pool_length_2,
            D=D,
            dropout_conv=dropout_conv,
            d_group=d_group,
            tcn_depth=tcn_depth,
            kernel_length_tcn=kernel_length_tcn,
            dropout_tcn=dropout_tcn,
            use_group_attn=use_group_attn,
            q_heads=q_heads,
            kv_heads=kv_heads,
            trans_depth=trans_depth,
            trans_dropout=trans_dropout,
        )
        super().__init__(model=model, n_classes=n_classes, **kwargs)
        self.session_shift_aug = bool(session_shift_aug)
        self.gain_jitter_std = float(gain_jitter_std)
        self.lambda_aug = float(lambda_aug)
        self.lambda_shallow_inv = float(lambda_shallow_inv)
        self.shallow_inv_mode = str(shallow_inv_mode).lower()

    def _prepare_training_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.hparams.get("random_channel_masking", False):
            x = random_channel_mask(x, self.hparams.get("keep_ratio", 0.9))
        if self.hparams.get("random_channel_selection", False):
            x, _ = select_random_channels(x, self.hparams.get("keep_ratio", 0.9))
        return x

    def _get_shallow_late_feature(self) -> torch.Tensor:
        conv_block = self.model.model.conv_block
        feat = getattr(conv_block, "last_shallow_late", None)
        if feat is None:
            raise RuntimeError("conv_block.last_shallow_late was not populated by the forward pass.")
        if feat.ndim == 4:
            feat = feat.squeeze(2)
        return feat

    @staticmethod
    def _mean_logvar(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = feat.float().mean(dim=(0, 2))
        logvar = feat.float().var(dim=(0, 2), unbiased=False).clamp_min(1e-8).log()
        return mu, logvar

    def _shallow_invariance_loss(
        self,
        clean_feat: torch.Tensor,
        aug_feat: torch.Tensor,
    ) -> torch.Tensor:
        mu_clean, logvar_clean = self._mean_logvar(clean_feat)
        mu_aug, logvar_aug = self._mean_logvar(aug_feat)
        if self.shallow_inv_mode == "mean_logvar":
            return F.mse_loss(mu_aug, mu_clean) + F.mse_loss(logvar_aug, logvar_clean)
        if self.shallow_inv_mode == "logvar_only":
            return F.mse_loss(logvar_aug, logvar_clean)
        if self.shallow_inv_mode == "mean_only":
            return F.mse_loss(mu_aug, mu_clean)
        raise ValueError(f"Unknown shallow_inv_mode: {self.shallow_inv_mode}")

    def shared_step(self, batch, batch_idx, mode: str = "train"):
        x, y = batch
        if mode == "train":
            x = self._prepare_training_input(x)

        y_hat = self.forward(x)
        loss_clean = F.cross_entropy(y_hat, y)
        loss = loss_clean
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)

        if mode == "train" and self.session_shift_aug:
            clean_feat = None
            if self.lambda_shallow_inv > 0.0:
                clean_feat = self._get_shallow_late_feature().detach()

            x_aug = channel_gain_jitter(x, self.gain_jitter_std)
            y_hat_aug = self.forward(x_aug)
            loss_aug = F.cross_entropy(y_hat_aug, y)
            loss = loss + self.lambda_aug * loss_aug
            self.log("train_loss_aug", loss_aug, prog_bar=False, on_step=False, on_epoch=True)
            self.log(
                "train_acc_aug",
                accuracy(y_hat_aug, y, task="multiclass", num_classes=self.hparams.n_classes),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

            if self.lambda_shallow_inv > 0.0:
                aug_feat = self._get_shallow_late_feature()
                loss_shinv = self._shallow_invariance_loss(clean_feat, aug_feat)
                loss = loss + self.lambda_shallow_inv * loss_shinv
                self.log("train_loss_shinv", loss_shinv, prog_bar=False, on_step=False, on_epoch=True)

        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        if mode == "train":
            self.log("train_loss_clean", loss_clean, prog_bar=False, on_step=False, on_epoch=True)
        return loss, acc
