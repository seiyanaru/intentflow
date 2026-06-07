from typing import Dict, Optional

import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import os

import numpy as np
from utils.interaug import interaug
def make_collate_fn(preproc):
    """Return a collate function that optionally applies interaug."""
    def collate(batch):
        xs, ys = zip(*batch)                  # tuples of tensors/ints
        x = torch.stack(xs)                   # [B, C, T]
        y = torch.tensor(ys, dtype=torch.long)

        if preproc.get("interaug", False):
            x, y = interaug([x, y])           # now shapes are OK
        return x, y
    return collate


class BaseDataModule(pl.LightningDataModule):
    dataset = None
    train_dataset = None
    test_dataset = None

    def __init__(self, preprocessing_dict: Dict, subject_id: int):
        super(BaseDataModule, self).__init__()
        self.preprocessing_dict = preprocessing_dict
        self.subject_id = subject_id

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        num_workers = self.preprocessing_dict.get("num_workers", os.cpu_count() // 2)
        loader_kwargs = dict(
            batch_size=self.preprocessing_dict["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=make_collate_fn(self.preprocessing_dict),
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4
        return DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        # IMPORTANT: Validation should NOT be the same as test (data leakage)
        # For BCIC2a, if session_E exists, we split session_T into train/val
        # If using BCICIV2a class, this will return test (temporary workaround until proper val split)
        # Recommended: Use BCICIV2aTVT class which has proper train/val/test split
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            num_workers = self.preprocessing_dict.get("num_workers", os.cpu_count() // 2)
            loader_kwargs = dict(
                batch_size=self.preprocessing_dict["batch_size"],
                num_workers=num_workers,
                pin_memory=True,
            )
            if num_workers > 0:
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = 4
            return DataLoader(self.val_dataset, **loader_kwargs)
        else:
            # Fallback: return test (WARNING: This causes data leakage!)
            print("WARNING: val_dataset not found. Returning test_dataloader (DATA LEAKAGE!)")
            print("         Consider using BCICIV2aTVT or implementing proper validation split.")
            return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        # test_batch_size overrides batch_size at test time (e.g. 1 for online OTTA simulation)
        test_bs = self.preprocessing_dict.get("test_batch_size", self.preprocessing_dict["batch_size"])
        num_workers = self.preprocessing_dict.get("num_workers", os.cpu_count() // 2)
        loader_kwargs = dict(
            batch_size=test_bs,
            num_workers=num_workers,
            pin_memory=True,
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4
        return DataLoader(self.test_dataset, **loader_kwargs)

    @staticmethod
    # Method 1 (per-channel & per-timepoint) across samples
    # def _z_scale(X, X_test):
    #     for ch_idx in range(X.shape[1]):
    #         sc = StandardScaler()
    #         X[:, ch_idx, :] = sc.fit_transform(X[:, ch_idx, :])
    #         X_test[:, ch_idx, :] = sc.transform(X_test[:, ch_idx, :])
    #     return X, X_test
    # Method 2 Per-channel across all samples and timepoints
    def _z_scale(X, X_test):
        # reshape to (samples*time, channels)
        s, c, t = X.shape
        X_2d      = X.transpose(1, 0, 2).reshape(c, -1).T
        X_test_2d = X_test.transpose(1, 0, 2).reshape(c, -1).T

        sc = StandardScaler().fit(X_2d)
        X      = sc.transform(X_2d).T.reshape(c, s, t).transpose(1, 0, 2)
        X_test = sc.transform(X_test_2d).T.reshape(c, X_test.shape[0], t).transpose(1, 0, 2)
        return X, X_test

    # @staticmethod
    # # Method 1 (per-channel & per-timepoint) across samples
    # def _z_scale_tvt(X_train, X_val, X_test):
    #     for ch in range(X_train.shape[1]):
    #         sc = StandardScaler()
    #         X_train[:, ch, :] = sc.fit_transform(X_train[:, ch, :])
    #         X_val[:, ch, :] = sc.transform(X_val[:, ch, :])
    #         X_test[:, ch, :] = sc.transform(X_test[:, ch, :])
    #     return X_train, X_val, X_test

    # Method 2 Per-channel across all samples and timepoints
    def _z_scale_tvt(X, X_val, X_test):
        # reshape to (samples*time, channels)
        s, c, t = X.shape
        X_2d      = X.transpose(1, 0, 2).reshape(c, -1).T
        X_val_2d = X_val.transpose(1, 0, 2).reshape(c, -1).T
        X_test_2d = X_test.transpose(1, 0, 2).reshape(c, -1).T

        sc = StandardScaler().fit(X_2d)
        X      = sc.transform(X_2d).T.reshape(c, s, t).transpose(1, 0, 2)
        X_val = sc.transform(X_val_2d).T.reshape(c, X_val.shape[0], t).transpose(1, 0, 2)
        X_test = sc.transform(X_test_2d).T.reshape(c, X_test.shape[0], t).transpose(1, 0, 2)
        return X, X_val, X_test

    @staticmethod
    def _ea_reference(
        X,
        eps=1e-6,
        shrinkage=0.0,
        power=1.0,
        channel_weights=None,
        preserve_diag=True,
        weight_strength=1.0,
    ):
        """Return R^{-1/2} for Euclidean Alignment from trials shaped (N,C,T)."""
        if X.ndim != 3:
            raise ValueError(f"EA expects trials shaped (N,C,T), got {X.shape}")
        covs = np.matmul(X, np.swapaxes(X, 1, 2)) / X.shape[-1]
        R = covs.mean(axis=0)
        if channel_weights is not None:
            R_raw = R.copy()
            weights = np.asarray(channel_weights, dtype=R.dtype)
            if weights.shape != (R.shape[0],):
                raise ValueError(
                    f"EA channel weights must be shaped ({R.shape[0]},), got {weights.shape}"
                )
            diag = np.diag(R).copy()
            scale = np.sqrt(np.clip(weights, 0.0, 1.0))
            R = scale[:, None] * R * scale[None, :]
            if preserve_diag:
                np.fill_diagonal(R, diag)
            R = (1.0 - weight_strength) * R_raw + weight_strength * R
        if shrinkage > 0.0:
            scale = np.trace(R) / R.shape[0]
            R = (1.0 - shrinkage) * R + shrinkage * scale * np.eye(R.shape[0])
        evals, evecs = np.linalg.eigh(R)
        evals = np.maximum(evals, eps)
        weights = evals ** (-0.5 * power)
        return (evecs * weights) @ evecs.T

    @staticmethod
    def _apply_ea(
        X,
        reference=None,
        eps=1e-6,
        shrinkage=0.0,
        power=1.0,
        channel_weights=None,
        preserve_diag=True,
        weight_strength=1.0,
    ):
        """Apply Euclidean Alignment to X using X or a provided reference set."""
        dtype = X.dtype
        ref = X if reference is None else reference
        r_isqrt = BaseDataModule._ea_reference(
            ref,
            eps=eps,
            shrinkage=shrinkage,
            power=power,
            channel_weights=channel_weights,
            preserve_diag=preserve_diag,
            weight_strength=weight_strength,
        )
        aligned = np.matmul(r_isqrt[None, :, :], X)
        return aligned.astype(dtype, copy=False)

    @staticmethod
    def _artifact_stress_cfg(preprocessing_dict):
        cfg = dict(preprocessing_dict.get("artifact_stress", {}))
        mode = cfg.get("mode", "none")
        cfg["enabled"] = bool(cfg.get("enabled", mode not in ("none", None, "")))
        cfg["mode"] = mode
        cfg["channels"] = cfg.get("channels", None)
        cfg["n_channels"] = int(cfg.get("n_channels", 2))
        cfg["level"] = float(cfg.get("level", 3.0))
        cfg["seed"] = int(cfg.get("seed", 0))
        cfg["p_trials"] = float(cfg.get("p_trials", 1.0))
        cfg["line_freq"] = float(cfg.get("line_freq", 50.0))
        cfg["burst_fraction"] = float(cfg.get("burst_fraction", 0.2))
        return cfg

    @staticmethod
    def _parse_stress_channels(channels, n_total):
        if channels is None or channels == "":
            return None
        if isinstance(channels, str):
            parsed = [int(ch.strip()) for ch in channels.split(",") if ch.strip()]
        else:
            parsed = [int(ch) for ch in channels]
        bad = [ch for ch in parsed if ch < 0 or ch >= n_total]
        if bad:
            raise ValueError(
                f"artifact_stress channels out of range for {n_total} channels: {bad}"
            )
        return parsed

    @staticmethod
    def _apply_artifact_stress(X, preprocessing_dict, subject_id=None):
        """Inject reproducible test-only channel corruption before EA."""
        cfg = BaseDataModule._artifact_stress_cfg(preprocessing_dict)
        if not cfg["enabled"]:
            return X

        mode = cfg["mode"]
        if mode in ("none", None, ""):
            return X
        valid_modes = {
            "gaussian",
            "highvar",
            "flatline",
            "dropout",
            "line",
            "burst",
            "scale",
            "mixed",
        }
        if mode not in valid_modes:
            raise ValueError(f"Unknown artifact_stress mode: {mode}")

        out = X.copy()
        rng = np.random.default_rng(cfg["seed"] + int(subject_id or 0) * 1009)
        n_trials, n_channels, n_times = out.shape
        channels = BaseDataModule._parse_stress_channels(cfg["channels"], n_channels)
        if channels is None:
            n_pick = max(1, min(cfg["n_channels"], n_channels))
            channels = sorted(rng.choice(n_channels, size=n_pick, replace=False).tolist())

        p_trials = min(max(cfg["p_trials"], 0.0), 1.0)
        if p_trials <= 0:
            return out
        trial_mask = rng.random(n_trials) < p_trials
        if not trial_mask.any():
            trial_mask[rng.integers(0, n_trials)] = True
        trial_idx = np.where(trial_mask)[0]
        level = cfg["level"]

        def add_gaussian(ch):
            noise = rng.normal(0.0, level, size=(len(trial_idx), n_times))
            out[trial_idx, ch, :] += noise.astype(out.dtype, copy=False)

        def add_flatline(ch):
            mean = out[trial_idx, ch, :].mean(axis=-1, keepdims=True)
            jitter = rng.normal(0.0, 0.01, size=(len(trial_idx), n_times))
            out[trial_idx, ch, :] = (mean + jitter).astype(out.dtype, copy=False)

        def add_line(ch):
            t = np.arange(n_times, dtype=np.float64) / float(
                preprocessing_dict.get("sfreq", 250)
            )
            phase = rng.uniform(0.0, 2.0 * np.pi, size=(len(trial_idx), 1))
            sine = level * np.sin(2.0 * np.pi * cfg["line_freq"] * t[None, :] + phase)
            out[trial_idx, ch, :] += sine.astype(out.dtype, copy=False)

        def add_burst(ch):
            burst_len = max(1, int(n_times * min(max(cfg["burst_fraction"], 0.01), 1.0)))
            for row, trial in enumerate(trial_idx):
                start = int(rng.integers(0, max(n_times - burst_len + 1, 1)))
                stop = min(start + burst_len, n_times)
                out[trial, ch, start:stop] += rng.normal(0.0, level, size=stop - start)

        def add_scale(ch):
            out[trial_idx, ch, :] *= level

        for pos, ch in enumerate(channels):
            ch_mode = mode
            if mode == "mixed":
                ch_mode = ("highvar", "flatline", "line", "burst", "scale")[pos % 5]
            if ch_mode in ("gaussian", "highvar"):
                add_gaussian(ch)
            elif ch_mode in ("flatline", "dropout"):
                add_flatline(ch)
            elif ch_mode == "line":
                add_line(ch)
            elif ch_mode == "burst":
                add_burst(ch)
            elif ch_mode == "scale":
                add_scale(ch)

        print(
            f"Applied artifact stress: mode={mode}, channels={channels}, "
            f"level={level:g}, p_trials={p_trials:g}, seed={cfg['seed']}"
        )
        return out.astype(X.dtype, copy=False)

    @staticmethod
    def _ea_enabled(preprocessing_dict):
        ea_cfg = preprocessing_dict.get("ea", {})
        if isinstance(ea_cfg, bool):
            return ea_cfg
        return bool(ea_cfg.get("enabled", False))

    @staticmethod
    def _ea_eps(preprocessing_dict):
        ea_cfg = preprocessing_dict.get("ea", {})
        if isinstance(ea_cfg, bool):
            return 1e-6
        return float(ea_cfg.get("eps", 1e-6))

    @staticmethod
    def _ea_shrinkage(preprocessing_dict):
        ea_cfg = preprocessing_dict.get("ea", {})
        if isinstance(ea_cfg, bool):
            return 0.0
        return float(ea_cfg.get("shrinkage", 0.0))

    @staticmethod
    def _ea_power(preprocessing_dict):
        ea_cfg = preprocessing_dict.get("ea", {})
        if isinstance(ea_cfg, bool):
            return 1.0
        return float(ea_cfg.get("power", 1.0))

    @staticmethod
    def _ea_weight_cfg(preprocessing_dict):
        ea_cfg = preprocessing_dict.get("ea", {})
        if isinstance(ea_cfg, bool):
            return {"enabled": False}
        cfg = dict(ea_cfg.get("channel_weighting", {}))
        mode = cfg.get("mode", "none")
        cfg["enabled"] = bool(cfg.get("enabled", mode not in ("none", None, "")))
        cfg["mode"] = mode
        cfg["w_min"] = float(cfg.get("w_min", 0.25))
        cfg["tau"] = float(cfg.get("tau", 2.0))
        cfg["relative_margin"] = float(cfg.get("relative_margin", 0.25))
        cfg["clip_z"] = float(cfg.get("clip_z", 6.0))
        cfg["preserve_diag"] = bool(cfg.get("preserve_diag", True))
        cfg["strength"] = float(cfg.get("strength", 1.0))
        cfg["post_weight_power"] = float(cfg.get("post_weight_power", 0.0))
        cfg["repair_strength"] = float(cfg.get("repair_strength", 0.0))
        cfg["repair_threshold"] = float(cfg.get("repair_threshold", 0.5))
        cfg["repair_topk"] = int(cfg.get("repair_topk", 3))
        cfg["sfreq"] = float(preprocessing_dict.get("sfreq", 250))
        return cfg

    @staticmethod
    def _channel_log_variance(X):
        return np.log(np.var(X, axis=-1) + 1e-8)

    @staticmethod
    def _channel_abs_kurtosis(X):
        centered = X - X.mean(axis=-1, keepdims=True)
        var = np.mean(centered * centered, axis=-1) + 1e-8
        fourth = np.mean(centered**4, axis=-1)
        return np.abs(fourth / (var * var) - 3.0)

    @staticmethod
    def _channel_line_noise_ratio(X, sfreq):
        freqs = np.fft.rfftfreq(X.shape[-1], d=1.0 / sfreq)
        spec = np.abs(np.fft.rfft(X, axis=-1)) ** 2
        total_mask = (freqs >= 1.0) & (freqs <= min(100.0, sfreq / 2.0 - 1.0))
        line_mask = ((freqs >= 48.0) & (freqs <= 52.0)) | (
            (freqs >= 58.0) & (freqs <= 62.0)
        )
        total = spec[..., total_mask].sum(axis=-1) + 1e-8
        line = spec[..., line_mask].sum(axis=-1) + 1e-8
        return np.log(line / total + 1e-8)

    @staticmethod
    def _channel_corr_with_reference(X):
        n_trials, n_channels, _ = X.shape
        out = np.zeros((n_trials, n_channels), dtype=np.float64)
        for ch in range(n_channels):
            ref = (X.sum(axis=1) - X[:, ch]) / max(n_channels - 1, 1)
            a = X[:, ch] - X[:, ch].mean(axis=-1, keepdims=True)
            b = ref - ref.mean(axis=-1, keepdims=True)
            denom = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1)) + 1e-8
            out[:, ch] = (a * b).sum(axis=-1) / denom
        return out

    @staticmethod
    def _robust_z(values, ref_values):
        mean = ref_values.mean(axis=0, keepdims=True)
        std = ref_values.std(axis=0, keepdims=True) + 1e-8
        return (values - mean) / std

    @staticmethod
    def _relative_outlier(values, margin):
        return np.clip(values - np.median(values) - margin, 0.0, None)

    @staticmethod
    def _cov_condition(R):
        evals = np.linalg.eigvalsh(R)
        evals = np.maximum(evals, 1e-12)
        return float(evals[-1] / evals[0])

    @staticmethod
    def _cov_leverage(X):
        covs = np.matmul(X, np.swapaxes(X, 1, 2)) / X.shape[-1]
        R = covs.mean(axis=0)
        full = np.log(BaseDataModule._cov_condition(R) + 1e-8)
        scores = []
        for ch in range(R.shape[0]):
            keep = [idx for idx in range(R.shape[0]) if idx != ch]
            reduced = R[np.ix_(keep, keep)]
            scores.append(
                max(0.0, full - np.log(BaseDataModule._cov_condition(reduced) + 1e-8))
            )
        scores = np.asarray(scores, dtype=np.float64)
        if scores.max() > 0:
            scores = scores / scores.max() * 3.0
        return scores

    @staticmethod
    def _ea_channel_weights(X, reference, preprocessing_dict):
        cfg = BaseDataModule._ea_weight_cfg(preprocessing_dict)
        if not cfg["enabled"]:
            return None
        mode = cfg["mode"]
        clip_z = cfg["clip_z"]
        margin = cfg["relative_margin"]
        sfreq = cfg["sfreq"]

        ref_var = BaseDataModule._channel_log_variance(reference)
        x_var = BaseDataModule._channel_log_variance(X)
        var_z = BaseDataModule._robust_z(x_var, ref_var)
        high_var = np.clip(var_z, 0.0, clip_z).mean(axis=0)
        flatline = np.clip(-var_z, 0.0, clip_z).mean(axis=0)
        bad = BaseDataModule._relative_outlier(high_var, margin)
        bad += BaseDataModule._relative_outlier(flatline, margin)

        if mode in ("artifact", "full"):
            ref_kurt = BaseDataModule._channel_abs_kurtosis(reference)
            x_kurt = BaseDataModule._channel_abs_kurtosis(X)
            ref_line = BaseDataModule._channel_line_noise_ratio(reference, sfreq)
            x_line = BaseDataModule._channel_line_noise_ratio(X, sfreq)
            kurt = np.clip(
                BaseDataModule._robust_z(x_kurt, ref_kurt), 0.0, clip_z
            ).mean(axis=0)
            line = np.clip(
                BaseDataModule._robust_z(x_line, ref_line), 0.0, clip_z
            ).mean(axis=0)
            bad += 0.5 * BaseDataModule._relative_outlier(kurt, margin)
            bad += 0.5 * BaseDataModule._relative_outlier(line, margin)

        if mode == "full":
            ref_corr = BaseDataModule._channel_corr_with_reference(reference)
            x_corr = BaseDataModule._channel_corr_with_reference(X)
            corr = np.clip(
                np.abs(BaseDataModule._robust_z(x_corr, ref_corr)), 0.0, clip_z
            ).mean(axis=0)
            leverage = BaseDataModule._cov_leverage(X)
            bad += 0.4 * BaseDataModule._relative_outlier(corr, margin)
            bad += 0.5 * BaseDataModule._relative_outlier(leverage, margin)

        weights = np.exp(-bad / cfg["tau"])
        return np.clip(weights, cfg["w_min"], 1.0)

    @staticmethod
    def _apply_channel_gate(X, channel_weights, power):
        if channel_weights is None or power <= 0.0:
            return X
        weights = np.asarray(channel_weights, dtype=X.dtype)
        gate = np.clip(weights, 0.0, 1.0) ** power
        return (X * gate[None, :, None]).astype(X.dtype, copy=False)

    @staticmethod
    def _repair_unreliable_channels(X, reference, channel_weights, weight_cfg):
        """Replace low-reliability channels with train-correlation interpolation."""
        strength = weight_cfg["repair_strength"]
        if channel_weights is None or strength <= 0.0:
            return X
        weights = np.asarray(channel_weights, dtype=np.float64)
        bad_channels = np.where(weights <= weight_cfg["repair_threshold"])[0]
        if bad_channels.size == 0:
            return X

        cov = np.matmul(reference, np.swapaxes(reference, 1, 2)).mean(axis=0)
        cov = cov / reference.shape[-1]
        diag = np.sqrt(np.maximum(np.diag(cov), 1e-8))
        corr = cov / (diag[:, None] * diag[None, :] + 1e-8)
        repaired = X.copy()
        good_mask = weights > weight_cfg["repair_threshold"]
        topk = max(1, weight_cfg["repair_topk"])

        used = []
        for ch in bad_channels:
            candidates = np.where(good_mask)[0]
            candidates = candidates[candidates != ch]
            if candidates.size == 0:
                continue
            order = np.argsort(np.abs(corr[ch, candidates]))[::-1][:topk]
            idx = candidates[order]
            coeff = corr[ch, idx].astype(np.float64)
            denom = np.sum(np.abs(coeff)) + 1e-8
            estimate = np.tensordot(coeff / denom, repaired[:, idx, :], axes=(0, 1))
            repaired[:, ch, :] = (
                (1.0 - strength) * repaired[:, ch, :] + strength * estimate
            )
            used.append(int(ch))

        if used:
            print(
                f"Repaired unreliable channels before EA: channels={used}, "
                f"strength={strength:g}, topk={topk}"
            )
        return repaired.astype(X.dtype, copy=False)

    @staticmethod
    def _ea_align_tt(X, X_test, preprocessing_dict):
        """Align train and test as separate EEG domains/sessions."""
        if not BaseDataModule._ea_enabled(preprocessing_dict):
            return X, X_test
        eps = BaseDataModule._ea_eps(preprocessing_dict)
        shrinkage = BaseDataModule._ea_shrinkage(preprocessing_dict)
        power = BaseDataModule._ea_power(preprocessing_dict)
        weight_cfg = BaseDataModule._ea_weight_cfg(preprocessing_dict)
        X_weights = BaseDataModule._ea_channel_weights(X, X, preprocessing_dict)
        X_test_weights = BaseDataModule._ea_channel_weights(
            X_test, X, preprocessing_dict
        )
        X_test = BaseDataModule._repair_unreliable_channels(
            X_test, X, X_test_weights, weight_cfg
        )
        X = BaseDataModule._apply_ea(
            X,
            eps=eps,
            shrinkage=shrinkage,
            power=power,
            channel_weights=X_weights,
            preserve_diag=weight_cfg["preserve_diag"],
            weight_strength=weight_cfg["strength"],
        )
        X = BaseDataModule._apply_channel_gate(
            X, X_weights, weight_cfg["post_weight_power"]
        )
        X_test = BaseDataModule._apply_ea(
            X_test,
            eps=eps,
            shrinkage=shrinkage,
            power=power,
            channel_weights=X_test_weights,
            preserve_diag=weight_cfg["preserve_diag"],
            weight_strength=weight_cfg["strength"],
        )
        X_test = BaseDataModule._apply_channel_gate(
            X_test, X_test_weights, weight_cfg["post_weight_power"]
        )
        print(
            f"Applied session-wise EA: train={X.shape}, test={X_test.shape}, "
            f"eps={eps:g}, shrinkage={shrinkage:g}, power={power:g}, "
            f"weight_mode={weight_cfg['mode']}, weight_strength={weight_cfg['strength']:g}, "
            f"post_weight_power={weight_cfg['post_weight_power']:g}"
        )
        return X, X_test

    @staticmethod
    def _ea_align_tvt(X, X_val, X_test, preprocessing_dict):
        """Align train/val with train reference and test as its own session."""
        if not BaseDataModule._ea_enabled(preprocessing_dict):
            return X, X_val, X_test
        eps = BaseDataModule._ea_eps(preprocessing_dict)
        shrinkage = BaseDataModule._ea_shrinkage(preprocessing_dict)
        power = BaseDataModule._ea_power(preprocessing_dict)
        weight_cfg = BaseDataModule._ea_weight_cfg(preprocessing_dict)
        X_ref = X
        X_weights = BaseDataModule._ea_channel_weights(X_ref, X_ref, preprocessing_dict)
        X_test_weights = BaseDataModule._ea_channel_weights(
            X_test, X_ref, preprocessing_dict
        )
        X_test = BaseDataModule._repair_unreliable_channels(
            X_test, X_ref, X_test_weights, weight_cfg
        )
        X = BaseDataModule._apply_ea(
            X,
            reference=X_ref,
            eps=eps,
            shrinkage=shrinkage,
            power=power,
            channel_weights=X_weights,
            preserve_diag=weight_cfg["preserve_diag"],
            weight_strength=weight_cfg["strength"],
        )
        X = BaseDataModule._apply_channel_gate(
            X, X_weights, weight_cfg["post_weight_power"]
        )
        X_val = BaseDataModule._apply_ea(
            X_val,
            reference=X_ref,
            eps=eps,
            shrinkage=shrinkage,
            power=power,
            channel_weights=X_weights,
            preserve_diag=weight_cfg["preserve_diag"],
            weight_strength=weight_cfg["strength"],
        )
        X_val = BaseDataModule._apply_channel_gate(
            X_val, X_weights, weight_cfg["post_weight_power"]
        )
        X_test = BaseDataModule._apply_ea(
            X_test,
            eps=eps,
            shrinkage=shrinkage,
            power=power,
            channel_weights=X_test_weights,
            preserve_diag=weight_cfg["preserve_diag"],
            weight_strength=weight_cfg["strength"],
        )
        X_test = BaseDataModule._apply_channel_gate(
            X_test, X_test_weights, weight_cfg["post_weight_power"]
        )
        print(
            f"Applied session-wise EA: train={X.shape}, val={X_val.shape}, "
            f"test={X_test.shape}, eps={eps:g}, shrinkage={shrinkage:g}, "
            f"power={power:g}, weight_mode={weight_cfg['mode']}, "
            f"weight_strength={weight_cfg['strength']:g}, "
            f"post_weight_power={weight_cfg['post_weight_power']:g}"
        )
        return X, X_val, X_test
    
    @staticmethod
    def _make_tensor_dataset(X, y):
        return TensorDataset(torch.Tensor(X), torch.Tensor(y).type(torch.LongTensor))
        # return TensorDataset(torch.tensor(X), torch.tensor(y).long())
        
    # @staticmethod
    # def _make_tensor_dataset(X, y, preprocessing_dict=None, mode="train"):
    #     if preprocessing_dict and mode == "train":
    #         return AugmentedTensorDataset(
    #             X, y,
    #             interaug=preprocessing_dict.get("interaug", False),
    #         )
    #     return TensorDataset(torch.tensor(X), torch.tensor(y).long())


# from utils.interaug import interaug
# class AugmentedTensorDataset(TensorDataset):
#     def __init__(self, X, y, interaug=False):
#         super().__init__(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
#         self.interaug = interaug

#     def __getitem__(self, index):
#         x, y = super().__getitem__(index)
        
#         if self.interaug:
#             x, y = interaug([x, y])

#         return x, y
