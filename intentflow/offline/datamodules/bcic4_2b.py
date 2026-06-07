from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4


class BCICIV2b(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["hand(L)", "hand(R)"]
    channels = 3
    classes = 2

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2b",
                                  preprocessing_dict=self.preprocessing_dict,
                                  data_path=self.preprocessing_dict.get("data_path"))

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        splitted_ds = self.dataset.split("session")
        available_sessions = list(splitted_ds.keys())
        print(f"Available sessions for subject {self.subject_id}: {available_sessions}")

        # MOABB (BNCI2014004) uses keys '0train'..'2train' / '3test'/'4test'.
        # Local GDF path (if ever re-enabled) uses 'session_0'..'session_4'.
        def _pick(*candidates):
            for c in candidates:
                if c in splitted_ds:
                    return c
            raise ValueError(
                f"Subject {self.subject_id}: none of {candidates} found. "
                f"Available: {available_sessions}"
            )
        train_session_keys = [_pick("0train", "session_0"), _pick("1train", "session_1"), _pick("2train", "session_2")]
        test_session_keys = [_pick("3test", "session_3"), _pick("4test", "session_4")]

        # TCFormer official window: stop=-0.5 trims MOABB trial end (4.5s) -> 4.0s / 1000 samples.
        # Sessions 1-2 (4s MI) yield shorter segments; enforce 1000 via crop/pad for batch-stacking.
        expected_length = 1000

        def _load(keys):
            xs, ys = [], []
            for session_key in keys:
                for i, run in enumerate(splitted_ds[session_key].datasets):
                    run_X, run_y = [], []
                    for j in range(len(run)):
                        x, target, _ = run[j]
                        if x.shape[-1] > expected_length:
                            x = x[..., :expected_length]
                        elif x.shape[-1] < expected_length:
                            pad = expected_length - x.shape[-1]
                            x = np.pad(x, ((0, 0), (0, pad)), "constant")
                        run_X.append(x)
                        run_y.append(target)
                    xs.append(np.stack(run_X))
                    ys.append(np.array(run_y))
                    print(f"Loaded {session_key} run {i}: X shape {xs[-1].shape}")
            return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

        X, y = _load(train_session_keys)
        X_test, y_test = _load(test_session_keys)
        print(f"Train: X {X.shape}, y {y.shape}; Test: X {X_test.shape}, y {y_test.shape}")

        # TCFormer official protocol: val == test, last-epoch evaluation (no best-on-val selection).
        # enable_checkpointing=False in train_pipeline ensures no implicit leakage from this choice.
        X_val, y_val = X_test.copy(), y_test.copy()

        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        X, X_val, X_test = BaseDataModule._ea_align_tvt(
            X, X_val, X_test, self.preprocessing_dict
        )

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class BCICIV2bLOSO(BCICIV2b):
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super(BCICIV2bLOSO, self).__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(
            subject_ids=self.all_subject_ids, dataset="2b",
            preprocessing_dict=self.preprocessing_dict,
            data_path=self.preprocessing_dict.get("data_path"))

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
        train_datasets = [
            splitted_ds[str(subj_id)].split("session")[f"session_{session}"] for
            subj_id in train_subjects for session in [0, 1, 2]]
        val_datasets = [
            splitted_ds[str(subj_id)].split("session")[f"session_{session}"] for
            subj_id in train_subjects for session in [3, 4]]
        test_datasets = [
            splitted_ds[str(self.subject_id)].split("session")[f"session_{session}"]
            for session in [3, 4]]

        # load the data
        X = np.concatenate([run.windows.load_data()._data for train_dataset in
                            train_datasets for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for train_dataset in train_datasets for run in
                            train_dataset.datasets], axis=0)
        X_val = np.concatenate([run.windows.load_data()._data for val_dataset in
                            val_datasets for run in val_dataset.datasets], axis=0)
        y_val = np.concatenate([run.y for val_dataset in val_datasets for run in
                            val_dataset.datasets], axis=0)
        X_test = np.concatenate([run.windows.load_data()._data for test_dataset in test_datasets
                                 for run in test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for test_dataset in test_datasets for run in
                                 test_dataset.datasets], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_test = BaseDataModule._z_scale_tvt(X, X_val, X_test)

        X, X_val, X_test = BaseDataModule._ea_align_tvt(
            X, X_val, X_test, self.preprocessing_dict
        )

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])
