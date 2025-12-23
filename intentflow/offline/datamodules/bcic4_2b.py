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
        # split the data
        splitted_ds = self.dataset.split("session")
        
        # BCIC IV-2b uses session_0, session_1, session_2 for training
        # and session_3, session_4 for evaluation
        available_sessions = list(splitted_ds.keys())
        print(f"Available sessions for subject {self.subject_id}: {available_sessions}")
        
        # Get training sessions (0, 1, 2)
        train_session_keys = [k for k in available_sessions if k in ["session_0", "session_1", "session_2"]]
        
        if not train_session_keys:
            raise ValueError(f"No training sessions found. Available: {available_sessions}")
        
        # Get test sessions (3, 4) if available
        test_session_keys = [k for k in available_sessions if k in ["session_3", "session_4"]]
        
        if not test_session_keys:
            print("Warning: No evaluation sessions (3, 4) found. Will split training data.")

        # Expected sample length: 3.0s @ 250Hz = 750 samples
        expected_length = 750
        
        # Load training data
        X_list = []
        y_list = []
        for session_key in train_session_keys:
            train_dataset = splitted_ds[session_key]
            for i, run in enumerate(train_dataset.datasets):
                try:
                    run_X = []
                    run_y = []
                    for j in range(len(run)):
                        x, target, _ = run[j]
                        # Standardize length
                        if x.shape[-1] > expected_length:
                            x = x[..., :expected_length]
                        elif x.shape[-1] < expected_length:
                            padding = expected_length - x.shape[-1]
                            x = np.pad(x, ((0, 0), (0, padding)), 'constant')
                        
                        run_X.append(x)
                        run_y.append(target)
                    
                    run_X = np.stack(run_X)
                    run_y = np.array(run_y)
                    
                    X_list.append(run_X)
                    y_list.append(run_y)
                    print(f"Loaded {session_key} run {i}: X shape {run_X.shape}")
                except Exception as e:
                    print(f"Error loading {session_key} run {i}: {e}")
                    continue

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        print(f"Total training data: X shape {X.shape}, y shape {y.shape}")
        
        # Load test data if available
        if test_session_keys:
            X_test_list = []
            y_test_list = []
            for session_key in test_session_keys:
                test_dataset = splitted_ds[session_key]
                for i, run in enumerate(test_dataset.datasets):
                    try:
                        run_X = []
                        run_y = []
                        for j in range(len(run)):
                            x, target, _ = run[j]
                            # Standardize length
                            if x.shape[-1] > expected_length:
                                x = x[..., :expected_length]
                            elif x.shape[-1] < expected_length:
                                padding = expected_length - x.shape[-1]
                                x = np.pad(x, ((0, 0), (0, padding)), 'constant')
                            
                            run_X.append(x)
                            run_y.append(target)
                        
                        X_test_list.append(np.stack(run_X))
                        y_test_list.append(np.array(run_y))
                        print(f"Loaded {session_key} run {i}: X shape {run_X.shape}")
                    except Exception as e:
                        print(f"Error loading {session_key} run {i}: {e}")
                        continue
            
            if X_test_list:
                X_test = np.concatenate(X_test_list, axis=0)
                y_test = np.concatenate(y_test_list, axis=0)
                print(f"Total test data: X shape {X_test.shape}, y shape {y_test.shape}")
            else:
                # No valid test data, split training
                from sklearn.model_selection import train_test_split
                X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            # No test sessions, split training data
            from sklearn.model_selection import train_test_split
            X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
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

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])
