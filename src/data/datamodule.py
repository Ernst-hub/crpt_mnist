from typing import (
    Any,
)

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        small: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = 0
        self.prepare_data_per_node = True
        self.small = small

    def prepare_data(
        self,
    ):
        # Load the data
        tr0 = np.load(self.data_dir + "/train_0.npz")
        tr1 = np.load(self.data_dir + "/train_1.npz")
        tr2 = np.load(self.data_dir + "/train_2.npz")
        tr3 = np.load(self.data_dir + "/train_3.npz")
        tr4 = np.load(self.data_dir + "/train_4.npz")
        test = np.load(self.data_dir + "/test.npz")

        # select and concatenate data
        if self.small:
            x_train = tr0["images"]
            y_train = tr0["labels"]
        else:
            x_train = np.concatenate(
                (
                    tr0["images"],
                    tr1["images"],
                    tr2["images"],
                    tr3["images"],
                    tr4["images"],
                )
            )
            y_train = np.concatenate(
                (
                    tr0["labels"],
                    tr1["labels"],
                    tr2["labels"],
                    tr3["labels"],
                    tr4["labels"],
                )
            )

        x_test = test["images"]
        y_test = test["labels"]

        # convert to tensor from numpy
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).int()
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).int()

        # transform dataset, 0 mean and 1 std:
        train_mean = torch.mean(x_train)
        train_std = torch.std(x_train)
        x_train = (x_train - train_mean) / train_std
        x_test = (x_test - train_mean) / train_std
        assert (
            torch.mean(x_train).item() < 1e-3
        ), "Failed to normalize the data, mean is not apprx. 0"
        assert (
            torch.std(x_train).item() - 1 < 1e-3
        ), "Failed to normalize the data, std is not apprx. 1"

        # create dataset
        self.train_dataset = TensorDataset(
            x_train,
            y_train,
        )
        self.test_dataset = TensorDataset(
            x_test,
            y_test,
        )

    def train_dataloader(
        self,
    ):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(
        self,
    ):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(
        self,
    ) -> Any:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
