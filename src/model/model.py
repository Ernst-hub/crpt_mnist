from typing import Any, Tuple
import pytest

import torch
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim


class Classifier(LightningModule):
    def __init__(self):
        super().__init__()

        # x = 28 x 28 x 1 (H, W, C)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),  # x = 28 x 28 x 64
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),  # x = 28 x 28 x 32
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),  # x = 28 x 28 x 16
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3, 1, 1),  # x = 28 x 28 x 8
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 28 * 28, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        if x.ndim != 3:
            raise ValueError(f"Expected x to have 4 dimensions, got {x.ndim}")
        
        x = torch.unsqueeze(x, 1)
        
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError(f"Expected x to have shape (N, 1, 28, 28), got {x.shape}")
        
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (y == preds.argmax(dim=-1)).float().mean()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # self.logger.experiment is the same as wandb.log
        self.logger.experiment.log(
            {"logits": wandb.Histogram(preds.detach().cpu().numpy())}
        )

        return loss
    
    def on_train_epoch_end(self) -> Any:
        self.log("train_acc_epoch", self.trainer.callback_metrics["train_acc"])

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, prefix: str
    ) -> torch.Tensor:
        x, y = batch
        x = torch.unsqueeze(x, 1)
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        acc = (y == preds.argmax(dim=-1)).float().mean()
        
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        self.log(
            f"{prefix}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss, acc

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        x = torch.unsqueeze(x, 1)
        preds = self.forward(x)
        preds = preds.argmax(dim=-1)
        return preds, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)