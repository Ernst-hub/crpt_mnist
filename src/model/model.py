from typing import Any, List, Optional, Tuple

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, 1)
        preds = self(x)
        loss = self.criterion(preds, y)
        acc = (y == preds.argmax(dim=-1)).float().mean()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # self.logger.experiment is the same as wandb.log
        self.logger.experiment.log(
            {"logits": wandb.Histogram(preds.detach().cpu().numpy())}
        )

        return loss

    # def validation_step(self, batch, batch_idx):
    #     self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        x = torch.unsqueeze(x, 1)
        preds = self.forward(x)
        acc = (y == preds.argmax(dim=-1)).float().mean()
        self.log(
            f"{prefix}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return acc

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = torch.unsqueeze(x, 1)
        preds = self.forward(x)
        return preds, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
