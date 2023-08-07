from typing import Any, Optional, Tuple, List

import torch
from pytorch_lightning import LightningModule
from torch import nn, optim


class Classifier(LightningModule):
    def __init__(self, use_wandb: Optional[bool] = False) -> None:
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

        self.use_wandb = use_wandb

        self.confmat_target: List[torch.Tensor] = []
        self.confmat_pred: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> Any:
        return self.classifier(self.backbone(x))

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        x, y = batch
        if x.ndim != 3:
            raise ValueError(
                f"Expected x to have 4 dimensions, got {x.ndim}"
            )

        x = torch.unsqueeze(x, 1)

        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError(
                f"Expected x to have shape (N, 1, 28, 28), got {x.shape}"
            )

        preds = self(x)  # expects shape (N, 10)
        assert (
            preds.shape[1] == 10
        ), f"Expected preds to have shape (N, 10), got {preds.shape}"

        y = y.long()  # expects shape (N)
        assert (
            y.shape[0] == preds.shape[0]
        ), f"Expected y to have shape {preds.shape[0]}, got {y.shape[0]}"

        loss = self.criterion(preds, y)
        acc = (y == preds.argmax(dim=-1)).float().mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # self.logger.experiment is the same as wandb.log
        if self.use_wandb:
            pass
        #     self.logger.experiment.log( # type: ignore[attr]
        #         {"logits": wandb.Histogram(preds.detach().cpu().numpy())}
        #     )

        return loss

    def on_train_epoch_end(self) -> None:
        self.log(
            "train_acc_epoch", self.trainer.callback_metrics["train_acc"]
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        prefix: str,
    ) -> Any:
        x, y = batch
        x = torch.unsqueeze(x, 1)
        y = y.long()
        preds = self(x)
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

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int | None = None
    ) -> Any:
        x, y = batch
        x = torch.unsqueeze(x, 1)
        preds = self.forward(x)
        preds = preds.argmax(dim=-1)

        # append values to confusion matrix, convert to np array
        self.confmat_target.append(y)
        self.confmat_pred.append(preds)

        return preds, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
