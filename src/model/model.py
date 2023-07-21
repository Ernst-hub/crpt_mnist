from typing import List, Tuple
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim

class Classifier(LightningModule):
    def __init__(self):
        super().__init__()
        
        # x = 28 x 28 x 1 (H, W, C)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), # x = 28 x 28 x 64
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, 1, 1), # x = 28 x 28 x 32
            nn.LeakyReLU(), 
            nn.Conv2d(32, 16, 3, 1, 1), # x = 28 x 28 x 16
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3, 1, 1), # x = 28 x 28 x 8
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 28 * 28, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)