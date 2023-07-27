import sys

import pytorch_lightning as pl


class LitProgressBar(pl.callbacks.ProgressBar):
    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )  # don't forget this :)
        percent = (batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f"{percent:.01f} percent complete \r")
