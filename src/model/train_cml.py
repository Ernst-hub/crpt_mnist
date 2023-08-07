import logging
import os
import sys
from pathlib import Path

import click
import pytorch_lightning as pl
import torch
from dotenv import find_dotenv, load_dotenv

from matplotlib import pyplot as plt  # type ignore
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

sys.path.append("..")


# import wandb
from src.data.datamodule import MNISTDataModule
from src.model.helper import LitProgressBar
from src.model.model import Classifier


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """
    Runs training scripts to train the model
    args:
        input_filepath: path to the processed data
        log_path: path to the log file
        output_filepath: path to the trained model saving checkpoints
    """

    pl.seed_everything(42)

    mnist = MNISTDataModule(data_dir=input_filepath, batch_size=64)
    clf = Classifier(use_wandb=True)

    # set callbacks
    checkpoint_clb = pl.callbacks.ModelCheckpoint(
        dirpath=output_filepath,
        filename="best-checkpoint",
        save_top_k=1,
        auto_insert_metric_name=True,
        verbose=True,
        monitor="train_acc_epoch",
        mode="max",
    )

    bar_clb = LitProgressBar()

    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_clb, bar_clb],
        # logger=pl.loggers.WandbLogger(log_model=True, project="crpt_mnist"),
        max_epochs=2,
        precision=16,  # speed up training by being rough in memory
        default_root_dir=os.getcwd(),
    )
    trainer.fit(clf, mnist)

    # run predictions

    trainer.predict(clf, mnist)

    targets = torch.cat(clf.confmat_target, dim=0)
    preds = torch.cat(clf.confmat_pred, dim=0)

    print(targets.shape)
    print(preds.shape)

    targets = targets.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()

    report = classification_report(
        y_true=targets, y_pred=preds, zero_division=1
    )
    with open("classification_report.txt", "w") as f:
        f.write(report)

    display_confusion_matrix = ConfusionMatrixDisplay(
        confusion_matrix(targets, preds),
    )

    display_confusion_matrix.plot()
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automatically by walking up directories until it's found
    load_dotenv(find_dotenv())

    main()
