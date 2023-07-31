import logging
import os
import sys

from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT

sys.path.append(_PROJECT_ROOT)


import pytorch_lightning as pl
import torch

from src.data.datamodule import MNISTDataModule
from src.model.model import Classifier


def test_training(verbose: bool = False, num_epochs: int = 1) -> None:
    """assert that the model is correctly trained"""

    pl.seed_everything(42)

    if verbose:
        logging.info("Running test_training")

    MNIST = MNISTDataModule(data_dir=_PATH_DATA, batch_size=64)
    model = Classifier()

    # set callbacks

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        precision=16,  # speed up training by beign rough in memory
        logger=pl.loggers.WandbLogger(log_model=True, project="crpt_mnist"),
        default_root_dir=os.getcwd(),
    )

    # trainer.loggers[0].watch(model)

    # Log initial loss with random weights
    random_loss = trainer.test(model, MNIST)
    random_loss = random_loss[0]["test_loss_epoch"]

    trainer.fit(model, MNIST)
    assert (
        trainer.logged_metrics.get("train_loss_epoch") is not None
    ), "No training loss logged"

    qualified_loss = trainer.test(model, MNIST)
    qualified_loss = qualified_loss[0]["test_loss_epoch"]
    assert trainer.current_epoch == num_epochs, "Training did not finish"

    assert (
        trainer.logged_metrics.get("test_loss_epoch") is not None
    ), "No test loss logged"
    assert (
        trainer.logged_metrics.get("test_acc_epoch") is not None
    ), "No test accuracy logged"
    assert qualified_loss < random_loss, "Model did not improve during training."

    if (random_loss - qualified_loss) / qualified_loss < 0.2:
        logging.warning("model loss improved little after training.")


test_training(num_epochs=1)
