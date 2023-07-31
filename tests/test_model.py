import sys

import pytest
import pytorch_lightning as pl
import torch

from tests import _MODEL_PATH, _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT

sys.path.append(_PROJECT_ROOT)

from src.data.datamodule import MNISTDataModule
from src.model.model import Classifier


def test_model(verbose=False):
    """assert that the model is outputting correct shapes"""

    MNIST = MNISTDataModule(data_dir=_PATH_DATA, batch_size=64)
    model = Classifier.load_from_checkpoint(
        checkpoint_path=_MODEL_PATH + "/best-checkpoint.ckpt"
    )

    MNIST.prepare_data()

    trainer = pl.Trainer()
    ps = trainer.predict(model=model, datamodule=MNIST)

    if verbose:
        print(len(ps))
        print(type(ps[0]))
        print(ps[0])
        print(type(ps[0][0]))
        print(ps[0][0])

    assert (
        ps[0][0].shape == ps[0][1].shape
    ), "Output shape does not match the target shape"


test_model()
