import sys

import pytorch_lightning as pl
import torch

from tests import _MODEL_PATH, _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT

sys.path.append(_PROJECT_ROOT)

from src.data.datamodule import MNISTDataModule
from src.model.model import Classifier


def test_model():
    """assert that the model is outputting correct shapes"""

    MNIST = MNISTDataModule(data_dir=_PATH_DATA, batch_size=64)
    model = Classifier.load_from_checkpoint(
        checkpoint_path=_MODEL_PATH + "/test-checkpoint.ckpt"
    )

    MNIST.prepare_data()
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    trainer = pl.Trainer(
        accelerator= "cpu",
    )
    
    ps = trainer.predict(model=model, datamodule=MNIST)

    assert (
        ps[0][0].shape == ps[0][1].shape
    ), "Output shape does not match the target shape"


test_model()
