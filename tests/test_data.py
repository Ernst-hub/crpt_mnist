import logging
import sys

from tests import _PATH_DATA, _PROJECT_ROOT

sys.path.append(_PROJECT_ROOT)

from src.data.datamodule import MNISTDataModule


def test_data():
    logger = logging.getLogger(__name__)
    logger.info("Running test_data")
    small_data = True
    MNIST = MNISTDataModule(
        data_dir=_PATH_DATA, batch_size=64, small=small_data
    )
    MNIST.prepare_data()

    N_train = 25000
    N_test = 5000
    print(len(MNIST.train_dataset))
    print(len(MNIST.test_dataset))

    if not small_data:
        assert (
            len(MNIST.train_dataset) == N_train
        ), "Wrong number of training samples"
        assert (
            len(MNIST.test_dataset) == N_test
        ), "Wrong number of test samples"

    logging.info("correct number of examples")

    x, y = MNIST.train_dataset[0]

    assert x.shape == (28, 28), "Wrong shape of X"
    assert y.shape == (), "Wrong shape of y"

    logging.info("correct shape of data")


test_data()
