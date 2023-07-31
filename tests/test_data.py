import logging
import sys

from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT

sys.path.append(_PROJECT_ROOT)

from src.data.datamodule import MNISTDataModule


def test_data():
    logger = logging.getLogger(__name__)
    logger.info("Running test_data")
    MNIST = MNISTDataModule(data_dir=_PATH_DATA, batch_size=64)
    MNIST.prepare_data()

    N_train = 25000
    N_test = 5000
    print(len(MNIST.train_dataset))
    print(len(MNIST.test_dataset))

    assert len(MNIST.train_dataset) == N_train, "Wrong number of training samples"
    assert len(MNIST.test_dataset) == N_test, "Wrong number of test samples"

    logging.info("correct number of examples")

    # print a data point:
    x, y = MNIST.train_dataset[0]

    assert x.shape == (28, 28), "Wrong shape of X"
    assert y.shape == (), "Wrong shape of y"

    logging.info("correct shape of data")


test_data()
