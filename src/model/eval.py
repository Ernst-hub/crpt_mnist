import logging
import sys
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

sys.path.append("..")
import pytorch_lightning as pl

from data.make_dataset import MNISTDataModule
from model import Classifier


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path())
def main(input_filepath: str, model_path: str) -> None:
    """
    Runs training scripts to train the model
    args:
        input_filepath: path to the processed data
        log_path: path to the log file
        output_filepath: path to the trained model saving checkpoints
    """

    logger = logging.getLogger(__name__)
    logger.info("Running main")

    # mnist = MNISTDataModule(data_dir=input_filepath)
    mnist = MNISTDataModule(data_dir=input_filepath, batch_size=64)

    logger.info("DataModule loaded")

    clf = Classifier.load_from_checkpoint(
        checkpoint_path=model_path + "/best-checkpoint.ckpt"
    )

    trainer = pl.Trainer()
    trainer.test(model=clf, datamodule=mnist)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automatically by walking up directories until it's found
    load_dotenv(find_dotenv())

    main()
