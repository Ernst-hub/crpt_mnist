import os
import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv

import torch
from model import Classifier
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/Users/kristianernst/Work/Learning/MLOps/DTU/S4/exercise/crpt_mnist/data/raw", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def prepare_data(self):
        # Load the data
        tr0 = np.load(self.data_dir + '/train_0.npz')
        tr1 = np.load(self.data_dir + '/train_1.npz')
        tr2 = np.load(self.data_dir + '/train_2.npz')
        tr3 = np.load(self.data_dir + '/train_3.npz')
        tr4 = np.load(self.data_dir + '/train_4.npz')
        test = np.load(self.data_dir + '/test.npz')
        
        # select and concatenate data
        x_train = np.concatenate((tr0['images'], tr1['images'], tr2['images'], tr3['images'], tr4['images']))
        y_train = np.concatenate((tr0['labels'], tr1['labels'], tr2['labels'], tr3['labels'], tr4['labels']))
        x_test = test['images']
        y_test = test['labels']
        
        # convert to tensor from numpy
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).int()
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).int()
        
        # transform dataset, 0 mean and 1 std:
        train_mean = torch.mean(x_train)
        train_std = torch.std(x_train)
        x_train = (x_train - train_mean) / train_std
        x_test = (x_test - train_mean) / train_std
        assert torch.mean(x_train).item() < 1e-3, "Failed to normalize the data, mean is not apprx. 0"
        assert torch.std(x_train).item() - 1 < 1e-3, "Failed to normalize the data, std is not apprx. 1"
        
        # create dataset
        self.train_dataset = TensorDataset(x_train, y_train)
        self.test_dataset = TensorDataset(x_test, y_test)
     
    def train_dataloader(self):
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

class LitProgressBar(pl.callbacks.ProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
        percent = (batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()
        sys.stdout.write(f'{percent:.01f} percent complete \r')
        


@click.command()

@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath: str, output_filepath: str) -> None:
    """
    Runs training scripts to train the model
    args:
        input_filepath: path to the processed data
        log_path: path to the log file
        output_filepath: path to the trained model saving checkpoints
    """    
    pl.seed_everything(42)
    
    mnist = MNISTDataModule(data_dir=input_filepath)
    
    clf = Classifier()
    
    # set callbacks
    checkpoint_clb = pl.callbacks.ModelCheckpoint(
        dirpath=output_filepath,
        filename='best-checkpoint',
        save_top_k=1,
        auto_insert_metric_name=True,
        verbose=True,
        monitor = 'loss'
    )
    bar_clb = LitProgressBar()
    
    
    # train
    trainer = pl.Trainer(callbacks=[checkpoint_clb, bar_clb], logger=pl.loggers.WandbLogger(log_model=True, project='crpt_mnist'), max_epochs = 100)
    trainer.fit(clf, mnist)
    
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    project_dir = Path(__file__).resolve().parents[2]
    
    # find .env automatically by walking up directories until it's found
    load_dotenv(find_dotenv())
    
    main()