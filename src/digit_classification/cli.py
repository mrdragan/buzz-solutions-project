import os
import typer 
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .model import DigitClassifier
from .data import MNISTDataModule


os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Seed everything to make it reproducible
pl.seed_everything(42)

app = typer.Typer()

# Download data
@app.command()
def download_data(data_dir: str = typer.Option(..., "--data-dir")):
    """
    Downloads the mnist data
    """
    data_module = _setup_data(data_dir)
    data_module.prepare_data()

# Train Model
@app.command()
def train(
    data_dir: str = typer.Option(..., "--data-dir"),
    output_dir: str = typer.Option(..., "--output-dir"),

    # Model params
    learning_rate: float = 0.001,
    hidden_dims: List[int] = typer.Option([32, 64]),
    no_hidden: bool = typer.Option(False, "--no-hidden"),
    pool_indexes: List[int] = typer.Option([0, 1]),

    # Data params
    batch_size: int = 64,
    num_workers: int = 4,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2
):
    """
    Trains the classification model
    """
    if no_hidden:
        hidden_dims = []

    trainer = _setup_trainer(output_dir=output_dir, inference_mode=False)

    model = _setup_model(learning_rate=learning_rate,
                         hidden_dims=hidden_dims,
                         pool_indexes=pool_indexes)

    data_module = _setup_data(data_dir=data_dir, 
                              batch_size=batch_size,
                              num_workers=num_workers,
                              train_fraction=train_fraction,
                              val_fraction=val_fraction,
                              test_fraction=test_fraction)

    trainer.fit(model, datamodule=data_module)


@app.command()
def evaluate(
    data_dir: str = typer.Option(..., "--data-dir"),
    checkpoint_path: str = typer.Option(..., "--checkpoint-path"),

    # Data params
    batch_size: int = 64,
    num_workers: int = 4,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2
):
    """
    Evaluates the classification model on the test set
    """

    trainer = _setup_trainer(inference_mode=True)
    model = DigitClassifier.load_from_checkpoint(checkpoint_path=checkpoint_path)
    data_module = _setup_data(data_dir=data_dir, 
                              batch_size=batch_size,
                              num_workers=num_workers,
                              train_fraction=train_fraction,
                              val_fraction=val_fraction,
                              test_fraction=test_fraction)

    trainer.test(model, datamodule=data_module)

def _setup_trainer(output_dir, inference_mode=False):

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=20,
        callbacks=ModelCheckpoint(monitor="val_loss",
                                  mode="min",
                                  save_top_k=1),
        logger=TensorBoardLogger(save_dir=output_dir),
        inference_mode=inference_mode
    )

    return trainer

def _setup_model(learning_rate, hidden_dims, pool_indexes):

    model = DigitClassifier(learning_rate=learning_rate,
                            hidden_dims=hidden_dims,
                            pool_indexes=pool_indexes)

    return model

def _setup_data(data_dir, batch_size=32, num_workers=4,
                train_fraction=0.6, val_fraction=0.2, test_fraction=0.2):

    data_module = MNISTDataModule(data_dir=data_dir, 
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 train_fraction=train_fraction,
                                 val_fraction=val_fraction,
                                 test_fraction=test_fraction)

    return data_module

if __name__ == "__main__":
    app()