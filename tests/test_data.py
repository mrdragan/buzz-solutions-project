
from pathlib import Path
import shutil
from torchvision.datasets import MNIST
import pytorch_lightning as pl

from digit_classification.data import MNISTDataModule

def test_data_downloadable():
    path = "../test_location"
    print(f"Downloading data to {path}")
    dm1 = MNISTDataModule(path)
    dm1.prepare_data()  

    path = Path(path)

    print("Checking if path exists and is not empty")
    assert path.is_dir()
    assert any(path.iterdir())
    print("Succeeded")

    shutil.rmtree(path)


def test_data_split_is_reproducible():
    """
    This tests if the data is loaded twice if it is reproducible
    """
    pl.seed_everything(42)
    dm1 = MNISTDataModule("../test_location1")
    dm1.prepare_data()
    dm1.setup("fit")
    td1 = dm1.train_dataset
    vd1 = dm1.val_dataset
    dm1.setup("test")
    ted1 = dm1.test_dataset

    pl.seed_everything(42)
    dm2 = MNISTDataModule("../test_location2")
    dm2.prepare_data()
    dm2.setup("fit")
    td2 = dm2.train_dataset
    vd2 = dm2.val_dataset
    dm2.setup("test")
    ted2 = dm2.test_dataset

    print("Checking if data is repeatable")
    for one, two in zip(td1, td2):
        one = one[0]
        two = two[0]
        assert (one == two).all()
    print("Training data is reproducible")

    for one, two in zip(vd1, vd2):
        one = one[0]
        two = two[0]
        assert (one == two).all()
    print("Validation data is reproducible")

    for one, two in zip(ted1, ted2):
        one = one[0]
        two = two[0]
        assert (one == two).all()
    print("Test data is reproducable")

    # Now check if the double loads are okay
    train_dataset = MNIST(
        root="../test_location1",
        train=True,
        download=False,
    )

    eval_dataset = MNIST(
        root="../test_location2",
        train=True,
        download=False,
    )

    for one, two in zip(train_dataset, eval_dataset):
        assert one==two

    print("Loading data twice with MNIST Object is consistent")

    shutil.rmtree("../test_location1")
    shutil.rmtree("../test_location2")