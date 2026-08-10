from collections import Counter
import math
from pathlib import Path
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        train_fraction: float = 0.6,
        val_fraction: float = 0.2,
        test_fraction: float = 0.2
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

        assert math.isclose(self.train_fraction + self.val_fraction + self.test_fraction, 1.0), \
            "Train, validation, and test fractions must sum to one."

        # Define placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Specify counts
        self.num_samples = {
            0: 1200,
            5: 300,
            8: 3500,
        }

        # Specify label mapping
        self.label_map ={
            0: 0, 
            5: 1, 
            8: 2
        }

        # Define Transforms
        self.train_transforms = self.get_train_transforms()
        self.eval_transforms = self.get_eval_transforms()

    def prepare_data(self):
        # Only needs to be run once
        MNIST(
            self.data_dir,
            train=True,
            download=True,
        )

    def setup(self, stage):

        # Pull data from file - Two different pulls for different transforms
        train_dataset = MNIST(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.train_transforms,
        )

        eval_dataset = MNIST(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.eval_transforms,
        )

        # Generate artificial quantities of data and partition
        train_indices = []
        val_indices = []
        test_indices = []

        for label, num_samples in self.num_samples.items():
            label_indices = torch.where(train_dataset.targets == label)[0]

            # Randomly permute indices deterministically
            permutation = torch.randperm(len(label_indices))

            # Extract samples
            selected = label_indices[permutation[:num_samples]]

            # Split this class based on fractions
            n_train = int(num_samples * self.train_fraction)
            n_val = int(num_samples * self.val_fraction)

            train_indices.extend(selected[:n_train].tolist())
            val_indices.extend(
                selected[n_train:n_train + n_val].tolist()
            )
            test_indices.extend(
                selected[n_train + n_val:].tolist()
            )

        # Setup datasets as needed
        if stage == "fit":
            self.train_dataset = RemapLabels(
                Subset(train_dataset, train_indices), self.label_map)
            self.val_dataset = RemapLabels(
                Subset(eval_dataset, val_indices), self.label_map)

        elif stage == "validate":
            self.val_dataset = RemapLabels(
                Subset(eval_dataset, val_indices), self.label_map)

        elif stage == "test" or stage == "predict":
            self.test_dataset = RemapLabels(
                Subset(eval_dataset, test_indices), self.label_map)
 
        else:
            raise ValueError("Stage must be 'fit', 'validate', 'test', or 'predict'")

    def get_train_transforms(self):
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        return train_transforms

    def get_eval_transforms(self):
        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        return eval_transforms


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

class RemapLabels(torch.utils.data.Dataset):
    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, self.label_map[y]

if __name__=="__main__":
    from tqdm import tqdm
    module = MNISTDataModule("/media/mrdragan/Ubuntu/buzz-solutions/data2")
    module.prepare_data()
    module.setup('fit')
    train_dataloader = module.train_dataloader()
    val_dataloader = module.val_dataloader()

    for img, target in tqdm(train_dataloader):
        continue
    for img, target in tqdm(val_dataloader):
        continue