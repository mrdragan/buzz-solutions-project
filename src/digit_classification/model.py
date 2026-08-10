from typing import List
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy


class DigitClassifier(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        hidden_dims: List[int] = [32, 64],
        pool_indexes: List[int] = [0, 1]
    ):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Check for improper pool count 
        if len(pool_indexes) > 2:
            raise ValueError("Please specify a maximum of two pooling layers")

        conv_layers = []
        in_channels = 1
        # Specify hidden layers 
        for i, out_channels in enumerate(hidden_dims):
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            ])
            in_channels = out_channels

            # Add pooling layers 
            if i in pool_indexes:
                conv_layers.append(nn.MaxPool2d(2))

        # Define backbone
        self.features = nn.Sequential(*conv_layers)

        # Define classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes), # Inferred input size
        )

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.train_accuracy(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.val_accuracy(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.test_accuracy(logits, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "test_accuracy",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer