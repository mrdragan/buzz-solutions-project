import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy


class DigitClassifier(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Model architecture
        self.model = nn.Sequential(
            # ...
        )

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

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