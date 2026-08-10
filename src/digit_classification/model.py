from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (MulticlassAccuracy,
    MulticlassConfusionMatrix, MulticlassPrecisionRecallCurve)


class DigitClassifier(pl.LightningModule):

    def __init__(
        self,
        num_classes: int = 3,
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

            # Add pooling layers if specified
            if i in pool_indexes:
                conv_layers.append(nn.MaxPool2d(2))

        # Define backbone
        self.features = nn.Sequential(*conv_layers)

        # Define classifier layer
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

        self.conf_mat = MulticlassConfusionMatrix(num_classes=num_classes)
        self.pr_curve = MulticlassPrecisionRecallCurve(num_classes=num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.train_accuracy(logits, y)

        self.log("train_loss", loss, on_step=False, 
                 on_epoch=True, prog_bar=True)
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
        self.conf_mat.update(logits, y)
        self.pr_curve.update(logits, y)

        self.log("val_loss", loss, on_step=False, 
                 on_epoch=True, prog_bar=True)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self):
        self._plot_conf_mat()
        self._plot_pr_curves()

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.test_accuracy(logits, y)
        self.conf_mat(logits, y)
        self.pr_curve(logits, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log(
            "test_accuracy",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
        )

    def on_test_epoch_end(self):
        self._plot_conf_mat()
        self._plot_pr_curves()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        class_scores = torch.softmax(logits, dim=1)
        return class_scores


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer

    def _plot_conf_mat(self):
        cm = self.conf_mat.compute().cpu().numpy()

        fig, ax = plt.subplots()

        ax.imshow(cm)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["0", "5", "8"])
        ax.set_yticklabels(["0", "5", "8"])

        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        self.logger.experiment.add_figure(
            "val/confusion_matrix",
            fig,
            self.current_epoch,
        )

        plt.close(fig)
        self.conf_mat.reset()

    def _plot_pr_curves(self):
        precision, recall, thresholds = self.pr_curve.compute()

        fig, ax = plt.subplots()

        class_names = ["0", "5", "8"]

        for i, class_name in enumerate(class_names):
            ax.plot(
                recall[i].cpu(),
                precision[i].cpu(),
                label=f"Class {class_name}",
            )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Validation Precision-Recall Curves")
        ax.legend()
        ax.grid()

        self.logger.experiment.add_figure(
            "val/precision_recall_curve",
            fig,
            self.current_epoch,
        )

        plt.close(fig)
        self.pr_curve.reset()