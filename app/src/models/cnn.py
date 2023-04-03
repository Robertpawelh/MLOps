from typing import Tuple
from torch import nn
from src.model_utils.binary_metrics import BinaryMetrics
from typing import Dict
import torch
import pytorch_lightning as pl
import time
import torch

# TODO: early stopping
class CNN(pl.LightningModule):
    def __init__(
        self,
        model_parameters: Dict,
    ):
        super(CNN, self).__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self._loss_fn = nn.BCELoss()
        self._learning_rate = model_parameters['learning_rate']
        self.metrics = BinaryMetrics(1, self.device)

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx
    ) -> torch.Tensor:
        logits, y_true, metrics = self._common_step(batch)
        loss = self._loss_fn(logits, y_true)
        self.metrics.log_metrics(self, loss, metrics, self.trainer.current_epoch, 'train')

        return loss

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx
    ) -> torch.Tensor:
        probs, y_true, metrics = self._common_step(batch)
        loss = self._loss_fn(probs, y_true)
        self.metrics.log_metrics(self, loss, metrics, self.trainer.current_epoch, 'val')

        return loss

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx
    ) -> torch.Tensor:
        logits, y_true, metrics = self._common_step(batch)
        loss = self._loss_fn(logits, y_true)
        self.metrics.log_metrics(self, loss, metrics, self.trainer.current_epoch, 'test')

        return loss

    def _common_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        x, y_true = batch['image'], batch['labels'].reshape(-1, 1).float()

        start = time.time()
        probs = self(x)
        end = time.time()

        metrics = self.metrics.calculate(probs, y_true, end - start)
        return probs, y_true, metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self._learning_rate,
            weight_decay=5e-4,
        )
