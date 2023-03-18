from project.losses.vicreg_loss import VICRegLoss
from project.models.models import get_projector, ConvEnc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from einops import rearrange
from torchmetrics.functional import accuracy
from spikingjelly.clock_driven import functional


class SSLModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        epochs: int,
        timesteps: int,
        enc1: str = "cnn",
        enc2: str = "cnn",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.enc1 = enc1
        self.enc2 = enc2

        self.encoder = None
        self.projector = None
        self.encoder1 = None
        self.encoder2 = None

        self.projector = get_projector()

        if enc1 == enc2:
            self.encoder = ConvEnc(in_channels=2, timesteps=timesteps, mode=enc1)
        else:
            self.encoder1 = ConvEnc(in_channels=2, timesteps=timesteps, mode=enc1)
            self.encoder2 = ConvEnc(in_channels=2, timesteps=timesteps, mode=enc2)

        self.criterion = VICRegLoss()

    def forward(self, Y, enc=None):
        if enc is None:
            representation = self.encoder(Y)

        elif enc == 1:
            representation = self.encoder1(Y)
        else:
            representation = self.encoder2(Y)

        return representation

    def shared_step(self, batch):
        (X, Y_a, Y_b), label = batch

        if self.encoder is None:
            representation = self(Y_a, enc=1)
            Z_a = self.projector(representation)
            representation = self(Y_b, enc=2)
            Z_b = self.projector(representation)
        else:
            representation = self(Y_a)
            Z_a = self.projector(representation)
            representation = self(Y_b)
            Z_b = self.projector(representation)

        loss = self.criterion(Z_a, Z_b)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)  # better perf
