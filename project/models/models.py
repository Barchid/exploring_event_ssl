import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from project.models import sew_resnet
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from einops import rearrange


def get_encoder(in_channels: int) -> nn.Module:
    resnet18 = models.resnet18(progress=True)

    resnet18.fc = nn.Identity()

    if in_channels != 3:
        resnet18.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    return resnet18


def get_encoder_3d(in_channels: int) -> nn.Module:
    # resnet18 = models.video.r3d_18()
    resnet18 = models.video.mc3_18()
    resnet18.fc = nn.Identity()
    resnet18.stem[0] = nn.Conv3d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=(3, 7, 7),
        stride=(1, 2, 2),
        padding=(1, 3, 3),
        bias=False,
    )
    # resnet18 = models.video.r2plus1d_18()
    # resnet18.fc = nn.Identity()
    # resnet18.stem[0] = nn.Conv3d(in_channels=in_channels, out_channels=45, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
    return resnet18


def get_projector(in_channels: int = 512):
    projector = nn.Sequential(
        nn.Linear(in_channels, 3 * in_channels),
        nn.BatchNorm1d(3 * in_channels),
        nn.ReLU(),
        nn.Linear(3 * in_channels, 3 * in_channels),
        nn.BatchNorm1d(3 * in_channels),
        nn.ReLU(),
        nn.Linear(3 * in_channels, 3 * in_channels),
    )

    return projector


def get_encoder_snn(in_channels: int, T: int):
    resnet18 = sew_resnet.MultiStepSEWResNet(
        block=sew_resnet.MultiStepBasicBlock,
        layers=[2, 2, 2, 2],
        zero_init_residual=True,
        T=T,
        cnf="ADD",
        multi_step_neuron=neuron.MultiStepIFNode,
        detach_reset=True,
        surrogate_function=surrogate.ATan(),
    )

    if in_channels != 3:
        resnet18.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    return resnet18


class ConvEnc(nn.Module):
    def __init__(self, in_channels: int, timesteps: int, mode="snn"):
        super(ConvEnc, self).__init__()
        if mode not in ("3dcnn", "snn", "cnn"):
            raise ValueError(
                f"Mode must be one of these values: ('3dcnn', 'snn', 'cnn'). Got:{mode}"
            )

        if mode == "3dcnn":
            self.encoder = get_encoder_3d(in_channels=in_channels)
        elif mode == "snn":
            self.encoder = get_encoder_snn(in_channels=in_channels, T=timesteps)
        else:  # cnn
            self.encoder = get_encoder(in_channels=in_channels * timesteps)

        self.mode = mode
        self.timesteps = timesteps

    def forward(self, x):
        # x = B,T,C,H,W

        if self.mode == "snn":
            functional.reset_net(self.encoder)
            x = x.permute(1, 0, 2, 3, 4)  # from (B,T,C,H,W) to (T, B, C, H, W)
        elif self.mode == "3dcnn":
            x = x.permute(1, 2, 0, 3, 4)  # from (T,B,C,H,W) to (B,C,T,H,W)
        elif self.mode == "cnn" and len(x.shape) == 5:
            x = rearrange(
                x,
                "batch time channel height width -> batch (time channel) height width",
            )

        x = self.encoder(x)
        return x
