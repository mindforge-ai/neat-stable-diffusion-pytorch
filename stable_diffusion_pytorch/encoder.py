import torch
from torch import nn
from torch.nn import functional as F
from .decoder import AttentionBlock, ResidualBlock


class Encoder(nn.Sequential):
    def __init__(self, num_latent_channels=4):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, num_latent_channels, kernel_size=3, padding=1),
            nn.Conv2d(
                num_latent_channels, num_latent_channels, kernel_size=1, padding=0
            ),  # quant_conv in HF diffusers
        )

    def forward(self, x):
        print(x.size())
        for module in self:
            x = module(x)
            print(x.size())

        # Below is the ~equivalent of DiagonalGaussianDistribution in HF

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * x
        x *= 0.18215
        return x
