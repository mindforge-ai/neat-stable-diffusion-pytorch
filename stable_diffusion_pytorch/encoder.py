import torch
from torch import nn
from torch.nn import functional as F
from .decoder import EncoderAttentionBlock, ResidualBlock, AttentionBlock


class Encoder(nn.Module):
    def __init__(self, num_latent_channels=8):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.down = nn.Sequential(
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
        )
        self.mid = nn.Sequential(
            ResidualBlock(512, 512), EncoderAttentionBlock(512), ResidualBlock(512, 512)
        )
        self.norm_out = nn.GroupNorm(32, 512)
        self.silu = nn.SiLU()
        self.unknown_conv = nn.Conv2d(
            512, num_latent_channels, kernel_size=3, padding=1
        )
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1)
        """ self.conv_out = nn.Conv2d(
                num_latent_channels, num_latent_channels, kernel_size=1, padding=0
            )  # quant_conv in HF diffusers """

    def forward(self, x, noise=None, calculate_posterior=False):
        x = self.conv_in(x)
        x = self.down(x)
        x = self.mid(x)
        x = self.norm_out(x)
        x = self.silu(x)
        x = self.unknown_conv(x)
        # x = self.conv_out(x)

        x = self.quant_conv(x)  # called "moments" in HF diffusers

        if not calculate_posterior:
            return x

        # Below is the ~equivalent of DiagonalGaussianDistribution in HF

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        stdev = torch.exp(0.5 * log_variance)
        variance = torch.exp(log_variance)

        if noise is not None:
            x = mean + stdev * noise
            return x

""" class Encoder(nn.Module):
    def __init__(self, num_latent_channels=8):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.down = nn.Sequential(
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
        )
        self.mid = nn.Sequential(
            ResidualBlock(512, 512), EncoderAttentionBlock(512), ResidualBlock(512, 512)
        )
        self.norm_out = nn.GroupNorm(32, 512)
        self.silu = nn.SiLU()
        self.unknown_conv = nn.Conv2d(
            512, num_latent_channels, kernel_size=3, padding=1
        )
        self.quant_conv = nn.Conv2d(8, 8, kernel_size=1)
        self.conv_out = nn.Conv2d(
                num_latent_channels, num_latent_channels, kernel_size=1, padding=0
            )  # quant_conv in HF diffusers

    def forward(self, x, noise=None, calculate_posterior=False):
        x = self.conv_in(x)
        test = x
        torch.save(test, "../from-neat.pt")
        exit()
        x = self.down(x)
        x = self.mid(x)
        x = self.norm_out(x)
        x = self.silu(x)
        x = self.unknown_conv(x)
        # x = self.conv_out(x)

        x = self.quant_conv(x)  # called "moments" in HF diffusers

        if not calculate_posterior:
            return x

        # Below is the ~equivalent of DiagonalGaussianDistribution in HF

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        stdev = torch.exp(0.5 * log_variance)
        variance = torch.exp(log_variance)

        if noise is not None:
            x = mean + stdev * noise
            return x
"""