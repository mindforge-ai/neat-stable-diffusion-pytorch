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

    def forward(self, x, noise):
        x = self.conv_in(x)
        x = self.down(x)
        x = self.mid(x)
        x = self.norm_out(x)
        x = self.silu(x)
        x = self.unknown_conv(x)
        # x = self.conv_out(x)

        x = self.quant_conv(x)

        # Below is the ~equivalent of DiagonalGaussianDistribution in HF

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise

        # anecdotally, the below scaling seems ~insignificant, perhaps without it the image is a bit less smooth
        x *= 0.18215
        return x


# Legacy


class LegacyEncoder(nn.Sequential):
    def __init__(self):
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
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x, noise):
        for module in self:
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise

        x *= 0.18215
        return x
