import torch
from torch import nn
from torch.nn import functional as F

from .attention import SelfAttention, EncoderSelfAttention


class EncoderAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.self_attention = EncoderSelfAttention(1, channels)

    def forward(self, x):

        residue = x
        x = self.groupnorm(x)
        x = self.self_attention(x)
        x += residue

        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residue = x
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        x += residue
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = x * torch.sigmoid(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = x * torch.sigmoid(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1)
        self.conv_in = nn.Conv2d(4, 512, kernel_size=3, padding=1)
        self.mid = nn.Sequential(
            ResidualBlock(512, 512), EncoderAttentionBlock(512), ResidualBlock(512, 512)
        )
        # the original repo has some weird logic where it goes through each outer block in reverse, but each inner block in forward order, so not truly reversed
        self.up = nn.ModuleList(
            [
                # block 0
                ResidualBlock(256, 128),
                ResidualBlock(128, 128),
                ResidualBlock(128, 128),
                # block 1
                ResidualBlock(512, 256),
                ResidualBlock(256, 256),
                ResidualBlock(256, 256),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                # block 2
                ResidualBlock(512, 512),
                ResidualBlock(512, 512),
                ResidualBlock(512, 512),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                # block 3
                ResidualBlock(512, 512),
                ResidualBlock(512, 512),
                ResidualBlock(512, 512),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ]
        )
        self.norm_out = nn.GroupNorm(32, 128, eps=1e-6)
        self.silu = nn.SiLU()
        self.unknown_conv = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # without scaling the latents in the following line, the image turned out faded and 'sandy-translucent' like a desert
        x /= 0.18215
        x = self.post_quant_conv(x)

        x = self.conv_in(x)

        x = self.mid(x)

        # this is a gross implemenation of CompVis/stable-diffusion's "backwards" decoder, for some reason it's implemented in reverse.
        x = self.up[-4](x)
        x = self.up[-3](x)
        x = self.up[-2](x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up[-1](x)
        x = self.up[-8](x)
        x = self.up[-7](x)
        x = self.up[-6](x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up[-5](x)
        x = self.up[-12](x)
        x = self.up[-11](x)
        x = self.up[-10](x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up[-9](x)
        x = self.up[-15](x)
        x = self.up[-14](x)
        x = self.up[-13](x)

        x = self.norm_out(x)

        x = x * torch.sigmoid(x) # This is Swish (slightly different to SILU)

        x = self.unknown_conv(x)

        return x


# Legacy


class LegacyDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x
