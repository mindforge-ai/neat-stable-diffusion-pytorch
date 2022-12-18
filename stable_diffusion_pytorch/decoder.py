import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention, CLIPSelfAttention

class EncoderAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.self_attention = CLIPSelfAttention(1, channels)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.self_attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

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
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 512, kernel_size=3, padding=1)
        self.mid = nn.Sequential(
            ResidualBlock(512, 512),
            EncoderAttentionBlock(512),
            ResidualBlock(512, 512)
        )
        self.up = nn.ModuleList([
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            # nn.Upsample(scale_factor=2),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
        ])
        self.norm_out = nn.GroupNorm(32, 128)
        self.silu = nn.SiLU()
        self.unknown_conv = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x /= 0.18215
        x = self.conv_in(x)
        print("a", x.size())
        x = self.mid(x)
        print("b", x.size())
        for layer in reversed(self.up):
            x = layer(x)
            print(x.size())
        print("c", x.size())
        x = self.norm_out(x)
        x = self.silu(x)
        x = self.unknown_conv(x)
        return x