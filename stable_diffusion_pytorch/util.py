import torch
import numpy as np
import os
import math


def get_time_embedding(timestep, dtype, device):
    """ freqs = torch.pow(
        10000, -torch.arange(start=0, end=160, dtype=dtype, device=device) / 160
    )
    x = torch.tensor([timestep], dtype=dtype, device=device)[:, None] * freqs[None] """

    timestep = timestep.unsqueeze(0).to(device)

    embedding_dim = 320
    max_period = 10000
    downscale_freq_shift = 0
    scale = 1

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=dtype, device=device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timestep[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


def get_alphas_cumprod(beta_start=0.00085, beta_end=0.012, num_steps=1000):
    betas = (
        torch.linspace(
            beta_start**0.5, beta_end**0.5, num_steps, dtype=torch.float32
        )
        ** 2
    )  # scaled_linear
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return alphas_cumprod


def get_file_path(filename, url=None):
    module_location = os.path.dirname(os.path.abspath(__file__))
    parent_location = os.path.dirname(module_location)
    file_location = os.path.join(parent_location, "data", filename)
    return file_location


def move_channel(image, to):
    if to == "first":
        return image.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    elif to == "last":
        return image.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    else:
        raise ValueError("to must be one of the following: first, last")


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x
