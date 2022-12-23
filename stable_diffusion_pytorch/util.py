import torch
import numpy as np
import os


def get_time_embedding(timestep, dtype, device):
    freqs = torch.pow(
        10000, -torch.arange(start=0, end=160, dtype=dtype, device=device) / 160
    )
    x = torch.tensor([timestep], dtype=dtype, device=device)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def get_alphas_cumprod(beta_start=0.00085, beta_end=0.012, n_training_steps=1000):
    betas = (
        torch.linspace(
            beta_start**0.5, beta_end**0.5, n_training_steps, dtype=torch.float32
        )
        ** 2
    ) # scaled_linear
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
