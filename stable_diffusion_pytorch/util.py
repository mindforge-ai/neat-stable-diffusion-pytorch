import torch
import numpy as np
import os
import math


def get_time_embedding(timestep, dtype, device):
    half = 320 // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(
            start=0, end=half, dtype=torch.float32
        )  # for some reason, fp32 is enforced here
        / half
    ).to(
        device=device
    )  # results diverge to CompVis if the above calculation is done on GPU
    args = (
        torch.tensor([timestep], dtype=dtype, device=device)[:, None].float()
        * freqs[None]
    )
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if 320 % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


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
