from typing import Optional, Tuple

import torch
import numpy as np


class DDIMSampler:
    def __init__(
        self,
        num_inference_steps: int = 50,
        num_training_steps: int = 1000,
        steps_offset: int = 1,
    ):
        self.num_inference_steps = num_inference_steps
        self.num_training_steps = num_training_steps

        step_ratio = num_training_steps // num_inference_steps
        self.timesteps = torch.flip(torch.arange(0, num_inference_steps, dtype=torch.long), dims=[0]) * step_ratio

        if steps_offset > 0:
            self.timesteps = self.timesteps + steps_offset

        # Stable Diffusion has steps_offset 1, but implementing this causes an index error atm.

        # Stable Diffusion uses a `scaled_linear` schedule with beta_start 0.00085 and beta_end 0.012.
        self.betas = (
            torch.linspace(0.00085**0.5, 0.012**0.5, num_training_steps, dtype=torch.float64) ** 2
        )

        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(
            self.alphas, dim=0,
        )  # cumprod is a trick when adding noise, allowing us to find the noise at the chosen timestep without iterating from 0 to the timestep.

    def forward_sample(
        self,
        X: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add noise to X according to the DDPM schedule.
        """

        if noise is None:
            noise = torch.randn_like(X)  # Pure noise, using a random seed.

        self.alpha_bar = self.alpha_bar.to(timesteps.device)

        # We don't blindly scale the noise by the timestep, because the schedule has some variance.

        sqrt_alpha_bar = torch.sqrt(
            self.alpha_bar[timesteps]
        )  # might be necessary to flatten / unsqueeze this and the following one

        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[timesteps])

        return sqrt_alpha_bar * X + sqrt_one_minus_alpha_bar * noise

    def reverse_sample(
        self,
        predicted_noise: torch.Tensor,
        timestep: int,
        previous_sample: torch.Tensor,
    ) -> torch.Tensor:
        previous_timestep = (
            timestep - self.num_training_steps // self.num_inference_steps
        )
        
        alpha_prod_t = self.alpha_bar[timestep]
        prev_alpha_prod_t = self.alpha_bar[previous_timestep] if previous_timestep >= 0 else self.alpha_bar[0]

        beta_prod_t = 1 - alpha_prod_t
        prev_beta_prod_t = 1 - prev_alpha_prod_t

        pred_original_sample = (previous_sample - torch.sqrt(beta_prod_t) * predicted_noise) / torch.sqrt(alpha_prod_t)

        variance = (prev_beta_prod_t / beta_prod_t) * (1 - alpha_prod_t / prev_alpha_prod_t)

        std_dev_t = 0 * torch.sqrt(variance) # 0 = eta ??

        pred_sample_direction = torch.sqrt(1 - prev_alpha_prod_t - std_dev_t ** 2) * predicted_noise

        prev_sample = torch.sqrt(prev_alpha_prod_t) * pred_original_sample + pred_sample_direction

        return prev_sample


class NumpyDDIMSampler:
    def __init__(
        self,
        num_inference_steps: int = 50,
        num_training_steps: int = 1000,
        steps_offset: int = 1,
    ):
        self.num_inference_steps = num_inference_steps
        self.num_training_steps = num_training_steps

        step_ratio = num_training_steps // num_inference_steps
        self.timesteps = torch.flip(torch.arange(0, num_inference_steps, dtype=torch.long), dims=[0]) * step_ratio

        if steps_offset > 0:
            self.timesteps = self.timesteps + steps_offset

        # Stable Diffusion has steps_offset 1, but implementing this causes an index error atm.

        # Stable Diffusion uses a `scaled_linear` schedule with beta_start 0.00085 and beta_end 0.012.
        self.betas = (
            torch.linspace(0.00085**0.5, 0.012**0.5, num_training_steps, dtype=torch.float64) ** 2
        ).numpy()

        self.alphas = 1 - self.betas
        self.alpha_bar = torch.tensor(np.cumprod(
            self.alphas, axis=0,
        ), dtype=torch.float32, device=0)

    def forward_sample(
        self,
        X: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add noise to X according to the DDPM schedule.
        """

        if noise is None:
            noise = torch.randn_like(X)  # Pure noise, using a random seed.

        self.alpha_bar = self.alpha_bar.to(timesteps.device)

        # We don't blindly scale the noise by the timestep, because the schedule has some variance.

        sqrt_alpha_bar = torch.sqrt(
            self.alpha_bar[timesteps]
        )  # might be necessary to flatten / unsqueeze this and the following one

        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[timesteps])

        return sqrt_alpha_bar * X + sqrt_one_minus_alpha_bar * noise

    def reverse_sample(
        self,
        predicted_noise: torch.Tensor,
        timestep: int,
        previous_sample: torch.Tensor,
    ) -> torch.Tensor:
        previous_timestep = (
            timestep - self.num_training_steps // self.num_inference_steps
        )
        
        alpha_prod_t = a_t = self.alpha_bar[timestep] # self.alpha_bar[timestep]
        prev_alpha_prod_t = a_prev = self.alpha_bar[previous_timestep] if previous_timestep >= 0 else self.alpha_bar[0]


        beta_prod_t = 1 - alpha_prod_t
        prev_beta_prod_t = 1 - prev_alpha_prod_t

        pred_original_sample = (previous_sample - torch.sqrt(beta_prod_t) * predicted_noise) / torch.sqrt(alpha_prod_t)

        variance = (prev_beta_prod_t / beta_prod_t) * (1 - alpha_prod_t / prev_alpha_prod_t)

        std_dev_t = 0 * torch.sqrt(variance) # 0 = eta ??

        pred_sample_direction = torch.sqrt(1 - prev_alpha_prod_t - std_dev_t ** 2) * predicted_noise

        prev_sample = torch.sqrt(prev_alpha_prod_t) * pred_original_sample + pred_sample_direction

        return prev_sample