import torch
from typing import Optional, Tuple


class DDPMSampler:
    def __init__(self, num_inference_steps: int = 50, num_training_steps: int = 1000):
        self.num_inference_steps = num_inference_steps
        self.num_training_steps = num_training_steps

        self.timesteps = torch.linspace(
            num_training_steps - 1, 0, num_inference_steps, device=0, dtype=torch.long
        )

        # Stable Diffusion uses a `scaled_linear` schedule with beta_start 0.00085 and beta_end 0.012.
        self.betas = (
            torch.linspace(0.00085**0.5, 0.012**0.5, num_training_steps, device=0)
            ** 2
        )

        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(
            self.alphas, dim=0
        )  # cumprod is a trick  when adding noise, allowing us to find the noise at the chosen timestep without iterating from 0 to the timestep.

        # Question: is beta the variance?

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

        # We don't blindly scale the noise by the timestep, because the schedule has some variance.

        # do stuff here

        return True
