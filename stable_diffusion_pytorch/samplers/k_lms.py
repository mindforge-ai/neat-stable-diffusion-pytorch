import numpy as np
from .. import util
import torch
from scipy import integrate

np.set_printoptions(precision=20)


class KLMSSampler:
    def __init__(self, num_inference_steps=50, num_training_steps=1000, lms_order=4, inference_mode=False):
        self.betas = torch.linspace(0.00085**0.5, 0.012**0.5, num_training_steps) ** 2 # beta_start and beta_end of scaled_linear
        self.alphas = 1 - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        if inference_mode:
            timesteps = np.linspace(0, num_training_steps - 1, num_inference_steps, dtype=float)[::-1].copy()
        else:
            timesteps = np.linspace(
                0, num_training_steps - 1, num_training_steps, dtype=float
            )[::-1].copy()

        self.timesteps = torch.from_numpy(timesteps).to(device=0)

        sigmas = og_sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
        if inference_mode:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
            sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
            og_sigmas = np.concatenate([og_sigmas[::-1], [0.0]]).astype(np.float32)
        else:
            sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=0)

        self.derivatives = []

        self.initial_scale = og_sigmas.max() # initial_scale = init_noise_sigma, comes from the max of the full `training_sigmas`
        self.num_inference_steps = num_inference_steps
        self.num_training_steps = num_training_steps
        self.lms_order = lms_order
        self.step_count = 0
        self.outputs = []

    def scale_input_latent(self, input_latent: torch.Tensor, step_count=None):
        if step_count is None:
            step_count = self.step_count
        sigma = self.sigmas[step_count]
        scaled_input_latent = input_latent / ((sigma**2 + 1) ** 0.5)
        return scaled_input_latent

    def get_lms_coefficient(self, order, t, current_order):
        """
        Compute a linear multistep coefficient.

        Args:
            order (TODO):
            t (TODO):
            current_order (TODO):
        """

        def lms_derivative(tau):
            prod = 1.0
            for k in range(order):
                if current_order == k:
                    continue
                prod *= (tau - self.sigmas[t - k]) / (
                    self.sigmas[t - current_order] - self.sigmas[t - k]
                )
            return prod

        integrated_coeff = integrate.quad(
            lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4
        )[0]

        return integrated_coeff

    def step(self, latents, output):
        t = self.step_count
        self.step_count += 1
        sigma = self.sigmas[t]
        order = self.lms_order

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = latents - sigma * output

        # 2. Convert to an ODE derivative
        derivative = (latents - pred_original_sample) / sigma
        self.derivatives.append(derivative)
        if len(self.derivatives) > order:
            self.derivatives.pop(0)

        # 3. Compute linear multistep coefficients
        order = min(t + 1, order)
        lms_coeffs = [
            self.get_lms_coefficient(order, t, curr_order)
            for curr_order in range(order)
        ]

        # 4. Compute previous sample based on the derivatives path
        prev_sample = latents + sum(
            coeff * derivative
            for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives))
        )

        return prev_sample
