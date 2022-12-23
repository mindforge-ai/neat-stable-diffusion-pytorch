import numpy as np
from .. import util
import torch
from scipy import integrate

np.set_printoptions(precision=20)

class KLMSSampler():
    def __init__(self, n_inference_steps=50, n_training_steps=1000, lms_order=4):
        alphas_cumprod = util.get_alphas_cumprod(n_training_steps=n_training_steps)
        timesteps = np.linspace(0, n_training_steps - 1, n_inference_steps, dtype=float)[::-1].copy()
        sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas).to(device=0)
        self.timesteps = torch.from_numpy(timesteps).to(device=0)
        self.derivatives = []


        self.initial_scale = sigmas.max()
        self.n_inference_steps = n_inference_steps
        self.n_training_steps = n_training_steps
        self.lms_order = lms_order
        self.step_count = 0
        self.outputs = []

    def scale_input_latent(self, input_latent: torch.Tensor, step_count=None):
        if step_count is None:
            step_count = self.step_count
        sigma = self.sigmas[step_count]
        scaled_input_latent = input_latent / ((sigma**2 + 1) ** 0.5)
        return scaled_input_latent

    def set_strength(self, strength=1):
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = torch.linspace(self.n_training_steps - 1, 0, self.n_inference_steps)
        self.timesteps = self.timesteps[start_step:]
        self.initial_scale = self.sigmas[start_step]
        self.step_count = start_step

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
                    prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
                return prod

            integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]

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
        lms_coeffs = [self.get_lms_coefficient(order, t, curr_order) for curr_order in range(order)]

        # 4. Compute previous sample based on the derivatives path
        prev_sample = latents + sum(
            coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives))
        )

        return prev_sample