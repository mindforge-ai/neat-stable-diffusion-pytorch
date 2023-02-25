import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].
    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.
    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class PNDMSampler:
    """
    Pseudo numerical methods for diffusion models (PNDM) proposes using more advanced ODE integration techniques,
    namely Runge-Kutta method and a linear multi-step method.
    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.
    For more details, see the original paper: https://arxiv.org/abs/2202.09778
    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        skip_prk_steps (`bool`):
            allows the scheduler to skip the Runge-Kutta steps that are defined in the original paper as being required
            before plms steps; defaults to `False`.
        set_alpha_to_one (`bool`, default `False`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion process)
            or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)
        steps_offset (`int`, default `0`):
            an offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_timesteps: int = 50,
        steps_offset: int = 0,  # 1
    ):
        # this schedule is very specific to the latent diffusion model.
        self.betas = (
            torch.linspace(
                0.00085**0.5, 0.012**0.5, num_train_timesteps, dtype=torch.float32
            )
            ** 2  # beta_start and beta_end
        )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.final_alpha_cumprod = self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.cur_model_output = 0
        self.counter = 0
        self.cur_sample = None
        self.ets = []

        # setable values
        self.num_inference_timesteps = num_inference_timesteps
        self.num_train_timesteps = num_train_timesteps
        self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.prk_timesteps = None
        self.plms_timesteps = None
        self.timesteps = None

        self.steps_offset = steps_offset

    def set_timesteps(
        self, num_inference_timesteps: int, device: Union[str, torch.device] = None
    ):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """

        self.num_inference_timesteps = num_inference_timesteps
        step_ratio = self.num_train_timesteps // self.num_inference_timesteps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        self._timesteps = (np.arange(0, num_inference_timesteps) * step_ratio).round()
        self._timesteps += self.steps_offset

        # for some models like stable diffusion the prk steps can/should be skipped to
        # produce better results. When using PNDM with `self.config.skip_prk_steps` the implementation
        # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
        self.prk_timesteps = np.array([])
        self.plms_timesteps = np.concatenate(
            [self._timesteps[:-1], self._timesteps[-2:-1], self._timesteps[-1:]]
        )[::-1].copy()

        timesteps = np.concatenate([self.prk_timesteps, self.plms_timesteps]).astype(
            np.int64
        )
        self.timesteps = torch.from_numpy(timesteps).to(device)

        self.ets = []
        self.counter = 0
        self.cur_model_output = 0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ):
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        This function calls `step_prk()` or `step_plms()` depending on the internal variable `counter`.
        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        return self.step_plms(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            return_dict=return_dict,
        )

    def step_plms(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ):
        """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.
        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class
        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        if self.num_inference_timesteps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        prev_timestep = (
            timestep - self.num_train_timesteps // self.num_inference_timesteps
        )

        if self.counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(model_output)
        else:
            prev_timestep = timestep
            timestep = (
                timestep + self.num_train_timesteps // self.num_inference_timesteps
            )

        if len(self.ets) == 1 and self.counter == 0:
            model_output = model_output
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            model_output = (model_output + self.ets[-1]) / 2
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            model_output = (
                23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]
            ) / 12
        else:
            model_output = (1 / 24) * (
                55 * self.ets[-1]
                - 59 * self.ets[-2]
                + 37 * self.ets[-3]
                - 9 * self.ets[-4]
            )

        prev_sample = self._get_prev_sample(
            sample, timestep, prev_timestep, model_output
        )
        self.counter += 1

        return prev_sample

    def _get_prev_sample(self, sample, timestep, prev_timestep, model_output):
        # See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        # this function computes x_(t−δ) using the formula of (9)
        # Note that x_t needs to be added to both sides of the equation

        # Notation (<variable name> -> <name in paper>
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # model_output -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # corresponds to (α_(t−δ) - α_t) divided by
        # denominator of x_t in formula (9) and plus 1
        # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t))
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        # corresponds to denominator of e_θ(x_t, t) in formula (9)
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)

        # full formula (9)
        prev_sample = (
            sample_coeff * sample
            - (alpha_prod_t_prev - alpha_prod_t)
            * model_output
            / model_output_denom_coeff
        )

        return prev_sample

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples
