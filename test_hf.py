from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import torch

model = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=LMSDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        trained_betas=None,
        prediction_type="epsilon",
    ),
    safety_checker=None,
).to(0)

g_cuda = torch.Generator(device=0).manual_seed(65536)

images = model(
    "Georges Seurat painting of a lemur on Saturn",
    generator=g_cuda,
    guidance_scale=7.5,
    num_inference_steps=50,
).images
images[0].save("hf_groundtruth.png")
