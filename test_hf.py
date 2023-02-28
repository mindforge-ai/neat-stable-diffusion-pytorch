from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

model = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    safety_checker=None,
    scheduler=DDIMScheduler(beta_start=0.00085, beta_end=0.012, num_train_timesteps=1000, beta_schedule="scaled_linear", clip_sample=False, prediction_type="epsilon", set_alpha_to_one=False, steps_offset=1, trained_betas=None)
).to(0)

print(model.scheduler)

g_cuda = torch.Generator(device=0).manual_seed(65536)

images = model(
    "Georges Seurat painting of a lemur on Saturn",
    generator=g_cuda,
    guidance_scale=1,
    num_inference_steps=50,
).images
images[0].save("hf_groundtruth.png")
