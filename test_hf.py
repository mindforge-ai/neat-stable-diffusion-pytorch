from diffusers import StableDiffusionPipeline
import torch

model = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16
).to(0)

print(model.vae.decoder)
