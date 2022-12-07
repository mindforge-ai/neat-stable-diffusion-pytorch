from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

model = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_auth_token="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",
).to(0)

model.save_pretrained("data/v1-5-weights")
