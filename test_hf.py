from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

model = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_auth_token="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",
    revision="fp16",
    torch_dtype=torch.float16,
).to(0)

model.save_pretrained("data/v1-5-weights")
