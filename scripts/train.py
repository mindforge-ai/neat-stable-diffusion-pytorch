import torch
from stable_diffusion_pytorch import Tokenizer, CLIP, Encoder, Decoder, Diffusion
from stable_diffusion_pytorch import util
from stable_diffusion_pytorch.samplers import (
    KLMSSampler,
    KEulerSampler,
    KEulerAncestralSampler,
)
from .infer import sample

import argparse
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from PIL import Image
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import itertools
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def make_compatible(state_dict):
    keys = list(state_dict.keys())
    changed = False
    for key in keys:
        if "causal_attention_mask" in key:
            del state_dict[key]
            changed = True
        elif "_proj_weight" in key:
            new_key = key.replace("_proj_weight", "_proj.weight")
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True
        elif "_proj_bias" in key:
            new_key = key.replace("_proj_bias", "_proj.bias")
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True

    if changed:
        print(
            "Given checkpoint data were modified dynamically by make_compatible"
            " function on model_loader.py. Maybe this happened because you're"
            " running newer codes with older checkpoint files. This behavior"
            " (modify old checkpoints and notify rather than throw an error)"
            " will be removed soon, so please download latest checkpoints file."
        )

    return state_dict


def load_model(module, weights_path, device):
    model = module().to(device)
    state_dict = torch.load(weights_path)
    state_dict = make_compatible(state_dict)
    model.load_state_dict(state_dict)
    return model


def preload_models(device):
    return {
        "clip": load_model(CLIP, "data/ckpt/clip.pt", device),
        "encoder": load_model(Encoder, "data/ckpt/encoder.pt", device),
        "decoder": load_model(Decoder, "data/ckpt/decoder.pt", device),
        "unet": load_model(Diffusion, "data/ckpt/unet.pt", device),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=65536)
    parser.add_argument("--sampler-num-train-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--data-dir", type=str, default="data/inputs")
    parser.add_argument("--base-dir", type=str, default="data/ckpt")
    parser.add_argument("--output-dir", type=str, default="data/outputs")
    parser.add_argument("--sample-interval", type=int, default=100)

    args = parser.parse_args()

    text_column = TextColumn("{task.description}")
    bar_column = BarColumn(bar_width=None)
    m_of_n_complete_column = MofNCompleteColumn()
    time_elapsed_column = TimeElapsedColumn()
    time_remaining_column = TimeRemainingColumn()
    progress = Progress(
        text_column,
        bar_column,
        m_of_n_complete_column,
        time_elapsed_column,
        time_remaining_column,
        expand=True,
    )

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    tokenizer = Tokenizer()

    text_encoder = CLIP().to(args.device)
    text_encoder.load_state_dict(torch.load("data/ckpt/clip.pt"))

    vae = Encoder().to(args.device)
    vae.load_state_dict(torch.load("data/ckpt/encoder.pt"))

    decoder = Decoder().to(args.device)
    decoder.load_state_dict(torch.load("data/ckpt/decoder.pt"))

    unet = Diffusion().to(args.device)
    unet.load_state_dict(torch.load("data/ckpt/unet.pt"))

    optimizer = torch.optim.AdamW(
        itertools.chain(unet.parameters(), vae.parameters(), text_encoder.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    noise_scheduler = KLMSSampler(n_inference_steps=args.sampler_num_train_steps)

    processed_input_images = []
    for input_image in list(Path(args.data_dir).iterdir()):
        input_image = Image.open(input_image)
        input_image = input_image.resize((args.width, args.height))
        input_image = np.array(input_image)
        input_image = torch.tensor(input_image).float()
        input_image = util.rescale(input_image, (0, 255), (-1, 1))
        processed_input_images.append(input_image)

    for epoch in range(args.num_epochs):
        text_encoder.train()
        vae.train()
        unet.train()
        for step, batch in enumerate(
            processed_input_images
        ):  # iterate through Dataloader
            # shift the below logic into the dataloader
            input_images_tensor = torch.tensor(batch, device=args.device).unsqueeze(0)
            input_images_tensor = util.move_channel(input_images_tensor, to="first")

            batch_len, _, height, width = input_images_tensor.shape
            noise_shape = (1, 4, height // 8, width // 8)

            encoder_noise = torch.randn(
                noise_shape,
                generator=generator,
                device=args.device,
            )

            latents = vae(input_images_tensor, encoder_noise)

            latents_noise = torch.randn(
                noise_shape,
                generator=generator,
                device=latents.device,
            )

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                args.sampler_num_train_steps,
                (batch_len,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            noisy_latents = latents + latents_noise * noise_scheduler.sigmas[timesteps]

            # this data should come from dataloader
            tokens = tokenizer.encode_batch(["minecraft video"])
            tokens = torch.tensor(tokens, device=args.device)
            encoder_hidden_states = text_encoder(tokens)

            time_embedding = util.get_time_embedding(
                timesteps, dtype=torch.float32, device=args.device
            )

            # Predict the noise residual
            output = unet(noisy_latents, encoder_hidden_states, time_embedding)

            target = latents_noise

            loss = F.mse_loss(output.float(), target.float(), reduction="mean")

            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            # create global step from epoch and step
            global_step = epoch * len(processed_input_images) + step
            writer.add_scalar("Loss/train", loss.detach().item(), global_step)

            if global_step % args.sample_interval == 0:
                images = sample(
                    models={
                        "encoder": vae,
                        "decoder": decoder,
                        "unet": unet,
                        "text_encoder": text_encoder,
                    },
                    text_prompt="minecraft video",
                )
                writer.add_image("Sample", img_grid)

    writer.flush()
    writer.close()
