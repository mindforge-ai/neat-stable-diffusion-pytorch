import torch
from stable_diffusion_pytorch import (
    Tokenizer,
    CLIPTextEncoder,
    Encoder,
    Decoder,
    Diffusion,
)
from stable_diffusion_pytorch import util
from stable_diffusion_pytorch.samplers import (
    DDIMSampler,
)
from infer import sample

import argparse
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import itertools
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os

WEIGHTS_PATH = "../weights/sd15"
writer = SummaryWriter()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
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

    tokenizer = Tokenizer(
        vocab_path=f"{WEIGHTS_PATH}/vocab.json",
        merges_path=f"{WEIGHTS_PATH}/merges.txt",
    )

    text_encoder = CLIPTextEncoder().to(args.device)
    text_encoder.load_state_dict(torch.load(f"{WEIGHTS_PATH}/clip.pt"))

    vae = Encoder().to(args.device)
    vae.load_state_dict(torch.load(f"{WEIGHTS_PATH}/encoder.pt"))

    decoder = Decoder().to(args.device)
    decoder.load_state_dict(torch.load(f"{WEIGHTS_PATH}/decoder.pt"))

    unet = Diffusion().to(args.device)
    unet.load_state_dict(torch.load(f"{WEIGHTS_PATH}/unet.pt"))

    train_unet = True
    train_vae = False
    train_decoder = False
    train_text_encoder = True

    if train_unet:
        unet.train()
    else:
        unet.requires_grad_(False)
        unet.eval()

    if train_vae:
        vae.train()
    else:
        vae.requires_grad_(False)
        vae.eval()

    if train_decoder:
        decoder.train()
    else:
        decoder.requires_grad_(False)
        decoder.eval()

    if train_text_encoder:
        text_encoder.train()
    else:
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    params_to_optimize = []

    if train_unet:
        params_to_optimize.append(unet.parameters())
    if train_vae:
        params_to_optimize.append(vae.parameters())
    if train_decoder:
        params_to_optimize.append(decoder.parameters())
    if train_text_encoder:
        params_to_optimize.append(text_encoder.parameters())

    optimizer = torch.optim.AdamW(
        itertools.chain(*params_to_optimize),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    noise_scheduler = DDIMSampler(num_inference_steps=args.sampler_num_train_steps)

    processed_input_images = []
    for input_image in list(Path(args.data_dir).iterdir()):
        input_image = Image.open(input_image).convert("RGB")

        input_image = transforms.Compose(
            [
                transforms.Resize(
                    (args.height, args.width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop((args.height, args.width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )(input_image)

        processed_input_images.append(input_image)

    with progress:
        task = progress.add_task(
            "Training...", total=args.num_epochs * len(processed_input_images)
        )
        for epoch in range(args.num_epochs):
            # text_encoder.train()
            vae.train()
            unet.train()
            for step, batch in enumerate(processed_input_images):
                global_step = epoch * len(processed_input_images) + step

                # iterate through Dataloader
                # shift the below logic into the dataloader
                input_images = torch.tensor(batch, device=args.device).unsqueeze(0)

                batch_len, _, height, width = input_images.size()
                noise_shape = (1, 4, height // 8, width // 8)

                encoder_noise = torch.randn(
                    noise_shape,
                    generator=generator,
                    device=args.device,
                )

                latents = vae(
                    input_images, noise=encoder_noise, calculate_posterior=True
                )
                latents = latents * 0.18215  # vae scaling factor

                # Here we process our image into 'training data' for the diffusion model.

                # 1. We create pure noise:

                noise = torch.randn(
                    latents.size(),
                    generator=generator,
                    device=args.device,
                )

                # 2. We pick a random timestep for each image in the batch:

                timesteps = torch.randint(
                    0,
                    args.sampler_num_train_steps,
                    (batch_len,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # 3. We add noise, scaled by the timestep, to the latents:
                # At timestep 1000, it's pure noise. At timestep 0, it's the original latents. At timestep 500, it's half noise and half original latents.
                # Internally, there is some variance (the betas schedule) in the amount of noise added. So the above line is not exactly true.

                noisy_latents = noise_scheduler.forward_sample(
                    latents, timesteps, noise=noise
                ).to(
                    torch.float32
                )  # I don't know if this dtype conversion is correct.

                # this data should come from dataloader
                tokens = tokenizer.encode_batch(["a photo of zwx man"])
                tokens = torch.tensor(tokens, device=args.device)
                encoder_hidden_states = text_encoder(tokens)

                time_embedding = util.get_time_embedding(
                    timesteps, dtype=torch.float32, device=args.device
                )

                # Now we ask the model to predict the noise which was added to the 'underlying latent' (image).
                # We tell the UNET the timestep (roughly how much noise was added to the image). If we left out this information, I imagine the UNET would also have to learn how to handle unknown quantities of noise.
                # We also pass in the text prompt as extra conditioning on the diffusion process.
                output = unet(noisy_latents, encoder_hidden_states, time_embedding)

                # Notice how the UNET never 'sees' the original latents, only the noisy latents.
                # That's because Stable Diffusion is trained to predict the noise that was added to the image, rather than the image itself.
                # (Authors of the DDPM paper got better results this way.)
                target = noise

                loss = F.mse_loss(output.float(), target.float(), reduction="mean")

                loss.backward()
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()

                # create global step from epoch and step
                writer.add_scalar(
                    "loss/train", loss.detach().item(), global_step=global_step
                )

                progress.update(task, advance=1)

                with torch.inference_mode():
                    if (global_step + 1) % args.sample_interval == 0:
                        images = sample(
                            models={
                                "encoder": vae.eval(),
                                "decoder": decoder.eval(),
                                "unet": unet.eval(),
                                "text_encoder": text_encoder.eval(),
                            },
                            text_prompt="a painting of zwx man",
                            num_samples=1,
                            show_progress=False,
                        )
                        """ torch.save(vae.state_dict(), f"weights/{global_step}-vae.pt")
                        torch.save(unet.state_dict(), f"weights/{global_step}-unet.pt")
                        torch.save(
                            decoder.state_dict(), f"weights/{global_step}-decoder.pt"
                        )
                        torch.save(
                            text_encoder.state_dict(),
                            f"weights/{global_step}-text-encoder.pt",
                        ) """
                        writer.add_images(
                            "images/samples", images, global_step, dataformats="NHWC"
                        )
                        for index, image in enumerate(images):
                            image = transforms.ToPILImage()(image)
                            image.save(
                                f"./training-samples/sample_{global_step}_{index}.png"
                            )

    writer.flush()
    writer.close()
