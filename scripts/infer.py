import torch
from stable_diffusion_pytorch import Tokenizer, CLIP, Encoder, Decoder, Diffusion
from stable_diffusion_pytorch import util
from stable_diffusion_pytorch.samplers import (
    KLMSSampler,
    KEulerSampler,
    KEulerAncestralSampler,
)
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


def parse_arguments(input_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="Georges Seurat painting of a lemur on Saturn",
    )
    parser.add_argument(
        "--image-prompt",
        type=str,
        default=None,
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=65536)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--sampler", type=str, default="k_lms")
    parser.add_argument("--sampler-num-infer-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="samples")

    args = parser.parse_args(input_args)

    return args

original_keys = {
    # CLIPTextEncoder weights
    "cond_stage_model.transformer.text_model.embeddings.position_ids": "embedding.position_indices",
    "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": "embedding.token_embedding.weight",
    "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight": "embedding.position_embedding.weight",
    "cond_stage_model.transformer.text_model.final_layer_norm.weight": "final_layer_norm.weight",
    "cond_stage_model.transformer.text_model.final_layer_norm.bias": "final_layer_norm.bias",
    # VAE Encoder weights
    "first_stage_model.encoder.conv_in.weight": "conv_in.weight",
    "first_stage_model.encoder.conv_in.bias": "conv_in.bias",
    "first_stage_model.encoder.norm_out.weight": "norm_out.weight",
    "first_stage_model.encoder.norm_out.bias": "norm_out.bias",
    "first_stage_model.encoder.conv_out.weight": "unknown_conv.weight",
    "first_stage_model.encoder.conv_out.bias": "unknown_conv.bias",
    # VAE Decoder weights
    "first_stage_model.decoder.conv_in.weight": "conv_in.weight",
    "first_stage_model.decoder.conv_in.bias": "conv_in.bias",
    "first_stage_model.decoder.norm_out.weight": "norm_out.weight",
    "first_stage_model.decoder.norm_out.bias": "norm_out.bias",
    "first_stage_model.decoder.conv_out.weight": "unknown_conv.weight",
    "first_stage_model.decoder.conv_out.bias": "unknown_conv.bias",
    "first_stage_model.post_quant_conv.weight": "post_quant_conv.weight",
    "first_stage_model.post_quant_conv.bias": "post_quant_conv.bias",
}

# CLIPTextEncoder weights
for i in range(12):
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.k_proj.weight"] = f"stack.{i}.self_attention.k_proj.weight"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.k_proj.bias"] = f"stack.{i}.self_attention.k_proj.bias"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.v_proj.weight"] = f"stack.{i}.self_attention.v_proj.weight"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.v_proj.bias"] = f"stack.{i}.self_attention.v_proj.bias"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.q_proj.weight"] = f"stack.{i}.self_attention.q_proj.weight"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.q_proj.bias"] = f"stack.{i}.self_attention.q_proj.bias"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.out_proj.weight"] = f"stack.{i}.self_attention.out_proj.weight"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.out_proj.bias"] = f"stack.{i}.self_attention.out_proj.bias"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm1.weight"] = f"stack.{i}.layer_norm_1.weight"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm1.bias"] = f"stack.{i}.layer_norm_1.bias"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc1.weight"] = f"stack.{i}.linear_1.weight"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc1.bias"] = f"stack.{i}.linear_1.bias"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc2.weight"] = f"stack.{i}.linear_2.weight"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.fc2.bias"] = f"stack.{i}.linear_2.bias"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm2.weight"] = f"stack.{i}.layer_norm_2.weight"
    original_keys[f"cond_stage_model.transformer.text_model.encoder.layers.{i}.layer_norm2.bias"] = f"stack.{i}.layer_norm_2.bias"

# VAE Encoder weights
down_counter = 0
for outer_i in range(4):
    for inner_i in range(2):
        original_keys[f"first_stage_model.encoder.down.{outer_i}.block.{inner_i}.norm1.weight"] = f"down.{down_counter}.groupnorm_1.weight"
        original_keys[f"first_stage_model.encoder.down.{outer_i}.block.{inner_i}.norm1.bias"] = f"down.{down_counter}.groupnorm_1.bias"
        original_keys[f"first_stage_model.encoder.down.{outer_i}.block.{inner_i}.conv1.weight"] = f"down.{down_counter}.conv_1.weight"
        original_keys[f"first_stage_model.encoder.down.{outer_i}.block.{inner_i}.conv1.bias"] = f"down.{down_counter}.conv_1.bias"
        original_keys[f"first_stage_model.encoder.down.{outer_i}.block.{inner_i}.norm2.weight"] = f"down.{down_counter}.groupnorm_2.weight"
        original_keys[f"first_stage_model.encoder.down.{outer_i}.block.{inner_i}.norm2.bias"] = f"down.{down_counter}.groupnorm_2.bias"
        original_keys[f"first_stage_model.encoder.down.{outer_i}.block.{inner_i}.conv2.weight"] = f"down.{down_counter}.conv_2.weight"
        original_keys[f"first_stage_model.encoder.down.{outer_i}.block.{inner_i}.conv2.bias"] = f"down.{down_counter}.conv_2.bias"
        if outer_i in [1, 2] and inner_i == 0:
            original_keys[f"first_stage_model.encoder.down.{outer_i}.block.{inner_i}.nin_shortcut.weight"] = f"down.{down_counter}.residual_layer.weight" # first it's 3
            original_keys[f"first_stage_model.encoder.down.{outer_i}.block.{inner_i}.nin_shortcut.bias"] = f"down.{down_counter}.residual_layer.bias"
        down_counter += 1
    # might need if here, i.e. if outer_i < 0  
    original_keys[f"first_stage_model.encoder.down.{outer_i}.downsample.conv.weight"] = f"down.{down_counter}.weight" # first it's 2
    original_keys[f"first_stage_model.encoder.down.{outer_i}.downsample.conv.bias"] = f"down.{down_counter}.bias"
    down_counter += 1

original_keys[f"first_stage_model.encoder.mid.block_1.norm1.weight"] = f"mid.0.groupnorm_1.weight"
original_keys[f"first_stage_model.encoder.mid.block_1.norm1.bias"] = f"mid.0.groupnorm_1.bias"
original_keys[f"first_stage_model.encoder.mid.block_1.conv1.weight"] = f"mid.0.conv_1.weight"
original_keys[f"first_stage_model.encoder.mid.block_1.conv1.bias"] = f"mid.0.conv_1.bias"
original_keys[f"first_stage_model.encoder.mid.block_1.norm2.weight"] = f"mid.0.groupnorm_2.weight"
original_keys[f"first_stage_model.encoder.mid.block_1.norm2.bias"] = f"mid.0.groupnorm_2.bias"
original_keys[f"first_stage_model.encoder.mid.block_1.conv2.weight"] = f"mid.0.conv_2.weight"
original_keys[f"first_stage_model.encoder.mid.block_1.conv2.bias"] = f"mid.0.conv_2.bias"
original_keys[f"first_stage_model.encoder.mid.block_2.norm1.weight"] = f"mid.2.groupnorm_1.weight"
original_keys[f"first_stage_model.encoder.mid.block_2.norm1.bias"] = f"mid.2.groupnorm_1.bias"
original_keys[f"first_stage_model.encoder.mid.block_2.conv1.weight"] = f"mid.2.conv_1.weight"
original_keys[f"first_stage_model.encoder.mid.block_2.conv1.bias"] = f"mid.2.conv_1.bias"
original_keys[f"first_stage_model.encoder.mid.block_2.norm2.weight"] = f"mid.2.groupnorm_2.weight"
original_keys[f"first_stage_model.encoder.mid.block_2.norm2.bias"] = f"mid.2.groupnorm_2.bias"
original_keys[f"first_stage_model.encoder.mid.block_2.conv2.weight"] = f"mid.2.conv_2.weight"
original_keys[f"first_stage_model.encoder.mid.block_2.conv2.bias"] = f"mid.2.conv_2.bias"

original_keys[f"first_stage_model.encoder.mid.attn_1.norm.weight"] = f"mid.1.groupnorm.weight"
original_keys[f"first_stage_model.encoder.mid.attn_1.norm.bias"] = f"mid.1.groupnorm.bias"
original_keys[f"first_stage_model.encoder.mid.attn_1.q.weight"] = f"mid.1.self_attention.q_proj.weight"
original_keys[f"first_stage_model.encoder.mid.attn_1.q.bias"] = f"mid.1.self_attention.q_proj.bias"
original_keys[f"first_stage_model.encoder.mid.attn_1.k.weight"] = f"mid.1.self_attention.k_proj.weight"
original_keys[f"first_stage_model.encoder.mid.attn_1.k.bias"] = f"mid.1.self_attention.k_proj.bias"
original_keys[f"first_stage_model.encoder.mid.attn_1.v.weight"] = f"mid.1.self_attention.v_proj.weight"
original_keys[f"first_stage_model.encoder.mid.attn_1.v.bias"] = f"mid.1.self_attention.v_proj.bias"
original_keys[f"first_stage_model.encoder.mid.attn_1.proj_out.weight"] = f"mid.1.self_attention.out_proj.weight"
original_keys[f"first_stage_model.encoder.mid.attn_1.proj_out.bias"] = f"mid.1.self_attention.out_proj.bias"

# VAE Decoder weights

original_keys[f"first_stage_model.decoder.mid.block_1.norm1.weight"] = f"mid.0.groupnorm_1.weight"
original_keys[f"first_stage_model.decoder.mid.block_1.norm1.bias"] = f"mid.0.groupnorm_1.bias"
original_keys[f"first_stage_model.decoder.mid.block_1.conv1.weight"] = f"mid.0.conv_1.weight"
original_keys[f"first_stage_model.decoder.mid.block_1.conv1.bias"] = f"mid.0.conv_1.bias"
original_keys[f"first_stage_model.decoder.mid.block_1.norm2.weight"] = f"mid.0.groupnorm_2.weight"
original_keys[f"first_stage_model.decoder.mid.block_1.norm2.bias"] = f"mid.0.groupnorm_2.bias"
original_keys[f"first_stage_model.decoder.mid.block_1.conv2.weight"] = f"mid.0.conv_2.weight"
original_keys[f"first_stage_model.decoder.mid.block_1.conv2.bias"] = f"mid.0.conv_2.bias"
original_keys[f"first_stage_model.decoder.mid.block_2.norm1.weight"] = f"mid.2.groupnorm_1.weight"
original_keys[f"first_stage_model.decoder.mid.block_2.norm1.bias"] = f"mid.2.groupnorm_1.bias"
original_keys[f"first_stage_model.decoder.mid.block_2.conv1.weight"] = f"mid.2.conv_1.weight"
original_keys[f"first_stage_model.decoder.mid.block_2.conv1.bias"] = f"mid.2.conv_1.bias"
original_keys[f"first_stage_model.decoder.mid.block_2.norm2.weight"] = f"mid.2.groupnorm_2.weight"
original_keys[f"first_stage_model.decoder.mid.block_2.norm2.bias"] = f"mid.2.groupnorm_2.bias"
original_keys[f"first_stage_model.decoder.mid.block_2.conv2.weight"] = f"mid.2.conv_2.weight"
original_keys[f"first_stage_model.decoder.mid.block_2.conv2.bias"] = f"mid.2.conv_2.bias"

original_keys[f"first_stage_model.decoder.mid.attn_1.norm.weight"] = f"mid.1.groupnorm.weight"
original_keys[f"first_stage_model.decoder.mid.attn_1.norm.bias"] = f"mid.1.groupnorm.bias"
original_keys[f"first_stage_model.decoder.mid.attn_1.q.weight"] = f"mid.1.self_attention.q_proj.weight"
original_keys[f"first_stage_model.decoder.mid.attn_1.q.bias"] = f"mid.1.self_attention.q_proj.bias"
original_keys[f"first_stage_model.decoder.mid.attn_1.k.weight"] = f"mid.1.self_attention.k_proj.weight"
original_keys[f"first_stage_model.decoder.mid.attn_1.k.bias"] = f"mid.1.self_attention.k_proj.bias"
original_keys[f"first_stage_model.decoder.mid.attn_1.v.weight"] = f"mid.1.self_attention.v_proj.weight"
original_keys[f"first_stage_model.decoder.mid.attn_1.v.bias"] = f"mid.1.self_attention.v_proj.bias"
original_keys[f"first_stage_model.decoder.mid.attn_1.proj_out.weight"] = f"mid.1.self_attention.out_proj.weight"
original_keys[f"first_stage_model.decoder.mid.attn_1.proj_out.bias"] = f"mid.1.self_attention.out_proj.bias"

up_counter = 0
for outer_i in range(4):
    for inner_i in range(3):
        original_keys[f"first_stage_model.decoder.up.{outer_i}.block.{inner_i}.norm1.weight"] = f"up.{up_counter}.groupnorm_1.weight"
        original_keys[f"first_stage_model.decoder.up.{outer_i}.block.{inner_i}.norm1.bias"] = f"up.{up_counter}.groupnorm_1.bias"
        original_keys[f"first_stage_model.decoder.up.{outer_i}.block.{inner_i}.conv1.weight"] = f"up.{up_counter}.conv_1.weight"
        original_keys[f"first_stage_model.decoder.up.{outer_i}.block.{inner_i}.conv1.bias"] = f"up.{up_counter}.conv_1.bias"
        original_keys[f"first_stage_model.decoder.up.{outer_i}.block.{inner_i}.norm2.weight"] = f"up.{up_counter}.groupnorm_2.weight"
        original_keys[f"first_stage_model.decoder.up.{outer_i}.block.{inner_i}.norm2.bias"] = f"up.{up_counter}.groupnorm_2.bias"
        original_keys[f"first_stage_model.decoder.up.{outer_i}.block.{inner_i}.conv2.weight"] = f"up.{up_counter}.conv_2.weight"
        original_keys[f"first_stage_model.decoder.up.{outer_i}.block.{inner_i}.conv2.bias"] = f"up.{up_counter}.conv_2.bias"
        if outer_i in [0, 1, 2] and inner_i == 0:
            original_keys[f"first_stage_model.decoder.up.{outer_i}.block.{inner_i}.nin_shortcut.weight"] = f"up.{up_counter}.residual_layer.weight" # first it's 3
            original_keys[f"first_stage_model.decoder.up.{outer_i}.block.{inner_i}.nin_shortcut.bias"] = f"up.{up_counter}.residual_layer.bias"
        up_counter += 1
    if outer_i > 0:
        original_keys[f"first_stage_model.decoder.up.{outer_i}.upsample.conv.weight"] = f"up.{up_counter}.weight"
        original_keys[f"first_stage_model.decoder.up.{outer_i}.upsample.conv.bias"] = f"up.{up_counter}.bias"
        up_counter += 1

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
        elif key in original_keys.keys():
            new_key = original_keys[key]
            state_dict[new_key] = state_dict[key]
            if new_key in ["mid.1.self_attention.q_proj.weight", "mid.1.self_attention.k_proj.weight", "mid.1.self_attention.v_proj.weight", "mid.1.self_attention.out_proj.weight"]:
                state_dict[new_key] = state_dict[new_key].squeeze()
                changed = True
            del state_dict[key]
            changed = True
        elif key in ["first_stage_model.quant_conv.weight", "first_stage_model.quant_conv.bias"]:
            del state_dict[key]

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
        "clip": load_model(CLIP, "weights/v1-5-cond_stage_model.pt", device),
        "encoder": load_model(
            Encoder, "weights/v1-5-encoder.pt", device
        ),
        "decoder": load_model(Decoder, "weights/v1-5-decoder.pt", device),
        "diffusion": load_model(Diffusion, "data/ckpt/diffusion.pt", device),
    }


def main(args):
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

    prompts = [args.text_prompt]

    uncond_prompt = ""
    uncond_prompts = [uncond_prompt] if uncond_prompt else None

    upload_input_image = False
    input_images = None  # [Image.open(path)]

    strength = 0.6

    if args.cfg_scale == 1:
        use_cfg = False
    else:
        use_cfg = True

    models = preload_models(args.device)

    with torch.inference_mode(), progress:
        if not isinstance(prompts, (list, tuple)) or not prompts:
            raise ValueError("prompts must be a non-empty list or tuple")

        if uncond_prompts and not isinstance(uncond_prompts, (list, tuple)):
            raise ValueError(
                "uncond_prompts must be a non-empty list or tuple if provided"
            )
        if uncond_prompts and len(prompts) != len(uncond_prompts):
            raise ValueError(
                "length of uncond_prompts must be same as length of prompts"
            )
        uncond_prompts = uncond_prompts or [""] * len(prompts)

        if input_images and not isinstance(uncond_prompts, (list, tuple)):
            raise ValueError(
                "input_images must be a non-empty list or tuple if provided"
            )
        if input_images and len(prompts) != len(input_images):
            raise ValueError("length of input_images must be same as length of prompts")
        if not 0 < strength < 1:
            raise ValueError("strength must be between 0 and 1")

        if args.height % 8 or args.width % 8:
            raise ValueError("height and width must be a multiple of 8")

        generator = torch.Generator(device=args.device)
        generator.manual_seed(args.seed)

        tokenizer = Tokenizer()
        clip = models["clip"]
        if use_cfg:
            cond_tokens = tokenizer.encode_batch(prompts)
            cond_tokens = torch.tensor(
                cond_tokens, dtype=torch.long, device=args.device
            )
            cond_context = clip(cond_tokens)
            uncond_tokens = tokenizer.encode_batch(uncond_prompts)
            uncond_tokens = torch.tensor(
                uncond_tokens, dtype=torch.long, device=args.device
            )
            uncond_context = clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context])  # [2, 77, 768]
        else:
            tokens = tokenizer.encode_batch(prompts)
            tokens = torch.tensor(tokens, dtype=torch.long, device=args.device)
            context = clip(tokens)  # [1, 77, 768]
        del tokenizer, clip

        if args.sampler == "k_lms":
            sampler = KLMSSampler(n_inference_steps=args.sampler_num_infer_steps)
        elif sampler == "k_euler":
            sampler = KEulerSampler(n_inference_steps=args.sampler_num_infer_steps)
        elif sampler == "k_euler_ancestral":
            sampler = KEulerAncestralSampler(
                n_inference_steps=args.sampler_num_infer_steps, generator=generator
            )
        else:
            raise ValueError(
                "Unknown sampler value %s. "
                "Accepted values are {k_lms, k_euler, k_euler_ancestral}" % args.sampler
            )

        noise_shape = (len(prompts), 4, args.height // 8, args.width // 8)

        if args.image_prompt:
            encoder = models["encoder"]
            processed_input_images = []
            input_image = args.image_prompt
            if type(input_image) is str:
                input_image = Image.open(input_image)

            input_image = input_image.resize((args.width, args.height))
            input_image = np.array(input_image)
            input_image = torch.tensor(input_image, dtype=torch.float32)
            input_image = util.rescale(input_image, (0, 255), (-1, 1))
            processed_input_images.append(input_image)
            input_images_tensor = torch.stack(processed_input_images).to(args.device)
            input_images_tensor = util.move_channel(input_images_tensor, to="first")

            _, _, height, width = input_images_tensor.shape
            noise_shape = (len(prompts), 4, height // 8, width // 8)

            latents = encoder(input_images_tensor)
            print("latents out of encoder", latents.size())

            latents_noise = torch.randn(
                noise_shape, generator=generator, device=args.device
            )
            sampler.set_strength(
                strength=strength
            )  # proxy for timestep TODO: change to timestep
            latents += latents_noise * sampler.initial_scale

            del encoder, processed_input_images, input_images_tensor, latents_noise
        else:
            latents = torch.randn(noise_shape, generator=generator, device=args.device)
            latents *= sampler.initial_scale

        diffusion = models["diffusion"]

        timesteps = progress.track(sampler.timesteps, description="Denoising...")
        for timestep in timesteps:
            time_embedding = util.get_time_embedding(
                timestep, dtype=torch.float32, device=latents.device
            )

            input_latents = latents * sampler.get_input_scale()
            if use_cfg:
                input_latents = input_latents.repeat(
                    2, 1, 1, 1
                )  # Use same Gaussian noise for both latents

            output = diffusion(input_latents, context, time_embedding)
            if use_cfg:
                output_cond, output_uncond = output.chunk(
                    2
                )  # output_uncond is a 'random' image from the distribution of realistic images
                output = args.cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(latents, output)

        del diffusion

        decoder = models["decoder"]
        images = decoder(latents)
        del decoder

        images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = util.move_channel(images, to="last")
        images = images.to("cpu", torch.uint8).numpy()

        images = [Image.fromarray(image) for image in images]

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for image in images:
            image.save(f"{output_path}/{len(list(output_path.iterdir()))}.jpg")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
