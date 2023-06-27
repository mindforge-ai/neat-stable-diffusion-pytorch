# neat-stable-diffusion-pytorch

Minimal, annotated implementation of Stable Diffusion in PyTorch. STILL IN DEVELOPMENT.

With `cfg-scale` set to 1 (no classifier-free guidance), the txt2img setup in this repo acheives 100% match with the results from the [original Stable Diffusion codebase](https://github.com/CompVis/stable-diffusion). With classifier-free guidance, the results are a tiny bit different. I would say the difference is unnoticeable to the human eye though I will check CLIP similarity at some point later.

The reason they are different is that the CompVis stacks the text-conditioned latents and the unconditional latents in a slightly different way, and I think somewhere along the way PyTorch handles the reversed order differently... or something like that.

I would be so happy for anyone to jump in with any ideas, comments, or a PR; my hope is to get this repo to a stage where it provides a convincing, clean alternative to the `diffusers` and `CompVis` repos.

This implementation is a modified fork of [kjsman's stable-diffusion-pytorch](https://github.com/kjsman/stable-diffusion-pytorch).

I have yet to finish annotating the whole process, and removing redundant code, of which there is quite a bit at the moment.