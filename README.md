# neat-stable-diffusion-pytorch

Minimal, annotated implementation of Stable Diffusion in PyTorch. STILL IN DEVELOPMENT.

This setup achieves around ~99.4% similarity to the results in HuggingFace's `diffusers` repo without classifier-free guidance. It's a little bit less, probablty around ~99% when using classifier-free guidance. This is due to a few things HF are doing in their code which I haven't done here.
    - they use `torch.baddbmm` in the decoder's attention block which seems to scale the self-attention in a very slightly different way
    - they stack the unconditional latent and the conditional latent earlier than I do, but I think they are doing redundant computations because of this

I would be so happy for anyone to jump in with any ideas, comments, or a PR; my hope is to get this repo to a stage where it provides a convincing, clean alternative to the `diffusers` and `CompVis` repos.

This implementation is a modified fork of [kjsman's stable-diffusion-pytorch](https://github.com/kjsman/stable-diffusion-pytorch).

I have yet to finish annotating the whole process, and removing redundant code, of which there is quite a bit at the moment.

Notes for me:
    - training atm creates images that get progressively greyer