# Checkpoints

Model weights are intentionally **not** stored in this Git repository.
Download the public checkpoints from the links released with the paper, then either pass each path explicitly or arrange them as:

```text
checkpoints/
  vae/wan_2.1_vae.safetensors
  text_encoders/umt5-xxl-enc-fp8_e4m3fn-kijai.safetensors
  diffusion_models/Wan2.1-Fun-Control-14B_fp8_e4m3fn.safetensors
  lora/sierpinskicam.safetensors
```

Then run inference with either:

```bash
export SIERPINSKICAM_CHECKPOINT_DIR=$PWD/checkpoints
```

or pass `--vae`, `--t5`, `--dit`, and `--network-weights` explicitly.

## Download links

Replace the placeholders below with the final public URLs when the paper release assets are uploaded:

- SierpinskiCam LoRA/network checkpoint: TODO_PUBLIC_LINK
- Wan2.1 Fun-Control DiT checkpoint: TODO_PUBLIC_LINK
- Wan VAE checkpoint: TODO_PUBLIC_LINK
- UMT5 text encoder checkpoint: TODO_PUBLIC_LINK
