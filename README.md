# SierpinskiCam

Official inference code release for the SierpinskiCam.

**[SierpinskiCam: Camera-Controlled Video Retaking with Sierpinski Triangle Pattern Cues](https://arxiv.org/abs)**

[Suttisak Wizadwongsa*](), [Hyelin Nam*](https://hyelinnam.github.io/), [Supasorn Suwajanakorn](https://www.supasorn.com/), [Jeong Joon Park](https://jjparkcv.github.io/)


[![Project Website](https://img.shields.io/badge/Project-Website-blue)]() [![arXiv](https://img.shields.io/badge/arXiv-2311.18608-b31b1b.svg)]()

## Abstract
Generating novel renderings of a scene along user-defined camera trajectories from a single monocular video, dubbed video retaking, is a compelling but difficult problem in content creation and visual effects. Existing geometry-guided approaches reconstruct a 4D representation from the source video and render it along the target trajectory to condition video diffusion models. However, this guidance degrades as the target camera departs from the source trajectory, leaving newly revealed regions sparse or entirely missing. We propose SierpinskiCam, which addresses this limitation by augmenting geometry-based guidance with Sierpinski dome texture cues that contains rich trackable features even under large viewpoint changes. We further introduce a reference video conditioning mechanism that appends source-video tokens to the target-token sequence and separates the two streams with negative RoPE indices, enabling appearance grounding without architectural modification or per-video adaptation. Extensive experiments show that SierpinskiCam achieves significant gains in camera controllability, geometric consistency, and video quality across diverse and challenging retaking scenarios.


## What you can run

The public workflow is:

1. **Use the provided camera trajectory JSON** (`example_test_data/cameras/camera_extrinsics.json`).
2. **Generate SierpinskiCam conditioning/data** (`rgb`, `dense_tx`, and first-frame `img`) from the provided sample videos.
3. **Optionally cache prompt/text embeddings** for the target scenes.
4. **Run checkpoint-based inference** to produce an output video for at least one example scene.

## Repository layout

```text
SierpinskiCam/
  README.md
  pyproject.toml
  src/musubi_tuner/                       # Musubi/Wan inference stack used by SierpinskiCam
  example_test_data/
    cameras/camera_extrinsics.json        # ReCamMaster-style test camera paths
    input_videos/                         # five small sample videos for smoke tests
    textures/sierpinski_dome_16x16_2048.png # Sierpinski dome texture asset
  scripts/
    generate_camera_path.py               # optional custom camera-path generator
    create_sierpinskicam_conditioning.py   # conditioning/data generator
    cache_text.py                         # optional prompt/text-encoder cache helper
    run_sierpinskicam_inference.py         # one-video or batch inference
  examples/prompts/example_prompt.txt      # minimal prompt file
  checkpoints/                              # local checkpoint workspace; weights are git-ignored
  docs/                                    # static SierpinskiCam project page
  data/                                    # local generated conditioning/text-cache workspace (git-ignored)
  outputs/                                 # generated videos/latents (git-ignored)
```

The `docs/` folder is the deployable static project page. Runtime instructions live in this root README.

## Installation

Python `>=3.10,<3.13` is required. The default install path uses CUDA 12.4 PyTorch wheels.

```bash
git clone https://github.com/HyelinNAM/SierpinskiCam.git SierpinskiCam
cd SierpinskiCam

conda create -n sierpinskicam python=3.11 -y
conda activate sierpinskicam

pip install -r requirements.txt
```

`requirements.txt` installs this repository, the Wan/Musubi inference dependencies, Depth-Anything-3, and the small conditioning helpers. If your machine uses a different CUDA version, edit the two PyTorch lines at the top of `requirements.txt` before installing.

To regenerate conditioning from the sample videos, also point the script to the TrajectoryCrafter code directory that contains `utils.Warper`. In the original TrajectoryCrafter checkout, this is usually the `models/` directory, so the path should contain `utils.py` or a `utils/` module with `Warper`:

```bash
export TRAJECTORYCRAFTER_PATH=/path/to/TrajectoryCrafter/models
```

If you only run inference from already generated `data/conditioning/...` folders, TrajectoryCrafter and Depth-Anything-3 are not used at inference time.

## Checkpoints

Weights are **not** committed to this repository. Download the four files below before running inference.

| File to save | Download source |
| --- | --- |
| `checkpoints/vae/wan_2.1_vae.safetensors` | [Wan VAE](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors) |
| `checkpoints/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors` | [UMT5 text encoder](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/umt5-xxl-enc-fp8_e4m3fn.safetensors) |
| `checkpoints/diffusion_models/Wan2.1-Fun-Control-14B_fp8_e4m3fn.safetensors` | [Wan2.1 Fun-Control 14B fp8 DiT](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2.1-Fun-Control-14B_fp8_e4m3fn.safetensors) |
| `checkpoints/lora/sierpinskicam.safetensors` | [SierpinskiCam LoRA](https://drive.google.com/file/d/1D2LZoyAWSZR1Z_1_tahqjIgrBJIi4nb5/view?usp=drive_link) |

Expected local layout:

```text
checkpoints/
  vae/wan_2.1_vae.safetensors
  text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors
  diffusion_models/Wan2.1-Fun-Control-14B_fp8_e4m3fn.safetensors
  lora/sierpinskicam.safetensors
```

Then set:

```bash
export SIERPINSKICAM_CHECKPOINT_DIR=$PWD/checkpoints
```

You can also pass all paths explicitly with `--vae`, `--t5`, `--dit`, and `--network-weights`.

Expected LoRA checksum:

```text
md5: 05b3cf328a79b3e6b9c34fd387a599d2
```


## Quickstart: end-to-end inference

The commands below are the intended smoke path for a clean checkout. Replace placeholder paths with your local dataset and checkpoint locations.

### 1. Use the provided camera trajectories

For paper reproduction, use the provided ReCamMaster-format camera file directly:

```text
example_test_data/cameras/camera_extrinsics.json
```

This JSON contains 81 frames and 14 camera paths:

- `cam01`-`cam10`: copied from the official ReCamMaster example camera paths: <https://github.com/KlingAIResearch/ReCamMaster/blob/main/example_test_data/cameras/camera_extrinsics.json>
- `cam11`-`cam14`: the four advanced camera paths used in the SierpinskiCam paper evaluation.

No camera-trajectory generation step is required for reproduction. The `generate_camera_path.py` script is only an optional utility for custom camera paths.

### 2. Use the provided sample videos

The release includes five small source videos under the default sample-data input root:

```text
example_test_data/input_videos/
  01.mp4
  02.mp4
  03.mp4
  04.mp4
  05.mp4
```

The conditioning script reads videos directly and uses the first `--frame-count` frames (`49` by default). It also still accepts legacy per-scene frame folders under `--input-base` if you want to run custom data. Use `--pad-short-scenes` only if you intentionally want to repeat the final frame for short clips.

The Sierpinski dome texture is self-contained in `example_test_data/textures/sierpinski_dome_16x16_2048.png` and is used by default via `--texture-path`.

### 3. Generate SierpinskiCam conditioning/data

```bash
python scripts/create_sierpinskicam_conditioning.py \
  --trajectorycrafter-path "$TRAJECTORYCRAFTER_PATH"
```

Expected files:

```text
data/conditioning/cam01/rgb/01.mp4
data/conditioning/cam01/dense_tx/01.mp4
data/conditioning/cam01/img/01.jpg
```

Default paths used by the command above:

- input videos: `example_test_data/input_videos`
- Sierpinski texture: `example_test_data/textures/sierpinski_dome_16x16_2048.png`
- scene names: `01`, `02`, `03`, `04`, `05`
- camera file: `example_test_data/cameras/camera_extrinsics.json`
- conditioning output: `data/conditioning`
- camera name: `cam01`

### 4. Optional: cache prompt/text embeddings

Inference can encode the prompt on the fly. For repeated runs, precompute a scene cache first:

```bash
python scripts/cache_text.py \
  --checkpoint-root "$SIERPINSKICAM_CHECKPOINT_DIR"
```

By default this uses `examples/prompts/example_prompt.txt`, writes `<scene>_wan_te.safetensors` files to `data/text_cache`, and caches the five provided sample-video scene names (`01`-`05`). Pass the folder to inference with `--te-cache data/text_cache`. If every selected scene has a cache file, the inference script skips live T5 prompt encoding.

### 5. Validate inference paths without loading models

Run this after generating conditioning in Step 3; `--check-only` still expects populated `data/conditioning/<camera>/rgb`, `dense_tx`, and `img` entries.

```bash
python scripts/run_sierpinskicam_inference.py \
  --checkpoint-root "$SIERPINSKICAM_CHECKPOINT_DIR" \
  --te-cache data/text_cache \
  --check-only
```

### 6. Run one-video inference

Run this in a GPU environment after downloading checkpoints:

```bash
python scripts/run_sierpinskicam_inference.py \
  --checkpoint-root "$SIERPINSKICAM_CHECKPOINT_DIR" \
  --te-cache data/text_cache \
  --max-videos 1 \
  --sample-steps 30 \
  --blocks-to-swap 18
```

Expected output:

```text
outputs/smoke_cam01/01.mp4
```

If you only want to check latent generation first, add `--no-decode`.

## Useful script entry points

```bash
# Generate default cam01 conditioning for the five sample videos.
python scripts/create_sierpinskicam_conditioning.py \
  --trajectorycrafter-path "$TRAJECTORYCRAFTER_PATH"

# Optionally precompute text caches for repeated inference runs.
python scripts/cache_text.py \
  --checkpoint-root "$SIERPINSKICAM_CHECKPOINT_DIR"

# Run the default one-video smoke inference.
python scripts/run_sierpinskicam_inference.py \
  --checkpoint-root "$SIERPINSKICAM_CHECKPOINT_DIR" \
  --te-cache data/text_cache

# Optional: generate custom camera paths instead of using example_test_data/cameras/.
python scripts/generate_camera_path.py --output data/custom_camera_extrinsics.json
```

Useful inference options:

- `--guidance dense_tx`: conditioning folder name under `--base-path`
- `--reference rgb`: reference-video folder name under `--base-path`
- `--te-cache <dir>`: optional text-encoder cache containing `<scene>_wan_te.safetensors`
- `--only-video <scene>`: run one named scene
- `--max-videos N`: cap the number of processed scenes
- `--no-decode`: save latents only
- `--check-only`: validate paths and exit before CUDA/model imports

## License

This code is released under the Apache-2.0 license. Model checkpoints are distributed separately; follow the license terms of each downloaded checkpoint.

## Acknowledgements

We thank the authors and maintainers of [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) and [TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter), which were helpful references while building this codebase.

## Citation

If you find this work useful, please consider citing SierpinskiCam. BibTeX coming soon.
