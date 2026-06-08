# SierpinskiCam

Official inference code release for the SierpinskiCam paper.

SierpinskiCam is released here as a **paper-accompanying inference codebase**: the repository is meant to help researchers generate the method's camera/conditioning inputs, load the released checkpoints, and run video inference. It is intentionally not a full training or paper-metric reproduction repository.

## What you can run

The public workflow is:

1. **Generate camera trajectories** in the `camera_path.json` format used by the method.
2. **Generate SierpinskiCam conditioning/data** (`rgb`, `dense_tx`, and first-frame `img`) from an image-sequence dataset.
3. **Optionally cache prompt/text embeddings** for the target scenes.
4. **Run checkpoint-based inference** to produce an output video for at least one example scene.

## Repository layout

```text
SierpinskiCam/
  README.md
  pyproject.toml
  src/musubi_tuner/                       # Musubi/Wan inference stack used by SierpinskiCam
  scripts/
    generate_camera_path.py               # camera_path.json generator
    create_sierpinskicam_conditioning.py   # conditioning/data generator
    cache_sierpinskicam_text.py            # optional prompt/text-encoder cache helper
    run_sierpinskicam_inference.py         # one-video or batch inference
  examples/prompts/example_prompt.txt      # minimal prompt file
  checkpoints/README.md                    # external checkpoint download instructions
  docs/                                    # upstream Musubi/Wan reference docs
  data/                                    # local inputs/conditioning workspace (git-ignored)
  outputs/                                 # generated videos/latents (git-ignored)
```

## Installation

Python `>=3.10,<3.13` is required.

```bash
git clone <PUBLIC_REPO_URL> SierpinskiCam
cd SierpinskiCam
```

Install a CUDA-enabled PyTorch build that matches your system, then install the package:

```bash
# Option A: uv with CUDA 12.4 wheels
uv sync --extra cu124

# Option B: uv with CUDA 12.8 wheels
uv sync --extra cu128

# Option C: pip after installing a compatible torch/torchvision build
pip install -e .
```

For conditioning generation, install the external geometry/depth dependencies used by the paper pipeline:

- Depth-Anything-3, importable as `depth_anything_3`
- TrajectoryCrafter `models/` directory containing `utils.Warper`
- MoviePy/ffmpeg for video writing

Point the conditioning script to TrajectoryCrafter with either `--trajectorycrafter-models` or:

```bash
export TRAJECTORYCRAFTER_MODELS=/path/to/TrajectoryCrafter/models
```

## Checkpoints

Weights are **not** committed to this repository. Download the public checkpoints released with the paper and follow `checkpoints/README.md`.

Expected local layout if you want to use `--checkpoint-root`:

```text
checkpoints/
  vae/wan_2.1_vae.safetensors
  text_encoders/umt5-xxl-enc-fp8_e4m3fn-kijai.safetensors
  diffusion_models/Wan2.1-Fun-Control-14B_fp8_e4m3fn.safetensors
  lora/sierpinskicam.safetensors
```

Then set:

```bash
export SIERPINSKICAM_CHECKPOINT_DIR=$PWD/checkpoints
```

You can also pass all paths explicitly with `--vae`, `--t5`, `--dit`, and `--network-weights`.

## Quickstart: end-to-end inference

The commands below are the intended smoke path for a clean checkout. Replace placeholder paths with your local dataset and checkpoint locations.

### 1. Generate camera trajectories

```bash
python scripts/generate_camera_path.py \
  --output data/camera_path.json \
  --total-frames 81
```

### 2. Prepare input frames

Create one folder per scene under `data/input_frames/`:

```text
data/input_frames/
  sample_scene/
    00000.jpg
    00001.jpg
    ...
```

The conditioning script uses the first `--frame-count` frames (`49` by default). Use `--pad-short-scenes` only if you intentionally want to repeat the final frame for short clips.

### 3. Generate SierpinskiCam conditioning/data

```bash
python scripts/create_sierpinskicam_conditioning.py \
  --input-base data/input_frames \
  --output-base data/conditioning \
  --camera-path data/camera_path.json \
  --trajectorycrafter-models "$TRAJECTORYCRAFTER_MODELS" \
  --camera-names cam01 \
  --scenes sample_scene \
  --save-outputs rgb,dense_tx
```

Expected files:

```text
data/conditioning/cam01/rgb/sample_scene.mp4
data/conditioning/cam01/dense_tx/sample_scene.mp4
data/conditioning/cam01/img/sample_scene.jpg
```

### 4. Optional: cache prompt/text embeddings

Inference can encode the prompt on the fly. For repeated runs, precompute a scene cache first:

```bash
python scripts/cache_sierpinskicam_text.py \
  --t5 "$SIERPINSKICAM_CHECKPOINT_DIR/text_encoders/umt5-xxl-enc-fp8_e4m3fn-kijai.safetensors" \
  --output-dir data/text_cache \
  --scenes sample_scene \
  --prompt-file examples/prompts/example_prompt.txt
```

This writes:

```text
data/text_cache/sample_scene_wan_te.safetensors
```

Pass the folder to inference with `--te-cache data/text_cache`. If every selected scene has a cache file, the inference script skips live T5 prompt encoding.

### 5. Validate inference paths without loading models

```bash
python scripts/run_sierpinskicam_inference.py \
  --base-path data/conditioning/cam01 \
  --output-dir outputs/smoke_cam01 \
  --checkpoint-root "$SIERPINSKICAM_CHECKPOINT_DIR" \
  --prompt-file examples/prompts/example_prompt.txt \
  --te-cache data/text_cache \
  --only-video sample_scene \
  --check-only
```

### 6. Run one-video inference

Run this in a GPU environment after downloading checkpoints:

```bash
python scripts/run_sierpinskicam_inference.py \
  --base-path data/conditioning/cam01 \
  --output-dir outputs/smoke_cam01 \
  --checkpoint-root "$SIERPINSKICAM_CHECKPOINT_DIR" \
  --prompt-file examples/prompts/example_prompt.txt \
  --te-cache data/text_cache \
  --only-video sample_scene \
  --max-videos 1 \
  --sample-steps 30 \
  --blocks-to-swap 18
```

Expected output:

```text
outputs/smoke_cam01/sample_scene.mp4
```

If you only want to check latent generation first, add `--no-decode`.

## Command reference

```bash
python scripts/generate_camera_path.py --help
python scripts/create_sierpinskicam_conditioning.py --help
python scripts/run_sierpinskicam_inference.py --help
```

Useful inference options:

- `--guidance dense_tx`: conditioning folder name under `--base-path`
- `--reference rgb`: reference-video folder name under `--base-path`
- `--te-cache <dir>`: optional text-encoder cache containing `<scene>_wan_te.safetensors`
- `--only-video <scene>`: run one named scene
- `--max-videos N`: cap the number of processed scenes
- `--no-decode`: save latents only
- `--check-only`: validate paths and exit before CUDA/model imports

## Validation checklist

These checks should pass without downloading model weights:

```bash
python scripts/generate_camera_path.py --output /tmp/sierpinskicam_camera_path.json
python scripts/create_sierpinskicam_conditioning.py --help
python scripts/cache_sierpinskicam_text.py --help
python scripts/run_sierpinskicam_inference.py --help
python -m compileall -q scripts src
```

Full validation requires a GPU, downloaded checkpoints, and a conditioning folder with `rgb/`, `dense_tx/`, and `img/` entries.

## Release scope

Included:

- camera path generation
- SierpinskiCam conditioning/data generation
- checkpoint-based inference
- Musubi/Wan code required by the inference path
- public docs, examples, checkpoint placeholders, and release hygiene files

Excluded:

- checkpoint weights
- generated videos, latents, metrics, and logs
- training runs and training artifacts
- paper metric reproduction scripts
- baseline wrappers and cluster-specific SLURM launchers
- private paper drafts, rebuttals, Codex/OMX folders, and private absolute paths

## License and attribution

See `LICENSE`, `THIRD_PARTY.md`, and `docs/musubi_tuner_README.md`. Model checkpoints are distributed separately; follow the license terms of each downloaded checkpoint.
