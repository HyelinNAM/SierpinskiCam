# SierpinskiCam

This repository is the inference-first code release for the SierpinskiCam paper. It is designed to accompany the paper release and to let external users run the core pipeline:

1. Generate a camera trajectory JSON.
2. Generate SierpinskiCam conditioning/data for the method.
3. Download checkpoints from public links and run one example inference video.

The codebase is derived from the Musubi/Wan inference stack plus SierpinskiCam-specific conditioning and inference helpers. It intentionally does **not** include training runs, paper metric reproduction, baseline experiment scripts, SLURM jobs, generated videos, model weights, logs, or internal paper/rebuttal artifacts.

## Repository layout

```text
SierpinskiCam/
  README.md
  pyproject.toml
  src/musubi_tuner/                       # core Wan/Musubi inference package used by SierpinskiCam
  scripts/
    generate_camera_path.py               # writes data/camera_path.json
    create_sierpinskicam_conditioning.py   # creates rgb/dense_tx conditioning videos
    run_sierpinskicam_inference.py         # runs checkpoint-based video inference
  examples/prompts/example_prompt.txt
  checkpoints/README.md                   # external checkpoint download placeholders
  docs/                                   # upstream Musubi/Wan reference docs
  data/                                   # local inputs/generated conditioning; ignored except .gitkeep
  outputs/                                # generated videos/latents; ignored except .gitkeep
```

## Installation

Python 3.10+ is recommended. Install the base package with a CUDA-enabled PyTorch extra appropriate for your machine:

```bash
git clone <PUBLIC_REPO_URL> SierpinskiCam
cd SierpinskiCam

# Example using uv and CUDA 12.4 wheels
uv sync --extra cu124

# or with pip after installing a compatible torch/torchvision build
pip install -e .
```

For SierpinskiCam conditioning generation you also need the external geometry/depth stack used by the paper pipeline:

- Depth-Anything-3, importable as `depth_anything_3`
- TrajectoryCrafter `models/` directory containing `utils.Warper`
- MoviePy / ffmpeg for writing videos

If these are installed separately, point the conditioning script to TrajectoryCrafter with:

```bash
export TRAJECTORYCRAFTER_MODELS=/path/to/TrajectoryCrafter/models
```

## Checkpoints

Weights are not committed to this repository. Download them from the public links released with the paper and follow `checkpoints/README.md`.

Expected local layout if using `SIERPINSKICAM_CHECKPOINT_DIR`:

```text
checkpoints/
  vae/wan_2.1_vae.safetensors
  text_encoders/umt5-xxl-enc-fp8_e4m3fn-kijai.safetensors
  diffusion_models/Wan2.1-Fun-Control-14B_fp8_e4m3fn.safetensors
  lora/sierpinskicam.safetensors
```

```bash
export SIERPINSKICAM_CHECKPOINT_DIR=$PWD/checkpoints
```

You can also pass `--vae`, `--t5`, `--dit`, and `--network-weights` explicitly to the inference script.

## End-to-end smoke workflow

The commands below show the intended public smoke path. Replace dataset/checkpoint paths with your local downloads.

### 1. Generate camera paths

```bash
python scripts/generate_camera_path.py \
  --output data/camera_path.json \
  --total-frames 81
```

### 2. Generate SierpinskiCam conditioning/data

Prepare an input frame dataset as one directory per scene:

```text
data/input_frames/
  sample_scene/
    00000.jpg
    00001.jpg
    ...
```

Then create conditioning for one camera and one scene:

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

This writes files such as:

```text
data/conditioning/cam01/rgb/sample_scene.mp4
data/conditioning/cam01/dense_tx/sample_scene.mp4
```

The inference script also expects a first-frame image under `img/`. If your conditioning generator did not create it, place one manually:

```text
data/conditioning/cam01/img/sample_scene.jpg
```

### 3. Run one-video inference

First validate paths without importing CUDA/model code:

```bash
python scripts/run_sierpinskicam_inference.py \
  --base-path data/conditioning/cam01 \
  --output-dir outputs/smoke_cam01 \
  --checkpoint-root "$SIERPINSKICAM_CHECKPOINT_DIR" \
  --prompt-file examples/prompts/example_prompt.txt \
  --only-video sample_scene \
  --check-only
```

Then run inference in a GPU environment:

```bash
python scripts/run_sierpinskicam_inference.py \
  --base-path data/conditioning/cam01 \
  --output-dir outputs/smoke_cam01 \
  --checkpoint-root "$SIERPINSKICAM_CHECKPOINT_DIR" \
  --prompt-file examples/prompts/example_prompt.txt \
  --only-video sample_scene \
  --max-videos 1 \
  --sample-steps 30 \
  --blocks-to-swap 18
```

Expected output:

```text
outputs/smoke_cam01/sample_scene.mp4
```

## Quick validation commands

These commands should work on a clean checkout without model downloads:

```bash
python scripts/generate_camera_path.py --output /tmp/sierpinskicam_camera_path.json
python scripts/create_sierpinskicam_conditioning.py --help
python scripts/run_sierpinskicam_inference.py --help
python -m compileall scripts src
```

Full video inference requires downloaded checkpoints, a valid conditioning folder, and a GPU environment.

## Release scope

Included:

- SierpinskiCam camera path generation
- SierpinskiCam conditioning/data generation entrypoint
- SierpinskiCam checkpoint-based inference entrypoint
- Musubi/Wan package code required by inference
- Public docs and checkpoint download placeholders

Excluded:

- model weight files
- generated outputs/videos/latents/metrics
- training run artifacts
- paper metric reproduction scripts
- baseline experiment wrappers
- SLURM scripts and cluster-specific launchers
- internal Codex/OMX coordination folders, logs, PDFs, rebuttals, and private paths

## License and attribution

See `THIRD_PARTY.md` and the upstream reference docs in `docs/`. This release includes code derived from Musubi Tuner and Wan-related components; respect all upstream licenses and model licenses when downloading checkpoints.
