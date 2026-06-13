#!/usr/bin/env python3
"""Create per-scene Wan/SierpinskiCam text-encoder caches.

The inference script looks for `<scene>_wan_te.safetensors` under `--te-cache`.
This helper writes that format from a prompt file or explicit prompt string.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_FILE = REPO_ROOT / "examples" / "prompts" / "example_prompt.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "text_cache"
DEFAULT_SCENES = "01,02,03,04,05"


def parse_args():
    parser = argparse.ArgumentParser(description="Cache SierpinskiCam/Wan T5 prompt embeddings for one or more scenes.")
    parser.add_argument("--checkpoint-root", default=os.environ.get("SIERPINSKICAM_CHECKPOINT_DIR"), help="Optional root containing text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors.")
    parser.add_argument("--t5", default=None, help="UMT5/T5 checkpoint path. Overrides --checkpoint-root derived path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where <scene>_wan_te.safetensors files are written.")
    parser.add_argument("--scenes", default=DEFAULT_SCENES, help="Comma-separated scene/video names to cache.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--prompt", help="Prompt text to cache.")
    group.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE), help="Prompt file; the first non-comment prompt is used by default.")
    parser.add_argument("--prompt-index", type=int, default=0, help="Prompt index from --prompt-file to cache.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda when available, else cpu.")
    parser.add_argument("--fp8-t5", action="store_true", help="Load the T5 model in fp8 mode.")
    parser.add_argument("--check-only", action="store_true", help="Validate paths and arguments without importing torch/model code.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.t5 is None and args.checkpoint_root:
        args.t5 = os.path.join(args.checkpoint_root, "text_encoders", "umt5-xxl-enc-fp8_e4m3fn.safetensors")
    if args.t5 is None:
        raise ValueError("Provide --t5 explicitly or set --checkpoint-root / SIERPINSKICAM_CHECKPOINT_DIR.")
    if not os.path.exists(args.t5):
        raise FileNotFoundError(f"--t5 does not exist: {args.t5}")
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    if not scenes:
        raise ValueError("--scenes must contain at least one scene name")
    if args.prompt_file and not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f"--prompt-file does not exist: {args.prompt_file}")
    if args.check_only:
        print("check-only passed")
        print(f"  output_dir: {args.output_dir}")
        print(f"  scenes: {scenes}")
        return

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

    import torch
    from safetensors.torch import save_file

    from musubi_tuner.hv_train_network import load_prompts
    from musubi_tuner.wan.configs import wan_t2v_14B
    from musubi_tuner.wan.modules.t5 import T5EncoderModel

    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompts = load_prompts(args.prompt_file)
        if args.prompt_index < 0 or args.prompt_index >= len(prompts):
            raise IndexError(f"--prompt-index {args.prompt_index} outside prompt file range 0..{len(prompts)-1}")
        prompt = prompts[args.prompt_index].get("prompt", "")

    config = wan_t2v_14B.t2v_14B
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading T5 on {device}: {args.t5}")
    text_encoder = T5EncoderModel(text_len=config.text_len, dtype=config.t5_dtype, device=device, weight_path=args.t5, fp8=args.fp8_t5)
    with torch.no_grad():
        ctx = text_encoder([prompt], device)[0].detach().to("cpu")

    for scene in scenes:
        out = output_dir / f"{scene}_wan_te.safetensors"
        save_file({"varlen_t5_bfloat16": ctx}, str(out))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
