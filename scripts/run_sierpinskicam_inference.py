import argparse
import glob
import os
import sys


def existing_path(path: str, kind: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one or more SierpinskiCam/Wan inference samples with explicit public paths."
    )
    parser.add_argument(
        "--base-path",
        required=True,
        help="Conditioning directory for one camera, containing rgb/, img/, and the selected guidance folder (usually dense_tx/).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated latents/videos.")
    parser.add_argument("--guidance", default="dense_tx", help="Conditioning folder under --base-path.")
    parser.add_argument("--reference", default="rgb", help="Reference-video folder under --base-path.")
    parser.add_argument("--te-cache", default=None, help="Optional text-encoder cache folder containing <video>_wan_te.safetensors files.")
    parser.add_argument(
        "--prompt-file",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "prompts", "example_prompt.txt")),
        help="Prompt file consumed by Musubi/Wan sampling utilities.",
    )
    parser.add_argument("--checkpoint-root", default=os.environ.get("SIERPINSKICAM_CHECKPOINT_DIR"), help="Optional root containing vae/, text_encoders/, diffusion_models/, and lora/ checkpoints.")
    parser.add_argument("--vae", default=None, help="Wan VAE checkpoint path. Overrides --checkpoint-root derived path.")
    parser.add_argument("--t5", default=None, help="UMT5 text encoder checkpoint path. Overrides --checkpoint-root derived path.")
    parser.add_argument("--dit", default=None, help="Wan Fun-Control DiT checkpoint path. Overrides --checkpoint-root derived path.")
    parser.add_argument("--network-weights", default=None, help="SierpinskiCam LoRA/network checkpoint path. Overrides --checkpoint-root derived path.")
    parser.add_argument("--task", default="i2v-14B-FC")
    parser.add_argument("--pe-mode", default="nf")
    parser.add_argument("--offset-pe", type=int, default=0)
    parser.add_argument("--lora-multiplier", type=float, default=0.95)
    parser.add_argument("--max-videos", type=int, default=1)
    parser.add_argument("--only-video", default=None)
    parser.add_argument("--sample-steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frame-count", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks-to-swap", type=int, default=18)
    parser.add_argument("--no-decode", action="store_true", help="Save latent only.")
    parser.add_argument("--check-only", action="store_true", help="Check paths and exit before CUDA/model imports.")
    return parser.parse_args()


def fill_checkpoint_paths(args):
    if args.checkpoint_root:
        derived = {
            "vae": os.path.join(args.checkpoint_root, "vae", "wan_2.1_vae.safetensors"),
            "t5": os.path.join(args.checkpoint_root, "text_encoders", "umt5-xxl-enc-fp8_e4m3fn-kijai.safetensors"),
            "dit": os.path.join(args.checkpoint_root, "diffusion_models", "Wan2.1-Fun-Control-14B_fp8_e4m3fn.safetensors"),
            "network_weights": os.path.join(args.checkpoint_root, "lora", "sierpinskicam.safetensors"),
        }
        for key, value in derived.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    missing = [name for name in ("vae", "t5", "dit", "network_weights") if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "Missing checkpoint paths: "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
            + ". Provide them explicitly or set --checkpoint-root / SIERPINSKICAM_CHECKPOINT_DIR."
        )
    return args

def check_inputs(args):
    args = fill_checkpoint_paths(args)
    existing_path(args.base_path, "base path")
    existing_path(os.path.join(args.base_path, args.reference), "reference folder")
    existing_path(os.path.join(args.base_path, args.guidance), "guidance folder")
    existing_path(os.path.join(args.base_path, "img"), "image folder")
    if args.te_cache:
        existing_path(args.te_cache, "text-encoder cache folder")
    existing_path(args.prompt_file, "prompt file")
    existing_path(args.vae, "VAE checkpoint")
    existing_path(args.t5, "T5 checkpoint")
    existing_path(args.dit, "DiT checkpoint")
    existing_path(args.network_weights, "SierpinskiCam LoRA/network checkpoint")

    video_files = sorted(glob.glob(os.path.join(args.base_path, args.reference, "*.mp4")))
    if args.only_video:
        stem = os.path.splitext(os.path.basename(args.only_video))[0]
        video_files = [v for v in video_files if os.path.splitext(os.path.basename(v))[0] == stem]
    if not video_files:
        raise FileNotFoundError(f"No reference videos found in {args.base_path}/{args.reference}")

    selected = video_files[: args.max_videos]
    print("SierpinskiCam inference inputs:")
    print(f"  base_path: {args.base_path}")
    print(f"  reference: {args.reference} ({len(video_files)} videos available)")
    print(f"  guidance: {args.guidance}")
    print(f"  selected: {[os.path.splitext(os.path.basename(v))[0] for v in selected]}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  te_cache: {args.te_cache or '(disabled)'}")
    return selected

def main():
    args = parse_args()
    selected_video_files = check_inputs(args)
    if args.check_only:
        print("check-only passed")
        return

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

    import torch
    from safetensors.torch import load_file, save_file

    from musubi_tuner.hv_train_network import (
        clean_memory_on_device,
        prepare_accelerator,
        load_prompts,
        read_config_from_file,
        save_videos_grid,
        setup_parser_common,
    )
    import musubi_tuner.networks.lora_wan as network_module
    from musubi_tuner.wan_train_network_koncat import WanNetworkTrainer, wan_setup_parser

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this shell. Run this script inside a GPU session/job.")

    trainer_parser = setup_parser_common()
    trainer_parser = wan_setup_parser(trainer_parser)
    trainer_args, _ = trainer_parser.parse_known_args([])
    trainer_args = read_config_from_file(trainer_args, trainer_parser)

    trainer_args.task = args.task
    trainer_args.vae = args.vae
    trainer_args.t5 = args.t5
    trainer_args.dit = args.dit
    trainer_args.network_weights = args.network_weights
    trainer_args.fp8_base = args.task == "i2v-14B-FC"
    trainer_args.offset_pe = args.offset_pe
    trainer_args.pe_mode = args.pe_mode
    trainer_args.sample_prompts = args.prompt_file
    trainer_args.blocks_to_swap = args.blocks_to_swap
    trainer_args.mixed_precision = "bf16"
    trainer_args.sdpa = True
    trainer_args.seed = args.seed
    trainer_args.network_module = "musubi_tuner.networks.lora_wan"

    output_latent_dir = os.path.join(args.output_dir, "latent")
    os.makedirs(output_latent_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = WanNetworkTrainer()
    trainer.handle_model_specific_args(trainer_args)
    accelerator = prepare_accelerator(trainer_args)

    vae = trainer.load_vae(trainer_args, vae_dtype=torch.float32, vae_path=trainer_args.vae)
    vae.requires_grad_(False)
    vae.eval()

    attn_mode = "torch"
    blocks_to_swap = trainer_args.blocks_to_swap if trainer_args.blocks_to_swap else 0
    loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

    transformer = trainer.load_transformer(
        accelerator,
        trainer_args,
        trainer_args.dit,
        attn_mode,
        trainer_args.split_attn,
        loading_device,
        torch.float8_e4m3fn if trainer_args.fp8_base else torch.bfloat16,
    )
    transformer.eval()
    transformer.requires_grad_(False)

    weights_sd = load_file(trainer_args.network_weights)
    network = network_module.create_arch_network_from_weights(
        args.lora_multiplier, weights_sd, unet=transformer, for_inference=True
    )
    network.merge_to(None, transformer, weights_sd, device=accelerator.device, non_blocking=True)

    if blocks_to_swap > 0:
        transformer.enable_block_swap(blocks_to_swap, accelerator.device, supports_backward=True)
        transformer.move_to_device_except_swap_blocks(accelerator.device)

    all_selected_caches_exist = bool(args.te_cache) and all(
        os.path.exists(os.path.join(args.te_cache, f"{os.path.splitext(os.path.basename(video_file))[0]}_wan_te.safetensors"))
        for video_file in selected_video_files
    )
    if all_selected_caches_exist:
        print("Using precomputed text-encoder caches for all selected videos; skipping live T5 prompt encoding.")
        sample_parameters = load_prompts(trainer_args.sample_prompts)
    else:
        sample_parameters = trainer.process_sample_prompts(trainer_args, accelerator, trainer_args.sample_prompts)
    sample_parameter = sample_parameters[0]

    default_t5_embeds = sample_parameter.get("t5_embeds", None)

    for index, video_file in enumerate(selected_video_files):
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f"Processing {index + 1}/{len(selected_video_files)}: {video_name}")

        cache_path = os.path.join(args.te_cache, f"{video_name}_wan_te.safetensors") if args.te_cache else None
        if cache_path and os.path.exists(cache_path):
            data = load_file(cache_path)
            sample_parameter["t5_embeds"] = data["varlen_t5_bfloat16"]
        elif default_t5_embeds is not None:
            sample_parameter["t5_embeds"] = default_t5_embeds
        else:
            sample_parameter.pop("t5_embeds", None)

        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(args.seed)

        transformer = accelerator.unwrap_model(transformer)
        transformer.switch_block_swap_for_inference()

        inference_args = {
            "accelerator": accelerator,
            "args": trainer_args,
            "sample_parameter": sample_parameter,
            "vae": vae,
            "dit_dtype": torch.float32,
            "transformer": transformer,
            "discrete_flow_shift": sample_parameter.get("discrete_flow_shift", 14.5),
            "sample_steps": args.sample_steps,
            "width": args.width,
            "height": args.height,
            "frame_count": args.frame_count,
            "generator": generator,
            "do_classifier_free_guidance": sample_parameter.get("negative_prompt", None) is not None,
            "guidance_scale": sample_parameter.get("guidance_scale", 5),
            "cfg_scale": sample_parameter.get("cfg_scale", None),
            "image_path": os.path.join(args.base_path, "img", f"{video_name}.jpg"),
            "control_video_path": os.path.join(args.base_path, args.guidance, f"{video_name}.mp4"),
            "ref_video_path": video_file,
            "control_end": 1,
        }

        latent_path = os.path.join(output_latent_dir, f"{video_name}.safetensors")
        if os.path.exists(latent_path) and os.path.getsize(latent_path) > 0:
            print(f"Skip existing latent: {latent_path}")
        else:
            with torch.no_grad(), accelerator.autocast():
                latent = trainer.do_inference(output_mode="latent", **inference_args)
                save_file(latent, latent_path)
                del latent
                clean_memory_on_device(accelerator.device)

        if args.no_decode:
            continue

        vae.to(accelerator.device)
        data = load_file(latent_path)
        latent = data["latent"].to(accelerator.device).unsqueeze(0)
        with accelerator.autocast(), torch.no_grad():
            video = vae.decode(latent)[0]
        video = video.unsqueeze(0).to(torch.float32).cpu()
        video = (video / 2 + 0.5).clamp(0, 1)
        save_videos_grid(video, os.path.join(args.output_dir, f"{video_name}.mp4"), fps=12)


if __name__ == "__main__":
    main()
