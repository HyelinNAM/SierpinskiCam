# Third-party code and attribution

This release is built from a Musubi Tuner working tree plus SierpinskiCam-specific scripts.

- Code under `src/musubi_tuner/wan` is modified from Wan2.1/Wan components and is Apache-2.0 licensed according to the upstream Musubi documentation.
- Code under `src/musubi_tuner/hunyuan_model`, `frame_pack`, `qwen_image`, and other Musubi modules follows the attribution and licenses documented in `docs/musubi_tuner_README.md` and the per-file headers.
- Some modules are copied or adapted from Hugging Face Diffusers or related projects, as noted in file headers.
- SierpinskiCam-specific scripts in `scripts/` are provided for the paper release pipeline.

Model checkpoints are not distributed in this repository. Users must follow the license terms of each externally downloaded checkpoint.
