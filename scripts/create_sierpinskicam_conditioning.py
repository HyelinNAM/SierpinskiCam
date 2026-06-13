import os
import tempfile
import random
import json
from pathlib import Path
import argparse

import sys

# Requires Depth-Anything-V3 and Trajectory-Crafter
# for DAVIS dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_PATH = REPO_ROOT / "example_test_data" / "cameras" / "camera_extrinsics.json"
DEFAULT_INPUT_BASE = REPO_ROOT / "example_test_data" / "input_videos"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "data" / "conditioning"
DEFAULT_TEXTURE_PATH = REPO_ROOT / "example_test_data" / "textures" / "sierpinski_dome_16x16_2048.png"
DEFAULT_SCENES = "01,02,03,04,05"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

class ChessPatternSphereRenderer:
    def __init__(self, sphere_radius: float, texture_type: str = "chess", squares_lat: int = 12, 
                 squares_lon: int = 12, texture_size: int = 1024, cache_dir: str = "texture_cache",
                 texture_path: str | None = None):
        self.sphere_radius = sphere_radius
        self.texture_type = texture_type
        self.squares_lat = squares_lat
        self.squares_lon = squares_lon
        self.texture_size = texture_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.texture_path = Path(texture_path) if texture_path else None
        self.texture = None
        self._load_texture()
        
    def _generate_chess_texture(self):
        texture = np.full((self.texture_size, self.texture_size, 3), 0, dtype=np.uint8)
        if self.texture_type == "sierpinski2":
            def draw_triangle(y, x, size, upward=True):
                height = size
                for row in range(height):
                    if upward:
                        start = x + size // 2 - row * size // height // 2
                        end = x + size // 2 + row * size // height // 2
                        texture[y + row, start:end] = [40, 40, 40]
                    else:
                        start = x + size // 2 - (height - 1 - row) * size // height // 2
                        end = x + size // 2 + (height - 1 - row) * size // height // 2
                        texture[y + row, start:end] = [40, 40, 40]

            def sierpinski(y, x, size, depth, upward=True):
                if depth == 0:
                    draw_triangle(y, x, size, upward)
                    return
                half = size // 2
                height = half
                if upward:
                    sierpinski(y, x + half // 2, half, depth - 1, upward)
                    sierpinski(y + height, x, half, depth - 1, upward)
                    sierpinski(y + height, x + half, half, depth - 1, upward)
                else:
                    sierpinski(y + height, x + half // 2, half, depth - 1, upward)
                    sierpinski(y, x, half, depth - 1, upward)
                    sierpinski(y, x + half, half, depth - 1, upward)

            grid_step = self.texture_size // self.squares_lat
            max_depth = 3
            for lat_i in range(self.squares_lat):
                for lon_i in range(self.squares_lon):
                    y = lat_i * grid_step
                    x = lon_i * grid_step
                    mod = lat_i % 4

                    if mod == 0:
                        sierpinski(y, x, grid_step, max_depth, True)
                        if lon_i % 2 == 0:
                            sierpinski(y, x - grid_step // 2, grid_step, max_depth, False)
                    elif mod == 1:
                        sierpinski(y, x, grid_step, max_depth, False)
                        if lon_i % 2 == 0:
                            sierpinski(y, x - grid_step // 2, grid_step, max_depth, True)
                    elif mod == 2:
                        sierpinski(y, x, grid_step, max_depth, True)
                        if lon_i % 2 == 1:
                            sierpinski(y, x - grid_step // 2, grid_step, max_depth, False)
                    else:
                        sierpinski(y, x, grid_step, max_depth, False)
                        if lon_i % 2 == 1:
                            sierpinski(y, x - grid_step // 2, grid_step, max_depth, True)
                        
            return texture
        
        lat_step = self.texture_size // self.squares_lat
        lon_step = self.texture_size // self.squares_lon
        
        for lat_i in range(self.squares_lat):
            for lon_i in range(self.squares_lon):
                y_start = lat_i * lat_step
                y_end = (lat_i + 1) * lat_step
                x_start = lon_i * lon_step
                x_end = (lon_i + 1) * lon_step
                
                is_colored = (lat_i + lon_i) % 2 == 1
                if is_colored:
                    texture[y_start:y_end, x_start:x_end] = [40, 40, 40]
        
        return texture
    
    def _load_texture(self):
        if self.texture_path is not None and self.texture_path.exists():
            self.texture = cv2.imread(str(self.texture_path))
            if self.texture is None:
                raise ValueError(f"Could not read texture image: {self.texture_path}")
            return

        filename = f"chess_{self.squares_lat}x{self.squares_lon}_{self.texture_type}.png"
        filepath = self.cache_dir / filename

        if filepath.exists():
            self.texture = cv2.imread(str(filepath))
            if self.texture is None:
                raise ValueError(f"Could not read cached texture image: {filepath}")
        else:
            self.texture = self._generate_chess_texture()
            cv2.imwrite(str(filepath), self.texture)

    def render(self, canvas, pose, K):
        device = pose.device
        height, width = canvas.shape[:2]
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        x_cam = (x_grid - K[0, 2]) / K[0, 0]
        y_cam = (y_grid - K[1, 2]) / K[1, 1]
        z_cam = torch.ones_like(x_cam)
        
        ray_dirs = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        ray_dirs_world = (pose[:3, :3].T @ ray_dirs.view(-1, 3).T).T.view(height, width, 3)
        cam_pos = -pose[:3, :3].T @ pose[:3, 3]

        a = torch.sum(ray_dirs_world**2, dim=-1)
        b = 2 * torch.sum(cam_pos.unsqueeze(0).unsqueeze(0) * ray_dirs_world, dim=-1)
        c = (torch.sum(cam_pos**2) - self.sphere_radius**2)
        
        discriminant = b**2 - 4*a*c
        valid_intersection = discriminant >= 0
        
        sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0))
        t1 = (-b - sqrt_disc) / a / 2
        t2 = (-b + sqrt_disc) / a / 2
        
        t = torch.where((t1 > 0) & (t2 > 0), torch.min(t1, t2), 
                    torch.where(t1 > 0, t1, t2))
        t = torch.where(valid_intersection & (t > 0), t, torch.inf)
        
        intersection_points = cam_pos.unsqueeze(0).unsqueeze(0) + t.unsqueeze(-1) * ray_dirs_world
        
        x_world = intersection_points[:, :, 0]
        y_world = intersection_points[:, :, 1]
        z_world = intersection_points[:, :, 2]
        
        r = torch.sqrt(x_world**2 + y_world**2 + z_world**2)
        lat = torch.acos(torch.clamp(y_world / r, -1, 1))
        lon = torch.atan2(z_world, x_world)
        lon = torch.where(lon < 0, lon + 2 * np.pi, lon)
        
        u = (lon / (2 * np.pi) * (self.texture.shape[1] - 1)).long()
        v = (lat / np.pi * (self.texture.shape[0] - 1)).long()
        
        u = torch.clamp(u, 0, self.texture.shape[1] - 1)
        v = torch.clamp(v, 0, self.texture.shape[0] - 1)
        
        texture_tensor = torch.tensor(self.texture, device=device)
        sampled_colors = texture_tensor[v, u]
        
        result = canvas.copy()
        mask = valid_intersection & (t != torch.inf)
        if mask.any():
            result[mask.cpu().numpy()] = sampled_colors[mask].cpu().numpy()

        return result

def save_frame(frame, temp_dir, video_file, frame_idx, target_size=None):
    if target_size is not None:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    frame_path = os.path.join(temp_dir, f"{os.path.splitext(video_file)[0]}_frame{frame_idx:06d}.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path

def read_video_frames(video_path, max_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def load_model(device, model_id="depth-anything/DA3NESTED-GIANT-LARGE"):
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
    return model


def to_homogeneous(matrices):
    n = matrices.shape[0]
    out = np.zeros((n, 4, 4), dtype=matrices.dtype)
    out[:, :3, :4] = matrices
    out[:, 3, 3] = 1.0
    return out

def save_prediction(prediction, output_path):
    save_data_npz = {
        'depth_maps': prediction.depth[0],
        'conf': prediction.conf[0],
        'extrinsics': prediction.extrinsics,
        'intrinsics': prediction.intrinsics,
    }
    np.savez_compressed(output_path, **save_data_npz)

def save_video(frames, output_path, fps=12):
    if len(frames) == 0:
        return

    frames_np = []
    for f in frames:
        if isinstance(f, np.ndarray):
            frames_np.append(f)
        else:
            raise ValueError("All frames must be numpy arrays")

    import moviepy.editor as mpy

    clip = mpy.ImageSequenceClip(frames_np, fps=fps)
    clip.write_videofile(output_path, codec='libx264')

def warp_frames(funwarp, frames, depths, pose_s, pose_t, K):
    warped_images = []
    masks = []
    video_length = frames.shape[0]
    
    for i in range(video_length):
        warped_frame2, mask2, warped_depth2, flow12 = funwarp.forward_warp(
            frames[i : i + 1],
            None,
            depths[i : i + 1],
            pose_s[i : i + 1],
            pose_t[i : i + 1],
            K[i : i + 1],
            None,
            False,
            twice=False,
        )
        warped_images.append(warped_frame2)
        masks.append(mask2)
    
    cond_video = (torch.cat(warped_images) + 1.0) / 2.0 * 255
    cond_masks = torch.cat(masks)
    
    return cond_video, cond_masks

def preprocess_fixed(pred, max_size=832):
    imgs = torch.from_numpy(pred.processed_images).permute(0, 3, 1, 2) / 255 * 2 - 1
    depth = torch.from_numpy(pred.depth[:, None])
    c2w = torch.from_numpy(to_homogeneous(pred.extrinsics))
    intrs = torch.from_numpy(pred.intrinsics)
    
    N, C, H, W = imgs.shape
    scale = max_size / max(H, W)
    if scale != 1.0:
        new_H = int(H * scale)
        new_W = int(W * scale)
        imgs = F.interpolate(imgs, size=(new_H, new_W), mode='bilinear', align_corners=False)
        depth = F.interpolate(depth, size=(new_H, new_W), mode='nearest')
        
        intrs = intrs.clone()
        intrs[:, 0, 0] *= scale
        intrs[:, 1, 1] *= scale
        intrs[:, 0, 2] *= scale
        intrs[:, 1, 2] *= scale
    
    return imgs, depth, c2w, intrs

def build_warp_arg(imgs, depth, c2w, traj, intrs, n=49):
    a = slice(0, n)

    first = (imgs[a], depth[a], c2w[a], traj[a], intrs[a])

    return first

def smooth_camera_path(extrinsics, intrs, alpha=0.5):
    T = extrinsics.shape[0]
    dtype = extrinsics.dtype
    
    extr_4x4 = np.tile(np.eye(4, dtype=dtype), (T, 1, 1))
    extr_4x4[:, :3, :4] = extrinsics[:, :3, :4]
    
    first_idx = 0
    mid_idx = T // 2
    last_idx = T - 1
    
    key_poses = extr_4x4[[first_idx, mid_idx, last_idx]]
    
    t = np.linspace(0, 1, T, dtype=dtype)
    
    def cubic_interpolate(p0, p1, p2, t):
        L0 = 2 * (t - 0.5) * (t - 1)
        L1 = -4 * t * (t - 1)
        L2 = 2 * t * (t - 0.5)
        return L0[:, None] * p0 + L1[:, None] * p1 + L2[:, None] * p2
    
    Rs = key_poses[:, :3, :3]
    ts = key_poses[:, :3, 3]
    
    t_smooth = cubic_interpolate(ts[0], ts[1], ts[2], t)
    
    def slerp_segment(R0, R1, t_val):
        R_rel = R0.T @ R1
        trace = np.trace(R_rel)
        theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if abs(theta) < 1e-6:
            return R0
        
        axis_skew = (R_rel - R_rel.T) / (2 * np.sin(theta))
        theta_t = theta * t_val
        R_t = R0 @ (np.eye(3, dtype=dtype) + 
                    np.sin(theta_t) * axis_skew + 
                    (1 - np.cos(theta_t)) * (axis_skew @ axis_skew))
        return R_t
    
    R_smooth = []
    for i in range(T):
        if i <= mid_idx:
            t_local = i / mid_idx if mid_idx > 0 else 0
            R_smooth.append(slerp_segment(Rs[0], Rs[1], t_local))
        else:
            t_local = (i - mid_idx) / (T - 1 - mid_idx) if T - 1 - mid_idx > 0 else 0
            R_smooth.append(slerp_segment(Rs[1], Rs[2], t_local))
    R_smooth = np.stack(R_smooth)
    
    extr_smooth = np.tile(np.eye(4, dtype=dtype), (T, 1, 1))
    extr_smooth[:, :3, :3] = R_smooth
    extr_smooth[:, :3, 3] = t_smooth
    
    extr_blended = alpha * extr_smooth + (1 - alpha) * extr_4x4
    
    intr_avg = intrs.mean(axis=0, keepdims=True)
    intr_avg = np.tile(intr_avg, (T, 1, 1))
    intr_blended = alpha * intr_avg + (1 - alpha) * intrs
    
    return extr_blended[:, :3, :4].astype(dtype), intr_blended.astype(intrs.dtype)

def render_dome_video(prediction, chess_renderer, target_height=480, target_width=832):
    frames = []
    
    #extrinsics, intrinsics = smooth_camera_path(prediction.extrinsics, prediction.intrinsics, 0.8)
    extrinsics, intrinsics = prediction.extrinsics, prediction.intrinsics
    max_len = min(len(prediction.extrinsics),49)
    for idx in range(max_len):
        pose = torch.from_numpy(extrinsics[idx])
        K = torch.from_numpy(intrinsics[idx]).clone()
        canvas = chess_renderer.render(
            np.zeros((target_height, target_width, 3), dtype=np.uint8),
            pose,
            K
        )
        frames.append(canvas)
    
    return np.array(frames)

def collect_sorted_frames(folder_path, limit=None):
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    imgs.sort()
    base = [os.path.join(folder_path, f) for f in imgs]

    if limit is None:
        return base
    if len(base) >= limit:
        return base[:limit]

    return base


def find_scene_video(input_base, scene):
    scene_path = os.path.join(input_base, scene)
    if os.path.isfile(scene_path) and scene_path.lower().endswith(VIDEO_EXTENSIONS):
        return scene_path

    stem, ext = os.path.splitext(scene)
    if ext.lower() in VIDEO_EXTENSIONS:
        candidate = os.path.join(input_base, scene)
        return candidate if os.path.isfile(candidate) else None

    for video_ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(input_base, f"{scene}{video_ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def discover_scenes(input_base):
    scenes = []
    for entry in os.listdir(input_base):
        path = os.path.join(input_base, entry)
        if os.path.isdir(path):
            scenes.append(entry)
        elif os.path.isfile(path) and entry.lower().endswith(VIDEO_EXTENSIONS):
            scenes.append(os.path.splitext(entry)[0])
    return sorted(set(scenes))


def collect_video_frames_as_paths(video_path, temp_dir, limit):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {video_path}")

    frame_paths = []
    scene_stem = os.path.splitext(os.path.basename(video_path))[0]
    while len(frame_paths) < limit:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_dir, f"{scene_stem}_frame{len(frame_paths):06d}.jpg")
        if not cv2.imwrite(frame_path, frame):
            cap.release()
            raise ValueError(f"Could not write temporary frame: {frame_path}")
        frame_paths.append(frame_path)
    cap.release()
    return frame_paths


def collect_input_frame_paths(input_base, scene, frame_count, temp_dir):
    frame_dir = os.path.join(input_base, scene)
    if os.path.isdir(frame_dir):
        return collect_sorted_frames(frame_dir, limit=frame_count)

    video_path = find_scene_video(input_base, scene)
    if video_path is not None:
        return collect_video_frames_as_paths(video_path, temp_dir, frame_count)

    raise FileNotFoundError(
        f"Could not find scene '{scene}' under {input_base}. "
        "Expected either a video file such as <scene>.mp4 or a frame directory named <scene>."
    )

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

def get_c2w(w2cs, transform_matrix, relative_c2w=True):
    if relative_c2w:
        target_cam_c2w = np.eye(4)
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
    return np.array([transform_matrix @ x for x in ret_poses], dtype=np.float32)


def max_point_distance(prediction, conf_threshold=0.5):
    N, H, W = prediction.depth.shape
    first_extrinsic = prediction.extrinsics[0]  # [3,4]
    
    first_extrinsic_4x4 = np.eye(4, dtype=np.float32)
    first_extrinsic_4x4[:3, :4] = first_extrinsic[:3, :4]
    world_to_first = np.linalg.inv(first_extrinsic_4x4)
    
    max_dist = 0.0
    
    for i in range(N):
        depth = prediction.depth[i]        # [H,W]
        conf = prediction.conf[i]          # [H,W]
        K = prediction.intrinsics[i]       # [3,3]
        extrinsic = prediction.extrinsics[i]  # [3,4]

        mask = conf > conf_threshold
        if not np.any(mask):
            continue
        
        ys, xs = np.nonzero(mask)
        z = depth[ys, xs]
        x = (xs - K[0, 2]) * z / K[0, 0]
        y = (ys - K[1, 2]) * z / K[1, 1]
        
        points_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # [num_points,4]
        
        extrinsic_4x4 = np.eye(4, dtype=np.float32)
        extrinsic_4x4[:3, :4] = extrinsic[:3, :4]
        points_world = (extrinsic_4x4 @ points_cam.T).T  # [num_points,4]
        
        points_aligned = (world_to_first @ points_world.T).T[:, :3]
        
        distances = np.linalg.norm(points_aligned, axis=1)
        max_dist = max(max_dist, distances.max())
    
    return max_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SierpinskiCam conditioning videos from input videos or image-sequence folders.")
    parser.add_argument(
        "--input-base",
        default=str(DEFAULT_INPUT_BASE),
        help="Directory containing scene videos (<scene>.mp4) or one frame directory per scene.",
    )
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE), help="Output directory for camera-specific rgb/dense/mask/dome/dense_tx folders.")
    parser.add_argument("--camera-path", default=str(DEFAULT_CAMERA_PATH), help="ReCamMaster-format camera JSON. Defaults to example_test_data/cameras/camera_extrinsics.json.")
    parser.add_argument("--trajectorycrafter-models", default=os.environ.get("TRAJECTORYCRAFTER_MODELS"), help="Path to TrajectoryCrafter/models containing utils.Warper. Can also be set with TRAJECTORYCRAFTER_MODELS.")
    parser.add_argument("--da3-model-id", default="depth-anything/DA3NESTED-GIANT-LARGE", help="Depth-Anything-3 model id or local path.")
    parser.add_argument("--camera-names", default="cam01", help="Comma-separated camera names to render. Defaults to cam01 for a quick smoke run; use cam01,...,cam14 for all provided paths.")
    parser.add_argument(
        "--scenes",
        default=DEFAULT_SCENES,
        help=(
            "Comma-separated scene names to process. Defaults to the five provided videos under "
            "example_test_data/input_videos; use --scenes all to process every discovered video/folder."
        ),
    )
    parser.add_argument("--save-outputs", default="rgb,dense_tx", help="Comma-separated outputs among rgb,dense,mask,dome,dense_tx.")
    parser.add_argument("--frame-count", type=int, default=49, help="Number of input frames used per scene.")
    parser.add_argument("--texture-path", default=str(DEFAULT_TEXTURE_PATH), help="Sierpinski dome texture image. Defaults to the self-contained example_test_data texture asset.")
    parser.add_argument("--texture-cache-dir", default="data/texture_cache", help="Local cache for generated Sierpinski dome textures.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda or cpu. Defaults to cuda if available.")
    parser.add_argument("--pad-short-scenes", action="store_true", help="Repeat the final frame if a scene is shorter than --frame-count.")
    parser.add_argument("--check-only", action="store_true", help="Validate paths/arguments and exit before model imports.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_base):
        raise FileNotFoundError(f"--input-base does not exist or is not a directory: {args.input_base}")
    if not os.path.isfile(args.camera_path):
        raise FileNotFoundError(f"--camera-path does not exist: {args.camera_path}")
    if args.texture_path and not os.path.isfile(args.texture_path):
        raise FileNotFoundError(f"--texture-path does not exist: {args.texture_path}")
    valid_outputs = {"rgb", "dense", "mask", "dome", "dense_tx"}
    folder_names = [name.strip() for name in args.save_outputs.split(",") if name.strip()]
    unknown_outputs = sorted(set(folder_names) - valid_outputs)
    if unknown_outputs:
        raise ValueError(f"Unknown --save-outputs entries: {unknown_outputs}. Valid: {sorted(valid_outputs)}")
    if not folder_names:
        raise ValueError("--save-outputs must include at least one output name")

    if args.camera_names:
        camera_names = [name.strip() for name in args.camera_names.split(",") if name.strip()]
    else:
        camera_names = [f"cam{i:02d}" for i in range(1, 11)]

    if args.scenes and args.scenes.lower() not in {"all", "none"}:
        scenes = [name.strip() for name in args.scenes.split(",") if name.strip()]
    else:
        scenes = discover_scenes(args.input_base)
    if not scenes:
        raise ValueError(f"No scenes found under --input-base: {args.input_base}")
    missing_scenes = [
        scene for scene in scenes
        if not os.path.isdir(os.path.join(args.input_base, scene)) and find_scene_video(args.input_base, scene) is None
    ]
    if missing_scenes:
        raise FileNotFoundError(f"Requested scenes not found under --input-base: {missing_scenes}")

    with open(args.camera_path, "r") as file:
        cam_data_for_check = json.load(file)
    missing = [name for name in camera_names if name not in cam_data_for_check.get("frame0", {})]
    if missing:
        raise ValueError(f"Camera names missing from camera path frame0: {missing}")

    if args.check_only:
        print("check-only passed")
        print(f"  input_base: {args.input_base}")
        print(f"  output_base: {args.output_base}")
        print(f"  camera_path: {args.camera_path}")
        print(f"  texture_path: {args.texture_path or '(generate procedurally)'}")
        print(f"  cameras: {camera_names}")
        print(f"  scenes: {scenes}")
        print(f"  outputs: {folder_names}")
        sys.exit(0)

    if args.trajectorycrafter_models is None:
        raise ValueError("Provide --trajectorycrafter-models or set TRAJECTORYCRAFTER_MODELS to the TrajectoryCrafter/models directory.")
    if not os.path.isdir(args.trajectorycrafter_models):
        raise FileNotFoundError(f"TrajectoryCrafter models path does not exist: {args.trajectorycrafter_models}")

    global cv2, torch, F, np
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.append(args.trajectorycrafter_models)
    from utils import Warper

    try:
        from fastprogress import progress_bar
    except Exception:
        progress_bar = lambda x: x

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    model = load_model(device, args.da3_model_id)
    funwarp = Warper(device=device.type)
    chess_renderer = ChessPatternSphereRenderer(
        squares_lat=16,
        squares_lon=16,
        sphere_radius=30,
        texture_size=2048,
        texture_type="sierpinski2",
        cache_dir=args.texture_cache_dir,
        texture_path=args.texture_path,
    )

    output_folders = {}
    for cam_name in camera_names:
        output_folders[cam_name] = {}
        for folder_name in folder_names:
            path = os.path.join(args.output_base, cam_name, folder_name)
            os.makedirs(path, exist_ok=True)
            output_folders[cam_name][folder_name] = path
        img_path = os.path.join(args.output_base, cam_name, "img")
        os.makedirs(img_path, exist_ok=True)
        output_folders[cam_name]["img"] = img_path

    os.makedirs(os.path.join(args.output_base, "cam"), exist_ok=True)

    frame_count = args.frame_count
    with open(args.camera_path, "r") as file:
        cam_data = json.load(file)

    traj_all = {}
    for cam_name in camera_names:
        cameras = [parse_matrix(cam_data[f"frame{i}"][cam_name]) for i in range(frame_count)]
        cameras = np.transpose(np.stack(cameras), (0, 2, 1))

        w2cs = []
        for cam in cameras:
            if cam.shape[0] == 3:
                cam = np.vstack((cam, np.array([[0, 0, 0, 1]])))
            cam = cam[:, [1, 2, 0, 3]]
            cam[:3, 1] *= -1.
            w2cs.append(np.linalg.inv(cam))

        true_c2w = get_c2w(w2cs, np.eye(4), True)
        for c2w in true_c2w:
            c2w[:3, 3] *= 1 / 100
        traj_all[cam_name] = np.linalg.inv(true_c2w)

    skip_output_name = "dense_tx" if "dense_tx" in folder_names else folder_names[0]
    for scene in progress_bar(scenes):
        expected_output_file = os.path.join(output_folders[camera_names[-1]][skip_output_name], f"{scene}.mp4")

        if os.path.exists(expected_output_file):
            continue

        with tempfile.TemporaryDirectory(prefix=f"sierpinskicam_{scene}_") as temp_dir:
            frame_paths = collect_input_frame_paths(args.input_base, scene, frame_count, temp_dir)
            if len(frame_paths) < frame_count:
                if args.pad_short_scenes and frame_paths:
                    print(f"Padding short scene {scene}: {len(frame_paths)} -> {frame_count}", flush=True)
                    frame_paths = frame_paths + [frame_paths[-1]] * (frame_count - len(frame_paths))
                else:
                    print(f"Skipping short scene {scene}: {len(frame_paths)} < {frame_count}", flush=True)
                    continue

            first_frame = cv2.imread(frame_paths[0])
            if first_frame is None:
                raise ValueError(f"Could not read first frame for scene {scene}: {frame_paths[0]}")
            for cam_name in camera_names:
                cv2.imwrite(os.path.join(output_folders[cam_name]["img"], f"{scene}.jpg"), first_frame)

            prediction = model.inference(frame_paths[:frame_count])
            save_prediction(prediction, os.path.join(args.output_base, f"cam/{scene}.npz"))
            prediction.extrinsics, prediction.intrinsics = smooth_camera_path(prediction.extrinsics, prediction.intrinsics, 0.8)
            imgs, depth, c2ws, intrs = preprocess_fixed(prediction)
            max_distance = max_point_distance(prediction)
            print("max_distance", max_distance)
            chess_renderer.sphere_radius = min(30, max_distance)

            for cam_name in camera_names:
                traj = torch.from_numpy(traj_all[cam_name])
                args0 = build_warp_arg(imgs, depth, c2ws, traj, intrs)
                cond_video, cond_masks = warp_frames(funwarp, *args0)
                prediction.extrinsics = traj_all[cam_name]
                dome_video = render_dome_video(prediction, chess_renderer, cond_video.shape[2], cond_video.shape[3])
                dense_tx_video = cond_masks * cond_video + (1 - cond_masks) * torch.from_numpy(dome_video[:frame_count * 2]).permute(0, 3, 1, 2)

                if "rgb" in folder_names:
                    input_video = ((255 / 2 * (imgs[:frame_count] + 1)).permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
                    save_video(input_video, os.path.join(output_folders[cam_name]["rgb"], f"{scene}.mp4"))
                if "dense" in folder_names:
                    cond_video_np = (cond_video.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
                    save_video(cond_video_np, os.path.join(output_folders[cam_name]["dense"], f"{scene}.mp4"))
                if "mask" in folder_names:
                    cond_masks_np = (cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                    save_video(cond_masks_np, os.path.join(output_folders[cam_name]["mask"], f"{scene}.mp4"))
                if "dome" in folder_names:
                    save_video(dome_video, os.path.join(output_folders[cam_name]["dome"], f"{scene}.mp4"))
                if "dense_tx" in folder_names:
                    dense_tx_video = (dense_tx_video.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
                    save_video(dense_tx_video, os.path.join(output_folders[cam_name]["dense_tx"], f"{scene}.mp4"))
