#!/usr/bin/env python3
"""Generate the default SierpinskiCam camera path JSON.

The output format follows the ReCamMaster-style `camera_path.json` used by the
conditioning script: frame keys map to named camera trajectories (`cam01`, ...).
"""

import argparse
import json
import math
from pathlib import Path

ROT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
POS = [5000, 1500, 100]


def fmt_row(v):
    return f"[{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} 0]"


def fmt_pos(v):
    return f"[{v[0]} {v[1]} {v[2]} 1]"


def pose_string(rot, pos):
    return f"{fmt_row(rot[0])} {fmt_row(rot[1])} {fmt_row(rot[2])} {fmt_pos(pos)}"


def pan_horizontal(frame, degrees, total_frames):
    if frame == 0:
        return pose_string(ROT, POS)
    angle = math.radians(degrees * frame / max(total_frames - 1, 1))
    c, s = math.cos(angle), math.sin(angle)
    r0 = [c * ROT[0][i] + s * ROT[1][i] for i in range(3)]
    r1 = [-s * ROT[0][i] + c * ROT[1][i] for i in range(3)]
    return pose_string([r0, r1, ROT[2]], POS)


def pan_vertical(frame, degrees, total_frames):
    if frame == 0:
        return pose_string(ROT, POS)
    angle = math.radians(degrees * frame / max(total_frames - 1, 1))
    c, s = math.cos(angle), math.sin(angle)
    r0 = [c * ROT[0][0] + s * ROT[0][2], ROT[0][1], -s * ROT[0][0] + c * ROT[0][2]]
    r2 = [c * ROT[2][0] + s * ROT[2][2], ROT[2][1], -s * ROT[2][0] + c * ROT[2][2]]
    return pose_string([r0, ROT[1], r2], POS)


def zoom_by_distance(frame, distance, total_frames):
    if frame == 0:
        return pose_string(ROT, POS)
    t = frame / max(total_frames - 1, 1)
    offset = 100 * distance * t
    new_pos = [POS[i] + offset * ROT[0][i] for i in range(3)]
    return pose_string(ROT, new_pos)


def orbit_around_front_object(frame, degrees, total_frames, object_distance=10):
    if frame == 0:
        return pose_string(ROT, POS)
    object_distance = object_distance * 100
    object_pos = [POS[i] + object_distance * ROT[0][i] for i in range(3)]
    angle = math.radians(degrees * frame / max(total_frames - 1, 1))
    c, s = math.cos(angle), math.sin(angle)
    d = object_distance
    rotated_offset = [(-d) * (c * ROT[0][i] - s * ROT[1][i]) for i in range(3)]
    new_pos = [object_pos[i] + rotated_offset[i] for i in range(3)]
    r0 = [c * ROT[0][i] - s * ROT[1][i] for i in range(3)]
    r1 = [s * ROT[0][i] + c * ROT[1][i] for i in range(3)]
    return pose_string([r0, r1, ROT[2]], new_pos)


def spiral_translation(frame, circles, total_frames, radius_growth=1, zoom=0.0):
    if frame == 0:
        return pose_string(ROT, POS)
    t = frame / max(total_frames - 1, 1)
    angle = 2 * math.pi * circles * t
    radius = 100 * radius_growth * t
    dy = radius * math.cos(angle)
    dz = radius * math.sin(angle)
    zoom_offset = 100 * zoom * t
    new_pos = [POS[0] + zoom_offset * ROT[0][0], POS[1] + dy + zoom_offset * ROT[0][1], POS[2] + dz + zoom_offset * ROT[0][2]]
    return pose_string(ROT, new_pos)


def spiral_rotation(frame, circles, total_frames, degrees=15, zoom=0):
    if frame == 0:
        return pose_string(ROT, POS)
    t = frame / max(total_frames - 1, 1)
    progress = t * circles
    angle_h = math.radians(degrees * t * math.cos(progress * 2 * math.pi))
    angle_v = math.radians(degrees * t * math.sin(progress * 2 * math.pi))
    c_h, s_h = math.cos(angle_h), math.sin(angle_h)
    r0_h = [c_h * ROT[0][i] + -s_h * ROT[1][i] for i in range(3)]
    r1_h = [s_h * ROT[0][i] + c_h * ROT[1][i] for i in range(3)]
    c_v, s_v = math.cos(angle_v), math.sin(angle_v)
    r0 = [c_v * r0_h[0] - s_v * ROT[0][2], r0_h[1], s_v * r0_h[0] + c_v * ROT[0][2]]
    r2 = [c_v * ROT[2][0] - s_v * ROT[2][2], ROT[2][1], s_v * ROT[2][0] + c_v * ROT[2][2]]
    zoom_offset = 100 * zoom * t
    new_pos = [POS[0] + zoom_offset * ROT[0][0], POS[1], POS[2]]
    return pose_string([r0, r1_h, r2], new_pos)


def create_camera_json(total_frames: int):
    camera_data = {}
    for frame in range(total_frames):
        camera_data[f"frame{frame}"] = {
            "cam01": pan_horizontal(frame, 15, total_frames),
            "cam02": pan_horizontal(frame, -30, total_frames),
            "cam03": pan_vertical(frame, 12, total_frames),
            "cam04": pan_vertical(frame, -6, total_frames),
            "cam05": zoom_by_distance(frame, 3, total_frames),
            "cam06": zoom_by_distance(frame, -6, total_frames),
            "cam07": orbit_around_front_object(frame, 8, total_frames),
            "cam08": orbit_around_front_object(frame, -4, total_frames),
            "cam09": spiral_translation(frame, 1.0, total_frames, 1),
            "cam10": spiral_rotation(frame, 1.0, total_frames, 10),
        }
    return camera_data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a SierpinskiCam camera_path.json file.")
    parser.add_argument("--output", type=Path, default=Path("data/camera_path.json"), help="Output JSON path.")
    parser.add_argument("--total-frames", type=int, default=81, help="Number of frames to write. Conditioning uses the first 49 by default.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(create_camera_json(args.total_frames), f, indent=4)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
