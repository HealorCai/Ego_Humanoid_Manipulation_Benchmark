#!/usr/bin/env python3
"""
Project fingertip positions onto dataset images without running simulation.

Example:
python test/project_robot_hand_skeleton_egovla_tip.py \
  --hdf5 /cpfs04/user/caizetao/data/EgoVLA_SIM/Close-Drawer/episode_1.hdf5 \
  --video_out /cpfs04/user/caizetao/projects/benchmark_ws/Ego_Humanoid_Manipulation_Benchmark/output/skel_episode_1.mp4 \
  --max_frames 300 --fps 30

"""

import argparse
import os

import h5py
import imageio.v2 as imageio
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Fixed camera intrinsics/extrinsics (world->camera).
K_DEFAULT = np.array(
    [
        [488.6662, 0.0, 640.0],
        [0.0, 488.6662, 360.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

EXTR_DEFAULT = np.array(
    [
        [-1.1089e-05, -1.0000e+00, -4.0362e-06, 7.8595e-06],
        [-7.6605e-01, 1.1089e-05, -6.4278e-01, 1.1617e+00],
        [6.4278e-01, -4.0362e-06, -7.6605e-01, 1.2444e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00],
    ],
    dtype=np.float32,
)


def _project_points(points_w: torch.Tensor, K: torch.Tensor, extr: torch.Tensor) -> torch.Tensor:
    # points_w: (N, 3), K: (3, 3), extr: (4, 4) world->camera
    ones = torch.ones((points_w.shape[0], 1), device=points_w.device, dtype=points_w.dtype)
    pw_h = torch.cat([points_w, ones], dim=-1)  # (N, 4)
    pc_h = (extr @ pw_h.t()).t()  # (N, 4)
    pc = pc_h[:, :3]
    z = pc[:, 2].clamp(min=1e-6)
    x = pc[:, 0] / z
    y = pc[:, 1] / z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return torch.stack([u, v, z], dim=-1)


def _save_debug_plot(save_path: str, tips_w: torch.Tensor, axis_len: float) -> None:
    pts = tips_w.detach().cpu().numpy()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="g", s=10, label="fingertips")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Fingertips (World)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project fingertip skeleton onto dataset images (no sim).")
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 dataset.")
    parser.add_argument("--hdf5_images", type=str, default="observations/images/main", help="Image dataset path.")
    parser.add_argument("--hdf5_left_tips", type=str, default="observations/left_finger_tip_pos", help="Left tips path.")
    parser.add_argument("--hdf5_right_tips", type=str, default="observations/right_finger_tip_pos", help="Right tips path.")
    parser.add_argument("--video_out", type=str, default=None, help="Path to save overlay video (mp4).")
    parser.add_argument("--max_frames", type=int, default=300, help="Max frames to record. <=0 means no limit.")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS.")
    parser.add_argument("--debug_save_dir", type=str, default=None, help="Save matplotlib 3D debug images to dir.")
    parser.add_argument("--debug_draw_every", type=int, default=1, help="Save debug every N frames.")
    parser.add_argument("--debug_draw_scale", type=float, default=0.1, help="Unused placeholder for compatibility.")
    args = parser.parse_args()

    K = torch.from_numpy(K_DEFAULT)
    extr = torch.from_numpy(EXTR_DEFAULT)

    writer = None
    frame_count = 0

    with h5py.File(args.hdf5, "r") as f:
        if args.hdf5_images not in f:
            raise KeyError(f"HDF5 images dataset not found: {args.hdf5_images}")
        if args.hdf5_left_tips not in f:
            raise KeyError(f"HDF5 left tips dataset not found: {args.hdf5_left_tips}")
        if args.hdf5_right_tips not in f:
            raise KeyError(f"HDF5 right tips dataset not found: {args.hdf5_right_tips}")

        img_ds = f[args.hdf5_images]
        left_tips_ds = f[args.hdf5_left_tips]
        right_tips_ds = f[args.hdf5_right_tips]
        total_frames = min(img_ds.shape[0], left_tips_ds.shape[0], right_tips_ds.shape[0])
        if args.max_frames > 0:
            total_frames = min(total_frames, args.max_frames)

        with tqdm(total=total_frames, desc="Overlay frames") as pbar:
            while frame_count < total_frames:
                left_tips = torch.from_numpy(left_tips_ds[frame_count])
                right_tips = torch.from_numpy(right_tips_ds[frame_count])
                tips_w = torch.cat([left_tips, right_tips], dim=0)  # (10, 3)

                proj = _project_points(tips_w, K, extr)  # (10, 3) u, v, z

                img = img_ds[frame_count].copy()
                img_h, img_w = img.shape[:2]
                sim_w, sim_h = 1280, 720
                sx = img_w / float(sim_w)
                sy = img_h / float(sim_h)

                for i in range(proj.shape[0]):
                    u, v, z = proj[i].tolist()
                    if z <= 0:
                        continue
                    ui = int(round(u * sx))
                    vi = int(round(v * sy))
                    if 0 <= ui < img_w and 0 <= vi < img_h:
                        color = (0, 255, 0) if i < 5 else (0, 0, 255)
                        cv2.circle(img, (ui, vi), 3, color, -1)

                if args.video_out:
                    if writer is None:
                        writer = imageio.get_writer(args.video_out, fps=args.fps)
                    writer.append_data(img)

                if args.debug_save_dir and (frame_count % max(1, args.debug_draw_every) == 0):
                    os.makedirs(args.debug_save_dir, exist_ok=True)
                    save_path = f"{args.debug_save_dir}/debug_{frame_count:06d}.png"
                    _save_debug_plot(save_path, tips_w, args.debug_draw_scale)

                frame_count += 1
                pbar.update(1)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
