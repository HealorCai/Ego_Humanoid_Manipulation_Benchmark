#!/usr/bin/env python3
"""Generate the hand-skeleton overlay video for Flip-Mug episode_2 only.

The image frames are passed directly to the shared overlay implementation.  No
resize is performed, so the output video has exactly the HDF5 image resolution.

Example:
    python test/demo_flip_mug_episode_2_overlay.py

Use a different (for example, 1280x720) source file when needed:
    python test/demo_flip_mug_episode_2_overlay.py --hdf5 /path/to/episode_2.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import h5py

from project_robot_hand_skeleton_egovla import run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HDF5 = Path("/cpfs04/user/caizetao/data/EgoVLA_SIM/raw_hdf5/Flip-Mug/episode_2.hdf5")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "Flip-Mug_episode_2_overlay.mp4"
DEFAULT_JOINT_NAMES = Path(__file__).resolve().parent / "joint_names.txt"
DEFAULT_SKELETON_CFG = Path(__file__).resolve().parent / "config/h1_inspire_both_hands_v2.yaml"
DEFAULT_USD = (
    PROJECT_ROOT
    / "source/extensions/humanoid.tasks/humanoid/tasks/data/h1/h1_inspire_convex_decomp.usd"
)
IMAGE_DATASET = "observations/images/main"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the native-resolution overlay video for Flip-Mug episode_2."
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--video_out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Frames to process; 0 processes the complete episode.",
    )
    return parser.parse_args()


def main() -> None:
    cli = _parse_args()
    hdf5_path = cli.hdf5.expanduser().resolve()
    video_out = cli.video_out.expanduser().resolve()

    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as hdf5_file:
        if IMAGE_DATASET not in hdf5_file:
            raise KeyError(f"Image dataset not found: {IMAGE_DATASET}")
        image_shape = hdf5_file[IMAGE_DATASET].shape
        if len(image_shape) != 4:
            raise ValueError(f"Expected image data shaped (T, H, W, C), got {image_shape}")
        height, width = image_shape[1:3]

    video_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {hdf5_path}")
    print(f"Frames: {image_shape[0]}, native resolution: {width}x{height}")
    print(f"Output: {video_out} (no resize)")

    run(
        SimpleNamespace(
            hdf5=str(hdf5_path),
            hdf5_images=IMAGE_DATASET,
            hdf5_left_tips="observations/left_finger_tip_pos",
            hdf5_right_tips="observations/right_finger_tip_pos",
            hdf5_qpos="observations/qpos",
            hdf5_left_ee_pose="observations/left_ee_pose",
            hdf5_right_ee_pose="observations/right_ee_pose",
            joint_names_file=str(DEFAULT_JOINT_NAMES),
            usd_path=str(DEFAULT_USD),
            use_fk_hand=True,
            skeleton_cfg=str(DEFAULT_SKELETON_CFG),
            ee_quat_order="wxyz",
            video_out=str(video_out),
            keypoints_out=None,
            keypoints_mode="pixel",
            keypoints_include_z=False,
            max_frames=cli.max_frames,
            fps=cli.fps,
            debug_save_dir=None,
            debug_draw_every=1,
            debug_draw_scale=0.1,
        )
    )

    print(f"Done: {video_out} ({width}x{height})")


if __name__ == "__main__":
    main()
