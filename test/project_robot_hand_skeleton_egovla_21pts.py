#!/usr/bin/env python3
"""
Project fingertip positions onto dataset images without running simulation.

Example:
python test/project_robot_hand_skeleton_egovla_21pts.py \
  --hdf5 /cpfs04/user/caizetao/data/EgoVLA_SIM/Close-Drawer/episode_0.hdf5 \
  --video_out /cpfs04/user/caizetao/projects/benchmark_ws/Ego_Humanoid_Manipulation_Benchmark/output/skel_episode_0_21.mp4 \
  --max_frames 300 --fps 30 \
  --use_fk_hand \
  --joint_names_file /cpfs04/user/caizetao/projects/benchmark_ws/Ego_Humanoid_Manipulation_Benchmark/test/joint_names.txt \
  --usd_path /cpfs04/user/caizetao/projects/benchmark_ws/Ego_Humanoid_Manipulation_Benchmark/source/extensions/humanoid.tasks/humanoid/tasks/data/h1/h1_inspire_convex_decomp.usd


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
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R

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

def _pose_to_mat(pos: np.ndarray, quat: np.ndarray, quat_order: str) -> np.ndarray:
    if quat_order == "wxyz":
        qw, qx, qy, qz = quat
        quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float32)
    elif quat_order == "xyzw":
        qx, qy, qz, qw = quat
        quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported quat_order: {quat_order}")
    R_mat = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_mat
    T[:3, 3] = pos.astype(np.float32)
    return T


def _axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis.astype(np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis / norm
    return R.from_rotvec(axis * float(angle)).as_matrix().astype(np.float32)


class _JointInfo:
    def __init__(self, parent_link: str, child_link: str, axis: np.ndarray, T_p0: np.ndarray, T_c0: np.ndarray):
        self.parent_link = parent_link
        self.child_link = child_link
        self.axis = axis
        self.T_p0 = T_p0
        self.T_c0 = T_c0


def _load_usd_hand_kinematics(usd_path: str, side_prefix: str) -> Tuple[str, Dict[str, _JointInfo], Dict[str, List[str]]]:
    try:
        from pxr import Usd, UsdPhysics
    except Exception as exc:  # pragma: no cover - depends on Isaac/pxr
        raise RuntimeError("pxr is required for USD kinematics. Please run in Isaac/Omniverse env.") from exc

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    base_link = f"{side_prefix}_hand_base_link"
    joints: Dict[str, _JointInfo] = {}
    children: Dict[str, List[str]] = {}

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if not body0 or not body1:
            continue
        link0 = str(body0[0]).split("/")[-1]
        link1 = str(body1[0]).split("/")[-1]
        if not (link0.startswith(side_prefix) and link1.startswith(side_prefix)):
            continue

        axis_attr = joint.GetAxisAttr().Get()
        if axis_attr is None:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif isinstance(axis_attr, str):
            axis_token = axis_attr.upper()
            if axis_token == "X":
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif axis_token == "Y":
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif axis_token == "Z":
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                raise ValueError(f"Unknown joint axis token: {axis_attr}")
        else:
            axis = np.array([axis_attr[0], axis_attr[1], axis_attr[2]], dtype=np.float32)

        pos0 = joint.GetLocalPos0Attr().Get()
        rot0 = joint.GetLocalRot0Attr().Get()
        pos1 = joint.GetLocalPos1Attr().Get()
        rot1 = joint.GetLocalRot1Attr().Get()

        def _gf_quat_to_wxyz(q) -> np.ndarray:
            if q is None:
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            try:
                qw = q.GetReal()
                qx, qy, qz = q.GetImaginary()
            except Exception:
                qw = q[0]
                qx, qy, qz = q[1], q[2], q[3]
            return np.array([float(qw), float(qx), float(qy), float(qz)], dtype=np.float32)

        def _gf_vec_to_np(v) -> np.ndarray:
            if v is None:
                return np.zeros(3, dtype=np.float32)
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)

        T_p0 = _pose_to_mat(_gf_vec_to_np(pos0), _gf_quat_to_wxyz(rot0), "wxyz")
        T_c0 = _pose_to_mat(_gf_vec_to_np(pos1), _gf_quat_to_wxyz(rot1), "wxyz")

        joint_name = prim.GetName()
        joints[joint_name] = _JointInfo(link0, link1, axis, T_p0, T_c0)
        children.setdefault(link0, []).append(joint_name)

    if base_link not in children:
        print(f"[warn] No joints found with base {base_link}. Check USD naming.")
    return base_link, joints, children


def _compute_hand_link_transforms(
    base_pose: np.ndarray,
    base_link: str,
    joints: Dict[str, _JointInfo],
    children: Dict[str, List[str]],
    joint_angles: Dict[str, float],
) -> Dict[str, np.ndarray]:
    T_world: Dict[str, np.ndarray] = {base_link: base_pose}
    stack = [base_link]
    while stack:
        parent = stack.pop()
        for joint_name in children.get(parent, []):
            info = joints[joint_name]
            angle = float(joint_angles.get(joint_name, 0.0))
            R = _axis_angle_to_mat(info.axis, angle)
            T_rot = np.eye(4, dtype=np.float32)
            T_rot[:3, :3] = R
            T_parent = T_world[parent]
            T_child = T_parent @ info.T_p0 @ T_rot @ np.linalg.inv(info.T_c0)
            if info.child_link not in T_world:
                T_world[info.child_link] = T_child
                stack.append(info.child_link)
    return T_world


def _compute_hand_link_positions(
    base_pose: np.ndarray,
    base_link: str,
    joints: Dict[str, _JointInfo],
    children: Dict[str, List[str]],
    joint_angles: Dict[str, float],
    target_links: List[str],
) -> Dict[str, np.ndarray]:
    T_world = _compute_hand_link_transforms(base_pose, base_link, joints, children, joint_angles)
    out: Dict[str, np.ndarray] = {}
    for link in target_links:
        if link in T_world:
            out[link] = T_world[link][:3, 3].copy()
    return out


def _load_joint_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Project fingertip skeleton onto dataset images (no sim).")
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 dataset.")
    parser.add_argument("--hdf5_images", type=str, default="observations/images/main", help="Image dataset path.")
    parser.add_argument("--hdf5_left_tips", type=str, default="observations/left_finger_tip_pos", help="Left tips path.")
    parser.add_argument("--hdf5_right_tips", type=str, default="observations/right_finger_tip_pos", help="Right tips path.")
    parser.add_argument("--hdf5_qpos", type=str, default="observations/qpos", help="Qpos dataset path.")
    parser.add_argument("--hdf5_left_ee_pose", type=str, default="observations/left_ee_pose", help="Left EE pose path.")
    parser.add_argument("--hdf5_right_ee_pose", type=str, default="observations/right_ee_pose", help="Right EE pose path.")
    parser.add_argument("--joint_names_file", type=str, default=None, help="Path to joint names file (one per line).")
    parser.add_argument("--usd_path", type=str, default=None, help="USD path for robot.")
    parser.add_argument("--use_fk_hand", action="store_true", help="Use qpos + USD to FK hand keypoints.")
    parser.add_argument("--ee_quat_order", type=str, default="wxyz", choices=["wxyz", "xyzw"], help="Quaternion order in ee_pose.")
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
    use_fk_hand = args.use_fk_hand
    joint_name_to_index: Dict[str, int] = {}
    left_base_link = "L_hand_base_link"
    right_base_link = "R_hand_base_link"
    left_kin = right_kin = None
    left_children = right_children = None
    left_target_links = right_target_links = None
    left_joint_names = right_joint_names = None

    if use_fk_hand:
        if args.joint_names_file is None:
            raise ValueError("--joint_names_file is required when --use_fk_hand is set.")
        if args.usd_path is None:
            raise ValueError("--usd_path is required when --use_fk_hand is set.")

        joint_names = _load_joint_names(args.joint_names_file)
        joint_name_to_index = {name: i for i, name in enumerate(joint_names)}
        left_base_link, left_kin, left_children = _load_usd_hand_kinematics(args.usd_path, "L")
        right_base_link, right_kin, right_children = _load_usd_hand_kinematics(args.usd_path, "R")

        left_target_links = [
            "L_index_proximal",
            "L_index_intermediate",
            "L_index_tip",
            "L_middle_proximal",
            "L_middle_intermediate",
            "L_middle_tip",
            "L_ring_proximal",
            "L_ring_intermediate",
            "L_ring_tip",
            "L_pinky_proximal",
            "L_pinky_intermediate",
            "L_pinky_tip",
            "L_thumb_proximal",
            "L_thumb_intermediate",
            "L_thumb_distal",
            "L_thumb_tip",
        ]
        right_target_links = [name.replace("L_", "R_") for name in left_target_links]

        left_joint_names = [
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_intermediate_joint",
            "L_thumb_distal_joint",
        ]
        right_joint_names = [name.replace("L_", "R_") for name in left_joint_names]

    with h5py.File(args.hdf5, "r") as f:
        if args.hdf5_images not in f:
            raise KeyError(f"HDF5 images dataset not found: {args.hdf5_images}")
        if args.hdf5_left_tips not in f:
            raise KeyError(f"HDF5 left tips dataset not found: {args.hdf5_left_tips}")
        if args.hdf5_right_tips not in f:
            raise KeyError(f"HDF5 right tips dataset not found: {args.hdf5_right_tips}")
        if args.hdf5_left_ee_pose not in f:
            raise KeyError(f"HDF5 left ee pose dataset not found: {args.hdf5_left_ee_pose}")
        if args.hdf5_right_ee_pose not in f:
            raise KeyError(f"HDF5 right ee pose dataset not found: {args.hdf5_right_ee_pose}")
        if use_fk_hand and args.hdf5_qpos not in f:
            raise KeyError(f"HDF5 qpos dataset not found: {args.hdf5_qpos}")

        img_ds = f[args.hdf5_images]
        left_tips_ds = f[args.hdf5_left_tips]
        right_tips_ds = f[args.hdf5_right_tips]
        left_ee_pose_ds = f[args.hdf5_left_ee_pose]
        right_ee_pose_ds = f[args.hdf5_right_ee_pose]
        qpos_ds = f[args.hdf5_qpos] if use_fk_hand else None
        total_frames = min(
            img_ds.shape[0],
            left_tips_ds.shape[0],
            right_tips_ds.shape[0],
            left_ee_pose_ds.shape[0],
            right_ee_pose_ds.shape[0],
            qpos_ds.shape[0] if qpos_ds is not None else img_ds.shape[0],
        )
        if args.max_frames > 0:
            total_frames = min(total_frames, args.max_frames)

        with tqdm(total=total_frames, desc="Overlay frames") as pbar:
            while frame_count < total_frames:
                left_tips = torch.from_numpy(left_tips_ds[frame_count])
                right_tips = torch.from_numpy(right_tips_ds[frame_count])
                left_ee_pose = left_ee_pose_ds[frame_count]
                right_ee_pose = right_ee_pose_ds[frame_count]
                left_wrist_w = torch.from_numpy(left_ee_pose[:3]).unsqueeze(0)
                right_wrist_w = torch.from_numpy(right_ee_pose[:3]).unsqueeze(0)
                tips_w = torch.cat([left_tips, right_tips], dim=0)  # (10, 3)

                points_w_list = [left_wrist_w, right_wrist_w, tips_w]
                point_colors: List[Tuple[int, int, int]] = []
                point_colors.extend([(255, 255, 0), (0, 255, 255)])  # wrists
                point_colors.extend([(0, 255, 0)] * 5 + [(0, 0, 255)] * 5)  # tips

                if use_fk_hand:
                    qpos = qpos_ds[frame_count]
                    left_joint_angles = {}
                    right_joint_angles = {}
                    for name in left_joint_names:
                        if name in joint_name_to_index:
                            left_joint_angles[name] = float(qpos[joint_name_to_index[name]])
                    for name in right_joint_names:
                        if name in joint_name_to_index:
                            right_joint_angles[name] = float(qpos[joint_name_to_index[name]])

                    left_base_T = _pose_to_mat(left_ee_pose[:3], left_ee_pose[3:7], args.ee_quat_order)
                    right_base_T = _pose_to_mat(right_ee_pose[:3], right_ee_pose[3:7], args.ee_quat_order)

                    left_T_world = _compute_hand_link_transforms(
                        left_base_T, left_base_link, left_kin, left_children, left_joint_angles
                    )
                    right_T_world = _compute_hand_link_transforms(
                        right_base_T, right_base_link, right_kin, right_children, right_joint_angles
                    )

                    left_link_pos = {
                        k: left_T_world[k][:3, 3].copy()
                        for k in left_target_links
                        if k in left_T_world
                    }
                    right_link_pos = {
                        k: right_T_world[k][:3, 3].copy()
                        for k in right_target_links
                        if k in right_T_world
                    }

                    left_fk_pts = [left_link_pos[k] for k in left_target_links if k in left_link_pos]
                    right_fk_pts = [right_link_pos[k] for k in right_target_links if k in right_link_pos]
                    if left_fk_pts:
                        points_w_list.append(torch.from_numpy(np.stack(left_fk_pts, axis=0)))
                        point_colors.extend([(0, 200, 0)] * len(left_fk_pts))
                    if right_fk_pts:
                        points_w_list.append(torch.from_numpy(np.stack(right_fk_pts, axis=0)))
                        point_colors.extend([(0, 0, 200)] * len(right_fk_pts))

                    # Virtual points on intermediate links for index/middle/ring/pinky (exclude thumb):
                    # offset 2cm along local Y axis.
                    virtual_pts = []
                    for side_prefix, T_world in [("L", left_T_world), ("R", right_T_world)]:
                        for finger in ["index", "middle", "ring", "pinky"]:
                            link_name = f"{side_prefix}_{finger}_intermediate"
                            if link_name in T_world:
                                y_offset = -0.02 if side_prefix == "L" else 0.02
                                offset_local = np.array([0.0, y_offset, 0.0, 1.0], dtype=np.float32)
                                p = T_world[link_name] @ offset_local
                                virtual_pts.append(p[:3].copy())
                    if virtual_pts:
                        points_w_list.append(torch.from_numpy(np.stack(virtual_pts, axis=0)))
                        point_colors.extend([(255, 128, 0)] * len(virtual_pts))

                points_w = torch.cat(points_w_list, dim=0)
                proj = _project_points(points_w, K, extr)

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
                        if i == 0:
                            color = (255, 255, 0)  # left wrist
                            radius = 4
                        elif i == 1:
                            color = (0, 255, 255)  # right wrist
                            radius = 4
                        else:
                            color = point_colors[i] if i < len(point_colors) else (255, 255, 255)
                            radius = 3
                        cv2.circle(img, (ui, vi), radius, color, -1)

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
