#!/usr/bin/env python3
"""Overlay exported hand keypoints on recorded images. See test/README.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


def _read_mode(attrs) -> str:
    mode = attrs.get('mode', 'pixel')
    if isinstance(mode, bytes):
        mode = mode.decode('utf-8', errors='ignore')
    if isinstance(mode, np.ndarray) and mode.shape == ():
        mode = mode.item()
    if isinstance(mode, np.ndarray) and mode.dtype.kind in ('S', 'U') and mode.size == 1:
        mode = mode.reshape(-1)[0]
    if not isinstance(mode, str):
        mode = str(mode)
    return mode


def _decode_point_names(attrs):
    pn = attrs.get('point_names', None)
    if pn is None:
        return None
    if isinstance(pn, np.ndarray):
        out = []
        for x in pn.reshape(-1):
            if isinstance(x, bytes):
                out.append(x.decode('utf-8', errors='ignore'))
            else:
                out.append(str(x))
        return out
    return None


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def _draw_points_pil(img: np.ndarray, pts_xy: np.ndarray, colors: list[tuple[int, int, int]], radius: int) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        raise RuntimeError('Pillow is required for visualization (pip install pillow).') from exc

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    for (x, y), color in zip(pts_xy, colors):
        if x < 0 or y < 0:
            continue
        left = float(x) - radius
        top = float(y) - radius
        right = float(x) + radius
        bottom = float(y) + radius
        draw.ellipse([left, top, right, bottom], outline=color, fill=color)
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description='Overlay exported hand keypoints on raw images and save an mp4.')
    ap.add_argument('--raw_hdf5', required=True, type=str, help='Raw EgoVLA_SIM hdf5 containing images.')
    ap.add_argument('--keypoints_hdf5', required=True, type=str, help='Exported keypoints hdf5 (dataset: keypoints).')
    ap.add_argument('--out_mp4', required=True, type=str, help='Output mp4 path.')
    ap.add_argument('--hdf5_images', default='observations/images/main', type=str, help='Image dataset key in raw_hdf5.')
    ap.add_argument('--fps', default=30, type=int)
    ap.add_argument('--max_frames', default=300, type=int, help='<=0 means all frames')
    ap.add_argument('--radius', default=3, type=int)
    ap.add_argument('--alpha', default=1.0, type=float, help='(reserved)')
    args = ap.parse_args()

    raw_hdf5 = Path(args.raw_hdf5)
    kp_hdf5 = Path(args.keypoints_hdf5)
    out_mp4 = Path(args.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)


    with h5py.File(raw_hdf5, 'r') as f_raw, h5py.File(kp_hdf5, 'r') as f_kp:
        if args.hdf5_images not in f_raw:
            raise KeyError(f"Images dataset not found: {args.hdf5_images}")
        if 'keypoints' not in f_kp:
            raise KeyError("Keypoints dataset not found: 'keypoints'")

        img_ds = f_raw[args.hdf5_images]
        kps_ds = f_kp['keypoints']

        mode = _read_mode(f_kp.attrs)
        point_names = _decode_point_names(f_kp.attrs)

        total = min(int(img_ds.shape[0]), int(kps_ds.shape[0]))
        if args.max_frames > 0:
            total = min(total, int(args.max_frames))

        # keypoints shape: (T, 2, 21, D)
        if kps_ds.ndim != 4 or kps_ds.shape[1] != 2:
            raise ValueError(f"Unexpected keypoints shape: {kps_ds.shape}, expected (T,2,21,D)")

        # Build per-point colors: right=red, left=green, optionally highlight base link.
        num_pts = int(kps_ds.shape[2])
        colors = []
        for hand in range(2):
            base = (255, 64, 64) if hand == 0 else (64, 255, 64)
            for j in range(num_pts):
                c = base
                if point_names is not None:
                    # right then left in attrs; after reshape it is (hand, j)
                    name = point_names[hand * num_pts + j] if hand * num_pts + j < len(point_names) else ''
                    if name.endswith('hand_base_link'):
                        c = (64, 200, 255)
                    elif name.endswith('_tip'):
                        c = (255, 255, 0)
                colors.append(c)

        writer = imageio.get_writer(str(out_mp4), fps=int(args.fps))
        try:
            for t in range(total):
                img = np.array(img_ds[t])
                img = _ensure_rgb(img)
                h, w = int(img.shape[0]), int(img.shape[1])

                kps = np.array(kps_ds[t], dtype=np.float32)  # (2,21,D)
                xy = kps[..., :2].reshape(-1, 2)

                if mode == 'norm':
                    # -1 stays -1; only scale valid points.
                    valid = (xy[:, 0] >= 0) & (xy[:, 1] >= 0)
                    xy_scaled = xy.copy()
                    xy_scaled[valid, 0] = xy_scaled[valid, 0] * float(w)
                    xy_scaled[valid, 1] = xy_scaled[valid, 1] * float(h)
                    xy = xy_scaled
                elif mode != 'pixel':
                    raise ValueError(f"Unsupported keypoints mode: {mode}")

                img_out = _draw_points_pil(img, xy, colors, radius=int(args.radius))
                writer.append_data(img_out)
        finally:
            writer.close()

    print(f"[ok] wrote: {out_mp4}")


if __name__ == '__main__':
    main()
