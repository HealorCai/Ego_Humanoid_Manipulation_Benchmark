#!/usr/bin/env python3
"""Export recorded HDF5 images to video. See test/README.md."""

import argparse

import h5py
import imageio.v2 as imageio
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Write HDF5 image dataset to video.")
    parser.add_argument("--input", required=True, help="Path to .hdf5 file")
    parser.add_argument("--dataset", default="observations/images/main", help="Dataset path for images")
    parser.add_argument("--output", required=True, help="Output video path (mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.input, "r") as f:
        if args.dataset not in f:
            raise KeyError(f"Dataset not found: {args.dataset}")
        ds = f[args.dataset]
        if ds.ndim != 4 or ds.shape[-1] != 3:
            raise ValueError(f"Expected (T, H, W, 3), got {ds.shape}")


        with imageio.get_writer(args.output, fps=args.fps) as writer:
            for i in range(ds.shape[0]):
                writer.append_data(ds[i])


if __name__ == "__main__":
    main()
