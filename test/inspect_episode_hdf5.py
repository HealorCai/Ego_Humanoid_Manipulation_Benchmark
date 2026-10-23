#!/usr/bin/env python3
"""
Inspect an HDF5 file: print groups, datasets, shapes, dtypes, and attributes.

python test/inspect_episode_hdf5.py /cpfs04/user/caizetao/data/EgoVLA_SIM/Flip-Mug/episode_0.hdf5

"""

import argparse
from typing import Any

import h5py
import numpy as np


def _format_value(val: Any, max_len: int = 200) -> str:
    if isinstance(val, (bytes, np.bytes_)):
        try:
            val = val.decode("utf-8")
        except Exception:
            return repr(val)
    if isinstance(val, np.ndarray):
        return f"ndarray shape={val.shape} dtype={val.dtype}"
    s = repr(val)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _print_attrs(obj: h5py.HLObject, indent: int = 2) -> None:
    if not obj.attrs:
        return
    pad = " " * indent
    for k in obj.attrs:
        print(f"{pad}@{k}: {_format_value(obj.attrs[k])}")


def _summarize_dataset(ds: h5py.Dataset, sample: int) -> None:
    print(f"  dataset: shape={ds.shape} dtype={ds.dtype}")
    _print_attrs(ds, indent=4)
    if sample <= 0:
        return
    try:
        if ds.size == 0:
            return
        if ds.ndim == 1:
            data = ds[: min(sample, ds.shape[0])]
            print(f"    sample: {_format_value(data)}")
        elif ds.ndim == 2:
            data = ds[: min(sample, ds.shape[0]), : min(sample, ds.shape[1])]
            print(f"    sample: {_format_value(data)}")
        else:
            # For higher dims, print a small slice summary only
            slc = (slice(0, 1),) * ds.ndim
            data = ds[slc]
            print(f"    sample slice shape={data.shape} dtype={data.dtype}")
    except Exception as exc:
        print(f"    sample: <error reading sample: {exc}>")


def _visit(name: str, obj: h5py.HLObject, sample: int) -> None:
    if isinstance(obj, h5py.Group):
        print(f"group: {name or '/'}")
        _print_attrs(obj, indent=2)
    elif isinstance(obj, h5py.Dataset):
        print(f"dataset: {name}")
        _summarize_dataset(obj, sample)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an HDF5 file structure.")
    parser.add_argument("path", help="Path to .hdf5 file")
    parser.add_argument("--sample", type=int, default=0, help="Print small data samples")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as f:
        print(f"file: {args.path}")
        _print_attrs(f, indent=2)
        f.visititems(lambda name, obj: _visit(name, obj, args.sample))
        breakpoint()


if __name__ == "__main__":
    main()

'''
(egovlasim) root@dsw-387707-5fb9d9d5cc-x6xp5:~/projects/benchmark_ws/Ego_Humanoid_Manipulation_Benchmark# python test/inspect_episode_hdf5.py /cpfs04/user/caizetao/data/EgoVLA_SIM/Flip-Mug/episode_0.hdf5
file: /cpfs04/user/caizetao/data/EgoVLA_SIM/Flip-Mug/episode_0.hdf5
  @sim: True
dataset: action
  dataset: shape=(155, 50) dtype=float32
group: observations
group: observations/images
dataset: observations/images/main
  dataset: shape=(155, 384, 384, 3) dtype=uint8
dataset: observations/left_ee_pose
  dataset: shape=(155, 7) dtype=float32
dataset: observations/left_finger_tip_pos
  dataset: shape=(155, 5, 3) dtype=float32
dataset: observations/left_target_ee_pose
  dataset: shape=(155, 7) dtype=float32
dataset: observations/qpos
  dataset: shape=(155, 50) dtype=float32
dataset: observations/qvel
  dataset: shape=(155, 50) dtype=float32
dataset: observations/right_ee_pose
  dataset: shape=(155, 7) dtype=float32
dataset: observations/right_finger_tip_pos
  dataset: shape=(155, 5, 3) dtype=float32
dataset: observations/right_target_ee_pose
  dataset: shape=(155, 7) dtype=float32
'''