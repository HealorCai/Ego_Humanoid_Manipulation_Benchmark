#!/usr/bin/env python3
"""Print simulator robot joint names. See test/README.md."""
def force_conda_aws():
    import sys

    sys.path = [
        p for p in sys.path
        if "omni.kit.pip_archive" not in p and "pip_prebundle" not in p
    ]

    for name, mod in list(sys.modules.items()):
        file = getattr(mod, "__file__", "") or ""
        if (
            name == "botocore" or name.startswith("botocore.")
            or name == "boto3" or name.startswith("boto3.")
            or name == "s3transfer" or name.startswith("s3transfer.")
        ) and ("omni.kit.pip_archive" in file or "pip_prebundle" in file):
            del sys.modules[name]

    print("[force_conda_aws]")
    import botocore
    print("botocore   :", botocore.__file__, botocore.__version__)
    import boto3
    print("boto3      :", boto3.__file__, boto3.__version__)
    import s3transfer
    print("s3transfer :", s3transfer.__file__, s3transfer.__version__)

force_conda_aws()

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print robot joint names.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (e.g., Humanoid-Close-Drawer-v0).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--out", type=str, default=None, help="Optional output file (one name per line).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.isaac.lab_tasks  # noqa: F401
import humanoid.tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    joint_names = env.unwrapped.robot.data.joint_names
    for name in joint_names:
        print(name)

    if args_cli.out:
        from pathlib import Path
        Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args_cli.out, "w", encoding="utf-8") as f:
            for name in joint_names:
                f.write(f"{name}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
