"""Script to run a humanoid environment with random action agent."""
"""Script is derived from IsaacLab random agent example."""
'''
python test/project_robot_hand_skeleton_egovla_env.py --task Humanoid-Close-Drawer-v0 --num_envs=1 --enable_cameras --headless
'''

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

import importlib
import sys

# if "omni.isaac.lab" not in sys.modules:
#     sys.modules["omni.isaac.lab"] = importlib.import_module("isaaclab")

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random agent for Ego_Humanoid_Manipulation_Benchmark environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

# if "omni.isaac.lab_tasks" not in sys.modules:
#     sys.modules["omni.isaac.lab"] = importlib.import_module("isaaclab")
#     sys.modules["omni.isaac.lab_tasks"] = importlib.import_module("isaaclab_tasks")

import omni.isaac.lab_tasks  # noqa: F401
import humanoid.tasks
from omni.isaac.lab_tasks.utils import parse_env_cfg

def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1 # torch.Size([4, 50])
            env_results = env.step(actions)
            rgb_obs = env_results[0]["fixed_rgb"][0].cpu().numpy()[:, :, :]
            print(actions.shape)
            print(rgb_obs.shape)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
