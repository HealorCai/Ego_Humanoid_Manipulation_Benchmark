# EgoVLA dataset tools

Run commands from the benchmark repository root. Offline tools read recorded data; only `print_robot_joint_names.py` requires Isaac Sim.

| File | Purpose |
| --- | --- |
| `project_robot_hand_skeleton_egovla_ds.py` | Batch skeleton-video and keypoint export |
| `project_robot_hand_skeleton_egovla.py` | Single-episode projection; imported by the batch exporter |
| `config/h1_inspire_both_hands_v2.yaml`, `joint_names.txt` | Skeleton definition and joint order |
| `inspect_episode_hdf5.py` | Inspect dataset shapes and attributes |
| `hdf5_images_to_video.py` | Export recorded RGB frames |
| `visualize_hand_keypoints_on_video.py` | Check exported keypoints over recorded RGB frames |
| `summarize_skeleton_videos.py` | Summarize video counts, frames, and durations |
| `print_robot_joint_names.py` | Inspect joint order in the simulator |

## Skeleton preprocessing

```bash
python test/project_robot_hand_skeleton_egovla_ds.py \
  --root data/EgoVLA_SIM/raw_hdf5 \
  --out_root data/EgoVLA_SIM/skeleton_video_384x384 \
  --keypoints_out_root data/EgoVLA_SIM/skeleton_keypoints \
  --norm_keypoints --keypoints_include_z --use_fk_hand \
  --joint_names_file test/joint_names.txt \
  --usd_path source/extensions/humanoid.tasks/humanoid/tasks/data/h1/h1_inspire_convex_decomp.usd \
  --skeleton_cfg test/config/h1_inspire_both_hands_v2.yaml \
  --workers 8 --skip_existing
```

Input layout: `<task>/episode_*.hdf5`. The exporter does not resize images. The Skel-WAM release copy is documented in [Skeleton preprocessing](../../../Skel-WAM/docs/skeleton_preprocessing.md).

## Inspect and visualize

```bash
python test/inspect_episode_hdf5.py data/EgoVLA_SIM/raw_hdf5/Close-Drawer/episode_0.hdf5

python test/hdf5_images_to_video.py \
  --input data/EgoVLA_SIM/raw_hdf5/Close-Drawer/episode_0.hdf5 \
  --output output/episode_0.mp4

python test/visualize_hand_keypoints_on_video.py \
  --raw_hdf5 data/EgoVLA_SIM/raw_hdf5/Close-Drawer/episode_0.hdf5 \
  --keypoints_hdf5 data/EgoVLA_SIM/skeleton_keypoints/Close-Drawer/episode_0.hdf5 \
  --out_mp4 output/keypoints_episode_0.mp4

python test/summarize_skeleton_videos.py \
  --root data/EgoVLA_SIM/skeleton_video_384x384 \
  --csv output/skeleton_video_stats.csv
```

For simulator joint names:

```bash
"$IsaacLab_PATH/isaaclab.sh" -p test/print_robot_joint_names.py \
  --task Humanoid-Close-Drawer-v0 --num_envs 1 --headless --enable_cameras \
  --out output/joint_names.txt
```

Use `scripts/random_agent.py` for simulator smoke tests and [Skel-WAM evaluation](../../../Skel-WAM/docs/egovla_sim_evaluation.md) for policy evaluation. Historical projection variants, the episode-specific demo, and generated CSV have been removed; Git retains their history.
