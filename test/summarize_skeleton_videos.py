"""Summarize video counts, frames, and durations by task. See test/README.md."""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


DEFAULT_FPS = 20.0


@dataclass
class VideoInfo:
    frames: int
    duration_s: float


def _run_ffprobe(path: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        return None, None, None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, None, None

    stream = None
    if isinstance(data.get("streams"), list) and data["streams"]:
        stream = data["streams"][0]
    fmt = data.get("format") or {}

    nb_frames = None
    if stream and "nb_frames" in stream:
        try:
            nb_frames = int(stream["nb_frames"])
        except (TypeError, ValueError):
            nb_frames = None

    fps = None
    if stream and "avg_frame_rate" in stream:
        afr = stream["avg_frame_rate"]
        try:
            num, den = afr.split("/")
            num = float(num)
            den = float(den)
            if den != 0:
                fps = num / den
        except Exception:
            fps = None

    duration = None
    for key in ("duration",):
        if stream and key in stream and stream[key]:
            try:
                duration = float(stream[key])
                break
            except (TypeError, ValueError):
                duration = None
    if duration is None and fmt.get("duration"):
        try:
            duration = float(fmt["duration"])
        except (TypeError, ValueError):
            duration = None

    return nb_frames, duration, fps


def _video_info(path: str, fallback_fps: float) -> Optional[VideoInfo]:
    nb_frames, duration, fps = _run_ffprobe(path)

    if nb_frames is None and duration is None:
        return None

    if fps is None or fps <= 0:
        fps = fallback_fps

    if duration is None and nb_frames is not None:
        duration = nb_frames / fps

    if nb_frames is None and duration is not None:
        nb_frames = int(round(duration * fps))

    if duration is None or nb_frames is None:
        return None

    return VideoInfo(frames=nb_frames, duration_s=duration)


def _collect_mp4_files(root: str) -> List[str]:
    mp4s: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".mp4"):
                mp4s.append(os.path.join(dirpath, name))
    return mp4s


def _summarize_videos(
    files: List[str], fps: float, cache: Dict[str, VideoInfo]
) -> Tuple[int, float, int]:
    total_frames = 0
    total_duration = 0.0
    count = 0
    for path in files:
        info = cache.get(path)
        if info is None:
            info = _video_info(path, fps)
            if info is None:
                continue
            cache[path] = info
        total_frames += info.frames
        total_duration += info.duration_s
        count += 1
    return total_frames, total_duration, count


def _format_seconds(seconds: float) -> str:
    if seconds < 0:
        return "0s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60
    if hours > 0:
        return f"{hours}h{minutes}m{sec:.1f}s"
    if minutes > 0:
        return f"{minutes}m{sec:.1f}s"
    return f"{sec:.1f}s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize episodes and video stats per task for skeleton_video_384x384."
    )
    parser.add_argument(
        "--root",
        default="data/EgoVLA_SIM/skeleton_video_384x384",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Fallback fps when metadata is missing.",
    )
    parser.add_argument(
        "--csv",
        default="output/skeleton_video_stats.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    tasks = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ]
    tasks.sort()

    cache: Dict[str, VideoInfo] = {}

    lines: List[str] = []
    lines.append(
        "task,subtask,episodes,frames,total_seconds,avg_episode_seconds,avg_episode_hms"
    )

    grand_frames = 0
    grand_seconds = 0.0
    grand_episodes = 0

    for task in tasks:
        task_dir = os.path.join(root, task)
        subtask_dirs = [
            d
            for d in os.listdir(task_dir)
            if os.path.isdir(os.path.join(task_dir, d)) and not d.startswith(".")
        ]
        subtask_dirs.sort()

        task_files = _collect_mp4_files(task_dir)
        task_frames, task_seconds, task_eps = _summarize_videos(task_files, args.fps, cache)

        grand_frames += task_frames
        grand_seconds += task_seconds
        grand_episodes += task_eps

        avg_task = (task_seconds / task_eps) if task_eps else 0.0
        lines.append(
            f"{task},(all),{task_eps},{task_frames},{task_seconds:.3f},{avg_task:.3f},{_format_seconds(avg_task)}"
        )

        for sub in subtask_dirs:
            sub_dir = os.path.join(task_dir, sub)
            sub_files = _collect_mp4_files(sub_dir)
            sub_frames, sub_seconds, sub_eps = _summarize_videos(sub_files, args.fps, cache)
            avg_sub = (sub_seconds / sub_eps) if sub_eps else 0.0
            lines.append(
                f"{task},{sub},{sub_eps},{sub_frames},{sub_seconds:.3f},{avg_sub:.3f},{_format_seconds(avg_sub)}"
            )

    avg_all = (grand_seconds / grand_episodes) if grand_episodes else 0.0
    lines.append(
        f"ALL,(all),{grand_episodes},{grand_frames},{grand_seconds:.3f},{avg_all:.3f},{_format_seconds(avg_all)}"
    )

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
