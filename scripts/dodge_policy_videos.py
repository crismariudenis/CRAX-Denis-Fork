#!/usr/bin/env python3
"""Videos of a trained safe_dodge_humanoid policy, one MP4 per camera.

Loads the last checkpoint of a training run and plays it for a fixed number of
steps. A fall ends the episode as usual, but the env resets and the video keeps
going, so every video has the full length (the return/cost overlay restarts
from 0 each episode). All cameras are rendered from the same rollout.

Usage:
    python scripts/dodge_policy_videos.py                       # newest curriculum run, level 3
    python scripts/dodge_policy_videos.py --levels 1 2 3        # one set of videos per level
    python scripts/dodge_policy_videos.py --run models/<run_name> --steps 1000
    python scripts/dodge_policy_videos.py --runs_dir models/<folder>   # every run in a folder (e.g. all seeds)
"""
import argparse
import os
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `training` imports when run as a script

# MuJoCo needs an OpenGL backend for offscreen rendering; EGL works headless.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crax import envs
from training.agents.ppo import checkpoint as ppo_checkpoint

ENV_NAME = "safe_dodge_humanoid"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, default=None,
                   help="Training run folder under models/ (default: the newest "
                        f"{ENV_NAME}_curriculum_* run)")
    p.add_argument("--runs_dir", type=str, default=None,
                   help=f"Record every {ENV_NAME}_curriculum_* run in this folder instead (e.g. one per seed)")
    p.add_argument("--levels", type=int, nargs="+", default=[3], choices=[1, 2, 3],
                   help="Levels to record the policy on, one set of videos each")
    p.add_argument("--steps", type=int, default=1000,
                   help="Video length in env steps (one step = 0.015 s, so 1000 = 15 s)")
    p.add_argument("--cameras", type=str, nargs="+", default=["front", "side", "orbit"])
    p.add_argument("--fps", type=int, default=60, help="~67 is real time; lower = slow motion")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def find_runs(run: str | None, runs_dir: str | None) -> list[Path]:
    """The run folders: the given one, every run in runs_dir, or the newest dodge curriculum run."""
    if run is not None:
        path = Path(run)
        return [path if path.is_absolute() or path.exists() else REPO_ROOT / "models" / run]
    search_dir = Path(runs_dir) if runs_dir is not None else REPO_ROOT / "models"
    runs = sorted(search_dir.glob(f"{ENV_NAME}_curriculum_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No {ENV_NAME}_curriculum_* run in {search_dir}; pass --run.")
    return runs if runs_dir is not None else runs[-1:]


def latest_checkpoint(run_dir: Path) -> Path:
    """The checkpoint with the highest step (folders are zero-padded step numbers)."""
    steps = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"No checkpoints in {run_dir}.")
    return steps[-1]


def overlay_metrics(frames, returns, costs):
    """Draws the running return and cost onto each frame."""
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
    annotated = []
    for frame, ret, cost in zip(frames, returns, costs):
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        for i, (text, color) in enumerate([(f"Return: {ret:.2f}", (50, 220, 50)),
                                           (f"Cost: {cost:.2f}", (230, 60, 60))]):
            draw.text((10, 10 + 30 * i), text, font=font, fill=color,
                      stroke_width=2, stroke_fill=(0, 0, 0))
        annotated.append(np.array(img))
    return annotated


def record_level(policy, level: int, args, out_prefix: str) -> None:
    """Plays the policy on one level for --steps and saves one video per camera."""
    env = envs.get_environment(ENV_NAME, level=level)
    reset_fn, step_fn = jax.jit(env.reset), jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)
    trajectory = [state.pipeline_state]
    total_reward, total_cost, falls = 0.0, 0.0, 0
    returns, costs = [total_reward], [total_cost]

    for _ in range(args.steps):
        rng, act_rng = jax.random.split(rng)
        action, _ = policy(state.obs, act_rng)
        state = step_fn(state, action)
        trajectory.append(state.pipeline_state)
        total_reward += float(state.reward)
        total_cost += float(state.info["cost"])
        returns.append(total_reward)
        costs.append(total_cost)
        if bool(state.done):
            # Keep recording: start a new episode, totals restart from 0
            falls += 1
            rng, reset_rng = jax.random.split(rng)
            state = reset_fn(reset_rng)
            total_reward, total_cost = 0.0, 0.0

    print(f"level {level}: {args.steps} steps, {falls} falls")
    for camera in args.cameras:
        frames = env.render(trajectory, height=args.height, width=args.width, camera=camera)
        frames = overlay_metrics(frames, returns, costs)
        out_path = REPO_ROOT / "videos" / f"{out_prefix}_lvl{level}_{camera}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out_path, np.stack(frames).astype(np.uint8), fps=args.fps)
        print(f"saved video: {out_path} ({len(frames)} frames @ {args.fps} fps)")


def main() -> None:
    args = parse_args()
    for run_dir in find_runs(args.run, args.runs_dir):
        ckpt = latest_checkpoint(run_dir)
        print(f"policy: {ckpt}")
        policy = jax.jit(ppo_checkpoint.load_policy(ckpt, deterministic=True))
        # Short name: sd (safe dodge) + a 4-character id of the run, the same every time for one run
        out_prefix = f"sd_{zlib.crc32(run_dir.name.encode()) & 0xFFFF:04x}"
        print(f"video names: {out_prefix}_lvl<level>_<camera>.mp4  (id {out_prefix[3:]} = {run_dir.name})")
        for level in args.levels:
            record_level(policy, level, args, out_prefix=out_prefix)


if __name__ == "__main__":
    main()
