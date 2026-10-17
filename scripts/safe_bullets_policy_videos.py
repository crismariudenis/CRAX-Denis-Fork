
#!/usr/bin/env python3
"""Generate videos from a trained SafeBulletsHumanoid curriculum policy."""

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crax import envs
from training.agents.ppo import checkpoint as ppo_checkpoint

ENV_NAME = "safe_bullets_humanoid"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=None,
                   help="Specific training run directory")
    p.add_argument("--runs_dir", default=None,
                   help="Directory containing curriculum run folders")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3],
                   choices=[1, 2, 3])
    p.add_argument("--stage", type=int, default=None,
                   help="Checkpoint stage number, if applicable")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--cameras", nargs="+",
                   default=["front", "side", "fixedfar", "orbit"],
                   help="Camera views; orbit makes a full 360-degree sweep")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env_kwargs", default="{}")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def find_runs(run, runs_dir):
    if run:
        path = Path(run)
        if not path.is_absolute() and not path.exists():
            path = REPO_ROOT / run
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory not found: {path}")
        return [path]

    search_dir = (
        Path(runs_dir) if runs_dir
        else REPO_ROOT / "models"
    )
    runs = sorted(
        search_dir.glob(f"{ENV_NAME}_curriculum_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(
            f"No {ENV_NAME}_curriculum_* runs found in {search_dir}. "
            "Pass --run with the exact run directory."
        )
    return runs if runs_dir else runs[-1:]


def latest_checkpoint(run_dir, stage):
    stage_dirs = sorted(
        [p for p in run_dir.glob("stage_*")
         if p.name[6:].isdigit()],
        key=lambda p: int(p.name[6:]),
    )

    if stage is not None:
        run_dir = run_dir / f"stage_{stage}"
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"Stage directory not found: {run_dir}"
            )
    elif stage_dirs:
        run_dir = stage_dirs[-1]

    checkpoints = sorted(
        [p for p in run_dir.iterdir()
         if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No numbered checkpoints found in {run_dir}"
        )

    return checkpoints[-1]


def render_orbit(env, trajectory, height, width):
    """Render a trajectory with a free camera making one full 360-degree sweep."""
    import mujoco

    model = env.sys.mj_model
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.elevation = -5.0
    camera.lookat[:] = [0.0, 0.0, 0.9]

    frames = []
    count = max(len(trajectory), 1)
    try:
        for i, state in enumerate(trajectory):
            data.qpos = state.q
            data.qvel = state.qd
            mujoco.mj_forward(model, data)
            camera.azimuth = -180.0 + 360.0 * i / count
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
    finally:
        renderer.close()
    return frames


def overlay_metrics(frames, returns, costs, dodged_counts):
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    output = []
    for frame, ret, cost, dodged in zip(frames, returns, costs, dodged_counts):
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        draw = ImageDraw.Draw(img)

        for i, label in enumerate(
            [f"Return: {ret:.2f}", f"Cost: {cost:.2f}",
             f"Bullets dodged: {int(dodged)}"]
        ):
            draw.text(
                (10, 10 + 30 * i),
                label,
                font=font,
                fill=((50, 220, 50) if i == 0 else
                      (230, 60, 60) if i == 1 else (80, 190, 255)),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

        output.append(np.asarray(img))

    return output


def record_level(policy, level, args, out_prefix):
    env = envs.get_environment(
        ENV_NAME,
        level=level,
        **json.loads(args.env_kwargs),
    )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)

    trajectory = [state.pipeline_state]
    total_reward = 0.0
    total_cost = 0.0
    returns = [0.0]
    costs = [0.0]
    dodged_counts = [0.0]
    falls = 0

    for _ in range(args.steps):
        rng, action_rng = jax.random.split(rng)
        action, _ = policy(state.obs, action_rng)
        state = step_fn(state, action)

        trajectory.append(state.pipeline_state)
        total_reward += float(state.reward)
        total_cost += float(state.info["cost"])
        returns.append(total_reward)
        costs.append(total_cost)
        # Environment info holds the episode-level cumulative dodge counter.
        dodged_counts.append(float(state.info.get("dodged_bullets", 0.0)))

        if bool(state.done):
            falls += 1
            rng, reset_rng = jax.random.split(rng)
            state = reset_fn(reset_rng)
            total_reward = 0.0
            total_cost = 0.0

    print(f"Level {level}: {args.steps} steps, {falls} falls")

    for camera in args.cameras:
        if camera == "orbit":
            frames = render_orbit(env, trajectory, args.height, args.width)
        else:
            frames = env.render(
                trajectory,
                height=args.height,
                width=args.width,
                camera=camera,
            )
        frames = overlay_metrics(frames, returns, costs, dodged_counts)

        out_path = (
            REPO_ROOT / "videos" /
            f"{out_prefix}_lvl{level}_{camera}.mp4"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        iio.imwrite(
            out_path,
            np.stack(frames).astype(np.uint8),
            fps=args.fps,
        )
        print(f"Saved video: {out_path}")


def main():
    args = parse_args()

    for run_dir in find_runs(args.run, args.runs_dir):
        ckpt = latest_checkpoint(run_dir, args.stage)
        print(f"Loading checkpoint: {ckpt}")

        policy = jax.jit(
            ppo_checkpoint.load_policy(
                ckpt, deterministic=True
            )
        )

        run_id = zlib.crc32(run_dir.name.encode()) & 0xFFFF
        prefix = f"sb_{run_id:04x}"

        if args.stage is not None:
            prefix += f"_stage{args.stage}"
        if args.tag:
            prefix += f"_{args.tag}"

        for level in args.levels:
            record_level(policy, level, args, prefix)


if __name__ == "__main__":
    main()
