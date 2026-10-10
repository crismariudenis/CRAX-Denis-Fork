#!/usr/bin/env python3
"""Live MuJoCo viewer for a CRAX environment that reloads when you edit it.

Opens an interactive window showing the environment running with random
actions. Every second the script checks the files under `crax/` (Python
sources and XML assets); when one changes, the viewer restarts with the new
version of the environment, keeping your camera angle.

Two physics modes:
  --physics env     (default) steps the real CRAX env, so reward/cost logic,
                    termination and backend match training. Each restart
                    re-JITs the env (~30-60 s for the humanoid).
  --physics mujoco  steps the env's MuJoCo model with plain CPU MuJoCo. Restarts
                    in a few seconds; best for editing the scene/XML. Reward
                    and cost are not computed.

Usage:
    python examples/02_live_view.py --env_name safe_height_humanoid
    python examples/02_live_view.py --env_name safe_height_humanoid --physics mujoco
    python examples/02_live_view.py --env_name safe_goal_point --level 2 \\
        --env_kwargs '{"episode_length": 500}'

Needs a display (it opens a GLFW window). Close the window or press Ctrl+C to quit.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent
WATCH_SUFFIXES = {".py", ".xml", ".obj", ".stl", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env_name", type=str, default="safe_height_humanoid",
                   help="Environment name from the CRAX registry")
    p.add_argument("--level", type=int, default=1, choices=[1, 2, 3], help="Difficulty level")
    p.add_argument("--env_kwargs", type=str, default="{}",
                   help="JSON dict of extra env constructor kwargs (see docs/PARAMETERS.md)")
    p.add_argument("--physics", choices=["env", "mujoco"], default="env",
                   help="Step the real CRAX env (JIT) or plain CPU MuJoCo on its model")
    p.add_argument("--actions", choices=["random", "zero"], default="random",
                   help="Action source: uniform random in [-1, 1], or all zeros")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--poll", type=float, default=1.0, help="Seconds between file-change checks")
    p.add_argument("--watch", type=str, nargs="*", default=[str(REPO_ROOT / "crax")],
                   help="Files/directories to watch for changes (default: crax/)")
    p.add_argument("--no_reload", action="store_true", help="Run once without watching files")
    # Internal: set by the watcher when it spawns the viewer process.
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_cam_file", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ----------------------------------------------------------------------------
# Watcher (parent process): restarts the viewer whenever a watched file changes.
# ----------------------------------------------------------------------------

def snapshot(paths):
    """Maps every watched file to its modification time."""
    mtimes = {}
    for root in map(Path, paths):
        files = [root] if root.is_file() else root.rglob("*")
        for f in files:
            if f.is_file() and f.suffix in WATCH_SUFFIXES and "__pycache__" not in f.parts:
                try:
                    mtimes[f] = f.stat().st_mtime
                except FileNotFoundError:
                    pass  # Editors often replace files via a temp file.
    return mtimes


def changed_files(old, new):
    return sorted(str(f.relative_to(REPO_ROOT)) if f.is_relative_to(REPO_ROOT) else str(f)
                  for f in set(old) | set(new) if old.get(f) != new.get(f))


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_watcher(args) -> None:
    cam_file = Path(tempfile.gettempdir()) / f"crax_live_view_cam_{os.getpid()}.json"

    mjpython = shutil.which("mjpython")
    if mjpython is None:
        raise RuntimeError(
            "mjpython not found. On macOS, run this script with mjpython."
        )

    child_cmd = [mjpython, __file__,"--_child","--_cam_file",str(cam_file),]
    child_cmd += [a for a in sys.argv[1:] if a != "--no_reload"]

    mtimes = snapshot(args.watch)
    proc = subprocess.Popen(child_cmd)
    print(f"[live_view] watching {', '.join(args.watch)} (every {args.poll:g}s); Ctrl+C to quit")
    try:
        while True:
            time.sleep(args.poll)
            new_mtimes = snapshot(args.watch)
            changed = changed_files(mtimes, new_mtimes)
            if changed:
                mtimes = new_mtimes
                print(f"[live_view] changed: {', '.join(changed)} -> reloading")
                stop(proc)
                proc = subprocess.Popen(child_cmd)
            elif proc.poll() is not None and proc.returncode == 0:
                print("[live_view] viewer closed")
                break
            # A crashed viewer (e.g. a syntax error in the env) stays down until the next edit.
    except KeyboardInterrupt:
        pass
    finally:
        stop(proc)
        cam_file.unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# Viewer (child process): builds the env and runs it in a passive MuJoCo viewer.
# ----------------------------------------------------------------------------

CAM_FIELDS = ("lookat", "distance", "azimuth", "elevation")


def load_camera(viewer, cam_file):
    if cam_file and Path(cam_file).exists():
        try:
            cam = json.loads(Path(cam_file).read_text())
        except (OSError, json.JSONDecodeError):
            return
        with viewer.lock():
            viewer.cam.lookat[:] = cam["lookat"]
            viewer.cam.distance = cam["distance"]
            viewer.cam.azimuth = cam["azimuth"]
            viewer.cam.elevation = cam["elevation"]


def save_camera(viewer, cam_file):
    if cam_file:
        cam = {k: getattr(viewer.cam, k) for k in CAM_FIELDS}
        cam["lookat"] = [float(x) for x in cam["lookat"]]
        Path(cam_file).write_text(json.dumps(cam))


def run_viewer(args) -> None:
    import jax
    import mujoco
    import mujoco.viewer
    import numpy as np

    from crax import envs

    env = envs.get_environment(args.env_name, level=args.level, **json.loads(args.env_kwargs))
    model = env.sys.mj_model
    data = mujoco.MjData(model)
    dt = float(env.dt)
    episode_length = getattr(env, "episode_length", 1000)
    print(f"[live_view] env={args.env_name} level={args.level} physics={args.physics} "
          f"dt={dt:.4f}s action_size={env.action_size}")

    # Reset through the env so the initial pose matches training (spawn height, noise, ...).
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step) if args.physics == "env" else None
    rng = jax.random.PRNGKey(args.seed)
    np_rng = np.random.default_rng(args.seed)

    def show(state):
        data.qpos[:] = np.asarray(state.pipeline_state.q)
        data.qvel[:] = np.asarray(state.pipeline_state.qd)
        mujoco.mj_forward(model, data)

    def sample_action():
        if args.actions == "zero":
            return np.zeros(env.action_size, dtype=np.float32)
        return np_rng.uniform(-1.0, 1.0, env.action_size).astype(np.float32)

    def reset():
        nonlocal rng
        rng, key = jax.random.split(rng)
        s = reset_fn(key)
        show(s)
        return s

    t0 = time.time()
    state = reset()
    if step_fn is not None:
        state = step_fn(state, sample_action())  # Compile before opening the window.
    print(f"[live_view] ready in {time.time() - t0:.1f}s")

    # CPU MuJoCo mode: one env step = n_frames physics steps with ctrl held fixed.
    n_substeps = max(1, round(dt / model.opt.timestep))
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        load_camera(viewer, args._cam_file)
        steps, ret, cost = 0, 0.0, 0.0
        last_cam_save = 0.0
        while viewer.is_running():
            tick = time.time()
            action = sample_action()
            with viewer.lock():
                if step_fn is not None:
                    state = step_fn(state, action)
                    ret += float(state.reward)
                    cost += float(state.info.get("cost", 0.0))
                    done = bool(state.done)
                    show(state)
                else:
                    limited = model.actuator_ctrllimited.astype(bool)
                    data.ctrl[:] = np.where(limited, ctrl_lo + (action + 1) / 2 * (ctrl_hi - ctrl_lo), action)
                    mujoco.mj_step(model, data, nstep=n_substeps)
                    done = False
                steps += 1
                if done or steps >= episode_length:
                    print(f"[live_view] episode end: steps={steps} return={ret:.2f} cost={cost:.2f}")
                    state = reset()
                    steps, ret, cost = 0, 0.0, 0.0
            if step_fn is not None:
                viewer.set_texts((None, None, f"step {steps}\nreturn {ret:.2f}\ncost {cost:.2f}", None))
            viewer.sync()
            if tick - last_cam_save > 1.0:
                save_camera(viewer, args._cam_file)
                last_cam_save = tick
            time.sleep(max(0.0, dt - (time.time() - tick)))  # Real-time playback.
        save_camera(viewer, args._cam_file)


if __name__ == "__main__":
    # MuJoCo's interactive viewer needs GLFW, not the EGL backend used for offscreen video.
    os.environ["MUJOCO_GL"] = "glfw"
    args = parse_args()
    if args._child or args.no_reload:
        run_viewer(args)
    else:
        run_watcher(args)
