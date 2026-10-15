#!/usr/bin/env python
"""Train sequential curriculum stages with warm-started policies."""

import functools
import os
from datetime import datetime
from pathlib import Path

import wandb

from training import curriculum
from training.curriculum import Stage
from training.config import build_base_parser
from training.run_utils import (
    setup_gpu_environment,
    get_algorithm_train_fn,
    filter_kwargs_for_fn,
    custom_progress_fn,
    record_episode_video,
    make_vision_network_factory,
    morphology_override,
    VISION_CAMERA_OVERRIDES,
)


def build_stages(config, env_name):
    """Build stages, including explicit bullet-count progression for SafeBullets."""
    if env_name != "safe_bullets_humanoid":
        levels = list(config.levels)
        if config.level_steps is not None:
            stage_steps = [int(value) for value in config.level_steps]
        else:
            total_steps = int(config.num_timesteps)
            base_steps = total_steps // len(levels)
            stage_steps = [base_steps] * len(levels)
            stage_steps[-1] += total_steps - sum(stage_steps)

        if any(steps <= 0 for steps in stage_steps):
            raise ValueError("Every curriculum stage must have a positive step budget.")

        return curriculum.create_difficulty_curriculum(
            env_name=env_name,
            levels=levels,
            steps_per_level=stage_steps,
        )

    # Keep the environment and policy architecture the same while increasing
    # the number of incoming bullets. The level is still passed to the env.
    bullet_counts = [0, 1, 2]
    if len(config.levels) != len(bullet_counts):
        raise ValueError(
            "For safe_bullets_humanoid, provide exactly three --levels values "
            "for the 0-, 1-, and 2-bullet stages."
        )

    if config.level_steps is not None:
        stage_steps = [int(value) for value in config.level_steps]
    else:
        total_steps = int(config.num_timesteps)
        base_steps = total_steps // len(bullet_counts)
        stage_steps = [base_steps] * len(bullet_counts)
        stage_steps[-1] += total_steps - sum(stage_steps)

    if any(steps <= 0 for steps in stage_steps):
        raise ValueError(
            "Every curriculum stage must have a positive step budget. "
            "Increase --num_timesteps or provide positive --level_steps."
        )

    stages = []
    for level, steps, num_bullets in zip(
        config.levels, stage_steps, bullet_counts
    ):
        stages.append(
            Stage(
                env_name=env_name,
                level=int(level),
                num_steps=int(steps),
                kwargs_override={
                    "num_bullets": num_bullets,
                    "reset_noise_scale": 0.0 if num_bullets == 0 else 0.01,
                },
            )
        )
    return stages


def main():
    parser = build_base_parser(
        description="Curriculum training across difficulty levels"
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Difficulty levels to train on",
    )
    parser.add_argument(
        "--level_steps",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Training steps per stage, one per --levels entry "
            "(e.g. 1e8 5e7 5e7). If omitted, --num_timesteps is split "
            "as evenly as possible across stages."
        ),
    )
    config = parser.parse_args()

    if not config.levels:
        parser.error("--levels must contain at least one level.")
    if config.level_steps is not None:
        if len(config.level_steps) != len(config.levels):
            parser.error(
                f"--level_steps needs one value per level ({len(config.levels)}), "
                f"got {len(config.level_steps)}"
            )
        if any(value <= 0 for value in config.level_steps):
            parser.error("--level_steps values must all be positive.")

    alg_name = config.alg
    env_name = config.env_name
    use_wandb = config.use_wandb

    if config.vision_camera is None:
        config.vision_camera = (
            morphology_override(env_name, VISION_CAMERA_OVERRIDES) or "vision"
        )

    setup_gpu_environment(vision=config.vision)

    for seed in config.seeds:
        stages = build_stages(config, env_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cli_cfg = vars(config)
        runtime_cfg = {"seed": seed, "timestamp": timestamp}
        cfg = {**cli_cfg, **runtime_cfg}
        run_name = f"{env_name}_curriculum_{alg_name}_seed{seed}_{timestamp}"

        print(f"\n{'=' * 50}")
        print(f"Running experiment {run_name}")
        print(f"{'=' * 50}\n")
        print("\nCurriculum stages:")
        for index, stage in enumerate(stages):
            overrides = stage.kwargs_override or {}
            bullet_note = ""
            if "num_bullets" in overrides:
                bullet_note = (
                    f", bullets={overrides['num_bullets']}, "
                    f"reset_noise_scale={overrides['reset_noise_scale']}"
                )
            print(
                f"  {index + 1}. {stage.env_name}, level={stage.level}: "
                f"{stage.num_steps:,} steps{bullet_note}"
            )

        if use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                id=run_name,
                config=cfg.copy(),
                group=config.wandb_group if config.wandb_group else env_name,
                job_type=alg_name,
                tags=config.wandb_tags,
            )

        if config.store_model:
            root_dir = Path(__file__).parent.parent.resolve()
            checkpoint_root = root_dir / config.model_dir / run_name
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            cfg["save_checkpoint_path"] = checkpoint_root

        progress_fn = functools.partial(
            custom_progress_fn,
            use_wandb=use_wandb,
            verbose=not config.quiet,
        )

        train_fn_base = get_algorithm_train_fn(alg_name)
        train_kwargs = filter_kwargs_for_fn(train_fn_base, cfg)

        if config.vision:
            vision_kwargs = {
                "cameras": config.vision_cameras,
                "height": config.vision_height,
                "width": config.vision_width,
                "obs_mode": config.vision_obs_mode,
                "frame_stack": config.vision_frame_stack,
                "grayscale": config.vision_grayscale,
                "num_render_workers": config.vision_render_workers,
            }
            train_kwargs["vision"] = True
            train_kwargs["vision_kwargs"] = vision_kwargs
            state_obs_key = (
                "state" if config.vision_obs_mode == "pixels+state" else ""
            )
            train_kwargs["network_factory"] = make_vision_network_factory(
                alg_name,
                policy_obs_key=state_obs_key,
                value_obs_key=state_obs_key,
            )
            train_kwargs["augment_pixels"] = config.vision_augment

        policy_fn, final_params, results, eval_env = curriculum.train_curriculum(
            stages=stages,
            train_fn=train_fn_base,
            train_kwargs=train_kwargs,
            progress_fn=progress_fn,
            seed=seed,
        )

        print("\n" + "=" * 60)
        print("CURRICULUM TRAINING SUMMARY")
        print("=" * 60)
        for result in results:
            print(f"\nStage {result.stage_idx + 1}: {result.env_name}")
            print(f"  Steps: {result.num_steps:,}")
            print(f"  Time: {result.training_time:.1f}s")
            if "eval/episode_reward" in result.final_metrics:
                print(f"  Final reward: {result.final_metrics['eval/episode_reward']:.2f}")
            if "eval/episode_cost" in result.final_metrics:
                print(f"  Final cost: {result.final_metrics['eval/episode_cost']:.2f}")

        if not config.skip_video:
            video_length = (
                config.video_length
                if config.video_length
                else config.episode_length
            )
            if video_length is None:
                video_length = getattr(eval_env, "episode_length", None)
            record_episode_video(
                env=eval_env,
                make_inference_fn=policy_fn,
                params=final_params,
                steps=video_length,
                cameras=config.cameras,
                width=config.video_width,
                height=config.video_height,
                fps=config.video_fps,
                frame_stride=config.video_frame_stride,
                out_name=run_name,
                log_to_wandb=config.use_wandb,
                seed=seed,
                num_episodes=config.num_video_episodes,
            )

        if use_wandb and wandb.run is not None:
            wandb.finish()

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
