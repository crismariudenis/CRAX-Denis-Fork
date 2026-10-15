# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for sequential curriculum training."""

import inspect
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

from crax import envs


class Stage(NamedTuple):
    """One stage in a curriculum."""
    env_name: str
    level: int
    num_steps: int
    kwargs_override: Optional[Dict[str, Any]] = None


class StageResult(NamedTuple):
    """Summary and metrics for one curriculum stage."""
    stage_idx: int
    env_name: str
    num_steps: int
    final_metrics: Dict[str, Any]
    training_time: float
    metrics_history: List[Dict[str, Any]]


def train_curriculum(
    stages: List[Stage],
    train_fn: Callable,
    train_kwargs: Dict[str, Any],
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    seed: int = 0,
):
    """Train stages in order, warm-starting each stage from the previous one.

    Training functions may return either:
      (make_policy, params, metrics, eval_env)
    or, when supporting full-state transfer:
      (make_policy, params, metrics, eval_env, training_state)
    """
    if not stages:
        raise ValueError("Curriculum must contain at least one stage.")

    all_results = []
    current_params = None
    current_training_state = None

    try:
        train_fn_parameters = inspect.signature(train_fn).parameters
    except (TypeError, ValueError):
        train_fn_parameters = {}

    supports_training_state = (
        "pretrained_state" in train_fn_parameters
        and "return_training_state" in train_fn_parameters
    )

    global_step = 0
    make_policy = None
    eval_env = None

    print(f"Starting curriculum training with {len(stages)} stages")
    print("=" * 60)

    for stage_idx, stage in enumerate(stages):
        print(f"\n[Stage {stage_idx + 1}/{len(stages)}] {stage.env_name}")
        print(f"  Level: {stage.level}")
        print(f"  Training budget: {stage.num_steps:,} steps")
        print("-" * 40)

        stage_kwargs = dict(train_kwargs)
        if stage.kwargs_override:
            stage_kwargs.update(stage.kwargs_override)

        # These values are used to construct the environment. Vision settings
        # are also passed to the trainer, so leave vision/vision_kwargs intact.
        env_creation_keys = (
            "backend",
            "vision",
            "vision_kwargs",
            "num_bullets",
            "reset_noise_scale",
        )
        env_creation_kwargs = {
            key: stage_kwargs[key]
            for key in env_creation_keys
            if key in stage_kwargs
        }

        # Environment-specific options are not arguments to the PPO trainer.
        stage_kwargs.pop("num_bullets", None)
        stage_kwargs.pop("reset_noise_scale", None)

        env = envs.get_environment(
            stage.env_name, level=stage.level, **env_creation_kwargs
        )
        eval_env = envs.get_environment(
            stage.env_name, level=stage.level, **env_creation_kwargs
        )

        episode_length = stage_kwargs.get("episode_length")
        if episode_length is None:
            episode_length = getattr(env, "episode_length", None)

        stage_kwargs["environment"] = env
        stage_kwargs["eval_env"] = eval_env
        stage_kwargs["num_timesteps"] = int(stage.num_steps)
        stage_kwargs["seed"] = seed + stage_idx
        stage_kwargs["episode_length"] = episode_length
        stage_kwargs.pop("env_name", None)

        # Each stage gets a separate checkpoint directory so checkpoints from
        # earlier stages are not overwritten by later stages.
        checkpoint_path = stage_kwargs.get("save_checkpoint_path")
        if checkpoint_path is not None:
            stage_kwargs["save_checkpoint_path"] = (
                Path(checkpoint_path) / f"stage_{stage_idx + 1}"
            )

        if current_params is not None:
            # Do not reload the original checkpoint on later stages.
            stage_kwargs["restore_checkpoint_path"] = None
            if supports_training_state and current_training_state is not None:
                stage_kwargs.pop("pretrained_params", None)
                stage_kwargs["pretrained_state"] = current_training_state
            else:
                stage_kwargs["pretrained_params"] = current_params

        if supports_training_state:
            stage_kwargs["return_training_state"] = True

        stage_metrics = []
        stage_start_step = global_step

        def stage_progress_fn(step: int, metrics: Dict[str, Any]):
            metrics_with_stage = dict(metrics)
            metrics_with_stage["curriculum_stage"] = stage_idx + 1
            metrics_with_stage["curriculum_env"] = stage.env_name
            metrics_with_stage["global_step"] = stage_start_step + step
            stage_metrics.append(metrics_with_stage)
            if progress_fn is not None:
                progress_fn(stage_start_step + step, metrics_with_stage)

        stage_kwargs["progress_fn"] = stage_progress_fn

        start_time = time.time()
        train_result = train_fn(**stage_kwargs)
        training_time = time.time() - start_time

        if not isinstance(train_result, (tuple, list)):
            raise TypeError(
                "train_fn must return a tuple/list with 4 or 5 elements; "
                f"got {type(train_result).__name__}."
            )
        if len(train_result) == 5:
            (
                make_policy,
                params,
                final_metrics,
                eval_env,
                current_training_state,
            ) = train_result
        elif len(train_result) == 4:
            make_policy, params, final_metrics, eval_env = train_result
            current_training_state = None
        else:
            raise ValueError(
                "train_fn must return 4 values, or 5 when returning training state; "
                f"got {len(train_result)}."
            )

        current_params = params
        global_step += int(stage.num_steps)

        result = StageResult(
            stage_idx=stage_idx,
            env_name=stage.env_name,
            num_steps=int(stage.num_steps),
            final_metrics=final_metrics,
            training_time=training_time,
            metrics_history=stage_metrics,
        )
        all_results.append(result)

        print(f"\n  Stage {stage_idx + 1} complete:")
        print(f"    Training time: {training_time:.1f}s")
        if "eval/episode_reward" in final_metrics:
            print(f"    Final reward: {final_metrics['eval/episode_reward']:.2f}")
        if "eval/episode_cost" in final_metrics:
            print(f"    Final cost: {final_metrics['eval/episode_cost']:.2f}")

    print("\n" + "=" * 60)
    print("Curriculum training complete!")
    print(f"Total stages: {len(stages)}")
    print(f"Total allocated steps: {global_step:,}")

    return make_policy, current_params, all_results, eval_env


def create_difficulty_curriculum(
    env_name: str,
    levels: List[int],
    steps_per_level: Union[int, List[int]],
) -> List[Stage]:
    """Create one stage for each requested difficulty level."""
    if not levels:
        raise ValueError("At least one difficulty level is required.")

    if isinstance(steps_per_level, int):
        steps_per_level = [steps_per_level] * len(levels)

    if len(steps_per_level) != len(levels):
        raise ValueError(
            f"Got {len(steps_per_level)} step counts for {len(levels)} levels."
        )
    if any(int(steps) <= 0 for steps in steps_per_level):
        raise ValueError("Every curriculum stage must have a positive step budget.")

    return [
        Stage(env_name=env_name, level=level, num_steps=int(steps))
        for level, steps in zip(levels, steps_per_level)
    ]


def split_budget_curriculum(
    env_names: List[str],
    total_steps: int,
    weights: Optional[List[float]] = None,
) -> List[Stage]:
    """Split a total budget across environments, respecting optional weights."""
    if not env_names:
        raise ValueError("At least one environment is required.")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    if weights is None:
        weights = [1.0] * len(env_names)
    if len(weights) != len(env_names):
        raise ValueError("weights must have the same length as env_names.")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative and sum to a positive value.")

    total_weight = sum(weights)
    step_budgets = [
        int(total_steps * weight / total_weight) for weight in weights
    ]
    # Assign integer-rounding remainder to the final stage.
    step_budgets[-1] += total_steps - sum(step_budgets)

    return [
        Stage(env_name=name, level=1, num_steps=steps)
        for name, steps in zip(env_names, step_budgets)
        if steps > 0
    ]
