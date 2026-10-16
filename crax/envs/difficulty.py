"""
Centralized difficulty mapping for safety environments.

This module defines a small, extensible system to translate a difficulty
level (1, 2, 3) into environment-specific parameter overrides.

It is intentionally lightweight and modular: add new env handlers or tweak
mappings in a single place without touching training code or env classes.

"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

# ============================================================================
# Task-based difficulty configurations
# These define difficulty levels for each task type, independent of agent
# ============================================================================

_TASK_DIFFICULTY_CONFIGS: dict[str, dict[int, dict[str, Any]]] = {
    # Goal navigation task - navigate to goal while avoiding hazards
    "goal": {
        1: {
            "goal_type": "cylinder",
            "goal_count": 2,
            "goal_size": 0.2,
            "goal_height": 0.2,
            "hazard_specs": [
                {"type": "cylinder", "count": 12, "size": 0.4, "height": 0.01, "collidable": False},
                {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                 "fixed": True},
            ],
        },
        2: {
            "goal_type": "cylinder",
            "goal_count": 2,
            "goal_size": 0.18,
            "goal_height": 0.2,
            "hazard_specs": [
                {"type": "cylinder", "count": 8, "size": 0.4, "height": 0.01, "collidable": False},
                {"type": "cylinder", "count": 8, "size": 0.3, "height": 0.4, "collidable": True},
                {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                 "fixed": True},
            ],
        },
        3: {
            "goal_type": "cylinder",
            "goal_count": 2,
            "goal_size": 0.16,
            "goal_height": 0.2,
            "hazard_specs": [
                {"type": "cube", "count": 6, "size": 0.3, "height": 0.01, "collidable": False},
                {"type": "cube", "count": 4, "size": 0.25, "height": 0.5, "collidable": True},
                {"type": "cylinder", "count": 6, "size": 0.35, "height": 0.01, "collidable": False},
                {"type": "cylinder", "count": 4, "size": 0.25, "height": 0.4, "collidable": True},
                {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                 "fixed": True},
            ],
        },
    },

    # Circle task - navigate in circles while staying within boundaries
    "circle": {
        # Level 1 (vertical walls)
        1: {
            "boundary_x": 1.125,
            "boundary_y": None,
        },
        # Level 2 (square boundary, 1 randomly placed hazard)
        2: {
            "boundary_x": 1.05,
            "boundary_y": 1.05,
            "hazard_specs": [
                {"type": "cylinder", "count": 1, "size": 0.15, "height": 0.15, "alpha_transparent": 1.0,
                 "collidable": False, "fixed": False},
            ],
        },
        # Level 3 (smaller boundary, 2 randomly placed hazards)
        3: {
            "boundary_x": 0.975,
            "boundary_y": 0.975,
            "hazard_specs": [
                {"type": "cylinder", "count": 2, "size": 0.2, "height": 0.2, "alpha_transparent": 1.0,
                 "collidable": False, "fixed": False},
            ],
        },
    },

    # Button task - press the correct button among multiple buttons
    "button": {
        # Level 1: Hazards and gremlins, constrained buttons
        1: {
            "placement_extents": (-2.0, -2.0, 2.0, 2.0),
            "buttons_constrained": True,
            "hazard_specs": [
                {"type": "cylinder", "count": 4, "size": 0.2, "height": 0.2, "collidable": True, "fixed": False},
                {"type": "gremlin", "count": 4, "size": 0.1, "height": 0.1, "travel": 0.35, "collidable": True,
                 "fixed": False},
            ],
        },
        # Level 2: More hazards and gremlins
        2: {
            "placement_extents": (-2.5, -2.5, 2.5, 2.5),
            "buttons_constrained": True,
            "hazard_specs": [
                {"type": "cylinder", "count": 8, "size": 0.2, "height": 0.2, "collidable": True, "fixed": False},
                {"type": "gremlin", "count": 6, "size": 0.1, "height": 0.1, "travel": 0.35, "collidable": True,
                 "fixed": False},
            ],
        },
        # Level 3: More hazards and larger orbits; reserve enough area for keepouts
        3: {
            "placement_extents": (-3.0, -3.0, 3.0, 3.0),
            "buttons_constrained": True,
            "hazard_specs": [
                {"type": "cylinder", "count": 12, "size": 0.2, "height": 0.2, "collidable": True, "fixed": False},
                {"type": "gremlin", "count": 8, "size": 0.1, "height": 0.1, "travel": 0.45, "collidable": True,
                 "fixed": False},
            ],
        },
    },

    # Push task - push a block to a goal
    "push": {
        # Level 1: Stationary goal
        1: {
            "goal_velocity": 0.0,
        },
        # Level 2: Slow moving goal
        2: {
            "goal_velocity": 0.3,
        },
        # Level 3: Fast moving goal
        3: {
            "goal_velocity": 0.6,
        },
    },

    # Pathway task - traverse hazard corridor (formerly "run")
    "pathway": {
        1: {"max_gap": 6.0},
        2: {"max_gap": 4.0},
        3: {"max_gap": 2.0},
    },

    # Height task - maintain head below height threshold
    "height": {
        1: {"max_height": 1.20},
        2: {"max_height": 1.10},
        3: {"max_height": 1.00},
    },

    # Dodge task - keep every body part out of a tilted hazard plane
    "dodge": {
        # Level 1: Stand upright on the platform, no plane and no fall cost. The hidden plane moves like
        # level 2's (a quarter of them sliding through like level 3's), so the observation normalizer
        # knows every level's plane ranges before they appear. Without the sliding ones, the plane's
        # sideways speed was 0 for levels 1-2, its normalizer std hit the 1e-6 floor, and level 3's
        # 1.5 m/s came out as ~1e6: the level-2 policy fell at once on level 3
        1: {"plane_enabled": False, "fall_cost": 0.0, "upright_reward_weight": 5.0,
            "plane_height_range": [1.0, 1.9], "plane_tilt_range": 20.0, "plane_offset_range": 0.3,
            "plane_resample_interval": 250, "plane_drop_speed": 0.75,
            "plane_slide_fraction": 0.25, "plane_slide_speed": 1.5, "plane_slide_distance": 3.0},
        # Levels 2-3 duck with the knees, not the waist: the torso stays vertical (upright reward) and
        # ducking to a 1.0 m torso costs only 10% of the height reward, so a knee duck beats a waist bend.
        # A fall costs 100, more than standing under every plane (~90 on level 2), so falling on purpose
        # never pays
        # Level 2: Every 3.75 s a plane at a random height, tilt and offset comes down from above and stops;
        # a quarter of them slide through from 3 m away instead (the plane is ~2.2 m half-size here)
        2: {"plane_enabled": True, "plane_height_range": [1.0, 1.9], "plane_tilt_range": 20.0,
            "plane_offset_range": 0.3, "plane_resample_interval": 250, "plane_drop_speed": 0.75,
            "plane_slide_fraction": 0.25, "plane_slide_speed": 1.5, "plane_slide_distance": 3.0,
            "upright_reward_weight": 3.0, "height_reward_full_z": 1.0, "height_reward_full_frac": 0.9,
            "fall_cost": 100.0},
        # Level 3: 1.5 m square planes fly through from random directions, one every ~2.7 s
        3: {"plane_enabled": True, "plane_slide_speed": 1.5, "plane_slide_distance": 2.0,
            "plane_half_size": 0.75, "plane_tilt_range": 20.0,
            "plane_height_range": [1.0, 1.7],
            "upright_reward_weight": 3.0, "height_reward_full_z": 1.0, "height_reward_full_frac": 0.9,
            "fall_cost": 100.0},
        # Level 4: "Bullets" as in The Matrix. Level 3's 1.5 m square, but every 3 s, at 6 m/s, at the
        # front (within 20 deg) and head/chest height, its leading edge starting ~1.65 m out: ~0.25 s
        # warning. In that time a squat (the head falls at most at g) gets the head down ~0.3 m at best;
        # leaning back, driven by the hips and knees, is not limited that way. The tilt (up to 20 deg)
        # decides which way is clear. Nothing rewards a lean: whether it is found is the open question.
        # Only the falls count differently: anything but the feet on the floor is a fall (the torso may go
        # down to 0.3 m), and the upright reward is waived while a plane is within 1 m, so tipping back to
        # dodge loses none of it. The height reward is unchanged, so it still has to stand back up.
        # A plane crosses the body in ~0.3 s (vs ~1.2 s on level 3), so the cost weight is raised to keep
        # a hit at ~6-12 (estimate): ignoring all 5 is over the limit (25), falling (100) still never pays
        4: {"plane_enabled": True, "plane_resample_interval": 200, "plane_slide_fraction": 1.0,
            "plane_slide_speed": 6.0, "plane_slide_distance": 2.4,
            "plane_slide_heading": 180.0, "plane_slide_heading_range": 20.0,
            "plane_half_size": 0.75, "plane_tilt_range": 20.0,
            "plane_height_range": [1.3, 1.5], "plane_cost_weight": 3.0,
            "fall_on_floor_contact": True, "healthy_z_range": [0.3, 2.0],
            "upright_reward_weight": 3.0, "upright_free_distance": 1.0,
            "height_reward_full_z": 1.0, "height_reward_full_frac": 0.9,
            "fall_cost": 100.0},
    },

    # Lift task for Ant - keep certain feet off the ground
    "lift_ant": {
        # Level 1: Front-left leg must stay off ground
        1: {"restricted_feet": ["front_left"]},
        # Level 2: Diagonal legs (front-left and back-right) must stay off
        2: {"restricted_feet": ["front_left", "back_right"]},
        # Level 3: Only back-right can touch (all others restricted)
        3: {"restricted_feet": ["front_left", "front_right", "back_left"]},
    },

    # Lift task for Spider - keep certain feet off the ground
    "lift_spider": {
        # Level 1: Keep 2 legs up (front-left + back-right diagonal)
        1: {"restricted_feet": ["front_left", "back_right"]},
        # Level 2: Keep 3 legs up (alternating tripod)
        2: {"restricted_feet": ["front_left", "mid_right", "back_left"]},
        # Level 3: Keep 4 legs up (only center legs may touch)
        3: {"restricted_feet": ["front_left", "front_right", "back_left", "back_right"]},
    },

    # Reach task - reach target while avoiding hazards
    "reach": {
        1: {"num_hazards": 4},
        2: {"num_hazards": 7},
        3: {"num_hazards": 10},
    },

    # Velocity task - maintain velocity below threshold
    "velocity": {
        # The actual threshold is computed in safe_velocity.py based on (agent, level)
        1: {"level": 1},
        2: {"level": 2},
        3: {"level": 3},
    },
}

# ============================================================================
# Environment name to task mapping
# Maps both old and new naming conventions to task configurations
# ============================================================================

_ENV_TO_TASK: dict[str, str] = {
    # New naming convention: safe_[task]_[agent]
    "safe_goal_point": "goal",
    "safe_circle_point": "circle",
    "safe_button_point": "button",
    "safe_push_point": "push",
    "safe_pathway_walker2d": "pathway",
    "safe_height_humanoid": "height",
    "safe_dodge_humanoid": "dodge",
    "safe_lift_ant": "lift_ant",
    "safe_lift_spider": "lift_spider",
    "safe_reacher": "reach",
}


def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges src into dst and returns dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


def get_task_for_env(env_name: str) -> str | None:
    """Returns the task type for an environment name.

    Args:
        env_name: Name of the environment (new or old naming convention)

    Returns:
        Task type string (e.g., "goal", "circle") or None if not found
    """
    return _ENV_TO_TASK.get(env_name)


def get_supported_levels(env_name: str) -> list[int]:
    """Returns the list of supported difficulty levels for an environment.

    Args:
        env_name: Name of the environment

    Returns:
        List of supported levels (e.g., [1, 2, 3]) or empty list if not supported
    """
    task = _ENV_TO_TASK.get(env_name)
    if task is None or task not in _TASK_DIFFICULTY_CONFIGS:
        return []
    return sorted(_TASK_DIFFICULTY_CONFIGS[task].keys())


def supports_difficulty(env_name: str) -> bool:
    """Check if an environment supports difficulty levels."""
    task = _ENV_TO_TASK.get(env_name)
    return task is not None and task in _TASK_DIFFICULTY_CONFIGS


def apply_difficulty(env_name: str, env_kwargs: dict[str, Any] | None, level: int) -> dict[str, Any]:
    """Apply difficulty-level overrides to environment kwargs.

    Args:
        env_name: Name of the environment
        env_kwargs: User-provided environment kwargs (can be None)
        level: Difficulty level (1, 2, or 3)

    Returns:
        Merged kwargs dict with difficulty overrides applied first, then env_kwargs
    """
    task = _ENV_TO_TASK.get(env_name)

    if task is None:
        print(f"Warning: Environment '{env_name}' does not have a task mapping for difficulty levels.")
        return env_kwargs or {}

    if task not in _TASK_DIFFICULTY_CONFIGS:
        print(f"Warning: Task '{task}' does not have difficulty configurations defined.")
        return env_kwargs or {}

    if level not in _TASK_DIFFICULTY_CONFIGS[task]:
        print(f"Warning: Level {level} not defined for task '{task}'. Available levels: {list(_TASK_DIFFICULTY_CONFIGS[task].keys())}")
        return env_kwargs or {}

    env_kwargs = deepcopy(env_kwargs or {})
    overrides = deepcopy(_TASK_DIFFICULTY_CONFIGS[task][level])

    # All envs use flat kwargs: merge overrides then env_kwargs (env_kwargs wins)
    out = _merge_dict(deepcopy(overrides), deepcopy(env_kwargs))
    return out


def register_task_difficulty(task_name: str, level: int, config: dict[str, Any]) -> None:
    """Register or update a difficulty configuration for a task.

    Args:
        task_name: Name of the task (e.g., "goal", "circle")
        level: Difficulty level (typically 1, 2, or 3)
        config: Configuration dict to apply at this level
    """
    if task_name not in _TASK_DIFFICULTY_CONFIGS:
        _TASK_DIFFICULTY_CONFIGS[task_name] = {}
    _TASK_DIFFICULTY_CONFIGS[task_name][level] = config


def register_env_task_mapping(env_name: str, task_name: str) -> None:
    """Register a mapping from environment name to task type.

    Args:
        env_name: Name of the environment
        task_name: Name of the task this environment uses
    """
    _ENV_TO_TASK[env_name] = task_name
