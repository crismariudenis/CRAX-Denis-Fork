#!/usr/bin/env bash
# Local SafeBulletsHumanoid curriculum smoke test + video generation.
# Run from anywhere; paths are resolved relative to the repository root.

set -Eeuo pipefail
cd "$(dirname "$0")"

# Keep this small for a local Mac smoke test.
ENV_NAME="safe_bullets_humanoid"
ALG="ppo_lag"
LEVELS=(1 2 3)
SEED=42

# This is 1,000 training timesteps PER level (3,000 total).
# Use --num_timesteps 1000 instead if you specifically want 1,000 total,
# but that only gives ~333 steps per level and may be too short for PPO updates.
STEPS_PER_LEVEL=1000

NUM_ENVS=8
EPISODE_LENGTH=1000
UNROLL_LENGTH=10
BATCH_SIZE=32
NUM_MINIBATCHES=2
NUM_UPDATES_PER_BATCH=1
NUM_EVALS=0
NUM_EVAL_ENVS=1
DISCOUNTING=0.97
ENTROPY_COST=0.01
LEARNING_RATE=3e-4
SAFETY_BOUND=0.1

export WANDB_MODE=disabled
MODEL_DIR="models/local_safe_bullets_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/local_smoke_test"

echo "=========================================="
echo "Local SafeBulletsHumanoid smoke test"
echo "Environment: ${ENV_NAME}"
echo "Algorithm: ${ALG}"
echo "Levels: ${LEVELS[*]}"
echo "Timesteps per level: ${STEPS_PER_LEVEL}"
echo "Total timesteps: $((STEPS_PER_LEVEL * ${#LEVELS[@]}))"
echo "Model directory: ${MODEL_DIR}"
echo "=========================================="

# Train and save checkpoints. Small batch settings are intended for a smoke test,
# not for meaningful policy learning.
python -m training.train_curriculum \
    --env_name "$ENV_NAME" \
    --alg "$ALG" \
    --levels "${LEVELS[@]}" \
    --level_steps "$STEPS_PER_LEVEL" "$STEPS_PER_LEVEL" "$STEPS_PER_LEVEL" \
    --num_envs "$NUM_ENVS" \
    --num_evals "$NUM_EVALS" \
    --num_eval_envs "$NUM_EVAL_ENVS" \
    --episode_length "$EPISODE_LENGTH" \
    --num_minibatches "$NUM_MINIBATCHES" \
    --batch_size "$BATCH_SIZE" \
    --unroll_length "$UNROLL_LENGTH" \
    --num_updates_per_batch "$NUM_UPDATES_PER_BATCH" \
    --discounting "$DISCOUNTING" \
    --entropy_cost "$ENTROPY_COST" \
    --learning_rate "$LEARNING_RATE" \
    --safety_bound "$SAFETY_BOUND" \
    --seeds "$SEED" \
    --use_wandb false \
    --model_dir "$MODEL_DIR" \
    --out_dir "$OUT_DIR" \
    --store_model true \
    --skip_rollout \
    --skip_video

echo "=========================================="
echo "Generating smoke-test videos..."
echo "=========================================="

# The updated video script must support the 'orbit' camera and any desired
# dodge-counter overlay. It will select the latest checkpoint in each run.
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
python scripts/safe_bullets_policy_videos.py \
    --runs_dir "$MODEL_DIR" \
    --levels "${LEVELS[@]}" \
    --steps 1000 \
    --fps 30 \
    --width 640 \
    --height 480 \
    --cameras front side fixedfar orbit \
    --tag local_smoke

echo "=========================================="
echo "Local training and video generation complete!"
echo "Models: $MODEL_DIR"
echo "Videos: videos/"
echo "=========================================="