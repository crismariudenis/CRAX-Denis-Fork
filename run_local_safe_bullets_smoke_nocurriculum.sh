
#!/usr/bin/env bash
# Local SafeBulletsHumanoid no-curriculum smoke test + video generation.
# Run from anywhere; paths are resolved relative to the repository root.

set -Eeuo pipefail
cd "$(dirname "$0")"

# Train directly on Level 3 (no curriculum).
ENV_NAME="safe_bullets_humanoid"
ALG="ppo_lag"
LEVEL=3
SEED=42

# Total training budget for this single level.
NUM_TIMESTEPS=1000

# Small settings for a local Mac smoke test.
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

MODEL_DIR="models/local_safe_bullets_level${LEVEL}_nocurr_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/local_smoke_test"

echo "=========================================="
echo "Local SafeBulletsHumanoid no-curriculum smoke test"
echo "Environment: ${ENV_NAME}"
echo "Algorithm: ${ALG}"
echo "Level: ${LEVEL}"
echo "Total timesteps: ${NUM_TIMESTEPS}"
echo "Model directory: ${MODEL_DIR}"
echo "=========================================="

# Train directly on Level 3.
python -m training.train_curriculum \
    --env_name "$ENV_NAME" \
    --alg "$ALG" \
    --levels "$LEVEL" \
    --num_timesteps "$NUM_TIMESTEPS" \
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
echo "Generating Level 3 smoke-test videos..."
echo "=========================================="

# Use GLFW by default for local Mac rendering.
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

python scripts/safe_bullets_policy_videos.py \
    --runs_dir "$MODEL_DIR" \
    --levels "$LEVEL" \
    --steps 1000 \
    --fps 30 \
    --width 640 \
    --height 480 \
    --cameras front side fixedfar orbit \
    --tag local_smoke_nocurr

echo "=========================================="
echo "Local no-curriculum smoke test complete!"
echo "Model directory: $MODEL_DIR"
echo "Videos: videos/"
echo "=========================================="
