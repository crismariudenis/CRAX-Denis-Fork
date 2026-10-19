
#!/usr/bin/env bash
# SafeBulletsHumanoid: train directly on level 3 (no curriculum).
# Generates videos after training completes.

set -e
cd "$(dirname "$0")/.."

# Configuration
ENV_NAME="safe_bullets_humanoid"
ALG="ppo_lag"
LEVEL=3
SEED=42

NUM_TIMESTEPS=100000000
NUM_ENVS=2048
NUM_EVALS=5
NUM_EVAL_ENVS=128
EPISODE_LENGTH=1000
NUM_MINIBATCHES=32
BATCH_SIZE=1024
UNROLL_LENGTH=10
NUM_UPDATES_PER_BATCH=4
DISCOUNTING=0.97
ENTROPY_COST=0.01
LEARNING_RATE=3e-4
SAFETY_BOUND=0.1

# Disable Weights & Biases
USE_WANDB=false
export WANDB_MODE=disabled

MODEL_DIR="models/safe_bullets_level3_nocurr_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/experimental_results"

echo "=========================================="
echo "SafeBulletsHumanoid - No Curriculum"
echo "=========================================="
echo "Environment: $ENV_NAME"
echo "Algorithm: $ALG"
echo "Level: $LEVEL (direct training)"
echo "Timesteps: $NUM_TIMESTEPS"
echo "Model directory: $MODEL_DIR"
echo "Weights & Biases: disabled"
echo "=========================================="

# Train directly on level 3
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
    --seed "$SEED" \
    --use_wandb "$USE_WANDB" \
    --model_dir "$MODEL_DIR" \
    --out_dir "$OUT_DIR" \
    --store_model true

echo "=========================================="
echo "Generating Level 3 videos..."
echo "=========================================="

python scripts/safe_bullets_policy_videos.py \
    --runs_dir "$MODEL_DIR" \
    --levels "$LEVEL" \
    --steps 1000 \
    --fps 60 \
    --cameras front side fixedfar orbit

echo "=========================================="
echo "Training and video generation complete!"
echo "=========================================="
