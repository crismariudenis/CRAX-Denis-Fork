#!/bin/bash
# SafeBulletsHumanoid Curriculum Training Script
# Runs PPO-Lagrangian across 3 difficulty levels

set -e  # Exit on error

# Configuration
ENV_NAME="safe_bullets_humanoid"
ALG="ppo_lag"
LEVELS="1 2 3"
SEED=42

# Training hyperparameters (adjust for your hardware)
NUM_TIMESTEPS=100000000      # 100M steps total (~33M per level)
NUM_ENVS=2048                # Parallel environments (reduce if OOM)
NUM_EVALS=5                  # Evaluation checkpoints
NUM_EVAL_ENVS=128            # Parallel eval envs
EPISODE_LENGTH=1000          # Steps per episode
NUM_MINIBATCHES=32
BATCH_SIZE=1024
UNROLL_LENGTH=10
NUM_UPDATES_PER_BATCH=4
DISCOUNTING=0.97
ENTROPY_COST=0.01
LEARNING_RATE=3e-4
SAFETY_BOUND=0.1             # Cost budget per episode

# Wandb (set to false to disable)
USE_WANDB=true
WANDB_PROJECT="crax"

# Output directories
MODEL_DIR="models"
OUT_DIR="runs/experimental_results"

echo "=========================================="
echo "SafeBulletsHumanoid Curriculum Training"
echo "=========================================="
echo "Env: $ENV_NAME"
echo "Algorithm: $ALG"
echo "Levels: $LEVELS"
echo "Total timesteps: $NUM_TIMESTEPS"
echo "Num envs: $NUM_ENVS"
echo "Seed: $SEED"
echo "=========================================="

# Activate conda environment (adjust path as needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate crax

# Run curriculum training
python -m training.train_curriculum \
  --env_name "$ENV_NAME" \
  --alg "$ALG" \
  --levels $LEVELS \
  --num_timesteps $NUM_TIMESTEPS \
  --num_envs $NUM_ENVS \
  --num_evals 5 \
  --num_eval_envs 128 \
  --episode_length 1000 \
  --num_minibatches 32 \
  --batch_size 1024 \
  --unroll_length 10 \
  --num_updates_per_batch 4 \
  --discounting 0.97 \
  --entropy_cost 0.01 \
  --learning_rate 3e-4 \
  --safety_bound 0.1 \
  --seed $SEED \
  --use_wandb $USE_WANDB \
  --wandb_project "crax" \
  --model_dir "$MODEL_DIR" \
  --out_dir "$OUT_DIR" \
  --store_model true

echo "=========================================="
echo "Training complete!"
echo "=========================================="