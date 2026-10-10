#!/usr/bin/env bash
# Full safe_dodge_humanoid curriculum (levels 1 -> 2 -> 3), then videos of every run on every level.
#   nohup bash scripts/dodge_videos.sh > dodge_run.log 2>&1 &
set -e
cd "$(dirname "$0")/.."

ALG=ppo_pid
LAMBDA_CLIP=10            # max Lagrange multiplier
LEVEL_STEPS="8e7 6e7 6e7" # env steps per level
NUM_ENVS=2048             # lower (with BATCH_SIZE) if the GPU runs out of memory
BATCH_SIZE=1024           # batch_size x num_minibatches must be divisible by NUM_ENVS
NUM_MINIBATCHES=32
SEEDS="69"
VIDEO_STEPS=1000          # 15 s
FPS=60
MODEL_DIR=models/dodge_$(date +%Y%m%d_%H%M%S)
# Brax's humanoid PPO settings, the ones level 1 learned to stand with
PPO_ARGS="--learning_rate 3e-4 --entropy_cost 1e-3 --discounting 0.97 --unroll_length 10
          --num_updates_per_batch 8 --deterministic_eval true --num_evals 20"

# 1. Training (the curriculum's own video step crashes, so it is skipped)
python -m training.train_curriculum --env_name safe_dodge_humanoid --alg $ALG --levels 1 2 3 \
    --pid_lambda_clip $LAMBDA_CLIP \
    --level_steps $LEVEL_STEPS --num_envs $NUM_ENVS --batch_size $BATCH_SIZE \
    --num_minibatches $NUM_MINIBATCHES --seeds $SEEDS --model_dir $MODEL_DIR \
    --use_wandb false --skip_video $PPO_ARGS

# 2. Videos
python scripts/dodge_policy_videos.py --runs_dir $MODEL_DIR --levels 1 2 3 \
    --steps $VIDEO_STEPS --fps $FPS --cameras front side orbit
