#!/bin/bash
# SafeBulletsHumanoid Curriculum Training - QUICK COMPILE TEST
# Minimal run to verify compilation and param transfer (1-2 min)

set -e

ENV_NAME="safe_bullets_humanoid"
ALG="ppo_lag"
LEVELS="1 2 3"
SEED=42

# Minimal settings for compile check only
NUM_TIMESTEPS=100
NUM_ENVS=8
NUM_EVALS=1
NUM_EVAL_ENVS=4
EPISODE_LENGTH=50
NUM_MINIBATCHES=2
BATCH_SIZE=16
UNROLL_LENGTH=2
NUM_UPDATES_PER_BATCH=1
DISCOUNTING=0.97
ENTROPY_COST=0.01
LEARNING_RATE=3e-4
SAFETY_BOUND=0.1
USE_WANDB=false

echo "=========================================="
echo "SafeBulletsHumanoid - QUICK COMPILE TEST"
echo "=========================================="
echo "This will run ~100 steps total (~33 per level)"
echo "Expected time: < 1 minute"
echo "=========================================="

python -m training.train_curriculum \
  --env_name "safe_bullets_humanoid" \
  --alg "ppo_lag" \
  --levels 1 2 3 \
  --num_timesteps 100 \
  --num_envs 8 \
  --num_evals 1 \
  --num_eval_envs 4 \
  --episode_length 50 \
  --num_minibatches 2 \
  --batch_size 16 \
  --unroll_length 2 \
  --num_updates_per_batch 1 \
  --discounting 0.97 \
  --entropy_cost 0.01 \
  --learning_rate 3e-4 \
  --safety_bound 0.1 \
  --seed 42 \
  --use_wandb false \
  --skip_video \
  --skip_rollout

echo "=========================================="
echo "Compile test passed!"
echo "=========================================="