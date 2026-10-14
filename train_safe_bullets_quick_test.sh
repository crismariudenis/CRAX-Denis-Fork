#!/bin/bash
#SBATCH --job-name=safe_bullets_quick_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:10:00
#SBATCH --output=safe_bullets_quick_test_%j.out
#SBATCH --error=safe_bullets_quick_test_%j.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd $HOME/repos/CRAX-Denis-Fork
source env/bin/activate
export MUJOCO_GL=egl

ENV_NAME="safe_bullets_humanoid"
ALG="ppo_lag"
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
echo "SafeBulletsHumanoid - QUICK COMPILE TEST (Per-Level Videos)"
echo "=========================================="
echo "This will run ~100 steps per level with separate videos"
echo "Expected time: ~3 minutes"
echo "=========================================="

# Run separate training for each level to get per-level videos
for LEVEL in 1 2 3; do
    echo "=========================================="
    echo "Training Level $LEVEL"
    echo "=========================================="
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S_%f)
    RUN_NAME="${ENV_NAME}_Level${LEVEL}_${ALG}_seed${SEED}_${TIMESTAMP}"
    
    python -m training.train_env \
      --env_name "${ENV_NAME}" \
      --alg "${ALG}" \
      --difficulty ${LEVEL} \
      --num_timesteps ${NUM_TIMESTEPS} \
      --num_envs ${NUM_ENVS} \
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
      --seed ${SEED} \
      --use_wandb false \
      --skip_rollout \
      --video_length 50 \
      --num_video_episodes 1
done

echo "=========================================="
echo "All levels complete! Videos in ./videos/"
echo "=========================================="
      --use_wandb false \
      --skip_rollout \
      --video_length 50 \
      --num_video_episodes 1
done

echo "=========================================="
echo "All levels complete! Videos in ./videos/"
echo "=========================================="