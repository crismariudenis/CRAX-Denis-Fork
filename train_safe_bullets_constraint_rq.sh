#!/bin/bash
#SBATCH --job-name=bullets_rq2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --array=0,10
#SBATCH --output=logs/rq/%x_%A_%a.out
#SBATCH --error=logs/rq/%x_%A_%a.err

# RQ2 (constraint vs. reward engineering), NO curriculum: every run trains on the
# final level (6 bullets) only.  One SLURM array task = one (condition, seed) run:
#
#   condition 0: b1_pid      PPO-PID, budget SAFETY_BOUND      (constrained agent)
#   condition 1: b2_cw1      PPO-Cost, reward - 1    * cost    (fixed penalty)
#   condition 2: b2_cw10     PPO-Cost, reward - 10   * cost
#   condition 3: b2_cw100    PPO-Cost, reward - 100  * cost
#   condition 4: b2_cw1000   PPO-Cost, reward - 1000 * cost
#
# task id = condition * NUM_SEEDS + seed   ->  5 conditions x 5 seeds = --array=0-24
#
# Usage (from the repo root; logs/rq must exist before sbatch):
#   mkdir -p logs/rq && sbatch train_safe_bullets_constraint_rq.sh
#   sbatch --array=0-4 train_safe_bullets_constraint_rq.sh      # only b1_pid, seeds 0-4
#   sbatch --array=3,8  train_safe_bullets_constraint_rq.sh     # re-run single tasks
# Quick check without SLURM (tiny CPU run):
#   SLURM_ARRAY_TASK_ID=0 NUM_TIMESTEPS=2000 NUM_ENVS=8 NUM_EVAL_ENVS=4 EPISODE_LENGTH=50 \
#   NUM_MINIBATCHES=2 BATCH_SIZE=4 UNROLL_LENGTH=5 NUM_UPDATES_PER_BATCH=1 NUM_EVALS=2 \
#   bash train_safe_bullets_constraint_rq.sh
#
# Requires the patched training/agents/ppo/ppo_cost.py (accepts num_evals, num_eval_envs, ...).

set -e

if command -v module >/dev/null 2>&1; then
    module load 2023
    module load Python/3.11.3-GCCcore-12.3.0
fi

cd "${SLURM_SUBMIT_DIR:-$HOME/repos/CRAX-Denis-Fork}"
[ -f env/bin/activate ] && source env/bin/activate

export MUJOCO_GL=egl
export WANDB_MODE=disabled

# Configuration (identical for every condition; override via environment variables)
ENV_NAME="safe_bullets_humanoid"
DIFFICULTY=3                                  # final level: 6 bullets
NUM_SEEDS=5
CONDITIONS=(b1_pid b2_cw1 b2_cw10 b2_cw100 b2_cw1000)

NUM_TIMESTEPS=${NUM_TIMESTEPS:-500000}
NUM_ENVS=${NUM_ENVS:-256}
NUM_EVALS=${NUM_EVALS:-10}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-128}
EPISODE_LENGTH=${EPISODE_LENGTH:-1000}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-32}
BATCH_SIZE=${BATCH_SIZE:-1024}
UNROLL_LENGTH=${UNROLL_LENGTH:-10}
NUM_UPDATES_PER_BATCH=${NUM_UPDATES_PER_BATCH:-4}
DISCOUNTING=0.97
ENTROPY_COST=0.01
LEARNING_RATE=3e-4
SAFETY_BOUND=${SAFETY_BOUND:-0.1}             # episodic cost budget d (per-step bound = d / episode length)
PID_LAMBDA_CLIP=10
 
TASK=${SLURM_ARRAY_TASK_ID:?"run via sbatch --array or set SLURM_ARRAY_TASK_ID"}
COND=${CONDITIONS[$((TASK / NUM_SEEDS))]}
SEED=$((TASK % NUM_SEEDS))
[ -n "$COND" ] || { echo "array task $TASK is out of range"; exit 1; }
 
case "$COND" in
    b1_pid)  ALG_ARGS=(--alg ppo_pid --safety_bound "$SAFETY_BOUND" --pid_lambda_clip "$PID_LAMBDA_CLIP") ;;
    b2_cw*)  ALG_ARGS=(--alg ppo_cost --cost-weight "${COND#b2_cw}") ;;
esac
 
# runs_root for scripts/rq_analyze.py is models/rq (<condition>/seed<k>)
MODEL_DIR="models/rq/$COND/seed$SEED"
 
echo "=========================================="
echo "RQ2 run: $COND  seed $SEED  (array task $TASK)"
echo "Environment: $ENV_NAME  difficulty $DIFFICULTY (final level only, no curriculum)"
echo "Timesteps: $NUM_TIMESTEPS   envs: $NUM_ENVS   safety bound: $SAFETY_BOUND"
echo "Model directory: $MODEL_DIR"
echo "=========================================="
 
python -m training.train_env \
    --env_name "$ENV_NAME" \
    "${ALG_ARGS[@]}" \
    --difficulty "$DIFFICULTY" \
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
    --seeds "$SEED" \
    --use_wandb false \
    --skip_video --skip_rollout \
    --model_dir "$MODEL_DIR" \
    --store_model
 
echo "Done: $COND seed $SEED"