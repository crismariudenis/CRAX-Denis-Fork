#!/usr/bin/env bash
# Full safe_dodge_humanoid curriculum (levels 1 -> 2 -> 3), then videos: each stage's own policy on its
# level (sd_<id>_stage<n>_lvl<n>_*), and the final policy on every level (sd_<id>_lvl<level>_*).
# 1e8 steps in all by default (level 2 gets the most: it never met its budget in 6e7 before), the same as scripts/dodge_no_curriculum.sh, so the two compare directly.
#   nohup bash scripts/dodge_videos.sh > dodge_curriculum.log 2>&1 &
set -e
# Keep the machine from going to sleep while this runs: after a suspend the GPU job hangs.
# Skipped where systemd-inhibit isn't allowed (e.g. a cluster)
if [ -z "$DODGE_NO_SLEEP" ] && command -v systemd-inhibit >/dev/null \
        && systemd-inhibit --what=sleep:idle true 2>/dev/null; then
    DODGE_NO_SLEEP=1 exec systemd-inhibit --what=sleep:idle --why="dodge training" bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
# The venv's python, so the script also runs from a shell where the venv isn't active
PYTHON=python
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python

ALG=ppo_pid
LAMBDA_CLIP=10            # max Lagrange multiplier
SAFETY_BOUND=${SAFETY_BOUND:-25}              # episodic cost limit (see dodge_no_curriculum.sh)
LEVELS=${LEVELS:-"1 2 3"}                     # e.g. LEVELS=1 LEVEL_STEPS=3e7 bash scripts/dodge_videos.sh
LEVEL_STEPS=${LEVEL_STEPS:-"2e7 5e7 3e7"}     # env steps per level (level 1 stands by ~12M; level 3
                                              # from scratch by ~20M)
NUM_ENVS=2048             # lower (with BATCH_SIZE) if the GPU runs out of memory
BATCH_SIZE=1024           # batch_size x num_minibatches must be divisible by NUM_ENVS
NUM_MINIBATCHES=32
SEEDS="69"
VIDEO_STEPS=1000          # 15 s
FPS=60
MODEL_DIR=models/dodge_$(date +%Y%m%d_%H%M%S)
# Brax's humanoid PPO settings, the ones level 1 learned to stand with
PPO_ARGS="--learning_rate 3e-4 --entropy_cost 1e-3 --discounting 0.97 --unroll_length 10
          --num_updates_per_batch 8 --deterministic_eval true --num_evals 10"  # per stage

# 1. Training (the curriculum's own video step crashes, so it is skipped)
$PYTHON -m training.train_curriculum --env_name safe_dodge_humanoid --alg $ALG --levels $LEVELS \
    --pid_lambda_clip $LAMBDA_CLIP --safety_bound $SAFETY_BOUND \
    --level_steps $LEVEL_STEPS --num_envs $NUM_ENVS --batch_size $BATCH_SIZE \
    --num_minibatches $NUM_MINIBATCHES --seeds $SEEDS --model_dir $MODEL_DIR \
    --use_wandb false --skip_video $PPO_ARGS

# 2. Videos of each stage's policy on the level it trained on
STAGE=1
for LEVEL in $LEVELS; do
    $PYTHON scripts/dodge_policy_videos.py --runs_dir $MODEL_DIR --stage $STAGE --levels $LEVEL \
        --steps $VIDEO_STEPS --fps $FPS --cameras front side orbit
    STAGE=$((STAGE + 1))
done

# 3. Videos of the final policy on every level
$PYTHON scripts/dodge_policy_videos.py --runs_dir $MODEL_DIR --levels $LEVELS \
    --steps $VIDEO_STEPS --fps $FPS --cameras front side orbit
