#!/usr/bin/env bash
# safe_dodge_humanoid without a curriculum: one run straight on level 3, then videos.
# 1e8 steps (as many as the curriculum in all): the first 1.5e8 run had stood up, crouched, stood up again and got stuck
# standing (cost ~29 > 25) by ~80M, so the rest added nothing.
# Same PPO settings and total steps as scripts/dodge_videos.sh (the curriculum), so the two compare
# directly. Runs from before 2026-10-15 used other level settings (no upright reward on level 3,
# fall cost 25) and don't compare with these.
#   nohup bash scripts/dodge_no_curriculum.sh > dodge_no_curriculum.log 2>&1 &
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
# Episodic cost limit. At 25 level 3 meets it with a slight knee bend (plane cost ~20) and never ducks
# the low planes; lower it to make ducking pay. Pass the same value to dodge_videos.sh to compare
SAFETY_BOUND=${SAFETY_BOUND:-25}
LEVEL=${LEVEL:-3}
# Total steps = the curriculum's (scripts/dodge_videos.sh) summed over its levels, so the two stay
# equal; pass the same LEVEL_STEPS to both to change them together
LEVEL_STEPS=${LEVEL_STEPS:-"2e7 5e7 3e7"}
STEPS=$(echo $LEVEL_STEPS | awk '{for (i = 1; i <= NF; i++) s += $i; printf "%g", s}')
NUM_ENVS=2048             # lower (with BATCH_SIZE) if the GPU runs out of memory
BATCH_SIZE=1024           # batch_size x num_minibatches must be divisible by NUM_ENVS
NUM_MINIBATCHES=32
SEEDS="69"
VIDEO_STEPS=1000          # 15 s
FPS=60
MODEL_DIR=models/dodge_nocurr_$(date +%Y%m%d_%H%M%S)
# Brax's humanoid PPO settings, the ones level 1 learned to stand with.
# 20 evals = one every 4M steps, as often as the 1.5e8 run's 40.
PPO_ARGS="--learning_rate 3e-4 --entropy_cost 1e-3 --discounting 0.97 --unroll_length 10
          --num_updates_per_batch 8 --deterministic_eval true --num_evals 20"

# 1. Training: a one-level curriculum, so the run folder and checkpoints look like dodge_videos.sh's
#    and dodge_policy_videos.py finds them unchanged
$PYTHON -m training.train_curriculum --env_name safe_dodge_humanoid --alg $ALG --levels $LEVEL \
    --pid_lambda_clip $LAMBDA_CLIP --safety_bound $SAFETY_BOUND \
    --level_steps $STEPS --num_envs $NUM_ENVS --batch_size $BATCH_SIZE \
    --num_minibatches $NUM_MINIBATCHES --seeds $SEEDS --model_dir $MODEL_DIR \
    --use_wandb false --skip_video $PPO_ARGS

# 2. Videos on level 3 (the policy never saw levels 1-2: it falls there)
$PYTHON scripts/dodge_policy_videos.py --runs_dir $MODEL_DIR --levels $LEVEL \
    --steps $VIDEO_STEPS --fps $FPS --cameras front side orbit
