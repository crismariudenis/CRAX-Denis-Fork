#!/usr/bin/env bash
# Full safe_dodge_humanoid training on a server, then videos of the result.
#   1. Curriculum training through levels 1 -> 2 -> 3 (1: no plane, 2: still plane,
#      3: sliding plane), each level starting from the previous level's policy. One run per seed.
#   2. Videos of every run's final policy on every level from the front, side and orbit cameras,
#      15 s each; a fall resets the episode and the video keeps going.
#      Output: videos/sd_<run id>_lvl<level>_<camera>.mp4 (the log says which run each id is)
#
# Edit the settings below, then start it so it keeps running after you log out:
#   nohup bash scripts/dodge_videos.sh > dodge_run.log 2>&1 &
#   tail -f dodge_run.log
set -e
cd "$(dirname "$0")/.."

TIMESTEPS=1e8     # total env steps per run, split evenly over the levels (~33M each)
NUM_ENVS=1024     # parallel envs, lower if the GPU runs out of memory (any number works)
# PPO needs batch_size x num_minibatches divisible by NUM_ENVS; tying the batch to NUM_ENVS
# always satisfies that (at 1024 it equals the defaults, 1024 x 32)
BATCH_SIZE=$NUM_ENVS
NUM_MINIBATCHES=32
SEEDS="69"     # one full curriculum run per seed, one after the other
VIDEO_STEPS=1000  # one step is 0.015 s, so 1000 steps = 15 s per video
FPS=60            # ~67 is real time; lower = slow motion
# This run's own checkpoint folder, so the videos only pick up runs started by this script
MODEL_DIR=models/dodge_$(date +%Y%m%d_%H%M%S)

# 1. Training. The curriculum's built-in video step is skipped because it crashes (it records
# from the batched eval env); step 2 records from the saved checkpoints instead.
python -m training.train_curriculum --env_name safe_dodge_humanoid --alg ppo_lag --levels 1 2 3 \
    --num_timesteps $TIMESTEPS --num_envs $NUM_ENVS --batch_size $BATCH_SIZE \
    --num_minibatches $NUM_MINIBATCHES --seeds $SEEDS --model_dir $MODEL_DIR \
    --use_wandb false --skip_video

# 2. Videos. To re-record later without training: the same command with the run's models/ folder
python scripts/dodge_policy_videos.py --runs_dir $MODEL_DIR --levels 1 2 3 \
    --steps $VIDEO_STEPS --fps $FPS --cameras front side orbit
