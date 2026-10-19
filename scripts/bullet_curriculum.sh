#!/usr/bin/env bash
# safe_bullet_dodge_humanoid curriculum (levels 1 -> 2 -> 3 -> 4), then videos of each stage on its level
# and of the final policy on every level. Runs scripts/dodge_videos.sh with the bullet env's defaults;
# compare with scripts/bullet_no_curriculum.sh (same total steps).
#   nohup bash scripts/bullet_curriculum.sh > bullet_curriculum.log 2>&1 &
# Defaults:
#   - cost limit 10: the cost per hit is 100 / shots per episode on every level, so 10 = ~10% of shots hit
#   - multiplier cap 30, as for dodge
#   - 1e8 steps: 1.5e7 to stand (dodge level 1 stood by ~12M), 2e7 / 3e7 / 3.5e7 on the bullet levels
# Override any of them as for dodge_videos.sh, e.g. LEVELS="1 2 3" LEVEL_STEPS="2e7 3e7 5e7"
export ENV_NAME=safe_bullet_dodge_humanoid
export SAFETY_BOUND=${SAFETY_BOUND:-10}
export LAMBDA_CLIP=${LAMBDA_CLIP:-30}
export LEVELS=${LEVELS:-"1 2 3 4"}
export LEVEL_STEPS=${LEVEL_STEPS:-"1.5e7 2e7 3e7 3.5e7"}
exec bash "$(dirname "$0")/dodge_videos.sh" "$@"
