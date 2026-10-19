#!/usr/bin/env bash
# safe_bullet_dodge_humanoid without a curriculum: one run straight on the last curriculum level (4, the
# Matrix, by default), with as many steps as scripts/bullet_curriculum.sh in all, then videos.
# Runs scripts/dodge_no_curriculum.sh with the bullet env's defaults.
#   nohup bash scripts/bullet_no_curriculum.sh > bullet_no_curriculum.log 2>&1 &
# Same limit (10, ~10% of shots hit), cap (30) and total steps (sum of LEVEL_STEPS) as the curriculum,
# so the two compare directly; pass the same overrides to both, e.g. LEVEL=3 LEVEL_STEPS="2e7 3e7 5e7"
export ENV_NAME=safe_bullet_dodge_humanoid
export SAFETY_BOUND=${SAFETY_BOUND:-10}
export LAMBDA_CLIP=${LAMBDA_CLIP:-30}
export LEVEL=${LEVEL:-4}
export LEVEL_STEPS=${LEVEL_STEPS:-"1.5e7 2e7 3e7 3.5e7"}
exec bash "$(dirname "$0")/dodge_no_curriculum.sh" "$@"
