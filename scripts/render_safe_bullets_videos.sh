
#!/bin/bash
#SBATCH --job-name=safe_bullets_video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=safe_bullets_video_%j.out
#SBATCH --error=safe_bullets_video_%j.err

set -euo pipefail

cd "$HOME/repos/CRAX-Denis-Fork"
source env/bin/activate

export MUJOCO_GL=egl

python -u scripts/safe_bullets_policy_videos.py \
  --run models/safe_bullets_20261014_210907/safe_bullets_humanoid_curriculum_ppo_lag_seed42_20261014_210914_930645 \
  --levels 1 2 3 \
  --steps 1000 \
  --fps 60 \
  --cameras front side fixedfar

