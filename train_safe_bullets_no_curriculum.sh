#!/bin/bash
#SBATCH --job-name=bullets_nocurr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --output=bullets_nocurr_%j.out
#SBATCH --error=bullets_nocurr_%j.err

set -e

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

cd "$HOME/repos/CRAX-Denis-Fork"
source env/bin/activate

export MUJOCO_GL=egl

srun bash scripts/bullets_videos_no_curriculum.sh