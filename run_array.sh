#!/bin/bash
#SBATCH --job-name=calo_grid
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd /gpfs/users/xiax/SoftwareCompensation/pytorch/projects_0/project_time_fixed_direction_ablation

module purge
module load anaconda3/2023.09-0/none-none

source ~/conda_rc
conda activate MLenv

python -u run_one_from_array.py --array-id "${SLURM_ARRAY_TASK_ID}"
