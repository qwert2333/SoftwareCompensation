#!/bin/bash
set -euo pipefail

cd /gpfs/users/xiax/SoftwareCompensation/pytorch/projects_0/project_time_fixed_direction_ablation

N_EXP=$(python - <<'PY'
from calo.experiment_grid import EXPERIMENTS
print(len(EXPERIMENTS))
PY
)

if [ "$N_EXP" -le 0 ]; then
  echo "No experiments found."
  exit 1
fi

echo "Experiments: $N_EXP"
echo "Total jobs: $N_EXP"
mkdir -p logs outputs

sbatch --array=0-$(($N_EXP-1)) run_array.sh
