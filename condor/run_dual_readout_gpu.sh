#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?experiment mode is required}"
DATA_DIR="${2:?input data directory is required}"
OUTPUT_DIR="${3:?output directory is required}"

ML_ROOT="${ML4HEP_PROJECT_DIR:-/eos/home-f/faguo/ML}"
REPO_DIR="$ML_ROOT/SoftwareCompensation"
REFERENCE_DIR="$REPO_DIR/outputs_dual_readout/standard_dual_readout/data"

case "$MODE" in
    s_only|sc_full|all) ;;
    *)
        echo "Unsupported experiment mode: $MODE" >&2
        exit 2
        ;;
esac

source "$ML_ROOT/setup_env.sh"
cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Python: $(command -v python)"
echo "Input: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Mode: $MODE"
nvidia-smi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.cuda.get_device_name(0))"

python -u run_dual_readout.py \
    --input-dir "$DATA_DIR" \
    --reference-dir "$REFERENCE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --experiment "$MODE" \
    --epochs 40 \
    --batch-size 8 \
    --num-workers 2 \
    --device cuda

date --iso-8601=seconds > "$OUTPUT_DIR/COMPLETED"
echo "Completed: $(cat "$OUTPUT_DIR/COMPLETED")"
