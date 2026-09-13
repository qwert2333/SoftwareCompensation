#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v condor_submit >/dev/null 2>&1; then
    echo "condor_submit is not available in PATH" >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "Refusing to submit from a dirty Git worktree." >&2
    echo "Commit the tested workflow first so the batch run is reproducible." >&2
    exit 2
fi

mkdir -p logs
condor_submit condor/dual_readout_gpu.sub
