#!/usr/bin/env bash
# Builds the aligned twins (python -m src.training.align_twins) of a verified
# campaign run under nohup, so it survives an SSH disconnect or a Codespaces idle-stop.
# Mirrors run_campaign.sh; also NOT invoked automatically.
#
# Resumable: a finished (M, split) twin file is skipped, so re-running the same
# command after a disconnect continues. Requires every joint-off design verified.
#
# Usage:
#   bash .devcontainer/run_twins.sh results/campaign_2026_11
#   bash .devcontainer/run_twins.sh results/campaign_2026_11 --workers 3
#
# The first argument is the run directory (the value of --run); anything after
# it is forwarded to `python -m src.training.align_twins --run <dir>`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV_NAME="thesis-codespace"

[ $# -ge 1 ] || {
  echo "[run_twins] ERROR: usage: run_twins.sh RUN_DIR [--workers N]" >&2
  exit 1
}

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

command -v p4c >/dev/null || {
  echo "[run_twins] ERROR: p4c not on PATH -- this is not the thesis-p4c image" \
       "(spec 2026-09-29 section 8.1)." >&2
  exit 1
}

mkdir -p logs
LOG="logs/twins_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[run_twins] launching: python -m src.training.align_twins --run $*"
echo "[run_twins] logging to $LOG"

nohup python -m src.training.align_twins --run "$@" > "$LOG" 2>&1 &
pid=$!

echo "[run_twins] started, PID $pid. Safe to disconnect now."
echo "[run_twins] check progress:   tail -f $LOG"
echo "[run_twins] then: bash .devcontainer/run_verify.sh <run> --workers 4 to compile the differing twins"
echo "[run_twins] after a disconnect/idle-stop, just re-run this same command."
