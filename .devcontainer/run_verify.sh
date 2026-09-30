#!/usr/bin/env bash
# Launches the p4c verifier (python -m src.verify) over a finished campaign run
# under nohup, so it survives an SSH disconnect or a Codespaces idle-stop.
# Mirrors run_campaign.sh; also NOT invoked automatically.
#
# Resumable: src.verify keeps its per-design logs and skips designs already
# verified, so re-running the same command after a disconnect continues.
#
# Usage:
#   bash .devcontainer/run_verify.sh results/campaign_2026_10
#   bash .devcontainer/run_verify.sh results/campaign_2026_10 --workers 3
#
# The first argument is the run directory (the value of --run); anything after
# it is forwarded to `python -m src.verify --run <dir>`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV_NAME="thesis-codespace"

[ $# -ge 1 ] || {
  echo "[run_verify] ERROR: usage: run_verify.sh RUN_DIR [--workers N]" >&2
  exit 1
}

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

command -v p4c >/dev/null || {
  echo "[run_verify] ERROR: p4c not on PATH -- this is not the thesis-p4c image" \
       "(spec 2026-09-29 section 8.1)." >&2
  exit 1
}

mkdir -p logs
LOG="logs/verify_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[run_verify] launching: python -m src.verify --run $*"
echo "[run_verify] logging to $LOG"

nohup python -m src.verify --run "$@" > "$LOG" 2>&1 &
pid=$!

echo "[run_verify] started, PID $pid. Safe to disconnect now."
echo "[run_verify] check progress:   tail -f $LOG"
echo "[run_verify] after a disconnect/idle-stop, just re-run this same command."
