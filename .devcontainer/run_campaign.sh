#!/usr/bin/env bash
# Launches the actual compute campaign (src/main.py --mode compute) under
# nohup, so it survives an SSH disconnect or a Codespaces idle-stop.
#
# NOT invoked automatically by postCreate.sh -- Tasks 28-29 (spinning up a
# billed Codespace and actually running ~10-16h+ of compute) are a real-money,
# real-cloud-resource action that needs a human's explicit go-ahead, not
# something a container-creation hook should trigger on its own.
#
# Resumable by construction, not by anything this script adds: --redo is off
# by default in src/main.py, so re-running the exact same command after a
# disconnect/idle-stop/reboot skips any (arm, M, split) whose file under
# <run>/rows/ already exists (written atomically via os.replace, so its
# existence is a reliable completion marker -- src/main.py).
#
# Usage:
#   bash .devcontainer/run_campaign.sh --run results/campaign_2026_10 --splits 0-9
#   bash .devcontainer/run_campaign.sh --run results/pilot --M 25 --splits 0-1   # quick pilot
#
# --run <dir> is required by src/main.py in compute mode. Any arguments are
# forwarded verbatim to `python -m src.main --mode compute`. (--n-splits no
# longer exists; use --splits A-B.) When done, verify with run_verify.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV_NAME="thesis-codespace"

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

for f in resources/apps_flow_features.csv resources/Wednesday-workingHours.pcap_ISCX.csv; do
    [ -f "$f" ] || {
        echo "[run_campaign] ERROR: $f is missing. Run .devcontainer/postCreate.sh" \
             "(or bash .devcontainer/postCreate.sh directly) first to fetch and" \
             "verify the datasets." >&2
        exit 1
    }
done

mkdir -p logs
LOG="logs/campaign_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[run_campaign] launching: python -m src.main --mode compute $*"
echo "[run_campaign] logging to $LOG"

nohup python -m src.main --mode compute "$@" > "$LOG" 2>&1 &
pid=$!

echo "[run_campaign] started, PID $pid. Safe to disconnect now."
echo "[run_campaign] check progress:   tail -f $LOG"
echo "[run_campaign] after a disconnect/idle-stop, just re-run this same command --"
echo "[run_campaign] completed (arm, M, split) files under <run>/rows/ are skipped automatically."
