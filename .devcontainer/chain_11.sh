#!/usr/bin/env bash
# Two-arm compiler-verified campaign, spec 2026-10-06 (alignment twins and the
# two-arm campaign), section 4: train `independent` + `joint-off` on 25 splits,
# then verify every design. Runs under nohup so it survives a disconnect or a
# Codespaces idle-stop; every stage is resumable (rows/ files and
# verify/<row_id>.json are atomic done markers), so after an interruption just
# re-run this same script -- finished (arm, M, split) jobs and verified designs
# are skipped.
#
# The twins (`python -m src.training.align_twins --run ...`) and the second
# verify pass are NOT chained here: align_twins requires this run to be fully
# verified first, and it does not exist yet while training runs.
#
# Usage (from the repo root, on the thesis-p4c Codespace image):
#   bash .devcontainer/chain_11.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN="results/campaign_2026_11"
ARMS="independent,joint-off"
TRAIN_WORKERS=3          # 4 logical cores: 3 training workers, the 4th for the OS/you
VERIFY_WORKERS=4

CONDA_ENV_NAME="thesis-codespace"
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

command -v p4c >/dev/null || {
  echo "[chain_11] ERROR: p4c not on PATH -- this is not the thesis-p4c image." >&2
  exit 1
}
for f in resources/apps_flow_features.csv resources/Wednesday-workingHours.pcap_ISCX.csv; do
  [ -f "$f" ] || { echo "[chain_11] ERROR: $f missing; run .devcontainer/postCreate.sh first." >&2; exit 1; }
done

mkdir -p logs
LOG="logs/chain_11_$(date -u +%Y%m%dT%H%M%SZ).log"

nohup bash -c "
  set -euo pipefail
  echo \"[chain_11] \$(date -u +%FT%TZ) train splits 0-12\"
  python -m src.main --mode compute --run $RUN --arm-slugs $ARMS --splits 0-12 --max-workers $TRAIN_WORKERS
  echo \"[chain_11] \$(date -u +%FT%TZ) train splits 13-24\"
  python -m src.main --mode compute --run $RUN --arm-slugs $ARMS --splits 13-24 --max-workers $TRAIN_WORKERS
  echo \"[chain_11] \$(date -u +%FT%TZ) verify\"
  python -m src.verify --run $RUN --workers $VERIFY_WORKERS
  echo \"[chain_11] \$(date -u +%FT%TZ) DONE -- next: git pull, then python -m src.training.align_twins --run $RUN, then verify again\"
" > "$LOG" 2>&1 &
pid=$!

echo "[chain_11] started, PID $pid, logging to $LOG. Safe to disconnect."
echo "[chain_11] progress:  tail -f $LOG ; ls $RUN/rows | wc -l   (300 files when training is done)"
