#!/usr/bin/env bash
# Runs once per Codespace (see devcontainer.json's postCreateCommand).
#
#   1. Creates/activates the `thesis-codespace` conda env from
#      .devcontainer/environment-linux.yml.
#   2. Downloads resources/apps_flow_features.csv and
#      resources/Wednesday-workingHours.pcap_ISCX.csv from the signed URLs
#      held in the THESIS_DATA_URL_APP / THESIS_DATA_URL_DDOS Codespace
#      secrets (never hardcoded here -- see devcontainer.json's "secrets"
#      block and D11 in the design spec).
#   3. Verifies each download against a known byte size AND sha256 before
#      accepting it -- a truncated/partial download (e.g. an expired signed
#      URL cut short mid-transfer) must fail loudly here, not get trained on
#      hours into a campaign.
#
# Idempotent by design: safe to re-run (e.g. on a container rebuild that
# reuses the same workspace) -- an already-present, already-verified file is
# left alone rather than re-downloaded, and `conda env create` falls back to
# `conda env update` if the env already exists. Does NOT touch results/ or
# resources/*.csv beyond what's described above, so it never fights the
# resumability src/main.py already implements (skip_existing / --redo,
# src/main.py:333-402).
#
# Does NOT launch the campaign itself -- see .devcontainer/run_campaign.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV_NAME="thesis-codespace"   # must match environment-linux.yml's `name:`
ENV_FILE=".devcontainer/environment-linux.yml"

log() { echo "[postCreate] $*"; }
fail() { echo "[postCreate] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Conda environment
# ---------------------------------------------------------------------------
[ -f "$ENV_FILE" ] || fail "$ENV_FILE not found -- run this from the repo root (cwd was $REPO_ROOT)"

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    log "conda env '$CONDA_ENV_NAME' already exists, updating in place"
    conda env update -n "$CONDA_ENV_NAME" -f "$ENV_FILE" --prune
else
    log "creating conda env '$CONDA_ENV_NAME' from $ENV_FILE"
    conda env create -f "$ENV_FILE"
fi

conda activate "$CONDA_ENV_NAME"
log "activated '$CONDA_ENV_NAME' ($(python --version))"

# Make the env the default for every future interactive/login shell in this
# container too -- activation above only applies to this script's process.
if ! grep -q "# Added by .devcontainer/postCreate.sh" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo "# Added by .devcontainer/postCreate.sh"
        echo "source /opt/conda/etc/profile.d/conda.sh"
        echo "conda activate $CONDA_ENV_NAME"
    } >> "$HOME/.bashrc"
    log "appended conda activation to ~/.bashrc"
fi

# ---------------------------------------------------------------------------
# 2 & 3. Fetch + verify datasets
# ---------------------------------------------------------------------------
mkdir -p resources

verify_file() {
    # verify_file PATH WANT_SIZE WANT_SHA256
    local path="$1" want_size="$2" want_sha="$3"
    [ -f "$path" ] || return 1
    local got_size
    got_size="$(stat -c%s "$path")"
    [ "$got_size" = "$want_size" ] || return 1
    local got_sha
    got_sha="$(sha256sum "$path" | awk '{print $1}')"
    [ "$got_sha" = "$want_sha" ]
}

# dest path | secret env var name | expected size (bytes) | expected sha256
# Sizes and hashes computed 2026-08-23 against the real files already
# present locally in resources/ this session (task-27-brief's exact values).
DATASET_SPECS="
resources/apps_flow_features.csv|THESIS_DATA_URL_APP|120758907|49d0f482ebfce870f1b7786593bfe72f0686af03817c12dba55e08b5e1a03667
resources/Wednesday-workingHours.pcap_ISCX.csv|THESIS_DATA_URL_DDOS|285642925|ed538e85b84181e8897dedb3d37d365982f44b27eccd67c477581a8b65f3d170
"

while IFS='|' read -r dest url_var want_size want_sha; do
    [ -n "$dest" ] || continue   # skip blank lines from the heredoc-ish list above

    if verify_file "$dest" "$want_size" "$want_sha"; then
        log "$dest already present and verified ($want_size bytes), skipping download"
        continue
    fi

    url_value="${!url_var:-}"
    [ -n "$url_value" ] || fail "$url_var is not set. Add it as a Codespace secret" \
        "(github.com -> Settings -> Codespaces -> Secrets, or repo/org secrets) named" \
        "exactly '$url_var', then rebuild the container."

    log "downloading $dest from \$$url_var ..."
    rm -f "$dest"
    if ! curl -fSL --retry 3 --retry-delay 5 -o "$dest" "$url_value"; then
        rm -f "$dest"
        fail "download of $dest failed (curl error). If \$$url_var is a signed URL," \
            "it may have expired -- generate a fresh one and update the Codespace secret."
    fi

    if ! verify_file "$dest" "$want_size" "$want_sha"; then
        got_size="$(stat -c%s "$dest" 2>/dev/null || echo '?')"
        got_sha="$(sha256sum "$dest" 2>/dev/null | awk '{print $1}')"
        rm -f "$dest"
        fail "$dest FAILED checksum verification after download." \
            "expected: $want_size bytes, sha256 $want_sha" \
            "got:      $got_size bytes, sha256 ${got_sha:-?}" \
            "This looks like a truncated/corrupted download (e.g. an expired signed URL" \
            "cut off mid-transfer). Refusing to leave a partial dataset in place -- the" \
            "partial file has been removed. Fix \$$url_var and rerun postCreate.sh."
    fi

    log "$dest verified OK ($want_size bytes, sha256 $want_sha)"
done <<< "$DATASET_SPECS"

log "all datasets present and verified"

# ---------------------------------------------------------------------------
# Reminders for the two things this file cannot itself guarantee
# ---------------------------------------------------------------------------
cat <<EOF

[postCreate] Setup complete.

  * Idle timeout: GitHub Codespaces has no devcontainer.json field for this.
    Set it to the 240-minute maximum yourself, either in your personal
    Settings -> Codespaces -> "Default idle timeout", or per-codespace at
    creation time:
        gh codespace create --idle-timeout 240m -r <owner>/<repo>

  * To run the campaign so it survives an SSH/Codespace disconnect:
        bash .devcontainer/run_campaign.sh [-- extra args to src.main]
    It launches 'python -m src.main --mode compute' under nohup and logs to
    logs/. --redo is off by default in src/main.py, so re-running the same
    command after an idle-stop/reconnect resumes rather than restarts
    (src/main.py:333-402).

EOF
