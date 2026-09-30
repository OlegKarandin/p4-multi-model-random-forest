#!/usr/bin/env bash
# Builds and pushes the thesis-p4c image from the WSL open-p4studio install.
# Run once, by the user, in WSL. Needs docker, and `gh` (or pass OWNER as $1).
# Credentials: docker/gh must already be logged in; nothing is embedded here.
#
# Usage: bash docker/p4c/build.sh [ghcr-owner]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WANT_COMMIT="0e81a468"

P4STUDIO="${P4STUDIO:-$HOME/open-p4studio}"
COMMIT="$(git -C "$P4STUDIO" rev-parse --short=8 HEAD)"
if [ "$COMMIT" != "$WANT_COMMIT" ] && [ "${ALLOW_OTHER_COMMIT:-}" != "1" ]; then
  echo "[build] ERROR: $P4STUDIO is at $COMMIT, expected $WANT_COMMIT." \
       "Set ALLOW_OTHER_COMMIT=1 to build anyway." >&2
  exit 1
fi
[ -d "$P4STUDIO/install/bin" ] || { echo "[build] ERROR: $P4STUDIO/install/bin missing" >&2; exit 1; }

OWNER="${1:-$(gh api user -q .login)}"
OWNER="$(echo "$OWNER" | tr '[:upper:]' '[:lower:]')"   # GHCR owners must be lowercase
IMAGE="ghcr.io/$OWNER/thesis-p4c:$COMMIT"

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp -a "$P4STUDIO/install" "$CTX/install"
cp "$REPO_ROOT/docker/p4c/Dockerfile" "$CTX/Dockerfile"
mkdir -p "$CTX/extra-libs"

# Shared libraries that ldd resolves for every ELF under install/bin, keeping
# only those outside install/ and outside the base image's standard set (libc
# and friends; boost comes from apt in the Dockerfile).
INSTALL_REAL="$(cd "$P4STUDIO/install" && pwd -P)"
LIBLIST="$CTX/extra-libs.list"
: > "$LIBLIST"
while IFS= read -r -d '' f; do
  file -b "$f" | grep -q '^ELF' || continue
  ldd "$f" 2>/dev/null | awk '/=> \// {print $3}'
done < <(find "$INSTALL_REAL/bin" -type f -print0) | sort -u | while IFS= read -r lib; do
  real="$(readlink -f "$lib")"
  case "$real" in
    "$INSTALL_REAL"/*) continue ;;
  esac
  case "$(basename "$lib")" in
    libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libgcc_s.so*|libstdc++.so*|ld-linux*|libutil.so*|libz.so*|libboost_*) continue ;;
  esac
  echo "$lib" >> "$LIBLIST"
done

echo "[build] out-of-tree libraries copied into extra-libs:"
while IFS= read -r lib; do
  [ -n "$lib" ] || continue
  cp -L "$lib" "$CTX/extra-libs/"
  echo "  $lib"
done < "$LIBLIST"
rm -f "$LIBLIST"

docker build --build-arg P4STUDIO_COMMIT="$COMMIT" -t "$IMAGE" "$CTX"
docker run --rm "$IMAGE" p4c --version
# `p4c --version` only runs the python wrapper; the real compiler is p4c-barefoot, and a
# missing shared library there shows up only when it starts (measured 2026-09-30: 30
# libabsl libs unresolved in the Codespace). Fail the build before pushing if any is.
docker run --rm "$IMAGE" bash -c 'ldd /opt/open-p4studio/install/bin/p4c-barefoot | { ! grep "not found"; }'
docker push "$IMAGE"

DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"
echo "[build] image:  $IMAGE"
echo "[build] digest: $DIGEST"
