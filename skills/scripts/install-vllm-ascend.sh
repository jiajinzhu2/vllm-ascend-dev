#!/usr/bin/env bash
set -euo pipefail

VLLM_DEV_ROOT="${VLLM_DEV_ROOT:-$HOME/vllm-workspace}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_ASCEND_REPO="${VLLM_ASCEND_REPO:-https://github.com/vllm-project/vllm-ascend.git}"
VLLM_ASCEND_REF="${VLLM_ASCEND_REF:-main}"
VLLM_SETUP_DRY_RUN="${VLLM_SETUP_DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_DIR="$VLLM_DEV_ROOT/vllm"
VLLM_ASCEND_DIR="$VLLM_DEV_ROOT/vllm-ascend"

log() {
    printf '[vllm-setup] %s\n' "$*"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log "missing required command: $1"
        exit 1
    fi
}

prepare_checkout() {
    local repo_dir="$1"
    local stamp_file="$2"
    local current_diff_hash=""
    local expected_diff_hash=""

    if [[ -z "$(git -C "$repo_dir" status --porcelain)" ]]; then
        return
    fi

    if [[ -f "$stamp_file" ]] && [[ -z "$(git -C "$repo_dir" ls-files --others --exclude-standard)" ]]; then
        current_diff_hash="$(git -C "$repo_dir" diff --binary | sha256sum | awk '{print $1}')"
        expected_diff_hash="$(<"$stamp_file")"
        if [[ "$current_diff_hash" == "$expected_diff_hash" ]]; then
            log "restoring installer-managed patch in $repo_dir"
            git -C "$repo_dir" restore --worktree -- .
            return
        fi
    fi

    if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
        log "refusing to modify dirty checkout: $repo_dir"
        git -C "$repo_dir" status --short
        exit 1
    fi
}

record_installer_diff() {
    local repo_dir="$1"
    local stamp_file="$2"
    git -C "$repo_dir" diff --binary | sha256sum | awk '{print $1}' > "$stamp_file"
}

clone_or_update() {
    local url="$1"
    local repo_dir="$2"
    local ref="$3"

    if [[ ! -d "$repo_dir/.git" ]]; then
        log "cloning $url into $repo_dir"
        git clone "$url" "$repo_dir"
    else
        prepare_checkout "$repo_dir" "$VLLM_DEV_ROOT/.vllm-ascend-installer.diff.sha256"
        log "updating $repo_dir"
        git -C "$repo_dir" fetch --prune origin
    fi

    git -C "$repo_dir" checkout --detach "origin/$ref"
}

patch_vllm_pyproject() {
    local file="$VLLM_DIR/pyproject.toml"
    log "applying vLLM packaging compatibility edits"
    sed -i -E '/^[[:space:]]*license-files[[:space:]]*=/d' "$file"
    sed -i -E '/^[[:space:]]*"?torch==2\.11\.0"?,?[[:space:]]*$/d' "$file"
    sed -i -E 's/^[[:space:]]*license[[:space:]]*=.*/license = {text = "Apache-2.0"}/' "$file"
}

remove_triton_ascend_requirement() {
    local file
    log "removing pinned triton-ascend declarations from vLLM Ascend"
    for file in "$VLLM_ASCEND_DIR/requirements.txt" "$VLLM_ASCEND_DIR/pyproject.toml"; do
        [[ -f "$file" ]] || continue
        sed -i -E '/triton-ascend[[:space:]]*==[[:space:]]*3\.2\.1/d' "$file"
    done
}

require_command git
require_command sed
require_command python
require_command sha256sum
mkdir -p "$VLLM_DEV_ROOT"

clone_or_update "$VLLM_ASCEND_REPO" "$VLLM_ASCEND_DIR" "$VLLM_ASCEND_REF"

verified_commit_file="$VLLM_ASCEND_DIR/.github/vllm-main-verified.commit"
if [[ ! -s "$verified_commit_file" ]]; then
    log "verified commit file not found: $verified_commit_file"
    exit 1
fi
verified_commit="$(tr -d '[:space:]' < "$verified_commit_file")"
if [[ ! "$verified_commit" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
    log "invalid verified vLLM commit: $verified_commit"
    exit 1
fi

if [[ ! -d "$VLLM_DIR/.git" ]]; then
    log "cloning $VLLM_REPO into $VLLM_DIR"
    git clone "$VLLM_REPO" "$VLLM_DIR"
else
    prepare_checkout "$VLLM_DIR" "$VLLM_DEV_ROOT/.vllm-installer.diff.sha256"
    git -C "$VLLM_DIR" fetch --prune origin
fi

log "checking out verified vLLM commit $verified_commit"
git -C "$VLLM_DIR" checkout --detach "$verified_commit"
patch_vllm_pyproject
remove_triton_ascend_requirement
record_installer_diff "$VLLM_DIR" "$VLLM_DEV_ROOT/.vllm-installer.diff.sha256"
record_installer_diff "$VLLM_ASCEND_DIR" "$VLLM_DEV_ROOT/.vllm-ascend-installer.diff.sha256"

if [[ "$VLLM_SETUP_DRY_RUN" == "1" ]]; then
    log "dry run complete; package uninstall/install skipped"
    git -C "$VLLM_DIR" diff --check
    git -C "$VLLM_ASCEND_DIR" diff --check
    exit 0
fi

log "uninstalling existing vLLM packages"
python -m pip uninstall -y vllm vllm-ascend || true

log "installing vLLM build prerequisite setuptools_rust"
python -m pip install setuptools_rust

log "installing editable vLLM"
(
    cd "$VLLM_DIR"
    VLLM_TARGET_DEVICE=empty python -m pip install -v -e . --no-build-isolation
)

log "restoring the CANN-compatible NumPy version"
python -m pip install 'numpy==1.26.4'

log "installing editable vLLM Ascend"
(
    cd "$VLLM_ASCEND_DIR"
    PYTHONPATH="$SCRIPT_DIR/python_compat${PYTHONPATH:+:$PYTHONPATH}" \
        python -m pip install -v -e . --no-build-isolation
)

log "installation complete"
python -m pip list | grep -E '^vllm([[:space:]-]|$)' || true
printf 'vLLM commit: %s\n' "$(git -C "$VLLM_DIR" rev-parse HEAD)"
printf 'vLLM Ascend commit: %s\n' "$(git -C "$VLLM_ASCEND_DIR" rev-parse HEAD)"
