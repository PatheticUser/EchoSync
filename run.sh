#!/usr/bin/env bash
# EchoSync AI — Bash Entrypoint
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv &>/dev/null; then
    echo "[EchoSync Error] 'uv' package manager not found in PATH." >&2
    echo "Install uv via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

exec uv run python scripts/run.py "$@"
