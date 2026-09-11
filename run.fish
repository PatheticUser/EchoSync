#!/usr/bin/env fish
# EchoSync AI — Fish Shell Entrypoint

set -l script_dir (status dirname)
cd "$script_dir"

if not command -v uv >/dev/null 2>&1
    echo "[EchoSync Error] 'uv' package manager not found in PATH." >&2
    echo "Install uv via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
end

exec uv run python scripts/run.py $argv
