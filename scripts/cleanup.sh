#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

if (( $# > 1 )) || [[ -n ${1:-} && $1 != --dry-run ]]; then
    printf 'Usage: bash scripts/cleanup.sh [--dry-run]\n' >&2
    exit 2
fi
CLEANUP_MODE=${1:-}

remove_path() {
    local target=$1
    [[ -e "$target" || -L "$target" ]] || return 0
    if [[ $CLEANUP_MODE == --dry-run ]]; then
        printf 'Would remove %s\n' "$target"
    else
        printf 'Removing %s\n' "$target"
        rm -rf -- "$target"
    fi
}

shopt -s nullglob
for target in .pytest_cache .ruff_cache .hypothesis build dist \
    ./*.log src/*.log artifacts/*.log artifacts/logs/*.log \
    pokemon-showdown/logs/*.log pokemon-showdown/logs/errors.txt \
    pokemon-showdown/logs/chatlog-access.txt; do
    remove_path "$target"
done

while IFS= read -r -d '' target; do
    remove_path "$target"
done < <(find src scripts tests -type d -name __pycache__ -prune -print0)

printf 'Preserved runs, checkpoints, replay data, teams, configuration, dependencies, and scratch.\n'
