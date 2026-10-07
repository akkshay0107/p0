#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

FORMAT=gen9championsvgc2026regmc
USAGE_MONTH=2026-09

EXPECTED=$(git ls-tree HEAD pokemon-showdown | awk '$1 == "160000" {print $3}')
if [[ -z "$EXPECTED" || ! -f pokemon-showdown/.git ]]; then
    echo "Initialize the pinned submodule: git submodule update --init --recursive" >&2
    exit 1
fi
REVISION=$(git -C pokemon-showdown rev-parse HEAD)
if [[ "$REVISION" != "$EXPECTED" ]] || [[ -n "$(git -C pokemon-showdown status --porcelain --untracked-files=no)" ]]; then
    echo "Showdown must be a clean checkout at the committed gitlink $EXPECTED." >&2
    exit 1
fi

npm --prefix pokemon-showdown ci --no-audit --no-fund
npm --prefix pokemon-showdown run build
node scripts/dump_champions_dex.js "$FORMAT" "$REVISION" --check

mkdir -p data
rm -f data/runtime_manifest.json
node scripts/dump_champions_dex.js "$FORMAT" "$REVISION"
uv run p0-build-vocab
uv run p0-build-spreads --month "$USAGE_MONTH" --fetch
node scripts/generate_showdown_raw_inventory.js
uv run python scripts/generate_replay_protocol_contract.py
uv run python scripts/write_runtime_manifest.py
printf 'Resource initialization complete.\n'
