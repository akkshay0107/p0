#!/usr/bin/env bash
set -euo pipefail

POOL_DIR="./artifacts/checkpoints/pool"
PPO_CHECKPOINT="./artifacts/checkpoints/ppo_checkpoint.pt"
BACKUP_DIR="./artifacts/backups"
SEEDS=(
  "seed_max_base_power"
  "seed_simple_heuristic"
  "seed_fuzzy_heuristic"
)

if [[ ! -d "$POOL_DIR" ]]; then
  echo "Pool directory missing, creating: $POOL_DIR"
  mkdir -p "$POOL_DIR"
fi

shopt -s nullglob dotglob
for path in "$POOL_DIR"/*; do
  name="$(basename "$path")"
  keep=0
  if [[ "$name" == "pool_state.json" ]]; then
    keep=1
  else
    for seed in "${SEEDS[@]}"; do
      if [[ "$name" == "$seed".* ]]; then
        keep=1
        break
      fi
    done
  fi

  if [[ $keep -eq 0 ]]; then
    rm -rf -- "$path"
  fi
done
shopt -u nullglob dotglob

rm -f -- "$PPO_CHECKPOINT"

# Restore from backups if they exist and are missing in POOL_DIR
for seed in "${SEEDS[@]}"; do
  if [[ ! -f "$POOL_DIR/$seed.pt" ]] && [[ -f "$BACKUP_DIR/$seed.pt" ]]; then
    echo "Restoring $seed.pt from backup..."
    cp "$BACKUP_DIR/$seed.pt" "$POOL_DIR/"
  fi
done

cat > "$POOL_DIR/pool_state.json" <<'JSON'
{
  "opponent_ids": [
    "seed_max_base_power",
    "seed_simple_heuristic",
    "seed_fuzzy_heuristic"
  ],
  "win_rates": {
    "seed_max_base_power": 0.5,
    "seed_simple_heuristic": 0.5,
    "seed_fuzzy_heuristic": 0.5
  }
}
JSON

echo "Reset complete: $POOL_DIR"
echo "Deleted: $PPO_CHECKPOINT"
