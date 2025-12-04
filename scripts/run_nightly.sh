#!/bin/bash
# Convenience script for running nightly training manually
# Usage: ./scripts/run_nightly.sh [config_name]
#   config_name: krk_fast, krk_stockfish, kpk_training (default: krk_fast)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_NAME="${1:-krk_fast}"
CONFIG_FILE="$PROJECT_ROOT/configs/nightly/${CONFIG_NAME}.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config not found: $CONFIG_FILE"
    echo "Available configs:"
    ls "$PROJECT_ROOT/configs/nightly/"*.json 2>/dev/null | xargs -n1 basename
    exit 1
fi

echo "Running nightly with config: $CONFIG_NAME"
echo "Config file: $CONFIG_FILE"
echo ""

cd "$PROJECT_ROOT"
uv run python demos/experiments/nightly_runner.py --config "$CONFIG_FILE"
