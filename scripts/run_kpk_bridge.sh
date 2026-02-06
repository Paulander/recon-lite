#!/bin/bash
# Wrapper script for KPK bridge training using near-promotion positions

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <num_games> [additional_args...]"
    echo "Example: $0 200 --plasticity --consolidate --weights-dir weights/latest"
    exit 1
fi

NUM_GAMES=$1
shift  # Remove first arg, pass rest to python script

FEN_FILE="data/bridge/near_promo.fens"

if [ ! -f "$FEN_FILE" ]; then
    echo "Error: FEN file not found: $FEN_FILE"
    exit 1
fi

echo "Running KPK bridge training with $NUM_GAMES games using positions from $FEN_FILE"

# Run the training with KPK near-promotion positions
uv run python demos/persistent/full_game_train.py \
    --batch "$NUM_GAMES" \
    --fen-file "$FEN_FILE" \
    "$@"