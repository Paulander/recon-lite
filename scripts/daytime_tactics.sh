#!/bin/bash
# =============================================================================
# ReCoN-lite Daytime Tactics Training
# =============================================================================
#
# Run focused tactics training during the workday with higher Stockfish depth.
# This is slower but more accurate than overnight training.
#
# Usage:
#   ./scripts/daytime_tactics.sh                    # Train all tactics
#   ./scripts/daytime_tactics.sh fork pin skewer    # Train specific tactics
#   ./scripts/daytime_tactics.sh --quick fork       # Quick test mode
#   ./scripts/daytime_tactics.sh --depth 10 fork    # Custom depth
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_ROOT"

# Defaults
ENGINE="/usr/games/stockfish"
DEPTH=6              # Higher than overnight (6 vs 2) for accuracy
LIMIT=500            # More positions per tactic
QUICK_MODE=false

# All available tactics with puzzle data
ALL_TACTICS=("fork" "pin" "skewer" "hangingPiece" "backRankMate" "discoveredAttack"
             "attraction" "deflection" "doubleCheck" "smotheredMate" "trappedPiece" 
             "quietMove" "sacrifice" "exposedKing" "interference" "zugzwang")

# Parse arguments
TACTICS_TO_TRAIN=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            DEPTH=4
            LIMIT=50
            shift
            ;;
        --depth)
            DEPTH="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [TACTICS...]"
            echo ""
            echo "Options:"
            echo "  --quick           Quick test mode (depth 4, 50 positions)"
            echo "  --depth N         Stockfish depth (default: 6)"
            echo "  --limit N         Positions per tactic (default: 500)"
            echo "  --engine PATH     Path to Stockfish"
            echo ""
            echo "Available tactics:"
            echo "  ${ALL_TACTICS[*]}"
            echo ""
            echo "Examples:"
            echo "  $0 fork pin           Train fork and pin detection"
            echo "  $0 --depth 10 fork    Train forks with depth 10"
            echo "  $0                    Train all tactics"
            exit 0
            ;;
        *)
            TACTICS_TO_TRAIN+=("$1")
            shift
            ;;
    esac
done

# If no tactics specified, train all
if [ ${#TACTICS_TO_TRAIN[@]} -eq 0 ]; then
    TACTICS_TO_TRAIN=("${ALL_TACTICS[@]}")
fi

# Directories
WEIGHTS_DIR="$PROJECT_ROOT/weights"
LOGS_DIR="$PROJECT_ROOT/logs"
mkdir -p "$WEIGHTS_DIR/tactics" "$WEIGHTS_DIR/latest/tactics" "$LOGS_DIR"

LOG_FILE="$LOGS_DIR/daytime_tactics_$TIMESTAMP.log"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

echo "=============================================="
echo "🎯 Daytime Tactics Training"
echo "=============================================="
log "Configuration:"
log "  Stockfish depth: $DEPTH"
log "  Positions per tactic: $LIMIT"
log "  Quick mode: $QUICK_MODE"
log "  Tactics to train: ${TACTICS_TO_TRAIN[*]}"
log "  Log file: $LOG_FILE"
echo ""

TOTAL_DETECTED=0
TOTAL_CORRECT=0
TOTAL_POSITIONS=0
TRAINED_COUNT=0

for tactic in "${TACTICS_TO_TRAIN[@]}"; do
    # Find FEN file
    FEN_FILE=""
    for candidate in \
        "$PROJECT_ROOT/data/puzzles/$tactic/lichess_$tactic.fen" \
        "$PROJECT_ROOT/data/puzzles/${tactic,,}/lichess_${tactic,,}.fen"; do
        if [ -f "$candidate" ]; then
            FEN_FILE="$candidate"
            break
        fi
    done
    
    if [ -z "$FEN_FILE" ]; then
        log "⚠️  Skipping $tactic (no FEN file found)"
        continue
    fi
    
    log "📝 Training: $tactic"
    
    # Run training - capture full output first to see errors
    TEMP_OUTPUT=$(mktemp)
    if ! uv run python demos/experiments/tactics_eval.py \
        --fen-file "$FEN_FILE" \
        --tactic-type "$tactic" \
        --limit "$LIMIT" \
        --consolidate \
        --consolidate-pack "$WEIGHTS_DIR/tactics/${tactic}_consol.json" \
        --engine "$ENGINE" --depth "$DEPTH" \
        2>&1 | tee "$TEMP_OUTPUT"; then
        log "   ⚠️  Error running training for $tactic"
        if [ -f "$TEMP_OUTPUT" ]; then
            log "   Error output:"
            tail -20 "$TEMP_OUTPUT" | while IFS= read -r line; do
                log "     $line"
            done
        fi
        rm -f "$TEMP_OUTPUT"
        continue
    fi
    
    # Extract result line
    result=$(grep "TACTICS_EVAL_RESULT" "$TEMP_OUTPUT" || echo "TACTICS_EVAL_RESULT: total=0 detected=0 correct=0 accuracy=0.000")
    
    # If no result found, log the output for debugging
    if ! echo "$result" | grep -q "TACTICS_EVAL_RESULT"; then
        log "   ⚠️  No TACTICS_EVAL_RESULT found for $tactic"
        if [ -f "$TEMP_OUTPUT" ]; then
            log "   Last 10 lines of output:"
            tail -10 "$TEMP_OUTPUT" | while IFS= read -r line; do
                log "     $line"
            done
        fi
    fi
    
    rm -f "$TEMP_OUTPUT"
    
    # Parse results
    total=$(echo "$result" | grep -o 'total=[0-9]*' | cut -d'=' -f2 || echo "0")
    detected=$(echo "$result" | grep -o 'detected=[0-9]*' | cut -d'=' -f2 || echo "0")
    correct=$(echo "$result" | grep -o 'correct=[0-9]*' | cut -d'=' -f2 || echo "0")
    accuracy=$(echo "$result" | grep -o 'accuracy=[0-9.]*' | cut -d'=' -f2 || echo "0")
    
    log "   ✓ $total positions, $detected detected, $correct correct (${accuracy}%)"
    
    TOTAL_POSITIONS=$((TOTAL_POSITIONS + total))
    TOTAL_DETECTED=$((TOTAL_DETECTED + detected))
    TOTAL_CORRECT=$((TOTAL_CORRECT + correct))
    TRAINED_COUNT=$((TRAINED_COUNT + 1))
    
    # Copy to latest
    if [ -f "$WEIGHTS_DIR/tactics/${tactic}_consol.json" ]; then
        cp "$WEIGHTS_DIR/tactics/${tactic}_consol.json" "$WEIGHTS_DIR/latest/tactics/"
    fi
done

echo ""
echo "=============================================="
log "✅ Training Complete"
echo "=============================================="
log "  Tactics trained: $TRAINED_COUNT"
log "  Total positions: $TOTAL_POSITIONS"
log "  Total detected: $TOTAL_DETECTED"
log "  Total correct: $TOTAL_CORRECT"

if [ "$TOTAL_POSITIONS" -gt 0 ]; then
    OVERALL_ACC=$(echo "scale=1; $TOTAL_CORRECT * 100 / $TOTAL_POSITIONS" | bc 2>/dev/null || echo "0")
    log "  Overall accuracy: ${OVERALL_ACC}%"
fi

log ""
log "Weights saved to: $WEIGHTS_DIR/latest/tactics/"
log "Full log: $LOG_FILE"

