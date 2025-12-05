#!/bin/bash
# =============================================================================
# ReCoN-lite Overnight Training Script (Modular Version)
# =============================================================================
# 
# This script runs modular overnight training:
# 1. BEFORE Evaluation: 50 full games vs random (baseline win rate)
# 2. Modular Training:
#    - Tactics training (each pattern type independently)
#    - KRK endgame training
#    - KPK endgame training
#    - Full game training with plasticity/consolidation
# 3. AFTER Evaluation: 50 full games vs random (final win rate)
# 4. Generates comprehensive comparison report
#
# Usage: ./scripts/overnight_training.sh [--quick] [--skip-eval] [--engine /path/to/stockfish]
#   --quick: Smaller batches for testing (~30 min)
#   --skip-eval: Skip BEFORE/AFTER full-game evaluation
#   --engine: Path to Stockfish (default: /usr/games/stockfish)
#
# =============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
START_TIME=$(date +%s)

# Configuration
QUICK_MODE=false
SKIP_EVAL=false
ENGINE="/usr/games/stockfish"

# Training parameters (normal mode)
KRK_GAMES=100
KPK_GAMES=200
FULLGAME_GAMES=100
FULLGAME_EVAL=50       # 50 games before/after for win rate
TACTICS_LIMIT=200      # Positions per tactic type

# Stockfish depth configuration (higher = more accurate but slower)
# - Depth 2: Fast (~10ms/pos), good for endgames where volume matters
# - Depth 6: Medium (~100ms/pos), good for tactics (2-3 move combos)
# - Depth 10+: Slow (~1s/pos), good for final validation
ENDGAME_DEPTH=2        # KRK/KPK training - depth matters less, volume matters more
TACTICS_DEPTH=6        # Tactics validation - needs to see 2-3 move combos
FULLGAME_DEPTH=4       # Full game training - balance speed and accuracy

GAME_TIMEOUT=180       # 3 minutes max per game

# Directories
WEIGHTS_DIR="$PROJECT_ROOT/weights"
BACKUP_DIR="$WEIGHTS_DIR/backups/$TIMESTAMP"
REPORTS_DIR="$PROJECT_ROOT/reports/overnight/$TIMESTAMP"
LOGS_DIR="$PROJECT_ROOT/logs"
CHECKPOINT_FILE="$REPORTS_DIR/checkpoint.txt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            KRK_GAMES=50
            KPK_GAMES=20
            FULLGAME_GAMES=20
            FULLGAME_EVAL=10
            TACTICS_LIMIT=50
            ENDGAME_DEPTH=2
            TACTICS_DEPTH=4
            FULLGAME_DEPTH=2
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--skip-eval] [--engine PATH]"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$BACKUP_DIR" "$REPORTS_DIR" "$LOGS_DIR" "$WEIGHTS_DIR/nightly" "$WEIGHTS_DIR/tactics" "$WEIGHTS_DIR/latest/tactics"

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log"
}

log_header() {
    echo "" | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log"
    echo "==============================================================================" | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log"
    log "$1"
    echo "==============================================================================" | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log"
}

save_checkpoint() {
    echo "$1" > "$CHECKPOINT_FILE"
    log "  [CHECKPOINT] $1"
}

safe_increment() {
    local var_name=$1
    local current_val=${!var_name}
    eval "$var_name=$((current_val + 1))"
}

calc_percentage() {
    local num=$1
    local denom=$2
    if [ "$denom" -gt 0 ] 2>/dev/null; then
        echo "scale=1; $num * 100 / $denom" | bc 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Run full game evaluation with new trainer
run_full_game_eval() {
    local n_games=$1
    local weights_path=$2
    local output_file=$3
    
    local wins=0
    local losses=0
    local draws=0
    
    # Use the new full_game_train.py for evaluation
    local weight_arg=""
    if [ -n "$weights_path" ] && [ -f "$weights_path" ]; then
        weight_arg="--consolidate-pack $weights_path"
    fi
    
    local result=$(timeout $((GAME_TIMEOUT * n_games)) uv run python demos/persistent/full_game_train.py \
        --batch "$n_games" \
        --vs-random \
        --max-moves 150 \
        --quiet \
        $weight_arg 2>&1 | tail -1)
    
    # Parse JSON result
    if echo "$result" | grep -q '"wins"'; then
        wins=$(echo "$result" | grep -o '"wins":[0-9]*' | cut -d':' -f2)
        losses=$(echo "$result" | grep -o '"losses":[0-9]*' | cut -d':' -f2)
        draws=$(echo "$result" | grep -o '"draws":[0-9]*' | cut -d':' -f2)
    fi
    
    echo "wins=$wins,losses=$losses,draws=$draws" > "$output_file"
    echo "$wins"
}

cd "$PROJECT_ROOT"

# =============================================================================
log_header "🌙 OVERNIGHT TRAINING - Starting at $(date)"
# =============================================================================

log "Configuration:"
log "  Quick mode: $QUICK_MODE"
log "  Skip evaluation: $SKIP_EVAL"
log "  Engine: $ENGINE"
log "  KRK games: $KRK_GAMES"
log "  KPK games: $KPK_GAMES"
log "  Full-game training: $FULLGAME_GAMES"
log "  Full-game eval: $FULLGAME_EVAL games (before/after)"
log "  Tactics positions: $TACTICS_LIMIT per type"
log "  Stockfish depths: endgame=$ENDGAME_DEPTH, tactics=$TACTICS_DEPTH, fullgame=$FULLGAME_DEPTH"
log "  Backup dir: $BACKUP_DIR"
log "  Reports dir: $REPORTS_DIR"

save_checkpoint "STARTED"

# =============================================================================
log_header "📦 STEP 1: Backing Up Initial Weights"
# =============================================================================

log "Copying consolidation state..."
if [ -f "$WEIGHTS_DIR/nightly/krk_consol.json" ]; then
    cp "$WEIGHTS_DIR/nightly/krk_consol.json" "$BACKUP_DIR/krk_consol_FIRST.json"
    log "  Saved: krk_consol_FIRST.json"
fi

if [ -f "$WEIGHTS_DIR/nightly/fullgame_consol.json" ]; then
    cp "$WEIGHTS_DIR/nightly/fullgame_consol.json" "$BACKUP_DIR/fullgame_consol_FIRST.json"
    log "  Saved: fullgame_consol_FIRST.json"
fi

# Backup all weight packs
for pack in "$WEIGHTS_DIR"/*.swp "$WEIGHTS_DIR/subgraphs"/*.swp "$WEIGHTS_DIR/tactics"/*.json; do
    if [ -f "$pack" ]; then
        cp "$pack" "$BACKUP_DIR/" 2>/dev/null || true
        log "  Backed up: $(basename "$pack")"
    fi
done

log "Initial backup complete."
save_checkpoint "BACKUP_COMPLETE"

# =============================================================================
# PHASE 1: BEFORE Evaluation
# =============================================================================

BEFORE_WINS=0
BEFORE_LOSSES=0
BEFORE_DRAWS=0
BEFORE_WIN_RATE="0"

if [ "$SKIP_EVAL" = false ]; then
    log_header "📊 PHASE 1: BEFORE Evaluation ($FULLGAME_EVAL full games)"
    
    log "Running $FULLGAME_EVAL games vs random (baseline)..."
    BEFORE_WINS=$(run_full_game_eval "$FULLGAME_EVAL" "" "$REPORTS_DIR/fullgame_BEFORE.txt")
    
    # Read back the results
    if [ -f "$REPORTS_DIR/fullgame_BEFORE.txt" ]; then
        eval $(cat "$REPORTS_DIR/fullgame_BEFORE.txt" | tr ',' '\n')
        BEFORE_LOSSES=${losses:-0}
        BEFORE_DRAWS=${draws:-0}
    fi
    
    BEFORE_WIN_RATE=$(calc_percentage $BEFORE_WINS $FULLGAME_EVAL)
    log "Full-game BEFORE: $BEFORE_WINS wins, $BEFORE_DRAWS draws, $BEFORE_LOSSES losses ($BEFORE_WIN_RATE% win rate)"
    
    save_checkpoint "BEFORE_EVAL_COMPLETE"
else
    log_header "📊 PHASE 1: SKIPPED (--skip-eval)"
fi

# =============================================================================
log_header "🎓 PHASE 2A: Tactics Training (Modular)"
# =============================================================================

TACTICS=("fork" "pin" "skewer" "hangingPiece" "backRankMate" "discoveredAttack" 
         "attraction" "deflection" "doubleCheck" "smotheredMate" "trappedPiece" "quietMove" "sacrifice")
TACTICS_TRAINED=0
TACTICS_DETECTED=0

log "Training tactical patterns..."

for tactic in "${TACTICS[@]}"; do
    FEN_FILE=""
    
    # Check various possible locations
    for candidate in \
        "$PROJECT_ROOT/data/puzzles/$tactic/lichess_$tactic.fen" \
        "$PROJECT_ROOT/data/puzzles/${tactic,,}/lichess_${tactic,,}.fen"; do
        if [ -f "$candidate" ]; then
            FEN_FILE="$candidate"
            break
        fi
    done
    
    if [ -n "$FEN_FILE" ]; then
        log "  Training $tactic from $FEN_FILE..."
        
        result=$(timeout 300 uv run python demos/experiments/tactics_eval.py \
            --fen-file "$FEN_FILE" \
            --tactic-type "$tactic" \
            --limit "$TACTICS_LIMIT" \
            --consolidate \
            --consolidate-pack "$WEIGHTS_DIR/tactics/${tactic}_consol.json" \
            --engine "$ENGINE" --depth "$TACTICS_DEPTH" \
            --output "$REPORTS_DIR/tactics_${tactic}.json" \
            --quiet 2>&1 | grep "TACTICS_EVAL_RESULT" || echo "")
        
        if [ -n "$result" ]; then
            log "    $result"
            safe_increment TACTICS_TRAINED
            # Parse detected count
            detected=$(echo "$result" | grep -o 'detected=[0-9]*' | cut -d'=' -f2 || echo "0")
            TACTICS_DETECTED=$((TACTICS_DETECTED + detected))
        fi
    else
        log "  Skipping $tactic (no FEN file found)"
    fi
done

log "Tactical training complete: $TACTICS_TRAINED tactics types, $TACTICS_DETECTED patterns detected"
save_checkpoint "TACTICS_TRAINING_COMPLETE"

# =============================================================================
log_header "🎓 PHASE 2B: KRK Endgame Training"
# =============================================================================

log "Training KRK with $KRK_GAMES games..."

uv run python demos/persistent/krk_persistent_demo.py \
    --batch "$KRK_GAMES" \
    --plasticity \
    --bandit \
    --consolidate \
    --consolidate-pack "$WEIGHTS_DIR/nightly/krk_consol.json" \
    --engine "$ENGINE" --depth "$ENDGAME_DEPTH" \
    --trace-out "$REPORTS_DIR/krk_training.jsonl" \
    --output-basename "krk_overnight_$TIMESTAMP" \
    2>&1 | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log" || log "KRK training encountered issues"

if [ -f "$WEIGHTS_DIR/nightly/krk_consol.json" ]; then
    cp "$WEIGHTS_DIR/nightly/krk_consol.json" "$BACKUP_DIR/krk_consol_after_KRK.json"
    log "  Checkpoint saved: krk_consol_after_KRK.json"
fi

save_checkpoint "KRK_TRAINING_COMPLETE"

# =============================================================================
log_header "🎓 PHASE 2C: KPK Endgame Training"
# =============================================================================

log "Training KPK with $KPK_GAMES games..."

uv run python demos/persistent/kpk_persistent_demo.py \
    --batch "$KPK_GAMES" \
    --plasticity \
    --consolidate \
    --consolidate-pack "$WEIGHTS_DIR/nightly/kpk_consol.json" \
    --engine "$ENGINE" --depth "$ENDGAME_DEPTH" \
    --trace-out "$REPORTS_DIR/kpk_training.jsonl" \
    --output-basename "kpk_overnight_$TIMESTAMP" \
    2>&1 | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log" || log "KPK training encountered issues"

save_checkpoint "KPK_TRAINING_COMPLETE"

# =============================================================================
log_header "🎓 PHASE 2D: Full Game Training"
# =============================================================================

log "Training full games with $FULLGAME_GAMES games..."

uv run python demos/persistent/full_game_train.py \
    --batch "$FULLGAME_GAMES" \
    --plasticity \
    --consolidate \
    --consolidate-pack "$WEIGHTS_DIR/nightly/fullgame_consol.json" \
    --stem-cells \
    --stem-cell-path "$WEIGHTS_DIR/nightly/stem_cells.json" \
    --engine "$ENGINE" \
    --depth "$FULLGAME_DEPTH" \
    --trace-out "$REPORTS_DIR/fullgame_training.jsonl" \
    --output-json "$REPORTS_DIR/fullgame_training_stats.json" \
    2>&1 | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log" || log "Full game training encountered issues"

save_checkpoint "FULLGAME_TRAINING_COMPLETE"

# =============================================================================
log_header "🧬 PHASE 2E: Pattern Promotion (Stem Cells)"
# =============================================================================

log "Promoting discovered patterns from stem cells..."

PROMOTED_COUNT=0
if [ -f "$WEIGHTS_DIR/nightly/stem_cells.json" ]; then
    promotion_result=$(uv run python tools/promote_patterns.py \
        --stem-cells "$WEIGHTS_DIR/nightly/stem_cells.json" \
        --output-dir "$WEIGHTS_DIR/promoted" \
        --min-consistency 0.6 \
        2>&1 | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log" | grep "PROMOTION_RESULT" || echo "")
    
    if [ -n "$promotion_result" ]; then
        PROMOTED_COUNT=$(echo "$promotion_result" | grep -o 'promoted=[0-9]*' | cut -d'=' -f2 || echo "0")
    fi
    
    log "Promoted $PROMOTED_COUNT patterns from stem cells"
    
    # Backup promoted patterns
    if [ -d "$WEIGHTS_DIR/promoted" ]; then
        cp -r "$WEIGHTS_DIR/promoted" "$BACKUP_DIR/promoted_$TIMESTAMP" 2>/dev/null || true
    fi
else
    log "No stem cells file found - skipping pattern promotion"
fi

save_checkpoint "PATTERN_PROMOTION_COMPLETE"

# =============================================================================
# PHASE 3: AFTER Evaluation
# =============================================================================

AFTER_WINS=0
AFTER_LOSSES=0
AFTER_DRAWS=0
AFTER_WIN_RATE="0"

if [ "$SKIP_EVAL" = false ]; then
    log_header "📊 PHASE 3: AFTER Evaluation ($FULLGAME_EVAL full games)"
    
    log "Running $FULLGAME_EVAL games vs random (with trained weights)..."
    AFTER_WINS=$(run_full_game_eval "$FULLGAME_EVAL" "$WEIGHTS_DIR/nightly/fullgame_consol.json" "$REPORTS_DIR/fullgame_AFTER.txt")
    
    # Read back the results
    if [ -f "$REPORTS_DIR/fullgame_AFTER.txt" ]; then
        eval $(cat "$REPORTS_DIR/fullgame_AFTER.txt" | tr ',' '\n')
        AFTER_LOSSES=${losses:-0}
        AFTER_DRAWS=${draws:-0}
    fi
    
    AFTER_WIN_RATE=$(calc_percentage $AFTER_WINS $FULLGAME_EVAL)
    log "Full-game AFTER: $AFTER_WINS wins, $AFTER_DRAWS draws, $AFTER_LOSSES losses ($AFTER_WIN_RATE% win rate)"
    
    save_checkpoint "AFTER_EVAL_COMPLETE"
fi

# =============================================================================
log_header "📦 PHASE 4: Saving Final Weights"
# =============================================================================

log "Saving final weight packs..."
if [ -f "$WEIGHTS_DIR/nightly/krk_consol.json" ]; then
    cp "$WEIGHTS_DIR/nightly/krk_consol.json" "$BACKUP_DIR/krk_consol_FINAL.json"
    log "  Saved: krk_consol_FINAL.json"
fi

if [ -f "$WEIGHTS_DIR/nightly/fullgame_consol.json" ]; then
    cp "$WEIGHTS_DIR/nightly/fullgame_consol.json" "$BACKUP_DIR/fullgame_consol_FINAL.json"
    log "  Saved: fullgame_consol_FINAL.json"
fi

if [ -f "$WEIGHTS_DIR/nightly/kpk_consol.json" ]; then
    cp "$WEIGHTS_DIR/nightly/kpk_consol.json" "$BACKUP_DIR/kpk_consol_FINAL.json"
    log "  Saved: kpk_consol_FINAL.json"
fi

# Update weights/latest/ with trained weights
log "Updating latest weights (weights/latest/)..."
if [ -f "$WEIGHTS_DIR/nightly/krk_consol.json" ]; then
    cp "$WEIGHTS_DIR/nightly/krk_consol.json" "$WEIGHTS_DIR/latest/krk_consol.json"
    log "  Latest: krk_consol.json"
fi
if [ -f "$WEIGHTS_DIR/nightly/kpk_consol.json" ]; then
    cp "$WEIGHTS_DIR/nightly/kpk_consol.json" "$WEIGHTS_DIR/latest/kpk_consol.json"
    log "  Latest: kpk_consol.json"
fi
if [ -f "$WEIGHTS_DIR/nightly/fullgame_consol.json" ]; then
    cp "$WEIGHTS_DIR/nightly/fullgame_consol.json" "$WEIGHTS_DIR/latest/fullgame_consol.json"
    log "  Latest: fullgame_consol.json"
fi
if [ -f "$WEIGHTS_DIR/nightly/stem_cells.json" ]; then
    cp "$WEIGHTS_DIR/nightly/stem_cells.json" "$WEIGHTS_DIR/latest/stem_cells.json"
    log "  Latest: stem_cells.json"
fi

# Copy all tactic weights to latest
for tactic_weight in "$WEIGHTS_DIR/tactics"/*.json; do
    if [ -f "$tactic_weight" ]; then
        cp "$tactic_weight" "$WEIGHTS_DIR/latest/tactics/"
        log "  Latest: tactics/$(basename "$tactic_weight")"
    fi
done

log "Latest weights updated. Use 'weights/latest/' for full_game_demo.py"

save_checkpoint "WEIGHTS_SAVED"

# =============================================================================
log_header "📈 PHASE 5: Generating Report"
# =============================================================================

REPORT_FILE="$REPORTS_DIR/overnight_summary.md"
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

WINS_CHANGE=$((AFTER_WINS - BEFORE_WINS))

cat > "$REPORT_FILE" << EOF
# 🌙 Overnight Training Report

**Generated:** $(date)
**Started:** $TIMESTAMP
**Duration:** ${DURATION_MIN} minutes

## 📊 Win Rate Comparison (vs Random)

| Metric | BEFORE | AFTER | Change |
|--------|--------|-------|--------|
| **Games** | $FULLGAME_EVAL | $FULLGAME_EVAL | - |
| **Wins** | $BEFORE_WINS | $AFTER_WINS | $WINS_CHANGE |
| **Losses** | $BEFORE_LOSSES | $AFTER_LOSSES | $((AFTER_LOSSES - BEFORE_LOSSES)) |
| **Draws** | $BEFORE_DRAWS | $AFTER_DRAWS | $((AFTER_DRAWS - BEFORE_DRAWS)) |
| **Win Rate** | ${BEFORE_WIN_RATE}% | ${AFTER_WIN_RATE}% | $(echo "$AFTER_WIN_RATE - $BEFORE_WIN_RATE" | bc)% |

## 🎓 Training Summary

### Endgame Training
- **KRK games:** $KRK_GAMES
- **KPK games:** $KPK_GAMES
- **Engine depth:** $ENDGAME_DEPTH (endgames), $FULLGAME_DEPTH (full games), $TACTICS_DEPTH (tactics)

### Full Game Training
- **Games trained:** $FULLGAME_GAMES
- **Stem cells:** Enabled
- **Patterns promoted:** $PROMOTED_COUNT

### Tactical Training
- **Tactic types trained:** $TACTICS_TRAINED
- **Total patterns detected:** $TACTICS_DETECTED
- **Positions per type:** $TACTICS_LIMIT

EOF

# Add per-tactic results
echo "## 🎯 Tactical Results" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

for tactic_result in "$REPORTS_DIR"/tactics_*.json; do
    if [ -f "$tactic_result" ]; then
        tactic_name=$(basename "$tactic_result" .json | sed 's/tactics_//')
        stats=$(python3 -c "
import json
try:
    d=json.load(open('$tactic_result'))
    s=d.get('stats',{})
    print(f\"- **{s.get('total',0)}** positions, **{s.get('accuracy',0)*100:.1f}%** accuracy\")
except:
    print('- Stats unavailable')
" 2>/dev/null || echo "- Stats unavailable")
        echo "- \`$tactic_name\`: $stats" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << EOF

## 📦 Weight Packs

| Pack | Path |
|------|------|
| KRK (before) | \`$BACKUP_DIR/krk_consol_FIRST.json\` |
| KRK (after) | \`$BACKUP_DIR/krk_consol_FINAL.json\` |
| Full Game | \`$BACKUP_DIR/fullgame_consol_FINAL.json\` |
| KPK | \`$BACKUP_DIR/kpk_consol_FINAL.json\` |

## 📁 Files Generated

- Traces: \`$REPORTS_DIR/\`
- Backups: \`$BACKUP_DIR/\`
- Logs: \`$LOGS_DIR/overnight_$TIMESTAMP.log\`

## 🔍 Compare Weight Changes

\`\`\`bash
# KRK changes
uv run python tools/pack_diff.py \\
  $BACKUP_DIR/krk_consol_FIRST.json \\
  $BACKUP_DIR/krk_consol_FINAL.json
\`\`\`

EOF

log "Report saved to: $REPORT_FILE"

# Try to generate pack diff
log "Generating weight diff..."
if [ -f "$BACKUP_DIR/krk_consol_FIRST.json" ] && [ -f "$BACKUP_DIR/krk_consol_FINAL.json" ]; then
    uv run python tools/pack_diff.py \
        "$BACKUP_DIR/krk_consol_FIRST.json" \
        "$BACKUP_DIR/krk_consol_FINAL.json" \
        >> "$REPORT_FILE" 2>&1 || log "  Pack diff failed (non-critical)"
fi

save_checkpoint "COMPLETE"

# =============================================================================
log_header "✅ OVERNIGHT TRAINING COMPLETE"
# =============================================================================

log ""
log "Results:"
log "  Duration: ${DURATION_MIN} minutes"
log "  Full-game win rate: ${BEFORE_WIN_RATE}% → ${AFTER_WIN_RATE}%"
log "  Tactics types trained: $TACTICS_TRAINED"
log ""
log "Files saved:"
log "  Report: $REPORT_FILE"
log "  KRK weights: $BACKUP_DIR/krk_consol_FINAL.json"
log "  Full game weights: $BACKUP_DIR/fullgame_consol_FINAL.json"
log "  Full log: $LOGS_DIR/overnight_$TIMESTAMP.log"
log ""
log "🌅 Training finished at $(date)"

# Show the summary report
echo ""
echo "==================== SUMMARY ===================="
cat "$REPORT_FILE"
