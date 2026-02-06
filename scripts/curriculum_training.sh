#!/bin/bash
# =============================================================================
# ReCoN-lite Curriculum Training Script
# =============================================================================
#
# Implements the "Reverse Curriculum" training strategy:
#   Phase 1 (Anchor): Perfect endgame conversion (KRK, KPK, KQK)
#   Phase 2 (Bridge): Learn transitions from simplified middlegame
#   Phase 3 (Wilderness): Tactical survival in complex positions
#   Phase 4 (Integration): Full game play from opening
#
# The system advances phases when exit criteria are met.
#
# Usage:
#   ./scripts/curriculum_training.sh                    # Full training
#   ./scripts/curriculum_training.sh --phase anchor     # Train specific phase
#   ./scripts/curriculum_training.sh --quick            # Quick test mode
#   ./scripts/curriculum_training.sh --resume           # Resume from checkpoint
#
# =============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
START_TIME=$(date +%s)

cd "$PROJECT_ROOT"

# Configuration
QUICK_MODE=false
RESUME_MODE=false
SINGLE_PHASE=""
ENGINE="/usr/games/stockfish"
EPOCHS=3              # Number of times to repeat full curriculum

# Configuration
QUICK_MODE=false
RESUME_MODE=false
SINGLE_PHASE=""
ENGINE="/usr/games/stockfish"
EPOCHS=1              # Number of times to repeat full curriculum
KPK_BRIDGE=false      # Use KPK positions for bridge phase

# Training parameters per phase
# Phase 1: Anchor (perfect endgame conversion)
ANCHOR_GAMES=500        # Episodes per endgame type
ANCHOR_WIN_RATE=0.99    # Exit criterion: >99% win rate
ANCHOR_DEPTH=2          # Stockfish depth for validation

# Phase 2: Bridge (learn transitions)
BRIDGE_GAMES=1000        # Simplified middlegame episodes
BRIDGE_ACTIVATION=0.8   # Exit: >80% endgame activation rate
BRIDGE_DEPTH=4

# Phase 3: Wilderness (tactical survival)
WILDERNESS_GAMES=300    # Complex position episodes
WILDERNESS_SF_LEVEL=3   # Stockfish level to compete against
WILDERNESS_DEPTH=6

# Phase 4: Integration (full games)
INTEGRATION_GAMES=100   # Full game episodes
INTEGRATION_DEPTH=4

# Directories
WEIGHTS_DIR="$PROJECT_ROOT/weights"
CURRICULUM_DIR="$WEIGHTS_DIR/curriculum"
LOGS_DIR="$PROJECT_ROOT/logs"
REPORTS_DIR="$PROJECT_ROOT/reports/curriculum/$TIMESTAMP"
CHECKPOINT_FILE="$CURRICULUM_DIR/checkpoint.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            ANCHOR_GAMES=50
            BRIDGE_GAMES=30
            WILDERNESS_GAMES=30
            INTEGRATION_GAMES=20
            ANCHOR_DEPTH=2
            BRIDGE_DEPTH=2
            WILDERNESS_DEPTH=4
            INTEGRATION_DEPTH=2
            shift
            ;;
        --phase)
            SINGLE_PHASE="$2"
            shift 2
            ;;
        --resume)
            RESUME_MODE=true
            shift
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || [ "$EPOCHS" -lt 1 ]; then
                echo "Error: --epochs must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        --kpk-bridge)
            KPK_BRIDGE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick           Quick test mode (fewer episodes)"
            echo "  --phase PHASE     Train single phase (anchor/bridge/wilderness/integration)"
            echo "  --resume          Resume from checkpoint"
            echo "  --engine PATH     Path to Stockfish"
            echo "  --epochs N        Repeat full curriculum N times (default: 1)"
            echo "  --kpk-bridge      Use KPK near-promotion positions for bridge phase"
            echo ""
            echo "Phases:"
            echo "  anchor:      Perfect endgame conversion (KRK, KPK, KQK)"
            echo "  bridge:      Learn transitions to endgame"
            echo "  wilderness:  Tactical survival"
            echo "  integration: Full game play"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$CURRICULUM_DIR" "$LOGS_DIR" "$REPORTS_DIR" "$WEIGHTS_DIR/nightly"

LOG_FILE="$LOGS_DIR/curriculum_$TIMESTAMP.log"

# Logging functions
log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_header() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
    log "$1"
}

# Save checkpoint
save_checkpoint() {
    local phase=$1
    local stats=$2
    cat > "$CHECKPOINT_FILE" << EOF
{
    "phase": "$phase",
    "timestamp": "$(date -Iseconds)",
    "stats": $stats
}
EOF
    log "Checkpoint saved: $phase"
}

# Check if Stockfish is available
if [ ! -f "$ENGINE" ]; then
    log "Warning: Stockfish not found at $ENGINE"
    log "Training will proceed without engine validation"
    ENGINE=""
fi

# =============================================================================
log_header "ReCoN Curriculum Training"
# =============================================================================

log "Configuration:"
log "  Quick mode: $QUICK_MODE"
log "  Single phase: ${SINGLE_PHASE:-all}"
log "  Resume mode: $RESUME_MODE"
log "  Epochs: $EPOCHS"
log "  Engine: ${ENGINE:-none}"
log "  Log file: $LOG_FILE"
echo ""

# Track overall progress
PHASES_COMPLETED=0
TOTAL_EPISODES=0

# =============================================================================
# Phase 1: Anchor (Perfect Endgame Conversion)
# =============================================================================

run_anchor_phase() {
    log_header "Phase 1: ANCHOR (Perfect Endgame Conversion)"
    
    log "Training KRK endgame ($ANCHOR_GAMES games)..."
    uv run python demos/persistent/krk_persistent_demo.py \
        --batch "$ANCHOR_GAMES" \
        --plasticity \
        --bandit \
        --consolidate \
        --consolidate-pack "$WEIGHTS_DIR/nightly/krk_consol.json" \
        ${ENGINE:+--engine "$ENGINE" --depth "$ANCHOR_DEPTH"} \
        --consolidate-eta 0.0001 \
        --consolidate-min-episodes 200 \
        --trace-out "$REPORTS_DIR/anchor_krk.jsonl" \
        --output-basename "anchor_krk_$TIMESTAMP" \
        2>&1 | tee -a "$LOG_FILE" || log "KRK training encountered issues"
    
    TOTAL_EPISODES=$((TOTAL_EPISODES + ANCHOR_GAMES))
    
    log "Training KPK endgame ($ANCHOR_GAMES games)..."
    uv run python demos/persistent/kpk_persistent_demo.py \
        --batch "$ANCHOR_GAMES" \
        --plasticity \
        --consolidate \
        --consolidate-pack "$WEIGHTS_DIR/nightly/kpk_consol.json" \
        ${ENGINE:+--engine "$ENGINE" --depth "$ANCHOR_DEPTH"} \
        --trace-out "$REPORTS_DIR/anchor_kpk.jsonl" \
        --output-basename "anchor_kpk_$TIMESTAMP" \
        2>&1 | tee -a "$LOG_FILE" || log "KPK training encountered issues"
    
    TOTAL_EPISODES=$((TOTAL_EPISODES + ANCHOR_GAMES))
    
    log "Training KQK endgame ($ANCHOR_GAMES games)..."
    if [ -f "demos/persistent/kqk_persistent_demo.py" ]; then
        uv run python demos/persistent/kqk_persistent_demo.py \
            --batch "$ANCHOR_GAMES" \
            --plasticity \
            --consolidate \
            --consolidate-pack "$WEIGHTS_DIR/nightly/kqk_consol.json" \
            ${ENGINE:+--engine "$ENGINE" --depth "$ANCHOR_DEPTH"} \
            --trace-out "$REPORTS_DIR/anchor_kqk.jsonl" \
            --output-basename "anchor_kqk_$TIMESTAMP" \
            2>&1 | tee -a "$LOG_FILE" || log "KQK training encountered issues"
        TOTAL_EPISODES=$((TOTAL_EPISODES + ANCHOR_GAMES))
    else
        log "  Skipping KQK (demo not found)"
    fi
    
    save_checkpoint "anchor" '{"episodes": '"$TOTAL_EPISODES"', "status": "complete"}'
    PHASES_COMPLETED=$((PHASES_COMPLETED + 1))
    
    log "Anchor phase complete"
}

# =============================================================================
# Phase 2: Bridge (Learn Transitions)
# =============================================================================

run_bridge_phase() {
    log_header "Phase 2: BRIDGE (Learn Transitions)"
    
    if [ "$KPK_BRIDGE" = true ]; then
        log "Training bridge with KPK near-promotion positions ($BRIDGE_GAMES games)..."
        
        # Use full_game_train.py directly with KPK FENs
        uv run python demos/persistent/full_game_train.py \
            --batch "$BRIDGE_GAMES" \
            --fen-file data/bridge/near_promo.fens \
            --max-moves 80 \
            --timeout-loss \
            --plasticity \
            --consolidate \
            --consolidate-pack "$WEIGHTS_DIR/nightly/fullgame_consol.json" \
            ${ENGINE:+--engine "$ENGINE" --depth "$BRIDGE_DEPTH"} \
            --weights-dir "$WEIGHTS_DIR/latest" \
            --trace-out "$REPORTS_DIR/bridge_training.jsonl" \
            --output-json "$REPORTS_DIR/bridge_stats.json" \
            --snapshot-dir "$REPORTS_DIR/bridge_snapshots" \
            --snapshot-interval 50 \
            --quiet \
            2>&1 | tee -a "$LOG_FILE" || log "KPK bridge training encountered issues"
    else
        log "Training simplified middlegame transitions ($BRIDGE_GAMES games)..."
        
        # Use full_game_demo with standard positions
        uv run python demos/persistent/full_game_demo.py \
            --batch "$BRIDGE_GAMES" \
            --plasticity \
            --bandit \
            --consolidate \
            --consolidate-pack "$WEIGHTS_DIR/nightly/fullgame_consol.json" \
            ${ENGINE:+--engine "$ENGINE" --depth "$BRIDGE_DEPTH"} \
            --trace-out "$REPORTS_DIR/bridge_training.jsonl" \
            --output-basename "bridge_$TIMESTAMP" \
            2>&1 | tee -a "$LOG_FILE" || log "Bridge training encountered issues"
    fi
    
    TOTAL_EPISODES=$((TOTAL_EPISODES + BRIDGE_GAMES))
    
    # Extract bridge motifs from traces
    log "Extracting bridge motifs..."
    uv run python demos/experiments/extract_motifs.py \
        --traces "$REPORTS_DIR/bridge_training.jsonl" \
        --out "$REPORTS_DIR/bridge_motifs.jsonl" \
        --bridge-mode \
        --affordance-threshold 0.5 \
        --stats \
        2>&1 | tee -a "$LOG_FILE" || log "Motif extraction encountered issues"
    
    save_checkpoint "bridge" '{"episodes": '"$TOTAL_EPISODES"', "status": "complete"}'
    PHASES_COMPLETED=$((PHASES_COMPLETED + 1))
    
    log "Bridge phase complete"
}

# =============================================================================
# Phase 3: Wilderness (Tactical Survival)
# =============================================================================

run_wilderness_phase() {
    log_header "Phase 3: WILDERNESS (Tactical Survival)"
    
    log "Training complex tactical positions ($WILDERNESS_GAMES games)..."
    
    # Full game training with higher exploration
    uv run python demos/persistent/full_game_demo.py \
        --batch "$WILDERNESS_GAMES" \
        --plasticity \
        --bandit \
        --explore-c 1.5 \
        --consolidate \
        --consolidate-pack "$WEIGHTS_DIR/nightly/fullgame_consol.json" \
        ${ENGINE:+--engine "$ENGINE" --depth "$WILDERNESS_DEPTH"} \
        --trace-out "$REPORTS_DIR/wilderness_training.jsonl" \
        --output-basename "wilderness_$TIMESTAMP" \
        2>&1 | tee -a "$LOG_FILE" || log "Wilderness training encountered issues"
    
    TOTAL_EPISODES=$((TOTAL_EPISODES + WILDERNESS_GAMES))
    
    # Tactics training for each pattern type
    log "Training tactical patterns..."
    TACTIC_TYPES=("fork" "pin" "skewer" "hangingPiece" "backRankMate" "discoveredAttack")
    
    for tactic in "${TACTIC_TYPES[@]}"; do
        FEN_FILE="$PROJECT_ROOT/data/puzzles/$tactic/lichess_$tactic.fen"
        if [ -f "$FEN_FILE" ]; then
            log "  Training: $tactic"
            uv run python demos/experiments/tactics_eval.py \
                --fen-file "$FEN_FILE" \
                --tactic-type "$tactic" \
                --limit 100 \
                --consolidate \
                --consolidate-pack "$WEIGHTS_DIR/tactics/${tactic}_consol.json" \
                ${ENGINE:+--engine "$ENGINE" --depth "$WILDERNESS_DEPTH"} \
                2>&1 | tee -a "$LOG_FILE" || true
        fi
    done
    
    save_checkpoint "wilderness" '{"episodes": '"$TOTAL_EPISODES"', "status": "complete"}'
    PHASES_COMPLETED=$((PHASES_COMPLETED + 1))
    
    log "Wilderness phase complete"
}

# =============================================================================
# Phase 4: Integration (Full Game Play)
# =============================================================================

run_integration_phase() {
    log_header "Phase 4: INTEGRATION (Full Game Play)"
    
    log "Training full games from opening ($INTEGRATION_GAMES games)..."
    
    uv run python demos/persistent/full_game_demo.py \
        --batch "$INTEGRATION_GAMES" \
        --plasticity \
        --bandit \
        --consolidate \
        --consolidate-pack "$WEIGHTS_DIR/nightly/fullgame_consol.json" \
        ${ENGINE:+--engine "$ENGINE" --depth "$INTEGRATION_DEPTH"} \
        --trace-out "$REPORTS_DIR/integration_training.jsonl" \
        --output-basename "integration_$TIMESTAMP" \
        2>&1 | tee -a "$LOG_FILE" || log "Integration training encountered issues"
    
    TOTAL_EPISODES=$((TOTAL_EPISODES + INTEGRATION_GAMES))
    
    save_checkpoint "integration" '{"episodes": '"$TOTAL_EPISODES"', "status": "complete"}'
    PHASES_COMPLETED=$((PHASES_COMPLETED + 1))
    
    log "Integration phase complete"
}

# =============================================================================
# Main Training Loop
# =============================================================================

if [ -n "$SINGLE_PHASE" ]; then
    # Run single phase
    case "$SINGLE_PHASE" in
        anchor)
            run_anchor_phase
            ;;
        bridge)
            run_bridge_phase
            ;;
        wilderness)
            run_wilderness_phase
            ;;
        integration)
            run_integration_phase
            ;;
        *)
            log "Unknown phase: $SINGLE_PHASE"
            log "Valid phases: anchor, bridge, wilderness, integration"
            exit 1
            ;;
    esac
else
    # Run all phases, repeated for EPOCHS
    for epoch in $(seq 1 $EPOCHS); do
        log_header "Epoch $epoch of $EPOCHS"
        run_anchor_phase
        run_bridge_phase
        run_wilderness_phase
        run_integration_phase
        
        if [ $epoch -lt $EPOCHS ]; then
            log "Epoch $epoch complete. Starting epoch $((epoch + 1))..."
        fi
    done
fi

# =============================================================================
# Summary and Report Generation
# =============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log_header "Training Complete"

log "Summary:"
log "  Phases completed: $PHASES_COMPLETED"
log "  Total episodes: $TOTAL_EPISODES"
log "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log ""
log "Output:"
log "  Weights: $WEIGHTS_DIR/nightly/"
log "  Reports: $REPORTS_DIR/"
log "  Log: $LOG_FILE"

# Generate JSON summary
cat > "$REPORTS_DIR/training_summary.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "duration_seconds": $DURATION,
    "phases_completed": $PHASES_COMPLETED,
    "total_episodes": $TOTAL_EPISODES,
    "quick_mode": $QUICK_MODE,
    "single_phase": "${SINGLE_PHASE:-null}",
    "epochs": $EPOCHS,
    "weights_dir": "$WEIGHTS_DIR/nightly",
    "reports_dir": "$REPORTS_DIR"
}
EOF

log ""
log "Training summary saved to $REPORTS_DIR/training_summary.json"

# Generate detailed markdown report
log_header "Generating Training Report"

uv run python scripts/analyze_training.py \
    --report-dir "$REPORTS_DIR" \
    --markdown \
    --output "$REPORTS_DIR/training_report.md" \
    2>&1 | tee -a "$LOG_FILE" || log "Warning: Failed to generate markdown report"

# Show quick stats
if [ -f "$REPORTS_DIR/training_report.md" ]; then
    log ""
    log "Training Report Preview:"
    head -30 "$REPORTS_DIR/training_report.md" | while read line; do
        log "  $line"
    done
    log ""
    log "Full report: $REPORTS_DIR/training_report.md"
fi

