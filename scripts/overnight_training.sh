#!/bin/bash
# =============================================================================
# ReCoN-lite Overnight Training Script
# =============================================================================
# 
# This script:
# 1. Backs up ALL current weights (so you can restore/compare later)
# 2. Runs BEFORE evaluation (baseline metrics)
# 3. Runs staged training: KRK → KPK → full-game eval
# 4. Saves intermediate weight snapshots
# 5. Runs AFTER evaluation (final metrics)
# 6. Generates comprehensive comparison report
#
# Usage: ./scripts/overnight_training.sh [--quick] [--engine /path/to/stockfish]
#   --quick: Smaller batches for testing (~30 min)
#   --engine: Path to Stockfish (default: /usr/games/stockfish)
#
# =============================================================================

set -e  # Exit on error
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Configuration - adjust these for longer runs
QUICK_MODE=false
ENGINE="/usr/games/stockfish"
KRK_GAMES=2000         # ~1.5 hours with Stockfish depth 2
KPK_GAMES=100          # ~1 hour
FULLGAME_EVAL=50       # For before/after comparison
DEPTH=2

# Directories
WEIGHTS_DIR="$PROJECT_ROOT/weights"
BACKUP_DIR="$WEIGHTS_DIR/backups/$TIMESTAMP"
REPORTS_DIR="$PROJECT_ROOT/reports/overnight/$TIMESTAMP"
LOGS_DIR="$PROJECT_ROOT/logs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            KRK_GAMES=20
            KPK_GAMES=10
            FULLGAME_EVAL=10
            shift
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$BACKUP_DIR" "$REPORTS_DIR" "$LOGS_DIR"

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log"
}

log_header() {
    echo ""
    echo "=============================================================================="
    log "$1"
    echo "=============================================================================="
}

cd "$PROJECT_ROOT"

# =============================================================================
log_header "🌙 OVERNIGHT TRAINING - Starting at $(date)"
# =============================================================================

log "Configuration:"
log "  Quick mode: $QUICK_MODE"
log "  Engine: $ENGINE"
log "  KRK games: $KRK_GAMES"
log "  KPK games: $KPK_GAMES"
log "  Full-game eval: $FULLGAME_EVAL"
log "  Backup dir: $BACKUP_DIR"
log "  Reports dir: $REPORTS_DIR"

# =============================================================================
log_header "📦 STEP 1: Backing Up Initial Weights"
# =============================================================================

# Save FIRST weight pack - the one we want to compare against
log "Copying consolidation state..."
if [ -f "$WEIGHTS_DIR/nightly/krk_consol.json" ]; then
    cp "$WEIGHTS_DIR/nightly/krk_consol.json" "$BACKUP_DIR/krk_consol_FIRST.json"
    log "  Saved: krk_consol_FIRST.json"
fi

# Backup all weight packs
for pack in "$WEIGHTS_DIR"/*.swp "$WEIGHTS_DIR/subgraphs"/*.swp; do
    if [ -f "$pack" ]; then
        cp "$pack" "$BACKUP_DIR/"
        log "  Backed up: $(basename "$pack")"
    fi
done

log "Initial backup complete."

# =============================================================================
log_header "📊 STEP 2: BEFORE Evaluation (Baseline Metrics)"
# =============================================================================

log "Running KRK baseline evaluation..."
uv run python demos/experiments/batch_eval.py \
    --mode krk \
    --fen-file data/endgames/krk/random.fen \
    --runs 50 \
    --max-plies 100 \
    --engine "$ENGINE" --depth "$DEPTH" \
    --trace-out "$REPORTS_DIR/krk_BEFORE.jsonl" \
    > "$REPORTS_DIR/krk_BEFORE_stats.txt" 2>&1 || true

log "Running full-game vs random baseline..."
BEFORE_WINS=0
BEFORE_DRAWS=0
BEFORE_LOSSES=0
for i in $(seq 1 $FULLGAME_EVAL); do
    # Use --json-result for reliable parsing
    result=$(uv run python demos/persistent/full_game_demo.py \
        --max-moves 200 --vs-random --quiet --json-result 2>&1 | grep -o '"outcome":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    case "$result" in
        "win") ((BEFORE_WINS++)) ;;
        "loss") ((BEFORE_LOSSES++)) ;;
        "draw") ((BEFORE_DRAWS++)) ;;
        *)
            # Fallback: try parsing RESULT: format
            result_line=$(uv run python demos/persistent/full_game_demo.py \
                --max-moves 200 --vs-random --quiet 2>&1 | grep "RESULT:" | tail -1 || echo "")
            case "$result_line" in
                *"1-0"*) ((BEFORE_WINS++)) ;;
                *"0-1"*) ((BEFORE_LOSSES++)) ;;
                *"1/2-1/2"*) ((BEFORE_DRAWS++)) ;;
            esac
            ;;
    esac
    log "  Game $i: $result"
done
BEFORE_WIN_RATE=$(echo "scale=2; $BEFORE_WINS * 100 / $FULLGAME_EVAL" | bc 2>/dev/null || echo "0")
log "Full-game BEFORE: $BEFORE_WINS wins, $BEFORE_DRAWS draws, $BEFORE_LOSSES losses ($BEFORE_WIN_RATE% win rate)"
echo "wins: $BEFORE_WINS, draws: $BEFORE_DRAWS, losses: $BEFORE_LOSSES, win_rate: $BEFORE_WIN_RATE%" > "$REPORTS_DIR/fullgame_BEFORE.txt"

# =============================================================================
log_header "🎓 STEP 3: KRK Training"
# =============================================================================

log "Training KRK with $KRK_GAMES games..."
uv run python demos/persistent/krk_persistent_demo.py \
    --batch "$KRK_GAMES" \
    --plasticity \
    --bandit \
    --consolidate \
    --consolidate-pack "$WEIGHTS_DIR/nightly/krk_consol.json" \
    --engine "$ENGINE" --depth "$DEPTH" \
    --trace-out "$REPORTS_DIR/krk_training.jsonl" \
    --output-basename "krk_overnight_$TIMESTAMP" \
    2>&1 | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log"

# Checkpoint after KRK
log "Saving KRK checkpoint..."
cp "$WEIGHTS_DIR/nightly/krk_consol.json" "$BACKUP_DIR/krk_consol_after_KRK.json"

# Generate KRK report
log "Generating KRK training report..."
uv run python tools/report_consolidation.py "$WEIGHTS_DIR/nightly/krk_consol.json" \
    -o "$REPORTS_DIR/krk_consolidation.md" 2>&1 || true

# =============================================================================
log_header "🎓 STEP 4: KPK Training"
# =============================================================================

log "Training KPK with $KPK_GAMES games..."
# Check if KPK persistent demo exists and has training support
if [ -f "$PROJECT_ROOT/demos/persistent/kpk_persistent_demo.py" ]; then
    uv run python demos/persistent/kpk_persistent_demo.py \
        --batch "$KPK_GAMES" \
        --engine "$ENGINE" --depth "$DEPTH" \
        --trace-out "$REPORTS_DIR/kpk_training.jsonl" \
        --output-basename "kpk_overnight_$TIMESTAMP" \
        2>&1 | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log" || log "KPK training skipped (not fully implemented)"
else
    log "KPK persistent demo not found, skipping..."
fi

# =============================================================================
log_header "🎯 STEP 4.5: Tactical Micro-Training (M8)"
# =============================================================================

# Create tactics weights directory
mkdir -p "$WEIGHTS_DIR/tactics"

# List of tactics to train on
TACTICS=("fork" "pin" "skewer" "hangingPiece" "backRankMate" "discoveredAttack")

log "Training tactical patterns..."

# First check if we have puzzle data
if [ -d "$PROJECT_ROOT/data/puzzles" ]; then
    TACTICS_FOUND=0
    TACTICS_RESULTS=""
    
    for tactic in "${TACTICS[@]}"; do
        # Look for FEN files for this tactic
        FEN_FILE=""
        
        # Check various possible locations
        for candidate in \
            "$PROJECT_ROOT/data/puzzles/$tactic/lichess_$tactic.fen" \
            "$PROJECT_ROOT/data/puzzles/${tactic,,}/lichess_${tactic,,}.fen" \
            "$PROJECT_ROOT/data/benchmarks/${tactic}_suite.fen"; do
            if [ -f "$candidate" ]; then
                FEN_FILE="$candidate"
                break
            fi
        done
        
        if [ -n "$FEN_FILE" ]; then
            log "  Training $tactic from $FEN_FILE..."
            
            # Run tactical evaluation with consolidation
            result=$(uv run python demos/experiments/tactics_eval.py \
                --fen-file "$FEN_FILE" \
                --tactic-type "$tactic" \
                --limit 200 \
                --consolidate \
                --consolidate-pack "$WEIGHTS_DIR/tactics/${tactic}_consol.json" \
                --output "$REPORTS_DIR/tactics_${tactic}.json" \
                --quiet 2>&1 | grep "TACTICS_EVAL_RESULT" || echo "")
            
            if [ -n "$result" ]; then
                log "    $result"
                TACTICS_RESULTS="$TACTICS_RESULTS\n$tactic: $result"
                ((TACTICS_FOUND++)) || true
            fi
        else
            log "  Skipping $tactic (no FEN file found)"
        fi
    done
    
    log "Tactical training complete: $TACTICS_FOUND tactics processed"
    
    # Also run combined tactics evaluation if available
    if [ -f "$PROJECT_ROOT/data/puzzles/combined_tactics.fen" ]; then
        log "  Running combined tactics evaluation..."
        uv run python demos/experiments/tactics_eval.py \
            --fen-file "$PROJECT_ROOT/data/puzzles/combined_tactics.fen" \
            --tactic-type "combined" \
            --limit 500 \
            --output "$REPORTS_DIR/tactics_combined.json" \
            --quiet 2>&1 | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log" || true
    fi
    
    # Run on built-in tactics suite if no puzzle data found
    if [ $TACTICS_FOUND -eq 0 ] && [ -f "$PROJECT_ROOT/data/benchmarks/tactics_suite.fen" ]; then
        log "  Using built-in tactics_suite.fen..."
        uv run python demos/experiments/tactics_eval.py \
            --fen-file "$PROJECT_ROOT/data/benchmarks/tactics_suite.fen" \
            --tactic-type "mixed" \
            --consolidate \
            --consolidate-pack "$WEIGHTS_DIR/tactics/mixed_consol.json" \
            --output "$REPORTS_DIR/tactics_suite.json" \
            --quiet 2>&1 | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log" || true
    fi
else
    log "No puzzle data found. Run tools/import_lichess_puzzles.py first for enhanced tactical training."
    
    # Fall back to built-in tactics suite
    if [ -f "$PROJECT_ROOT/data/benchmarks/tactics_suite.fen" ]; then
        log "  Using built-in tactics_suite.fen..."
        uv run python demos/experiments/tactics_eval.py \
            --fen-file "$PROJECT_ROOT/data/benchmarks/tactics_suite.fen" \
            --tactic-type "mixed" \
            --consolidate \
            --consolidate-pack "$WEIGHTS_DIR/tactics/mixed_consol.json" \
            --output "$REPORTS_DIR/tactics_suite.json" \
            --quiet 2>&1 | tee -a "$LOGS_DIR/overnight_$TIMESTAMP.log" || true
    fi
fi

# =============================================================================
log_header "📊 STEP 5: AFTER Evaluation (Final Metrics)"
# =============================================================================

log "Running KRK final evaluation..."
uv run python demos/experiments/batch_eval.py \
    --mode krk \
    --fen-file data/endgames/krk/random.fen \
    --runs 50 \
    --max-plies 100 \
    --engine "$ENGINE" --depth "$DEPTH" \
    --trace-out "$REPORTS_DIR/krk_AFTER.jsonl" \
    > "$REPORTS_DIR/krk_AFTER_stats.txt" 2>&1 || true

log "Running full-game vs random final (with trained weights)..."
AFTER_WINS=0
AFTER_DRAWS=0
AFTER_LOSSES=0
for i in $(seq 1 $FULLGAME_EVAL); do
    # Use --json-result for reliable parsing, and --weights to use trained consolidation pack
    result=$(uv run python demos/persistent/full_game_demo.py \
        --max-moves 200 --vs-random --quiet --json-result \
        --weights "$WEIGHTS_DIR/nightly/krk_consol.json" 2>&1 | grep -o '"outcome":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    case "$result" in
        "win") ((AFTER_WINS++)) ;;
        "loss") ((AFTER_LOSSES++)) ;;
        "draw") ((AFTER_DRAWS++)) ;;
        *)
            # Fallback: try parsing RESULT: format
            result_line=$(uv run python demos/persistent/full_game_demo.py \
                --max-moves 200 --vs-random --quiet \
                --weights "$WEIGHTS_DIR/nightly/krk_consol.json" 2>&1 | grep "RESULT:" | tail -1 || echo "")
            case "$result_line" in
                *"1-0"*) ((AFTER_WINS++)) ;;
                *"0-1"*) ((AFTER_LOSSES++)) ;;
                *"1/2-1/2"*) ((AFTER_DRAWS++)) ;;
            esac
            ;;
    esac
    log "  Game $i: $result"
done
AFTER_WIN_RATE=$(echo "scale=2; $AFTER_WINS * 100 / $FULLGAME_EVAL" | bc 2>/dev/null || echo "0")
log "Full-game AFTER: $AFTER_WINS wins, $AFTER_DRAWS draws, $AFTER_LOSSES losses ($AFTER_WIN_RATE% win rate)"
echo "wins: $AFTER_WINS, draws: $AFTER_DRAWS, losses: $AFTER_LOSSES, win_rate: $AFTER_WIN_RATE%" > "$REPORTS_DIR/fullgame_AFTER.txt"

# =============================================================================
log_header "📦 STEP 6: Saving Final Weights"
# =============================================================================

log "Saving final weight pack..."
cp "$WEIGHTS_DIR/nightly/krk_consol.json" "$BACKUP_DIR/krk_consol_FINAL.json"

# =============================================================================
log_header "📈 STEP 7: Generating Comparison Report"
# =============================================================================

REPORT_FILE="$REPORTS_DIR/overnight_summary.md"

cat > "$REPORT_FILE" << EOF
# 🌙 Overnight Training Report

**Generated:** $(date)
**Duration:** Started $TIMESTAMP

## 📦 Weight Packs for Comparison

| Pack | Path |
|------|------|
| **FIRST (before training)** | \`$BACKUP_DIR/krk_consol_FIRST.json\` |
| **FINAL (after training)** | \`$BACKUP_DIR/krk_consol_FINAL.json\` |
| **After KRK** | \`$BACKUP_DIR/krk_consol_after_KRK.json\` |

## 📊 Full-Game Win Rate vs Random

| Metric | BEFORE | AFTER | Change |
|--------|--------|-------|--------|
| **Wins** | $BEFORE_WINS | $AFTER_WINS | $(($AFTER_WINS - $BEFORE_WINS)) |
| **Draws** | $BEFORE_DRAWS | $AFTER_DRAWS | $(($AFTER_DRAWS - $BEFORE_DRAWS)) |
| **Losses** | $BEFORE_LOSSES | $AFTER_LOSSES | $(($AFTER_LOSSES - $BEFORE_LOSSES)) |
| **Win Rate** | ${BEFORE_WIN_RATE}% | ${AFTER_WIN_RATE}% | $(echo "$AFTER_WIN_RATE - $BEFORE_WIN_RATE" | bc)% |

## 🎓 Training Summary

- **KRK games:** $KRK_GAMES
- **KPK games:** $KPK_GAMES
- **Engine depth:** $DEPTH

## 🎯 Tactical Micro-Training (M8)

Trained tactical patterns from puzzle suites:
EOF

# Add tactical results to report
for tactic_result in "$REPORTS_DIR"/tactics_*.json; do
    if [ -f "$tactic_result" ]; then
        tactic_name=$(basename "$tactic_result" .json | sed 's/tactics_//')
        # Extract stats from JSON
        if command -v python3 &> /dev/null; then
            stats=$(python3 -c "import json; d=json.load(open('$tactic_result')); s=d.get('stats',{}); print(f\"- **{s.get('total',0)}** positions, **{s.get('accuracy',0)*100:.1f}%** accuracy\")" 2>/dev/null || echo "- Stats unavailable")
            echo "- \`$tactic_name\`: $stats" >> "$REPORT_FILE"
        fi
    fi
done

cat >> "$REPORT_FILE" << EOF

## 📁 Files Generated

- Traces: \`$REPORTS_DIR/\`
- Backups: \`$BACKUP_DIR/\`
- Logs: \`$LOGS_DIR/overnight_$TIMESTAMP.log\`

## 🔍 Compare Weight Changes

Run this command to see what changed:
\`\`\`bash
uv run python tools/pack_diff.py \\
  $BACKUP_DIR/krk_consol_FIRST.json \\
  $BACKUP_DIR/krk_consol_FINAL.json
\`\`\`

## 🎮 Replay Games with Different Packs

To compare gameplay:
\`\`\`bash
# With FIRST (before training)
uv run python demos/persistent/full_game_demo.py \\
  --vs-random --max-moves 200

# Manual comparison - load different consolidation packs
\`\`\`

EOF

log "Report saved to: $REPORT_FILE"

# Try to generate pack diff
log "Generating weight diff..."
if [ -f "$BACKUP_DIR/krk_consol_FIRST.json" ] && [ -f "$BACKUP_DIR/krk_consol_FINAL.json" ]; then
    uv run python tools/pack_diff.py \
        "$BACKUP_DIR/krk_consol_FIRST.json" \
        "$BACKUP_DIR/krk_consol_FINAL.json" \
        >> "$REPORT_FILE" 2>&1 || log "Pack diff failed (non-critical)"
fi

# Trace summary
log "Generating trace summaries..."
if [ -f "$REPORTS_DIR/krk_training.jsonl" ]; then
    uv run python tools/trace_summarize.py \
        "$REPORTS_DIR/krk_training.jsonl" \
        -o "$REPORTS_DIR/krk_trace_summary.json" 2>&1 || true
fi

# M8.4: Generate standardized nightly report
log "Generating standardized nightly report..."
uv run python tools/generate_nightly_report.py \
    --consol "$BACKUP_DIR/krk_consol_FINAL.json" \
    --traces "$REPORTS_DIR/krk_training.jsonl" \
    --first-consol "$BACKUP_DIR/krk_consol_FIRST.json" \
    --title "Overnight Training Report - $TIMESTAMP" \
    -o "$REPORTS_DIR/nightly_report.md" 2>&1 || true

# =============================================================================
log_header "✅ OVERNIGHT TRAINING COMPLETE"
# =============================================================================

log ""
log "Results:"
log "  Full-game win rate: ${BEFORE_WIN_RATE}% → ${AFTER_WIN_RATE}%"
log ""
log "Files saved:"
log "  Report: $REPORT_FILE"
log "  Weights FIRST: $BACKUP_DIR/krk_consol_FIRST.json"
log "  Weights FINAL: $BACKUP_DIR/krk_consol_FINAL.json"
log "  Full log: $LOGS_DIR/overnight_$TIMESTAMP.log"
log ""
log "🌅 Training finished at $(date)"

# Show the summary report
cat "$REPORT_FILE"