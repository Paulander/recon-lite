#!/bin/bash
# Full training cycle: run games, consolidate, generate report
# Usage: ./scripts/training_cycle.sh [--games N] [--engine PATH]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

GAMES=20
ENGINE=""
CONSOL_PACK="$PROJECT_ROOT/weights/nightly/krk_consol.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --games) GAMES="$2"; shift 2 ;;
        --engine) ENGINE="--engine $2"; shift 2 ;;
        --pack) CONSOL_PACK="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_ROOT"

echo "=== Training Cycle: $TIMESTAMP ==="
echo "Games: $GAMES"
echo "Consolidation pack: $CONSOL_PACK"
echo ""

# Step 1: Run training games with plasticity and consolidation
echo "📊 Step 1: Running $GAMES training games..."
uv run python demos/persistent/krk_persistent_demo.py \
    --batch "$GAMES" \
    --plasticity \
    --bandit \
    --consolidate \
    --consolidate-pack "$CONSOL_PACK" \
    --output-basename "krk_cycle_${TIMESTAMP}" \
    $ENGINE

# Step 2: Generate consolidation report
echo ""
echo "📈 Step 2: Generating consolidation report..."
REPORT_FILE="$PROJECT_ROOT/reports/nightly/cycle_${TIMESTAMP}.md"
uv run python tools/report_consolidation.py "$CONSOL_PACK" -o "$REPORT_FILE"
echo "  Report saved to: $REPORT_FILE"

# Step 3: Summary
echo ""
echo "=== Cycle Complete ==="
echo "Consolidation state: $CONSOL_PACK"
echo "Report: $REPORT_FILE"
echo "Visualization data: demos/outputs/persistent/krk_cycle_${TIMESTAMP}_viz.json"
