#!/bin/bash
# Setup script for ReCoN-lite nightly training runs
# Run this once to prepare the environment for automated training

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== ReCoN-lite Nightly Setup ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Create required directories
echo "📁 Creating directories..."
mkdir -p "$PROJECT_ROOT/reports/nightly/krk_fast"
mkdir -p "$PROJECT_ROOT/reports/nightly/krk_sf"
mkdir -p "$PROJECT_ROOT/reports/nightly/kpk"
mkdir -p "$PROJECT_ROOT/weights/nightly"
mkdir -p "$PROJECT_ROOT/weights/versions"
mkdir -p "$PROJECT_ROOT/logs"

# Check for sample FEN files
echo ""
echo "📄 Checking FEN data..."
if [ ! -d "$PROJECT_ROOT/data/endgames" ]; then
    echo "  ⚠️  data/endgames/ not found. Creating with sample positions..."
    mkdir -p "$PROJECT_ROOT/data/endgames/krk"
    mkdir -p "$PROJECT_ROOT/data/endgames/kpk"
    
    # Sample KRK positions
    cat > "$PROJECT_ROOT/data/endgames/krk/random.fen" << 'EOF'
# KRK sample positions for nightly training
4k3/6K1/8/8/8/8/R7/8 w - - 0 1
4k3/8/8/8/8/5K2/R7/8 w - - 0 1
8/8/8/4k3/8/8/R7/4K3 w - - 0 1
8/3k4/8/8/8/8/R7/4K3 w - - 0 1
4k3/8/8/8/8/8/7R/4K3 w - - 0 1
8/8/2k5/8/8/8/R7/4K3 w - - 0 1
3k4/8/8/8/8/4K3/8/R7 w - - 0 1
8/8/8/8/2k5/8/R7/4K3 w - - 0 1
EOF
    echo "  ✅ Created sample KRK positions"
    
    # Sample KPK positions
    cat > "$PROJECT_ROOT/data/endgames/kpk/sample.fen" << 'EOF'
# KPK sample positions for nightly training
8/8/8/4k3/8/8/4P3/4K3 w - - 0 1
8/8/2k5/8/8/6K1/4P3/8 w - - 0 1
8/8/8/8/4k3/8/4P3/4K3 w - - 0 1
8/8/4k3/8/4P3/8/8/4K3 w - - 0 1
EOF
    echo "  ✅ Created sample KPK positions"
else
    KRK_COUNT=$(find "$PROJECT_ROOT/data/endgames/krk" -name "*.fen" 2>/dev/null | wc -l)
    KPK_COUNT=$(find "$PROJECT_ROOT/data/endgames/kpk" -name "*.fen" 2>/dev/null | wc -l)
    echo "  ✅ Found $KRK_COUNT KRK FEN file(s), $KPK_COUNT KPK FEN file(s)"
fi

# Check Stockfish
echo ""
echo "🐟 Checking Stockfish..."
if command -v stockfish &> /dev/null; then
    echo "  ✅ stockfish found at: $(which stockfish)"
elif [ -x "/usr/games/stockfish" ]; then
    echo "  ✅ stockfish found at: /usr/games/stockfish"
else
    echo "  ⚠️  stockfish not found. Install with: sudo apt install stockfish"
    echo "     Training will use heuristic eval only (faster but less accurate)"
fi

# Create example crontab entry
echo ""
echo "📅 Crontab example (for automated nightly runs):"
echo ""
echo "  # Run KRK training every night at 2am"
echo "  0 2 * * * cd $PROJECT_ROOT && uv run python demos/experiments/nightly_runner.py --config configs/nightly/krk_stockfish.json >> logs/nightly.log 2>&1"
echo ""
echo "  To install, run: crontab -e"
echo ""

# Create run_nightly.sh convenience script
cat > "$PROJECT_ROOT/scripts/run_nightly.sh" << 'SCRIPT'
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
SCRIPT
chmod +x "$PROJECT_ROOT/scripts/run_nightly.sh"
echo "  ✅ Created scripts/run_nightly.sh"

# Create training_cycle.sh for full training loop
cat > "$PROJECT_ROOT/scripts/training_cycle.sh" << 'SCRIPT'
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
SCRIPT
chmod +x "$PROJECT_ROOT/scripts/training_cycle.sh"
echo "  ✅ Created scripts/training_cycle.sh"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick Start Commands:"
echo ""
echo "  # Quick training run (no Stockfish, ~2 min)"
echo "  ./scripts/run_nightly.sh krk_fast"
echo ""
echo "  # Full training cycle with consolidation"
echo "  ./scripts/training_cycle.sh --games 20"
echo ""
echo "  # With Stockfish evaluation"
echo "  ./scripts/training_cycle.sh --games 20 --engine /usr/games/stockfish"
echo ""
echo "  # View consolidation dashboard"
echo "  # Open demos/visualization/consolidation_dashboard.html in browser"
echo "  # Load weights/nightly/krk_consol.json"
echo ""

