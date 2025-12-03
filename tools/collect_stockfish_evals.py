#!/usr/bin/env python3
"""Collect Stockfish evaluations for distillation.

This tool annotates chess positions with Stockfish evaluations
to create a training dataset for the distilled evaluator.

Usage:
    uv run python tools/collect_stockfish_evals.py --traces reports/*.jsonl --out data/distillation/evals.jsonl
    uv run python tools/collect_stockfish_evals.py --pgn games.pgn --out data/distillation/evals.jsonl
    uv run python tools/collect_stockfish_evals.py --fens positions.fen --out data/distillation/evals.jsonl
"""

import argparse
import json
import sys
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass, asdict

import chess
import chess.engine


@dataclass
class EvalSample:
    """A single evaluation sample."""
    fen: str
    stockfish_eval: float  # In centipawns, capped at ±1500
    best_move: str         # UCI notation
    depth: int
    phase: str             # "opening", "middlegame", "endgame"
    material_balance: int  # White material - Black material
    
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_phase(board: chess.Board) -> str:
    """Estimate game phase from material count."""
    material = 0
    for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        material += len(board.pieces(pt, chess.WHITE))
        material += len(board.pieces(pt, chess.BLACK))
    
    if board.fullmove_number <= 10 and material >= 14:
        return "opening"
    elif material <= 6:
        return "endgame"
    return "middlegame"


def count_material(board: chess.Board) -> int:
    """Count material balance (positive = White ahead)."""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    white = sum(len(board.pieces(pt, chess.WHITE)) * v for pt, v in values.items())
    black = sum(len(board.pieces(pt, chess.BLACK)) * v for pt, v in values.items())
    return white - black


def eval_position_stockfish(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int = 15,
) -> Optional[EvalSample]:
    """Evaluate a single position with Stockfish."""
    try:
        result = engine.analyse(board, chess.engine.Limit(depth=depth))
        
        score = result.get("score")
        if score is None:
            return None
        
        # Convert to centipawns
        pov_score = score.white()
        if pov_score.is_mate():
            mate_in = pov_score.mate()
            if mate_in is not None:
                cp = 10000 if mate_in > 0 else -10000
            else:
                cp = 0
        else:
            cp = pov_score.score() or 0
        
        # Cap at ±1500 cp
        cp = max(-1500, min(1500, cp))
        
        # Get best move
        pv = result.get("pv", [])
        best_move = pv[0].uci() if pv else ""
        
        return EvalSample(
            fen=board.fen(),
            stockfish_eval=cp / 100.0,  # Convert to pawns
            best_move=best_move,
            depth=depth,
            phase=estimate_phase(board),
            material_balance=count_material(board),
        )
    except Exception as e:
        print(f"Error evaluating position: {e}")
        return None


def positions_from_traces(trace_paths: List[Path]) -> Iterator[chess.Board]:
    """Extract positions from trace JSONL files."""
    for path in trace_paths:
        if not path.exists():
            continue
        
        with open(path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    fen = data.get("fen") or data.get("position", {}).get("fen")
                    if fen:
                        board = chess.Board(fen)
                        yield board
                except (json.JSONDecodeError, ValueError):
                    continue


def positions_from_pgn(pgn_path: Path, sample_every: int = 5) -> Iterator[chess.Board]:
    """Extract positions from a PGN file, sampling every N moves."""
    import chess.pgn
    
    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            board = game.board()
            move_count = 0
            
            for move in game.mainline_moves():
                board.push(move)
                move_count += 1
                
                if move_count % sample_every == 0:
                    yield board.copy()


def positions_from_fens(fen_path: Path) -> Iterator[chess.Board]:
    """Read positions from a FEN file (one per line)."""
    with open(fen_path) as f:
        for line in f:
            fen = line.strip()
            if fen and not fen.startswith("#"):
                try:
                    board = chess.Board(fen)
                    yield board
                except ValueError:
                    continue


def generate_random_positions(count: int, min_moves: int = 5, max_moves: int = 60) -> Iterator[chess.Board]:
    """Generate random positions by playing random games."""
    for _ in range(count):
        board = chess.Board()
        num_moves = random.randint(min_moves, max_moves)
        
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            
            # Slight bias toward captures and checks for interesting positions
            captures = [m for m in legal_moves if board.is_capture(m)]
            checks = [m for m in legal_moves if board.gives_check(m)]
            
            if captures and random.random() < 0.3:
                move = random.choice(captures)
            elif checks and random.random() < 0.2:
                move = random.choice(checks)
            else:
                move = random.choice(legal_moves)
            
            board.push(move)
        
        if not board.is_game_over():
            yield board


def main():
    parser = argparse.ArgumentParser(description="Collect Stockfish evaluations")
    parser.add_argument("--traces", nargs="*", help="Trace JSONL files")
    parser.add_argument("--pgn", help="PGN file with games")
    parser.add_argument("--fens", help="FEN file with positions")
    parser.add_argument("--random", type=int, default=0, help="Generate random positions")
    parser.add_argument("--out", required=True, help="Output JSONL file")
    parser.add_argument("--depth", type=int, default=15, help="Stockfish search depth")
    parser.add_argument("--stockfish", default="stockfish", help="Path to Stockfish")
    parser.add_argument("--max-positions", type=int, default=10000, help="Max positions to evaluate")
    parser.add_argument("--balance-phases", action="store_true", help="Balance across phases")
    args = parser.parse_args()
    
    # Collect positions
    positions: List[chess.Board] = []
    
    if args.traces:
        trace_paths = [Path(p) for p in args.traces]
        positions.extend(positions_from_traces(trace_paths))
        print(f"Loaded {len(positions)} positions from traces")
    
    if args.pgn:
        pgn_pos = list(positions_from_pgn(Path(args.pgn)))
        positions.extend(pgn_pos)
        print(f"Loaded {len(pgn_pos)} positions from PGN")
    
    if args.fens:
        fen_pos = list(positions_from_fens(Path(args.fens)))
        positions.extend(fen_pos)
        print(f"Loaded {len(fen_pos)} positions from FENs")
    
    if args.random > 0:
        random_pos = list(generate_random_positions(args.random))
        positions.extend(random_pos)
        print(f"Generated {len(random_pos)} random positions")
    
    if not positions:
        print("No positions to evaluate! Use --traces, --pgn, --fens, or --random")
        sys.exit(1)
    
    # Deduplicate by FEN
    seen_fens = set()
    unique_positions = []
    for board in positions:
        fen = board.fen()
        if fen not in seen_fens:
            seen_fens.add(fen)
            unique_positions.append(board)
    
    print(f"Total unique positions: {len(unique_positions)}")
    
    # Sample if too many
    if len(unique_positions) > args.max_positions:
        if args.balance_phases:
            # Balance across phases
            by_phase = {"opening": [], "middlegame": [], "endgame": []}
            for board in unique_positions:
                phase = estimate_phase(board)
                by_phase[phase].append(board)
            
            per_phase = args.max_positions // 3
            sampled = []
            for phase, boards in by_phase.items():
                sampled.extend(random.sample(boards, min(per_phase, len(boards))))
            unique_positions = sampled
        else:
            unique_positions = random.sample(unique_positions, args.max_positions)
        
        print(f"Sampled down to {len(unique_positions)} positions")
    
    # Initialize Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    except Exception as e:
        print(f"Failed to start Stockfish at '{args.stockfish}': {e}")
        print("Install Stockfish or specify path with --stockfish")
        sys.exit(1)
    
    # Evaluate positions
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    try:
        for i, board in enumerate(unique_positions):
            if (i + 1) % 100 == 0:
                print(f"Evaluating position {i + 1}/{len(unique_positions)}...")
            
            sample = eval_position_stockfish(board, engine, depth=args.depth)
            if sample:
                samples.append(sample)
    finally:
        engine.quit()
    
    # Write output
    with open(out_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.as_dict()) + "\n")
    
    print(f"\nCollected {len(samples)} evaluations")
    print(f"Saved to {out_path}")
    
    # Stats
    phases = {}
    for s in samples:
        phases[s.phase] = phases.get(s.phase, 0) + 1
    print(f"Phase distribution: {phases}")


if __name__ == "__main__":
    main()

