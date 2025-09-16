#!/usr/bin/env python3
"""
Persistent KRK Chess Demo (ReCoN-driven)

Runs a single ReCoN engine instance across the whole game.
- Logs per-tick frames (network states, requests) for visualization
- Applies moves as actuators set env["chosen_move"]
- After applying a move, lets the opponent respond, then re-REQUESTS ROOT
- Outputs visualization JSON to demos/outputs/krk_persistent_visualization.json
"""

import argparse
import chess
import sys
from pathlib import Path
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite.engine import ReConEngine
from recon_lite.logger import RunLogger
from recon_lite.graph import NodeState
from demos.shared.krk_network import build_krk_network, create_random_krk_board


def play_persistent_game(initial_fen: str | None = None, max_plies: int = 200,
                         tick_watchdog: int = 300) -> dict:
    logger = RunLogger()
    board = chess.Board(initial_fen) if initial_fen else create_random_krk_board(white_to_move=True)

    g = build_krk_network()
    engine = ReConEngine(g)
    g.nodes["krk_root"].state = NodeState.REQUESTED

    # Attach graph edges for visualization
    graph_edges = [{"src": e.src, "dst": e.dst, "type": e.ltype.name} for e in g.edges]

    plies = 0
    rook_lost = False

    while not board.is_game_over() and plies < max_plies:
        # One decision cycle (White/ReCoN)
        env = {"board": board, "chosen_move": None}
        ticks = 0

        while ticks < tick_watchdog and env.get("chosen_move") is None and not board.is_game_over():
            ticks += 1
            now_req = engine.step(env)
            logger.snapshot(
                engine=engine,
                note=f"Persistent eval tick {ticks} (ply {plies+1})",
                env={"fen": board.fen(), "evaluation_tick": ticks, "ply": plies+1},
                thoughts="Persistent evaluation...",
                new_requests=list(now_req.keys()) if now_req else [],
            )
            if ticks == 1:
                logger.events[-1]["graph"] = {"edges": graph_edges}

        move_uci = env.get("chosen_move")
        if not move_uci:
            # Watchdog: pick a safe fallback to avoid failing fast
            from recon_lite_chess.actuators import choose_any_safe_move
            fallback = choose_any_safe_move(board)
            if fallback:
                move_uci = fallback
                logger.snapshot(
                    engine=None,
                    note=f"WATCHDOG fallback: {fallback}",
                    env={"fen": board.fen(), "ply": plies+1},
                    thoughts="No chosen_move by tick limit; applying fallback",
                    new_requests=[],
                )

        if move_uci:
            try:
                board.push_uci(move_uci)
            except Exception:
                break
            plies += 1

            if not any(p.piece_type == chess.ROOK and p.color == chess.WHITE for p in board.piece_map().values()):
                rook_lost = True

            logger.snapshot(
                engine=None,
                note=f"Applied move {plies}: {move_uci}",
                env={"fen": board.fen(), "ply": plies, "recons_move": move_uci},
                thoughts=f"Applied {move_uci} (persistent)",
                new_requests=[],
            )

            if board.is_game_over() or plies >= max_plies:
                break

            # Opponent plays immediately (random for now)
            opp_moves = list(board.legal_moves)
            if opp_moves:
                opp_uci = random.choice(opp_moves).uci()
                board.push_uci(opp_uci)
                logger.snapshot(
                    engine=None,
                    note=f"Opponent ply {plies}: {opp_uci}",
                    env={"fen": board.fen(), "ply": plies, "opponents_move": opp_uci},
                    thoughts="Random defense (persistent)",
                    new_requests=[],
                )
            if board.is_game_over() or plies >= max_plies:
                break

            # Kick off next decision cycle without resetting graph
            g.nodes["krk_root"].state = NodeState.REQUESTED
        else:
            # No move selected by watchdog/time—stop to avoid infinite loop
            break

    result = {
        "plies": plies,
        "checkmate": board.is_checkmate(),
        "stalemate": board.is_stalemate(),
        "rook_lost": rook_lost,
        "final_fen": board.fen(),
    }

    out_path = Path("demos/outputs/krk_persistent_visualization.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.to_json(str(out_path))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", type=str, default="", help="Optional FEN to start from")
    parser.add_argument("--max-plies", type=int, default=200, help="Maximum plies")
    args = parser.parse_args()

    start_fen = args.fen if args.fen else "4k3/6K1/8/8/8/8/R7/8 w - - 0 1"
    res = play_persistent_game(initial_fen=start_fen, max_plies=args.max_plies)
    print(res)


if __name__ == "__main__":
    main()
