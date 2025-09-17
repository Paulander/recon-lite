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
from recon_lite.graph import NodeState, Graph
from demos.shared.krk_network import build_krk_network, create_random_krk_board
from recon_lite_chess import (
    create_krk_root,
    create_phase0_establish_cut, create_phase0_choose_moves,
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
)
from recon_lite_chess.krk_nodes import wire_default_krk


def _build_basic_krk_graph() -> Graph:
    g = Graph()
    root  = create_krk_root("ROOT")
    p0    = create_phase0_establish_cut("PHASE0")
    ch0   = create_phase0_choose_moves("CHOOSE_P0")
    p1    = create_phase1_drive_to_edge("PHASE1")
    p2    = create_phase2_shrink_box("PHASE2")
    p3    = create_phase3_take_opposition("PHASE3")
    p4    = create_phase4_deliver_mate("PHASE4")
    for n in [root, p0, ch0, p1, p2, p3, p4]:
        g.add_node(n)
    wire_default_krk(g, "ROOT", {
        "root": "ROOT",
        "phase0": "PHASE0",
        "choose_p0": "CHOOSE_P0",
        "phase1": "PHASE1",
        "phase2": "PHASE2",
        "phase3": "PHASE3",
        "phase4": "PHASE4",
    })
    return g


def play_persistent_game(initial_fen: str | None = None, max_plies: int = 200,
                         tick_watchdog: int = 300, graph: str = "shared") -> dict:
    logger = RunLogger()
    if initial_fen:
        board = chess.Board(initial_fen)
    else:
        # create_random_krk_board returns a FEN string; wrap into a Board
        board = chess.Board(create_random_krk_board(white_to_move=True))

    g = build_krk_network() if graph == "shared" else _build_basic_krk_graph()
    engine = ReConEngine(g)
    root_id = "krk_root" if graph == "shared" else "ROOT"
    g.nodes[root_id].state = NodeState.REQUESTED

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

            # Reset states so the network re-evaluates this new position next cycle
            for n in g.nodes.values():
                n.state = NodeState.INACTIVE
            # Re-arm wait gate to detect new FEN (if present)
            if graph == "shared" and "wait_for_board_change" in g.nodes:
                g.nodes["wait_for_board_change"].meta.pop("last_fen", None)
            g.nodes[root_id].state = NodeState.REQUESTED
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


def run_batch(n_games: int = 10, graph: str = "shared", max_plies: int = 200) -> dict:
    stats = {
        "games": [],
        "mates": 0,
        "stalls": 0,
        "rook_losses": 0,
        "total_mate_plies": 0,
        "avg_mate_length": None,
    }
    for i in range(n_games):
        res = play_persistent_game(initial_fen=None, max_plies=max_plies, graph=graph)
        stats["games"].append(res)
        if res.get("checkmate"):
            stats["mates"] += 1
            stats["total_mate_plies"] += res.get("plies", 0)
        if res.get("rook_lost"):
            stats["rook_losses"] += 1
        # No explicit stall flag in persistent; watchdog fallback is logged only
    if stats["mates"]:
        stats["avg_mate_length"] = stats["total_mate_plies"]/stats["mates"]
    print(stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", type=str, default="", help="Optional FEN to start from")
    parser.add_argument("--max-plies", type=int, default=200, help="Maximum plies")
    parser.add_argument("--batch", type=int, default=0, help="Run N games in batch mode")
    parser.add_argument("--graph", type=str, choices=["shared","basic"], default="shared",
                        help="Graph to use: shared (default) or basic (top-level wiring)")
    args = parser.parse_args()

    if args.batch and args.batch > 0:
        run_batch(args.batch, graph=args.graph, max_plies=args.max_plies)
    else:
        start_fen = args.fen if args.fen else "4k3/6K1/8/8/8/8/R7/8 w - - 0 1"
        res = play_persistent_game(initial_fen=start_fen, max_plies=args.max_plies, graph=args.graph)
        print(res)


if __name__ == "__main__":
    main()
