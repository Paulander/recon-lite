#!/usr/bin/env python3
"""
KRK Chess Demo (ReCoN-driven)

What this does
--------------
- Builds a fresh KRK ReCoN graph each white move.
- Uses Phase-0 (establish cut + king/rook rendezvous), then Phase-1..4.
- The engine runs until a terminal sets env["chosen_move"].
- Logs per-move features to help diagnose stalls/oscillation.
- Can run a single interactive game or a batch of random KRK starts.

Usage
-----
Single game (interactive-ish prints):
    python krk_play_demo.py

Batch (10 games, summary stats):
    python krk_play_demo.py --batch 10
"""

import argparse
import random
import chess

from recon_lite.graph import Graph, LinkType, NodeState
from recon_lite.engine import ReConEngine
from recon_lite.logger import RunLogger

# KRK nodes + helpers
from recon_lite_chess import (
    create_krk_root,
    create_phase0_establish_cut, create_phase0_choose_moves,
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
)
from recon_lite_chess.krk_nodes import wire_default_krk
from recon_lite_chess.predicates import move_features, box_area


# -------- graph building --------

def build_krk_graph() -> Graph:
    """
    Build a KRK graph with Phase-0..4, and POR sequencing Phase0→1→2→3→4.
    Includes a Phase-0 terminal that must set env["chosen_move"].
    """
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


# -------- opponent --------

def opponent_random_move(board: chess.Board) -> str | None:
    """Black plays a random legal move."""
    moves = list(board.legal_moves)
    if not moves:
        return None
    mv = random.choice(moves)
    return mv.uci()


# -------- engine stepper --------

def choose_move_with_graph(board: chess.Board, logger: RunLogger, move_no: int,
                           max_ticks: int = 200) -> str | None:
    """
    Build a fresh KRK graph, request ROOT, tick until env["chosen_move"] is set
    or until max_ticks. Returns UCI string or None.
    """
    env = {"board": board, "chosen_move": None}

    g = build_krk_graph()
    eng = ReConEngine(g)
    g.nodes["ROOT"].state = NodeState.REQUESTED

    ticks = 0
    while ticks < max_ticks and env.get("chosen_move") is None:
        ticks += 1
        now_req = eng.step(env)

        # snapshot periodically to inspect why the graph might be idle
        if ticks % 10 == 0:
            logger.snapshot(
                engine=eng,
                note=f"Eval tick {ticks} (move {move_no})",
                env={"fen": board.fen(), "move_number": move_no, "evaluation_tick": ticks},
                thoughts=f"Phase sequencing with Phase-0 first. Waiting for terminal to set env['chosen_move'].",
                new_requests=list(now_req.keys()) if now_req else []
            )

    return env.get("chosen_move")


# -------- single game loop --------

def play_single_game(initial_fen: str | None = None, max_plies: int = 200) -> dict:
    """
    Plays a single KRK game: White (ReCoN) vs Black (random).
    Returns summary dict with details for later aggregation.
    """
    logger = RunLogger()
    board = chess.Board(initial_fen) if initial_fen else random_krk_board(white_to_move=True)

    stalls = 0
    rook_lost = False
    ply = 0
    mate = False

    print("🎮 KRK Demo — ReCoN (White) vs Random (Black)")
    print(board, "\n")

    while not board.is_game_over() and ply < max_plies:
        ply += 1

        # ---- White / ReCoN ----
        chosen_uci = choose_move_with_graph(board, logger, ply)

        if not chosen_uci:
            stalls += 1
            print(f"❌ Stall on ply {ply}: no chosen move, aborting.")
            break

        # feature logging BEFORE push
        mv = chess.Move.from_uci(chosen_uci)
        feats = move_features(board, mv)
        prev_area = box_area(board)

        try:
            board.push(mv)
        except Exception as e:
            print(f"❌ Illegal chosen move {chosen_uci}: {e}")
            stalls += 1
            break

        # rook loss check
        if not any(p.piece_type == chess.ROOK and p.color == chess.WHITE for p in board.piece_map().values()):
            rook_lost = True

        print(f"♔ White (ReCoN) plays: {chosen_uci}")
        print(board)
        print()

        logger.snapshot(
            engine=None,  # not strictly needed here
            note=f"ReCoN ply {ply}: {chosen_uci}",
            env={"fen": board.fen(), "ply": ply, "chosen_move": chosen_uci, "recons_move": chosen_uci,
                 "features": feats, "box_area_delta": feats["box_area_after"] - prev_area},
            thoughts=f"Chose {chosen_uci} | Δbox={feats['box_area_after']-prev_area} | king_progress={feats['king_progress']} | safe_check={feats['gives_safe_check']} | rook_safe={feats['rook_safe_after']}",
            new_requests=[]
        )

        if board.is_game_over():
            break

        # ---- Black / Random ----
        opp_uci = opponent_random_move(board)
        if not opp_uci:
            break

        board.push_uci(opp_uci)
        print(f"♚ Black (Random) plays: {opp_uci}")
        print(board)
        print()

        logger.snapshot(
            engine=None,
            note=f"Opponent ply {ply}: {opp_uci}",
            env={"fen": board.fen(), "ply": ply, "opponent_move": opp_uci, "opponents_move": opp_uci},
            thoughts="Random defense.",
            new_requests=[]
        )

    # Final outcome
    if board.is_checkmate():
        mate = True

    result = {
        "stalled": stalls > 0,
        "stall_count": stalls,
        "rook_lost": rook_lost,
        "checkmate": mate,
        "plies": ply,
        "final_fen": board.fen(),
    }

    # Save a compact log that your visualizer can read
    logger.to_json("demos/krk_visualization_data.json")
    print("\n💾 Log saved to demos/krk_visualization_data.json")
    print(f"🏁 Result: mate={mate}, stalls={stalls}, rook_lost={rook_lost}, plies={ply}")
    return result


# -------- batch mode --------

def random_krk_board(white_to_move: bool = True) -> chess.Board:
    """Create a random, legal KRK position with White to move by default."""
    b = chess.Board(None)
    squares = list(chess.SQUARES)
    random.shuffle(squares)

    wk = squares.pop()
    bk = squares.pop()
    # kings must not touch
    while chess.square_distance(wk, bk) <= 1:
        bk = squares.pop()

    r = squares.pop()

    b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    b.set_piece_at(r, chess.Piece(chess.ROOK, chess.WHITE))
    b.turn = chess.WHITE if white_to_move else chess.BLACK

    # If illegal (e.g. side to move in check illegally), reshuffle
    if b.is_valid():
        return b
    return random_krk_board(white_to_move)


def run_batch(n_games: int = 10) -> dict:
    stats = {
        "games": [],
        "stalls": 0,
        "rook_losses": 0,
        "mates": 0,
        "total_mate_plies": 0,
        "avg_mate_length": None,
    }
    for i in range(n_games):
        print(f"\n===== Game {i+1}/{n_games} =====")
        res = play_single_game(initial_fen=None, max_plies=200)
        stats["games"].append(res)
        if res["stalled"]:
            stats["stalls"] += 1
        if res["rook_lost"]:
            stats["rook_losses"] += 1
        if res["checkmate"]:
            stats["mates"] += 1
            stats["total_mate_plies"] += res["plies"]

    if stats["mates"]:
        stats["avg_mate_length"] = stats["total_mate_plies"] / stats["mates"]
    print("\n===== Batch Summary =====")
    print(stats)
    return stats


# -------- main --------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0, help="Run N random KRK games and print summary.")
    parser.add_argument("--fen", type=str, default="", help="Optional FEN to start from instead of random KRK.")
    args = parser.parse_args()

    if args.batch and args.batch > 0:
        run_batch(args.batch)
    else:
        start_fen = args.fen if args.fen else "4k3/6K1/8/8/8/8/R7/8 w - - 0 1"
        play_single_game(initial_fen=start_fen, max_plies=200)


if __name__ == "__main__":
    main()
