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
import sys
import pickle
import hashlib
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite.graph import Graph, LinkType, NodeState, NodeType
from recon_lite.engine import ReConEngine
from recon_lite.logger import RunLogger
from recon_lite.binding.manager import BindingInstance, BindingTable

# KRK nodes + helpers
from recon_lite_chess import (
    create_krk_root,
    create_phase0_establish_cut, create_phase0_choose_moves,
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
    create_king_drive_moves, create_box_shrink_moves,
    create_opposition_moves, create_mate_moves,
)
from recon_lite_chess.krk_nodes import wire_default_krk
from recon_lite_chess.predicates import move_features, box_area, dist_to_edge, enemy_nearest_edge_info
from recon_lite_chess.baseline_teacher import KRKTeacher
from recon_lite.learning.baseline import BaselineLearner, apply_sensor
from recon_lite_chess.graph.builder import build_graph_from_topology


# -------- graph building --------

def _graph_edges_payload(graph: Graph) -> list[dict]:
    edges = []
    for e in graph.edges:
        edges.append({
            "src": e.src,
            "dst": e.dst,
            "type": e.ltype.name,
            "weight": float(getattr(e, "w", 1.0)) if getattr(e, "w", None) is not None else 1.0,
        })
    return edges


def _find_our_rook_sq(board: chess.Board) -> chess.Square | None:
    color = board.turn
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type == chess.ROOK:
            return sq
    return None


def _square_token(square: chess.Square | None) -> str | None:
    if square is None:
        return None
    return f"square:{chess.square_name(square)}"


def _line_tokens(axis: str, index: int) -> list[str]:
    tokens: list[str] = []
    if axis == "file":
        for rank in range(8):
            tokens.append(_square_token(chess.square(index, rank)))
    else:
        for file in range(8):
            tokens.append(_square_token(chess.square(file, index)))
    return [tok for tok in tokens if tok is not None]


def _box_corner_tokens(enemy_sq: chess.Square | None) -> list[str]:
    if enemy_sq is None:
        return []
    ef, er = chess.square_file(enemy_sq), chess.square_rank(enemy_sq)
    distance = dist_to_edge(enemy_sq)
    min_file = max(0, ef - distance)
    max_file = min(7, ef + distance)
    min_rank = max(0, er - distance)
    max_rank = min(7, er + distance)
    corners = [
        chess.square(min_file, min_rank),
        chess.square(min_file, max_rank),
        chess.square(max_file, min_rank),
        chess.square(max_file, max_rank),
    ]
    return [token for token in (_square_token(sq) for sq in corners) if token is not None]


def _update_binding_table(table: BindingTable, board: chess.Board) -> dict:
    table.invalidate_on_board_change(board)
    color = board.turn
    our_king = board.king(color)
    enemy_king = board.king(not color)
    rook_sq = _find_our_rook_sq(board)

    with table.begin_tick("krk/core/kings") as session:
        if our_king is not None:
            session.reserve(BindingInstance("our_king", {_square_token(our_king)}))
        if enemy_king is not None:
            session.reserve(BindingInstance("enemy_king", {_square_token(enemy_king)}))

    with table.begin_tick("krk/p1/drive") as session:
        if rook_sq is not None:
            session.reserve(BindingInstance("rook_anchor", {_square_token(rook_sq)}))
        if enemy_king is not None:
            session.reserve(BindingInstance("target_enemy", {_square_token(enemy_king)}))

    with table.begin_tick("krk/p2/shrink") as session:
        if enemy_king is not None:
            try:
                fence = enemy_nearest_edge_info(board, enemy_king)
                line_tokens = _line_tokens(fence["axis"], fence["target_line"])
                if line_tokens:
                    session.reserve(BindingInstance("target_fence", set(line_tokens)))
            except Exception:
                pass
            corner_tokens = _box_corner_tokens(enemy_king)
            if corner_tokens:
                session.reserve(BindingInstance("box_corners", set(corner_tokens)))

    return table.snapshot()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_learner(learner_path: Path | None) -> tuple[BaselineLearner | None, dict]:
    if not learner_path:
        return None, {}
    learner_path = Path(learner_path)
    if not learner_path.exists():
        return None, {}
    with learner_path.open("rb") as f:
        learner = pickle.load(f)
    info = {
        "learner_path": str(learner_path),
        "learner_hash": _hash_file(learner_path),
    }
    return learner, info


def _goal_prototypes(learner: BaselineLearner | None, label: str = "mate_in_1") -> list:
    if not learner:
        return []
    return [g.s0 for g in learner.goal_memories if g.label == label]


def _state_vector(learner: BaselineLearner, features) -> list[float]:
    sensors = learner.get_mature_sensors()
    return [apply_sensor(s, features) for s in sensors]


def _goal_distance(learner: BaselineLearner, prototypes: list, features) -> float | None:
    if not prototypes:
        return None
    s = _state_vector(learner, features)
    if len(s) == 0:
        return None
    s = np.asarray(s, dtype=np.float32)
    if getattr(learner, "normalize_goals", True):
        s = s / (np.linalg.norm(s) + 1e-6)
    best = None
    for proto in prototypes:
        g = np.asarray(proto, dtype=np.float32)
        if getattr(learner, "normalize_goals", True):
            g = g / (np.linalg.norm(g) + 1e-6)
        dist = float(np.linalg.norm(s - g))
        if best is None or dist < best:
            best = dist
    return best


_diag_teacher = KRKTeacher()


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

    # Add move generator terminals for phases 1..4
    m1 = create_king_drive_moves("KING_DRIVE_MOVES")
    m2 = create_box_shrink_moves("BOX_SHRINK_MOVES")
    m3 = create_opposition_moves("OPPOSITION_MOVES")
    m4 = create_mate_moves("MATE_MOVES")

    for n in [root, p0, ch0, p1, p2, p3, p4, m1, m2, m3, m4]:
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

    # Wire move generators under their phases
    g.add_edge("PHASE1", "KING_DRIVE_MOVES", LinkType.SUB)
    g.add_edge("PHASE2", "BOX_SHRINK_MOVES", LinkType.SUB)
    g.add_edge("PHASE3", "OPPOSITION_MOVES", LinkType.SUB)
    g.add_edge("PHASE4", "MATE_MOVES", LinkType.SUB)

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
                           max_ticks: int = 200,
                           log_every_tick: bool = False,
                           linger_ticks_after_choice: int = 0,
                           topology_path: str | None = None,
                           binding_table: BindingTable | None = None,
                           learner: BaselineLearner | None = None,
                           learner_info: dict | None = None) -> str | None:
    env = {"board": board, "chosen_move": None}
    if binding_table is None:
        binding_table = BindingTable()
    env["binding"] = _update_binding_table(binding_table, board)

    g = build_graph_from_topology(topology_path) if topology_path else build_krk_graph()
    eng = ReConEngine(g)
    root_id = None
    if "ROOT" in g.nodes:
        root_id = "ROOT"
    elif "krk_entry" in g.nodes:
        root_id = "krk_entry"
    else:
        # Fallback: script node with no parent
        for nid, node in g.nodes.items():
            if node.ntype == NodeType.SCRIPT and g.parent_of(nid) is None:
                root_id = nid
                break
    if root_id:
        g.nodes[root_id].state = NodeState.REQUESTED
    logger.attach_graph(_graph_edges_payload(g))

    ticks = 0
    while ticks < max_ticks and env.get("chosen_move") is None:
        ticks += 1
        now_req = eng.step(env)

        if log_every_tick or ticks % 10 == 0:
            logger.snapshot(
                engine=eng,
                note=f"Eval tick {ticks} (move {move_no})",
                env={"fen": board.fen(), "move_number": move_no, "evaluation_tick": ticks, "binding": env.get("binding")},
                thoughts="Phase sequencing with Phase-0 first. Waiting for terminal to set env['chosen_move'].",
                new_requests=list(now_req.keys()) if now_req else []
            )

    if env.get("chosen_move") is not None:
        logger.snapshot(
            engine=eng,
            note=f"Decision at tick {ticks} (move {move_no})",
            env={"fen": board.fen(), "move_number": move_no, "evaluation_tick": ticks, "chosen_move": env.get("chosen_move"), "binding": env.get("binding")},
            thoughts="Decision made; capturing final node states.",
            new_requests=[]
        )
        linger = max(0, int(linger_ticks_after_choice))
        for i in range(linger):
            ticks += 1
            now_req = eng.step(env)
            logger.snapshot(
                engine=eng,
                note=f"Post-decision linger tick {i+1}/{linger} (move {move_no})",
                env={"fen": board.fen(), "move_number": move_no, "evaluation_tick": ticks, "chosen_move": env.get("chosen_move"), "binding": env.get("binding")},
                thoughts="Linger after choice to visualize downstream phase activation.",
                new_requests=list(now_req.keys()) if now_req else []
            )

    # Diagnostic snapshot: candidate moves + goal distance
    suggestions = env.get("actuator_suggestions", [])
    if suggestions:
        sorted_suggestions = sorted(suggestions, key=lambda s: s["score"], reverse=True)
        prototypes = _goal_prototypes(learner) if learner else []
        diag = []
        for cand in sorted_suggestions[:3]:
            move = cand["move"]
            uci = move.uci() if hasattr(move, "uci") else str(move)
            goal_dist = None
            mate_feature = None
            try:
                b2 = board.copy(stack=False)
                b2.push(move)
                features = _diag_teacher.features(b2)
                if features is not None and len(features) > 12:
                    mate_feature = float(features[12])
                if learner:
                    goal_dist = _goal_distance(learner, prototypes, features)
            except Exception:
                pass
            diag.append({
                "move": uci,
                "score": float(cand["score"]),
                "actuator": cand.get("actuator"),
                "goal_dist": goal_dist,
                "mate_feature": mate_feature,
            })

        payload = {
            "fen": board.fen(),
            "move_number": move_no,
            "suggested_move": env.get("suggested_move"),
            "suggested_actuator": env.get("suggested_actuator"),
            "move_confidence": env.get("move_confidence"),
            "candidates": diag,
        }
        if learner_info:
            payload.update(learner_info)
        logger.snapshot(
            engine=eng,
            note=f"Diagnostics (move {move_no})",
            env=payload,
            thoughts="Top actuator candidates with scores and goal distance.",
            new_requests=[]
        )

    if env.get("chosen_move") is not None:
        return env.get("chosen_move")
    if env.get("suggested_move") is not None:
        return env.get("suggested_move")
    return None


# -------- single game loop --------

def play_single_game(
    initial_fen: str | None = None,
    max_plies: int = 200,
    *,
    log_every_tick: bool = False,
    linger: int = 0,
    topology_path: str | None = None,
    output_path: Path | None = None,
    learner_path: Path | None = None,
) -> dict:
    """
    Plays a single KRK game: White (ReCoN) vs Black (random).
    Returns summary dict with details for later aggregation.
    """
    logger = RunLogger()
    learner, learner_info = _load_learner(learner_path)
    board = chess.Board(initial_fen) if initial_fen else random_krk_board(white_to_move=True)
    binding_table = BindingTable()
    binding_snapshot = _update_binding_table(binding_table, board)

    stalls = 0
    rook_lost = False
    ply = 0
    mate = False

    print("🎮 KRK Demo — ReCoN (White) vs Random (Black)")
    print(board, "\n")

    while not board.is_game_over() and ply < max_plies:
        ply += 1

        # ---- White / ReCoN ----
        chosen_uci = choose_move_with_graph(
            board,
            logger,
            ply,
            log_every_tick=log_every_tick,
            linger_ticks_after_choice=linger,
            topology_path=topology_path,
            binding_table=binding_table,
            learner=learner,
            learner_info=learner_info,
        )
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
        binding_snapshot = _update_binding_table(binding_table, board)

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
                 "features": feats, "box_area_delta": feats["box_area_after"] - prev_area, "binding": binding_snapshot},
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
        binding_snapshot = _update_binding_table(binding_table, board)

        logger.snapshot(
            engine=None,
            note=f"Opponent ply {ply}: {opp_uci}",
            env={"fen": board.fen(), "ply": ply, "opponent_move": opp_uci, "opponents_move": opp_uci, "binding": binding_snapshot},
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
    output_path = output_path or Path("demos/outputs/gameplay/krk_visualization_data.json")
    logger.to_json(str(output_path))
    print(f"\n💾 Log saved to {output_path}")
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


def run_batch(n_games: int = 10, *, topology_path: str | None = None) -> dict:
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
        res = play_single_game(initial_fen=None, max_plies=200, topology_path=topology_path)
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
    parser.add_argument("--topology", type=str, default="", help="Optional topology JSON to load instead of built-in graph.")
    parser.add_argument("--learner", type=str, default="", help="Optional BaselineLearner pickle for diagnostics.")
    parser.add_argument("--output", type=str, default="", help="Optional output JSON path for visualization log.")
    parser.add_argument("--max-plies", type=int, default=200, help="Maximum plies to play.")
    parser.add_argument("--log-every-tick", action="store_true", help="Log a snapshot on every engine tick.")
    parser.add_argument("--linger", type=int, default=0, help="Extra ticks to log after choosing a move.")
    args = parser.parse_args()

    if args.batch and args.batch > 0:
        run_batch(args.batch, topology_path=args.topology or None)
    else:
        start_fen = args.fen if args.fen else "4k3/6K1/8/8/8/8/R7/8 w - - 0 1"
        output_path = Path(args.output) if args.output else None
        learner_path = Path(args.learner) if args.learner else None
        play_single_game(
            initial_fen=start_fen,
            max_plies=args.max_plies,
            log_every_tick=args.log_every_tick,
            linger=args.linger,
            topology_path=args.topology or None,
            output_path=output_path,
            learner_path=learner_path,
        )


if __name__ == "__main__":
    main()
