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
from collections import deque
import chess
import sys
from pathlib import Path
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite.engine import ReConEngine
from recon_lite.logger import RunLogger
from recon_lite.graph import NodeState, Graph, LinkType
from demos.shared.krk_network import build_krk_network, create_random_krk_board
from recon_lite_chess import (
    create_krk_root,
    create_phase0_establish_cut, create_phase0_choose_moves,
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
    create_king_edge_detector, create_box_shrink_evaluator,
    create_opposition_evaluator, create_mate_deliver_evaluator,
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
    create_king_drive_moves, create_box_shrink_moves,
    create_opposition_moves, create_mate_moves
)
from recon_lite_chess.krk_nodes import wire_default_krk
from recon_lite_chess.predicates import dist_to_edge, on_rim, has_opposition_after, chebyshev


def _build_basic_krk_graph() -> Graph:
    g = Graph()
    root  = create_krk_root("ROOT")
    p0    = create_phase0_establish_cut("PHASE0")
    ch0   = create_phase0_choose_moves("CHOOSE_P0")
    p1    = create_phase1_drive_to_edge("PHASE1")
    p2    = create_phase2_shrink_box("PHASE2")
    p3    = create_phase3_take_opposition("PHASE3")
    p4    = create_phase4_deliver_mate("PHASE4")
    # Evaluators (no wait-for-board-change gate in basic)
    e_edge = create_king_edge_detector("king_at_edge")
    e_box  = create_box_shrink_evaluator("box_can_shrink")
    e_opp  = create_opposition_evaluator("can_take_opposition")
    e_mate = create_mate_deliver_evaluator("can_deliver_mate")

    # Actuators
    m_p1 = create_king_drive_moves("king_drive_moves")
    m_p2 = create_box_shrink_moves("box_shrink_moves")
    m_p3 = create_opposition_moves("opposition_moves")
    m_p4 = create_mate_moves("mate_moves")

    for n in [root, p0, ch0, p1, p2, p3, p4, e_edge, e_box, e_opp, e_mate, m_p1, m_p2, m_p3, m_p4]:
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

    # Basic wiring to match shared semantics (minus wait gate):
    # Phase 1
    g.add_edge("PHASE1", "king_drive_moves", LinkType.SUB)
    g.add_edge("PHASE1", "king_at_edge", LinkType.SUB)

    # Phase 2 depends on king at edge
    g.add_edge("PHASE2", "box_shrink_moves", LinkType.SUB)
    g.add_edge("king_at_edge", "PHASE2", LinkType.POR)

    # Phase 3 depends on box shrink possible
    g.add_edge("PHASE3", "opposition_moves", LinkType.SUB)
    g.add_edge("box_can_shrink", "PHASE3", LinkType.POR)
    g.add_edge("PHASE3", "box_can_shrink", LinkType.SUB)

    # Phase 4: deliver mate (do not POR-gate strictly on opposition; allow early mate)
    g.add_edge("PHASE4", "mate_moves", LinkType.SUB)
    # Removed POR from can_take_opposition -> PHASE4 to permit direct mates
    g.add_edge("PHASE4", "can_deliver_mate", LinkType.SUB)
    return g


# ---- Deterministic arbiter helpers (demo-local, no library changes) ----

def _find_our_rook_sq(board: chess.Board) -> chess.Square | None:
    color = board.turn
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type == chess.ROOK:
            return sq
    return None

def _rook_safe_now(board: chess.Board, rook_sq: chess.Square) -> bool:
    # Safe if enemy king cannot capture rook next move, or our king can immediately recapture
    color = board.turn
    enemy = not color
    ek = board.king(enemy)
    ok = board.king(color)
    if ek is None or ok is None or rook_sq is None:
        return False
    if chebyshev(ek, rook_sq) > 1:
        return True
    b = board.copy(stack=False)
    b.turn = enemy
    cap = chess.Move(ek, rook_sq)
    if cap in b.legal_moves:
        return chebyshev(ok, rook_sq) <= 1
    return True

def _cut_established(board: chess.Board) -> bool:
    """
    Conservative 'cut' heuristic: rook aligned with BK (file/rank), at least 2 away,
    rook not droppable now, and our king is reasonably close to rook or BK.
    """
    color = board.turn
    ek = board.king(not color)
    ok = board.king(color)
    rsq = _find_our_rook_sq(board)
    if ek is None or ok is None or rsq is None:
        return False
    same_file = chess.square_file(rsq) == chess.square_file(ek)
    same_rank = chess.square_rank(rsq) == chess.square_rank(ek)
    aligned = same_file or same_rank
    far_enough = chebyshev(rsq, ek) >= 2
    safe = _rook_safe_now(board, rsq)
    support_ok = chebyshev(ok, ek) <= 3 or chebyshev(ok, rsq) <= 3
    return aligned and far_enough and safe and support_ok

def _can_deliver_mate_now(board: chess.Board) -> bool:
    for mv in board.legal_moves:
        board.push(mv)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False

def _eligible_phase(board: chess.Board) -> str:
    """
    Centralized eligible phase selection from board features only.
    Highest-to-lowest precedence: 4 → 3 → 2 → 1 → 0.
    """
    enemy = not board.turn
    ek = board.king(enemy)
    if ek is None:
        return "phase0"
    if _can_deliver_mate_now(board):
        return "phase4"
    if on_rim(ek):
        for mv in board.legal_moves:
            if has_opposition_after(board, mv):
                return "phase3"
    if dist_to_edge(ek) == 0:
        return "phase2"
    if _cut_established(board):
        return "phase1"
    return "phase0"

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
    # Persistent env across plies to maintain fen history and pressure
    env = {"board": board, "chosen_move": None, "fen_history": deque(maxlen=12), "pressure_steps": 0}

    while not board.is_game_over() and plies < max_plies:
        # One decision cycle (White/ReCoN)
        env["board"] = board
        env["chosen_move"] = None
        ticks = 0
        min_decision_ticks = 3  # allow parallel phases a short window to compete
        proposals: list[dict] = []
        phase_rank = {"phase0": 0, "phase1": 1, "phase2": 2, "phase3": 3, "phase4": 4}

        while ticks < tick_watchdog and not board.is_game_over():
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

            eligible = _eligible_phase(board)

            # If any terminal proposed a move this tick, record its phase and keep evaluating briefly
            if env.get("chosen_move"):
                proposed = env["chosen_move"]
                # Try to identify which phase proposed this move
                phase_name = None
                for nid, node in engine.g.nodes.items():
                    ph = node.meta.get("phase")
                    sugg = node.meta.get("suggested_moves")
                    if ph and sugg and proposed in sugg:
                        phase_name = ph
                        break
                proposals.append({
                    "move": proposed,
                    "phase": phase_name or "unknown",
                    "rank": phase_rank.get(phase_name or "", -1)
                })
                # Clear to allow other phases to propose within the window
                env["chosen_move"] = None

            # Exit when eligible phase has at least one proposal and window elapsed
            if ticks >= min_decision_ticks:
                elig_props = [p for p in proposals if p["phase"] == eligible]
                if elig_props:
                    break

        # Choose the best proposal (prefer higher phase); fall back to any remaining choice
        move_uci = None
        if proposals:
            eligible = _eligible_phase(board)
            elig_props = [p for p in proposals if p["phase"] == eligible]
            pick_from = elig_props if elig_props else proposals
            pick_from.sort(key=lambda p: p["rank"])  # ensure deterministic order for equal ranks
            move_uci = max(pick_from, key=lambda p: p["rank"]) ["move"]
        else:
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

    # Reset only terminals (evaluators/actuators) and preserve confirmed phase states
    # This was overly aggressive earlier and ruined the whole graph: 
    # Per ReCoN engine (_request_child_if_ready): A node (e.g., PHASE1) is only REQUESTED if all POR predecessors (e.g., PHASE0) are TRUE/CONFIRMED (_all_por_predecessors_true
    phase_ids = ["ROOT", "PHASE0", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
    for n in g.nodes.values():
        if n.nid not in phase_ids and n.state == NodeState.CONFIRMED:
            n.state = NodeState.INACTIVE  # Reset confirmed terminals
        elif n.state not in (NodeState.CONFIRMED, NodeState.INACTIVE):
            n.state = NodeState.INACTIVE  # Reset transient states of non-phase nodes

    # Re-arm evaluators (clear meta for re-evaluation)
    for eval_id in ["king_at_edge", "box_can_shrink", "can_take_opposition", "can_deliver_mate"]:
        if eval_id in g.nodes:
            n = g.nodes[eval_id]
            n.meta.clear()  # Reset meta (e.g., caches) for fresh evaluation

    # Re-arm wait gate to detect new FEN (if present)
    if graph == "shared" and "wait_for_board_change" in g.nodes:
        g.nodes["wait_for_board_change"].meta.pop("last_fen", None)

    # Re-REQUEST root to trigger next cycle with persisted phase states
    g.nodes[root_id].state = NodeState.REQUESTED

    result = {
        "plies": plies,
        "checkmate": board.is_checkmate(),
        "stalemate": board.is_stalemate(),
        "rook_lost": rook_lost,
        "final_fen": board.fen(),
    }

    out_path = Path("demos/krk_visualization_data.json")
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
