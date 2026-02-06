"""Lightweight KPK (King+Pawn vs King) script scaffolding."""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import json

import chess

from recon_lite import Graph, LinkType, Node, NodeType
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint
from recon_lite_chess import create_wait_for_board_change
from recon_lite_chess.sensors import structure as struct_sensors
from recon_lite_chess.sensors import tactics as tactic_sensors

# Shared config cache for KPK weights
_CFG_CACHE = {"loaded": False, "weights": {"push_bias": 0.6, "king_distance_weight": 0.25, "safety_weight": 0.15}}


def _load_cfg():
    """Load KPK weights once from the SWP, fallback to defaults."""
    if _CFG_CACHE["loaded"]:
        return _CFG_CACHE["weights"]
    try:
        path = Path("weights/subgraphs/kpk_weight_pack.swp")
        data = json.loads(path.read_text())
        ws = data.get("move_selector", {})
        _CFG_CACHE["weights"].update({
            "push_bias": float(ws.get("push_bias", _CFG_CACHE["weights"]["push_bias"])),
            "king_distance_weight": float(ws.get("king_distance_weight", _CFG_CACHE["weights"]["king_distance_weight"])),
            "safety_weight": float(ws.get("safety_weight", _CFG_CACHE["weights"]["safety_weight"]))
        })
    except Exception:
        pass
    _CFG_CACHE["loaded"] = True
    return _CFG_CACHE["weights"]


def create_kpk_material_detector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        weights = _load_cfg()
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        node.meta["summary"] = summary
        env.setdefault("kpk", {})["material"] = summary
        is_kpk = bool(summary.get("is_kpk"))
        # Always complete - this is an observer, not a gate
        node.activation.value = 1.0 if is_kpk else 0.0
        return True, True  # Non-blocking sensor

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kpk_push_window(nid: str) -> Node:
    """
    Sensor that checks if pawn can be pushed safely.
    
    Always completes (returns True) because this is a detection node,
    not a gate. The result is stored in meta/env for the move selector to use.
    """
    def _predicate(node: Node, env: Dict[str, Any]):
        weights = _load_cfg()
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        color = summary.get("attacker_color")
        is_kpk = bool(summary.get("is_kpk"))
        can_push = is_kpk and tactic_sensors.can_push_pawn_safely(board, attacker_color=color)
        node.meta["can_push"] = can_push
        node.meta["is_kpk"] = is_kpk
        env.setdefault("kpk", {}).setdefault("tactics", {})["can_push"] = can_push
        # Always complete - this is an observer, not a gate
        node.activation.value = 1.0 if can_push else 0.0
        return True, True  # Non-blocking sensor

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kpk_opposition_probe(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        weights = _load_cfg()
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        color = summary.get("attacker_color")
        has_opposition = bool(summary.get("is_kpk")) and tactic_sensors.has_opposition_alignment(board, attacker_color=color)
        node.meta["has_opposition"] = has_opposition
        env.setdefault("kpk", {}).setdefault("tactics", {})["has_opposition"] = has_opposition
        # Always complete - this is an observer, not a gate
        node.activation.value = 1.0 if has_opposition else 0.0
        return True, True  # Non-blocking sensor

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kpk_promotion_probe(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        weights = _load_cfg()
        board = env.get("board")
        distance = struct_sensors.pawn_distance_to_promotion(board)
        node.meta["distance"] = distance
        env.setdefault("kpk", {}).setdefault("structure", {})["promotion_distance"] = distance
        close_to_promotion = distance <= 1
        # Always complete - this is an observer, not a gate
        node.activation.value = 1.0 if close_to_promotion else 0.0
        return True, True  # Non-blocking sensor

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


# ============================================================================
# ACTUATOR LEGS: Separate King and Pawn move proposers
# ============================================================================

def create_kpk_pawn_leg(nid: str) -> Node:
    """
    Pawn Leg: Proposes pawn moves only (push or promotion).
    
    Activation Level:
    - HIGH when pawn push is legal and safe
    - LOW when push is blocked or unsafe
    
    Stores proposal in env["kpk"]["legs"]["pawn"]
    """
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        
        if not summary.get("is_kpk"):
            node.meta["activation"] = 0.0
            return False, False
        
        color = summary.get("attacker_color")
        pawn_sq = summary.get("pawn_square")
        
        if color is None or pawn_sq is None:
            node.meta["activation"] = 0.0
            return False, False
        
        pawn_rank = chess.square_rank(pawn_sq)
        direction = 8 if color else -8
        push_sq = pawn_sq + direction
        
        # Check if promotion move
        is_promotion_rank = (color == chess.WHITE and pawn_rank == 6) or \
                           (color == chess.BLACK and pawn_rank == 1)
        
        if is_promotion_rank:
            push_move = chess.Move(pawn_sq, push_sq, promotion=chess.QUEEN)
        else:
            push_move = chess.Move(pawn_sq, push_sq)
        
        push_legal = push_move in board.legal_moves
        safe_push = push_legal and tactic_sensors.can_push_pawn_safely(board, attacker_color=color)
        
        # Calculate activation level
        if not push_legal:
            activation = 0.0
            proposal = None
        else:
            # Distance to promotion bonus
            distance_to_promo = 7 - pawn_rank if color == chess.WHITE else pawn_rank
            urgency = max(0, (5 - distance_to_promo) * 0.2)  # +0.2 per rank advanced
            safety = 0.3 if safe_push else 0.0
            
            activation = 0.5 + urgency + safety  # Base 0.5 if legal
            proposal = push_move.uci()
            
            # Promotion is maximum activation
            if is_promotion_rank:
                activation = 1.0
        
        # Store in env for arbiter
        leg_data = {
            "activation": activation,
            "proposal": proposal,
            "reason": "promotion" if is_promotion_rank else "pawn_push",
            "legal": push_legal,
            "safe": safe_push,
        }
        env.setdefault("kpk", {}).setdefault("legs", {})["pawn"] = leg_data
        node.meta["activation"] = activation
        node.meta["proposal"] = proposal
        
        # Success if we have a proposal
        return push_legal, push_legal

    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def create_kpk_king_leg(nid: str) -> Node:
    """
    King Leg: Proposes king moves only (support, opposition, shouldering).
    
    Activation Level:
    - HIGH when king move improves position (closer to pawn, better opposition)
    - LOW when pawn push is available and safe
    
    Heuristic Suppression (Leg Capping):
    - When env["heuristic_suppression"] is True, disable the "approach" heuristic
    - This forces the network to find alternative paths (e.g., triangulation)
    - Used to break "vibing" and force deeper hierarchical reasoning
    
    Stores proposal in env["kpk"]["legs"]["king"]
    """
    def _predicate(node: Node, env: Dict[str, Any]):
        weights = _load_cfg()
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        
        # HEURISTIC SUPPRESSION: Disable "approach" heuristic when enabled
        suppress_approach = env.get("heuristic_suppression", False)
        
        if not summary.get("is_kpk"):
            node.meta["activation"] = 0.0
            return False, False
        
        color = summary.get("attacker_color")
        pawn_sq = summary.get("pawn_square")
        attacker_king = summary.get("attacker_king")
        defender_king = summary.get("defender_king")
        
        if color is None or pawn_sq is None:
            node.meta["activation"] = 0.0
            return False, False
        
        pawn_file = chess.square_file(pawn_sq)
        pawn_rank = chess.square_rank(pawn_sq)
        
        # Check if pawn is blocked (lateral inhibition input)
        pawn_leg = env.get("kpk", {}).get("legs", {}).get("pawn", {})
        pawn_blocked = not pawn_leg.get("legal", True)
        pawn_unsafe = not pawn_leg.get("safe", True)
        
        # Evaluate all king moves
        best_move = None
        best_score = -1.0
        
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if not piece or piece.color != color or piece.piece_type != chess.KING:
                continue
            
            trial = board.copy(stack=False)
            trial.push(move)
            new_sq = trial.king(color)
            if new_sq is None:
                continue
            
            new_file = chess.square_file(new_sq)
            new_rank = chess.square_rank(new_sq)
            
            score = 0.0
            
            # Bonus for supporting pawn (adjacent file, ahead of pawn)
            file_dist = abs(new_file - pawn_file)
            if file_dist <= 1:
                score += 0.2
                # Extra if ahead of pawn (can escort)
                if color == chess.WHITE and new_rank > pawn_rank:
                    score += 0.15
                elif color == chess.BLACK and new_rank < pawn_rank:
                    score += 0.15
            
            # Bonus for opposition (same file as enemy king)
            if defender_king is not None:
                dk_file = chess.square_file(defender_king)
                dk_rank = chess.square_rank(defender_king)
                # Direct opposition
                if new_file == dk_file and abs(new_rank - dk_rank) == 2:
                    score += 0.3
                # Cutting off squares (SUPPRESSIBLE - this is the "vibe" heuristic)
                # When suppressed, the king leg must rely on TRIAL sensors for direction
                if attacker_king is not None and not suppress_approach:
                    cur_d = max(abs(chess.square_file(attacker_king) - dk_file),
                               abs(chess.square_rank(attacker_king) - dk_rank))
                    new_d = max(abs(new_file - dk_file), abs(new_rank - dk_rank))
                    if new_d < cur_d:
                        score += 0.1 * (cur_d - new_d)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        # Calculate final activation
        if best_move is None:
            activation = 0.0
            proposal = None
        else:
            # Boost activation if pawn is blocked or unsafe
            block_boost = 0.4 if pawn_blocked else 0.0
            unsafe_boost = 0.2 if pawn_unsafe else 0.0
            activation = min(1.0, best_score + block_boost + unsafe_boost)
            proposal = best_move.uci()
        
        # Store in env for arbiter
        leg_data = {
            "activation": activation,
            "proposal": proposal,
            "reason": "king_support",
            "pawn_blocked": pawn_blocked,
        }
        env.setdefault("kpk", {}).setdefault("legs", {})["king"] = leg_data
        node.meta["activation"] = activation
        node.meta["proposal"] = proposal
        
        return best_move is not None, best_move is not None

    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def create_kpk_arbiter(nid: str) -> Node:
    """
    Arbiter: Selects between Pawn and King leg proposals based on activation.
    
    Decision Rule:
    - Promotion always wins (pawn activation = 1.0)
    - Otherwise, highest activation wins
    - Tie-breaker: prefer pawn push (forward progress)
    """
    def _predicate(node: Node, env: Dict[str, Any]):
        legs = env.get("kpk", {}).get("legs", {})
        pawn_leg = legs.get("pawn", {})
        king_leg = legs.get("king", {})
        
        pawn_act = pawn_leg.get("activation", 0.0)
        king_act = king_leg.get("activation", 0.0)
        
        # Decision with tie-breaker for pawn
        if pawn_act >= king_act and pawn_leg.get("proposal"):
            winner = "pawn"
            proposal = pawn_leg.get("proposal")
            reason = pawn_leg.get("reason", "pawn_push")
        elif king_leg.get("proposal"):
            winner = "king"
            proposal = king_leg.get("proposal")
            reason = king_leg.get("reason", "king_support")
        else:
            # Fallback to any legal move
            board = env.get("board")
            legal = list(board.legal_moves) if board else []
            proposal = legal[0].uci() if legal else None
            winner = "fallback"
            reason = "no_proposal"
        
        # Store final decision
        env.setdefault("kpk", {}).setdefault("policy", {})["suggested_move"] = proposal
        node.meta["winner"] = winner
        node.meta["pawn_activation"] = pawn_act
        node.meta["king_activation"] = king_act
        node.meta["reason"] = reason
        
        return proposal is not None, proposal is not None

    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def create_kpk_move_selector(nid: str) -> Node:
    """
    KPK move selector with promotion-focused strategy.
    
    The goal of KPK is to PROMOTE the pawn. This is the success condition.
    Strategy priorities:
    1. Push pawn to promotion when safe (highest priority near 7th/8th rank)
    2. If pawn push blocked, use king to support/clear path
    3. Only use king moves when pawn push isn't viable
    """
    def _predicate(node: Node, env: Dict[str, Any]):
        weights = _load_cfg()
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        if not summary.get("is_kpk"):
            return False, False
        color = summary.get("attacker_color")
        pawn_sq = summary.get("pawn_square")
        suggestion = None
        if color is None or pawn_sq is None:
            return False, False

        pawn_rank = chess.square_rank(pawn_sq)
        pawn_file = chess.square_file(pawn_sq)
        
        # Calculate distance to promotion (ranks remaining)
        if color == chess.WHITE:
            distance_to_promo = 7 - pawn_rank  # White promotes on rank 7 (8th rank)
        else:
            distance_to_promo = pawn_rank  # Black promotes on rank 0 (1st rank)
        
        direction = 8 if color else -8
        push_sq = pawn_sq + direction
        
        # Check if this is a promotion move (pawn on 7th rank)
        is_promotion_rank = (color == chess.WHITE and pawn_rank == 6) or (color == chess.BLACK and pawn_rank == 1)
        
        if is_promotion_rank:
            # Promotion move - must specify promotion piece
            promo_move = chess.Move(pawn_sq, push_sq, promotion=chess.QUEEN)
            push_move = promo_move  # Use promotion move
            push_legal = promo_move in board.legal_moves
        else:
            # Regular pawn push
            push_move = chess.Move(pawn_sq, push_sq)
            push_legal = push_move in board.legal_moves
        
        safe_push = push_legal and tactic_sensors.can_push_pawn_safely(board, attacker_color=color)
        
        # CRITICAL: Promotion move! Always take it if legal
        if is_promotion_rank and push_legal:
            # This push is promotion! Highest priority - ALWAYS promote
            suggestion = push_move.uci()
            env.setdefault("kpk", {}).setdefault("policy", {})["suggested_move"] = suggestion
            node.meta["last_move"] = suggestion
            node.meta["reason"] = "promotion"
            return True, True
        
        # Calculate push score with distance bonus
        # The closer to promotion, the higher the priority
        base_push_score = weights.get("push_bias", 0.6)
        safety_bonus = weights.get("safety_weight", 0.15) if safe_push else 0.0
        # Massive bonus for pawns close to promotion
        promotion_urgency = (5 - distance_to_promo) * 0.3  # +0.3 per rank advanced
        push_score = (base_push_score + safety_bonus + promotion_urgency) if push_legal else -1.0

        attacker_king = summary.get("attacker_king")
        defender_king = summary.get("defender_king")
        
        # Evaluate king moves
        best_king = None
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.color == color and piece.piece_type == chess.KING:
                trial = board.copy(stack=False)
                trial.push(move)
                new_sq = trial.king(color)
                if new_sq is None:
                    continue
                    
                score = 0.0
                new_file = chess.square_file(new_sq)
                new_rank = chess.square_rank(new_sq)
                
                # Bonus for king supporting the pawn (within 1-2 files, ahead of pawn)
                file_dist_to_pawn = abs(new_file - pawn_file)
                if file_dist_to_pawn <= 1:
                    score += 0.2
                    # Extra bonus if king is ahead of pawn (can support promotion)
                    if color == chess.WHITE and new_rank > pawn_rank:
                        score += 0.1
                    elif color == chess.BLACK and new_rank < pawn_rank:
                        score += 0.1
                
                # Bonus for approaching enemy king (to cut off escape)
                if defender_king is not None:
                    cur_d = max(abs(chess.square_file(attacker_king) - chess.square_file(defender_king)),
                                abs(chess.square_rank(attacker_king) - chess.square_rank(defender_king))) if attacker_king is not None else 0
                    new_d = max(abs(new_file - chess.square_file(defender_king)),
                                abs(new_rank - chess.square_rank(defender_king)))
                    gain = max(0, cur_d - new_d)
                    score += gain * weights.get("king_distance_weight", 0.25)
                
                if best_king is None or score > best_king[0]:
                    best_king = (score, move)

        # Decision: Prioritize pawn push unless blocked or king move is significantly better
        if push_score >= 0 and (best_king is None or push_score > best_king[0] + 0.3):
            # Push the pawn!
            suggestion = push_move.uci()
            node.meta["reason"] = "pawn_push"
        elif best_king is not None:
            # King move to support
            suggestion = best_king[1].uci()
            node.meta["reason"] = "king_support"
        elif push_score >= 0:
            # Fallback to pawn push even if not ideal
            suggestion = push_move.uci()
            node.meta["reason"] = "pawn_push_fallback"

        if suggestion is None:
            # Last resort: any legal move
            legal = list(board.legal_moves)
            if legal:
                suggestion = legal[0].uci()
                node.meta["reason"] = "fallback"

        if suggestion is None:
            return True, False

        env.setdefault("kpk", {}).setdefault("policy", {})["suggested_move"] = suggestion
        node.meta["last_move"] = suggestion
        return True, True

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def build_kpk_network() -> Graph:
    """Original monolithic network (deprecated - use build_kpk_legs_network)."""
    g = Graph()

    # Terminals
    g.add_node(create_kpk_material_detector("kpk_material_check"))
    g.add_node(create_kpk_push_window("kpk_push_window"))
    g.add_node(create_kpk_opposition_probe("kpk_opposition_probe"))
    g.add_node(create_kpk_promotion_probe("kpk_promotion_probe"))
    g.add_node(create_kpk_move_selector("kpk_move_selector"))
    g.add_node(create_wait_for_board_change("kpk_wait_for_change"))

    # Script backbone
    g.add_node(Node(nid="kpk_root", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_detect", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_execute", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_finish", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_wait", ntype=NodeType.SCRIPT))

    # Hierarchy
    g.add_edge("kpk_root", "kpk_detect", LinkType.SUB)
    g.add_edge("kpk_root", "kpk_execute", LinkType.SUB)
    g.add_edge("kpk_root", "kpk_finish", LinkType.SUB)
    g.add_edge("kpk_root", "kpk_wait", LinkType.SUB)

    g.add_edge("kpk_detect", "kpk_material_check", LinkType.SUB)
    g.add_edge("kpk_detect", "kpk_push_window", LinkType.SUB)
    g.add_edge("kpk_execute", "kpk_move_selector", LinkType.SUB)
    g.add_edge("kpk_execute", "kpk_opposition_probe", LinkType.SUB)
    g.add_edge("kpk_finish", "kpk_promotion_probe", LinkType.SUB)
    g.add_edge("kpk_wait", "kpk_wait_for_change", LinkType.SUB)

    # Sequencing
    g.add_edge("kpk_detect", "kpk_execute", LinkType.POR)
    g.add_edge("kpk_execute", "kpk_finish", LinkType.POR)
    g.add_edge("kpk_finish", "kpk_wait", LinkType.POR)

    return g


def build_kpk_legs_network() -> Graph:
    """
    New architecture with separate King/Pawn actuator legs.
    
    Structure:
        kpk_root
        ├── kpk_detect (material check)
        ├── kpk_execute
        │   ├── kpk_pawn_leg (proposes pawn moves)
        │   ├── kpk_king_leg (proposes king moves)
        │   └── kpk_arbiter (selects winner)
        ├── kpk_finish (promotion check)
        └── kpk_wait (board change)
    
    Lateral Inhibition:
        pawn_leg --POR--> king_leg (pawn must evaluate first)
        both legs --POR--> arbiter (legs must complete before arbitration)
    """
    g = Graph()

    # === Sensor Terminals ===
    g.add_node(create_kpk_material_detector("kpk_material_check"))
    g.add_node(create_kpk_push_window("kpk_push_window"))
    g.add_node(create_kpk_opposition_probe("kpk_opposition_probe"))
    g.add_node(create_kpk_promotion_probe("kpk_promotion_probe"))
    g.add_node(create_wait_for_board_change("kpk_wait_for_change"))

    # === Actuator Legs ===
    g.add_node(create_kpk_pawn_leg("kpk_pawn_leg"))
    g.add_node(create_kpk_king_leg("kpk_king_leg"))
    g.add_node(create_kpk_arbiter("kpk_arbiter"))

    # === Script backbone ===
    g.add_node(Node(nid="kpk_root", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_detect", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_execute", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_finish", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_wait", ntype=NodeType.SCRIPT))

    # === Hierarchy (SUB links) ===
    g.add_edge("kpk_root", "kpk_detect", LinkType.SUB)
    g.add_edge("kpk_root", "kpk_execute", LinkType.SUB)
    g.add_edge("kpk_root", "kpk_finish", LinkType.SUB)
    g.add_edge("kpk_root", "kpk_wait", LinkType.SUB)

    # Detection phase
    g.add_edge("kpk_detect", "kpk_material_check", LinkType.SUB)
    g.add_edge("kpk_detect", "kpk_push_window", LinkType.SUB)

    # Execution phase with legs
    g.add_edge("kpk_execute", "kpk_pawn_leg", LinkType.SUB)
    g.add_edge("kpk_execute", "kpk_king_leg", LinkType.SUB)
    g.add_edge("kpk_execute", "kpk_arbiter", LinkType.SUB)
    g.add_edge("kpk_execute", "kpk_opposition_probe", LinkType.SUB)

    # Finish phase
    g.add_edge("kpk_finish", "kpk_promotion_probe", LinkType.SUB)
    g.add_edge("kpk_wait", "kpk_wait_for_change", LinkType.SUB)

    # === Sequencing (POR links) ===
    # Main phase sequencing
    g.add_edge("kpk_detect", "kpk_execute", LinkType.POR)
    g.add_edge("kpk_execute", "kpk_finish", LinkType.POR)
    g.add_edge("kpk_finish", "kpk_wait", LinkType.POR)

    # Lateral inhibition: pawn leg must evaluate BEFORE king leg
    # (so king can see if pawn is blocked)
    g.add_edge("kpk_pawn_leg", "kpk_king_leg", LinkType.POR)
    
    # Both legs must complete before arbiter decides
    g.add_edge("kpk_pawn_leg", "kpk_arbiter", LinkType.POR)
    g.add_edge("kpk_king_leg", "kpk_arbiter", LinkType.POR)

    return g


def run_kpk_episode_with_trace(
    board: chess.Board,
    *,
    max_plies: int = 100,
    max_ticks_per_move: int = 200,
    trace_db: Optional[TraceDB] = None,
    episode_id: str = "kpk-episode",
    pack_paths: Optional[list[Path]] = None,
) -> EpisodeRecord:
    """
    Run a single KPK episode using the batch evaluator and emit an EpisodeRecord.
    If `trace_db` is provided, the episode is appended immediately.
    """
    from demos.experiments.batch_eval import run_kpk_episode as _run

    _, _, ep = _run(
        board.fen(),
        max_plies=max_plies,
        max_ticks_per_move=max_ticks_per_move,
        pack_paths=pack_paths or [],
    )
    ep.episode_id = episode_id
    if pack_paths:
        ep.pack_meta = pack_fingerprint(pack_paths)
    if trace_db:
        trace_db.add_episode(ep)
    return ep


__all__ = [
    "build_kpk_network",
    "build_kpk_legs_network",
    "create_kpk_material_detector",
    "create_kpk_push_window",
    "create_kpk_opposition_probe",
    "create_kpk_promotion_probe",
    "create_kpk_move_selector",
    "create_kpk_pawn_leg",
    "create_kpk_king_leg",
    "create_kpk_arbiter",
]

