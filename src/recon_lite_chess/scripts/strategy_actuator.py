"""
Strategy Actuators for Pure ReCoN Move Selection.

These are minimal actuators that compute WHAT move achieves a goal,
but the DECISION of whether to use that move is made by learned weights.

Each strategy outputs:
- suggested_move: UCI string (e.g., "e7e8q")
- confidence: How confident in this move (0-1)

The generic arbiter picks the highest-weighted strategy's move.
"""
from typing import Dict, Any, Tuple, List, Optional
import chess

from recon_lite.graph import Node, NodeType


def create_promote_strategy(nid: str) -> Node:
    """
    Promote Strategy: Outputs promotion move if one exists.
    
    This actuator doesn't decide WHETHER to promote -
    it just provides the move IF promotion is possible.
    The decision is made by learned weights upstream.
    """
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if not board:
            return False, False
        
        # Find promotion moves
        for move in board.legal_moves:
            if move.promotion:
                node.meta["suggested_move"] = move.uci()
                node.meta["strategy"] = "promote"
                node.meta["confidence"] = 1.0  # Promotion is always high priority
                
                # Store in env for arbiter
                env.setdefault("strategy_outputs", {})[nid] = {
                    "move": move.uci(),
                    "confidence": 1.0,
                    "strategy": "promote",
                }
                return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_push_strategy(nid: str) -> Node:
    """
    PURE Push Strategy: Outputs ALL pawn moves with NEUTRAL weight.
    
    No heuristics - doesn't know "promotions best" or "higher rank better".
    The arbiter LEARNS which pawn moves lead to wins via M3/M4 plasticity.
    """
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if not board:
            return False, False
        
        color = board.turn
        
        # Output ALL pawn moves with NEUTRAL weight (no preference)
        pawn_moves = []
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                pawn_moves.append({
                    "move": move.uci(),
                    "source": nid,
                    "weight": 0.5,  # NEUTRAL - learned via plasticity
                    "is_promotion": move.promotion is not None,
                })
        
        if pawn_moves:
            # Add all to candidate_moves for arbiter selection
            env.setdefault("candidate_moves", []).extend(pawn_moves)
            
            # Also output first move as strategy output for backward compat
            node.meta["suggested_move"] = pawn_moves[0]["move"]
            node.meta["strategy"] = "push"
            node.meta["confidence"] = 0.5  # Neutral
            node.meta["move_count"] = len(pawn_moves)
            
            env.setdefault("strategy_outputs", {})[nid] = {
                "move": pawn_moves[0]["move"],
                "confidence": 0.5,
                "strategy": "push",
                "all_moves": pawn_moves,  # Arbiter can pick any
            }
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_king_support_strategy(nid: str) -> Node:
    """
    PURE King Strategy: Outputs ALL king moves with NEUTRAL weight.
    
    No heuristics - doesn't know "support pawn" or "stay close to file".
    The arbiter LEARNS which king moves lead to wins via M3/M4 plasticity.
    """
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if not board:
            return False, False
        
        color = board.turn
        
        # Output ALL king moves with NEUTRAL weight (no preference)
        king_moves = []
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KING and piece.color == color:
                king_moves.append({
                    "move": move.uci(),
                    "source": nid,
                    "weight": 0.5,  # NEUTRAL - learned via plasticity
                })
        
        if king_moves:
            # Add all to candidate_moves for arbiter selection
            env.setdefault("candidate_moves", []).extend(king_moves)
            
            # Also output first move as strategy output for backward compat
            node.meta["suggested_move"] = king_moves[0]["move"]
            node.meta["strategy"] = "king_support"
            node.meta["confidence"] = 0.5  # Neutral
            node.meta["move_count"] = len(king_moves)
            
            env.setdefault("strategy_outputs", {})[nid] = {
                "move": king_moves[0]["move"],
                "confidence": 0.5,
                "strategy": "king_support",
                "all_moves": king_moves,  # Arbiter can pick any
            }
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_generic_arbiter(nid: str) -> Node:
    """
    PURE Arbiter: Selects move via LEARNED softmax selection.
    
    Reads candidate_moves from strategies, applies learned_weights from
    node.meta (updated via M3/M4 plasticity), samples via softmax.
    
    No heuristics - all preference is learned from game outcomes.
    """
    import math
    import random as rnd
    
    def _softmax_sample(moves: List[dict], temperature: float = 1.0) -> Optional[dict]:
        """Sample from moves using softmax probability distribution over weights."""
        if not moves:
            return None
        
        weights = [m.get("weight", 0.5) for m in moves]
        max_w = max(weights)
        exp_weights = [math.exp((w - max_w) / temperature) for w in weights]
        total = sum(exp_weights)
        
        if total == 0:
            return rnd.choice(moves)
        
        probs = [w / total for w in exp_weights]
        
        # Sample according to probabilities
        r = rnd.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return moves[i]
        
        return moves[-1]  # Fallback
    
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        # Get candidate moves from pure strategies
        candidate_moves = env.get("candidate_moves", [])
        
        # Also check strategy_outputs for backward compat
        strategy_outputs = env.get("strategy_outputs", {})
        for sid, output in strategy_outputs.items():
            if output.get("all_moves"):
                # Pure strategy - already in candidate_moves
                continue
            elif output.get("move"):
                # Legacy strategy - add as candidate
                candidate_moves.append({
                    "move": output["move"],
                    "source": sid,
                    "weight": output.get("confidence", 0.5),
                })
        
        if not candidate_moves:
            return False, False
        
        # Apply LEARNED weight multipliers from node.meta
        # These weights are updated by M3/M4 plasticity based on game outcomes
        learned_weights = node.meta.get("learned_weights", {})
        for m in candidate_moves:
            source = m.get("source", "")
            # Apply learned multiplier if available
            if source in learned_weights:
                m["weight"] *= learned_weights[source]
            # Also check for move-specific weights
            move_key = f"{source}:{m['move']}"
            if move_key in learned_weights:
                m["weight"] *= learned_weights[move_key]
            # Boost promotions slightly to help early learning
            if m.get("is_promotion"):
                m["weight"] *= learned_weights.get("promotion_bonus", 1.0)
        
        # SOFTMAX selection - temperature controls exploration
        temperature = node.meta.get("temperature", 1.0)
        selected = _softmax_sample(candidate_moves, temperature)
        
        if selected:
            best_move = selected["move"]
            
            # Store for move execution
            env.setdefault("kpk", {}).setdefault("policy", {})["suggested_move"] = best_move
            node.meta["suggested_move"] = best_move
            node.meta["selected_source"] = selected.get("source", "unknown")
            node.meta["selection_method"] = "learned_softmax"
            node.meta["candidate_count"] = len(candidate_moves)
            
            # Store selection for plasticity update
            env["last_selected_move"] = selected
            
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)



# Factory functions for topology loading
def _create_promote_factory(nid: str) -> Node:
    return create_promote_strategy(nid)

def _create_push_factory(nid: str) -> Node:
    return create_push_strategy(nid)

def _create_king_support_factory(nid: str) -> Node:
    return create_king_support_strategy(nid)

