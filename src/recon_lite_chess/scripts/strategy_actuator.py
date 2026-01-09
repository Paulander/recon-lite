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
    Push Strategy: Outputs pawn push move if legal.
    
    Finds the best pawn push (prioritizing advancement and promotions).
    Promotions are included and get highest priority!
    """
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if not board:
            return False, False
        
        # Determine attacker color
        color = board.turn
        
        # Find pawn push moves (prioritize promotions!)
        best_push = None
        best_rank = -1
        best_is_promo = False
        
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                # It's a pawn move
                target_rank = chess.square_rank(move.to_square)
                if color == chess.BLACK:
                    target_rank = 7 - target_rank  # Normalize for black
                
                # Promotions get HIGHEST priority (rank 8 = best)
                is_promo = move.promotion is not None
                effective_rank = 8 if is_promo else target_rank
                
                if effective_rank > best_rank or (is_promo and not best_is_promo):
                    best_rank = effective_rank
                    best_push = move
                    best_is_promo = is_promo
        
        if best_push:
            node.meta["suggested_move"] = best_push.uci()
            node.meta["strategy"] = "push"
            # Promotions get 1.0 confidence, others scale by rank
            node.meta["confidence"] = 1.0 if best_is_promo else 0.5 + (best_rank / 14)
            
            env.setdefault("strategy_outputs", {})[nid] = {
                "move": best_push.uci(),
                "confidence": node.meta["confidence"],
                "strategy": "push",
            }
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_king_support_strategy(nid: str) -> Node:
    """
    King Support Strategy: Move king toward pawn to support promotion.
    
    Evaluates king moves for pawn support and path clearing.
    """
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if not board:
            return False, False
        
        color = board.turn
        king_sq = board.king(color)
        if king_sq is None:
            return False, False
        
        # Find pawn to support
        pawns = list(board.pieces(chess.PAWN, color))
        if not pawns:
            return False, False
        pawn_sq = pawns[0]
        pawn_file = chess.square_file(pawn_sq)
        pawn_rank = chess.square_rank(pawn_sq)
        
        # Find best king move (toward supporting pawn)
        best_move = None
        best_score = -100
        
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if not piece or piece.piece_type != chess.KING or piece.color != color:
                continue
            
            new_sq = move.to_square
            new_file = chess.square_file(new_sq)
            new_rank = chess.square_rank(new_sq)
            
            score = 0.0
            
            # Bonus for being close to pawn file
            file_dist = abs(new_file - pawn_file)
            score -= file_dist * 0.2
            
            # Bonus for being ahead of/beside pawn (support position)
            if color == chess.WHITE:
                if new_rank >= pawn_rank:
                    score += 0.3
            else:
                if new_rank <= pawn_rank:
                    score += 0.3
            
            # Bonus for optimal support distance (1-2 squares)
            if file_dist <= 1:
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move:
            node.meta["suggested_move"] = best_move.uci()
            node.meta["strategy"] = "king_support"
            node.meta["confidence"] = 0.5 + best_score
            
            env.setdefault("strategy_outputs", {})[nid] = {
                "move": best_move.uci(),
                "confidence": node.meta["confidence"],
                "strategy": "king_support",
            }
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_generic_arbiter(nid: str) -> Node:
    """
    Generic Arbiter: Picks move via BANDIT/SOFTMAX selection.
    
    PHASE 4: Instead of hard max, uses softmax over learned weights.
    This enables exploration and gradual convergence to best strategy.
    
    strategy_ids should be provided in node.meta["strategy_ids"] during
    topology loading or construction.
    """
    import math
    import random as rnd
    
    def _softmax_sample(scores: Dict[str, float], temperature: float = 1.0) -> Optional[str]:
        """Sample from scores using softmax probability distribution."""
        if not scores:
            return None
        
        # Compute softmax probabilities
        max_score = max(scores.values())
        exp_scores = {k: math.exp((v - max_score) / temperature) for k, v in scores.items()}
        total = sum(exp_scores.values())
        
        if total == 0:
            return rnd.choice(list(scores.keys()))
        
        probs = {k: v / total for k, v in exp_scores.items()}
        
        # Sample according to probabilities
        r = rnd.random()
        cumulative = 0.0
        for k, p in probs.items():
            cumulative += p
            if r <= cumulative:
                return k
        
        return list(scores.keys())[-1]  # Fallback
    
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        strategy_outputs = env.get("strategy_outputs", {})
        
        # PHASE 4: Also check legs proposals
        legs_proposals = env.get("legs", {})
        
        # Get strategy_ids from node meta (set during topology load)
        strategy_ids = node.meta.get("strategy_ids", [])
        
        # Collect all scores and moves
        candidate_scores: Dict[str, float] = {}
        candidate_moves: Dict[str, str] = {}
        
        # Check strategy_outputs (from strategy actuators)
        for sid in strategy_ids:
            output = strategy_outputs.get(sid)
            if output and output.get("move"):
                score = output.get("confidence", 0.5)
                candidate_scores[sid] = score
                candidate_moves[sid] = output["move"]
        
        # Also check legs proposals (PHASE 4: legs write to env["legs"])
        for leg_id, leg_data in legs_proposals.items():
            if isinstance(leg_data, dict) and leg_data.get("move"):
                # Get leg weight from edge
                weight = leg_data.get("weight", 0.5)
                candidate_scores[leg_id] = weight
                candidate_moves[leg_id] = leg_data["move"]
        
        if not candidate_moves:
            return False, False
        
        # BANDIT SELECTION: Softmax sample instead of hard max
        temperature = node.meta.get("temperature", 1.0)
        selected_id = _softmax_sample(candidate_scores, temperature)
        
        if selected_id and selected_id in candidate_moves:
            best_move = candidate_moves[selected_id]
            
            # Store for move execution
            env.setdefault("kpk", {}).setdefault("policy", {})["suggested_move"] = best_move
            node.meta["suggested_move"] = best_move
            node.meta["winning_strategy"] = selected_id
            node.meta["selection_method"] = "softmax"
            node.meta["candidate_count"] = len(candidate_moves)
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

