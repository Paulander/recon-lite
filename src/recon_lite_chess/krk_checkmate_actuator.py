"""
Simple KRK Actuator that maximizes is_checkmate delta.

For mate-in-1 positions, this is the correct objective:
- Select the move that achieves checkmate (is_checkmate: 0 -> 1)
- All other moves have delta 0
"""

import chess
import numpy as np
from recon_lite.graph import Node, NodeType
from recon_lite_chess.baseline_teacher import KRKTeacher

_teacher = KRKTeacher()

def create_checkmate_actuator(node_id=None):
    """Create actuator that selects move maximizing is_checkmate delta"""
    actual_id = node_id or "checkmate_actuator"
    
    def predicate(node, graph, env):
        board = env.get("board")
        if not board:
            return False, False
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return False, False
        
        v0 = _teacher.features(board)
        
        best_move = None
        best_delta = -999
        
        for move in legal_moves:
            b2 = board.copy()
            b2.push(move)
            v1 = _teacher.features(b2)
            
            # is_checkmate is feature 13
            delta_checkmate = v1[13] - v0[13]
            
            if delta_checkmate > best_delta:
                best_delta = delta_checkmate
                best_move = move
        
        if best_move is None:
            return False, False
        
        env["suggested_move"] = best_move
        env["move_confidence"] = best_delta
        
        return True, True
    
    return Node(
        nid=actual_id,
        ntype=NodeType.TERMINAL,
        predicate=predicate
    )
