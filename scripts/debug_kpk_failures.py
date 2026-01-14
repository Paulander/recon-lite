#!/usr/bin/env python3
"""Diagnostic script to analyze KPK failure cases."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite_chess.graph.unified_builder import build_unified_graph
from recon_lite import ReConEngine, NodeState
import chess

def create_random_kpk_board(white_to_move: bool = True) -> chess.Board:
    """Create a random valid KPK position."""
    import random
    
    while True:
        defender_king_sq = random.randint(0, 63)
        attacker_king_sq = random.randint(0, 63)
        if attacker_king_sq == defender_king_sq:
            continue
        
        dk_file = chess.square_file(defender_king_sq)
        dk_rank = chess.square_rank(defender_king_sq)
        ak_file = chess.square_file(attacker_king_sq)
        ak_rank = chess.square_rank(attacker_king_sq)
        
        if abs(dk_file - ak_file) <= 1 and abs(dk_rank - ak_rank) <= 1:
            continue
        
        pawn_rank = random.randint(1, 6)
        pawn_file = random.randint(0, 7)
        pawn_sq = chess.square(pawn_file, pawn_rank)
        
        if pawn_sq in (defender_king_sq, attacker_king_sq):
            continue
        
        board = chess.Board.empty()
        
        if white_to_move:
            board.set_piece_at(attacker_king_sq, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))
            board.set_piece_at(defender_king_sq, chess.Piece(chess.KING, chess.BLACK))
            board.turn = chess.WHITE
        else:
            board.set_piece_at(attacker_king_sq, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.BLACK))
            board.set_piece_at(defender_king_sq, chess.Piece(chess.KING, chess.WHITE))
            board.turn = chess.BLACK
        
        if not board.is_valid():
            continue
        if board.is_game_over():
            continue
        
        return board

def play_game(graph, engine, board, max_moves=100, max_ticks=10):
    """Play a game and return outcome details."""
    moves = []
    promoted = False
    starting_fen = board.fen()
    
    for move_num in range(max_moves):
        if board.is_game_over():
            break
        
        # Reset and request kpk_root
        for n in graph.nodes.values():
            n.state = NodeState.INACTIVE
        graph.nodes["kpk_root"].state = NodeState.REQUESTED
        
        env = {"board": board}
        suggested = None
        
        for tick in range(max_ticks):
            engine.step(env)
            suggested = env.get("kpk", {}).get("policy", {}).get("suggested_move")
            if suggested:
                break
        
        if suggested:
            try:
                move = chess.Move.from_uci(suggested)
                if move in board.legal_moves:
                    if move.promotion:
                        promoted = True
                    board.push(move)
                    moves.append(move.uci())
                    continue
            except:
                pass
        
        # Fallback
        legal = list(board.legal_moves)
        if legal:
            board.push(legal[0])
            moves.append(f"fallback:{legal[0].uci()}")
    
    # Determine outcome
    if board.is_checkmate():
        outcome = "checkmate"
    elif promoted:
        outcome = "promoted"
    elif board.is_stalemate():
        outcome = "stalemate"
    else:
        outcome = "timeout"
    
    return {
        "starting_fen": starting_fen,
        "outcome": outcome,
        "moves": len(moves),
        "promoted": promoted,
        "move_list": moves[:10],  # First 10 moves
        "fallbacks": sum(1 for m in moves if m.startswith("fallback")),
    }

def main():
    print("Building unified graph...")
    g = build_unified_graph(include_endgames=True, include_tactics=False)
    engine = ReConEngine(g)
    
    print("\n=== Running 30 games to analyze failures ===\n")
    
    wins = []
    losses = []
    
    for i in range(30):
        board = create_random_kpk_board()
        result = play_game(g, engine, board)
        
        if result["promoted"] or result["outcome"] == "checkmate":
            wins.append(result)
        else:
            losses.append(result)
            print(f"LOSS #{len(losses)}: {result['starting_fen']}")
            print(f"    Outcome: {result['outcome']}, Moves: {result['moves']}, Fallbacks: {result['fallbacks']}")
            print(f"    First moves: {result['move_list'][:5]}")
            print()
    
    print(f"\n=== Summary ===")
    print(f"Wins: {len(wins)} ({100*len(wins)/30:.1f}%)")
    print(f"Losses: {len(losses)} ({100*len(losses)/30:.1f}%)")
    
    if losses:
        print(f"\nLoss breakdown:")
        outcomes = {}
        for l in losses:
            outcomes[l["outcome"]] = outcomes.get(l["outcome"], 0) + 1
        for o, count in outcomes.items():
            print(f"  {o}: {count}")
        
        avg_fallbacks = sum(l["fallbacks"] for l in losses) / len(losses)
        print(f"\nAvg fallbacks per loss: {avg_fallbacks:.1f}")

if __name__ == "__main__":
    main()
