#!/usr/bin/env python3
"""Analyze the 2 g-pawn failure positions."""
import chess

fens = [
    '8/8/8/8/8/6K1/6P1/5k2 w - - 0 1',
    '8/8/8/8/8/6K1/6P1/3k4 w - - 0 1',
]

for fen in fens:
    board = chess.Board(fen)
    print(f'Position: {fen}')
    print(board)
    print()
    
    wk = bk = pawn = None
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING and piece.color == chess.WHITE:
            wk = sq
        elif piece.piece_type == chess.KING and piece.color == chess.BLACK:
            bk = sq
        elif piece.piece_type == chess.PAWN:
            pawn = sq
    
    print(f'  White King: {chess.square_name(wk)}')
    print(f'  Black King: {chess.square_name(bk)}')
    print(f'  Pawn: {chess.square_name(pawn)}')
    
    pawn_rank = chess.square_rank(pawn)
    steps_to_promote = 7 - pawn_rank - 1
    print(f'  Steps to promote: {steps_to_promote}')
    
    bk_dist_to_g8 = max(abs(chess.square_file(bk) - 6), abs(chess.square_rank(bk) - 7))
    print(f'  Black king distance to g8: {bk_dist_to_g8}')
    
    # Key squares rule for opposition
    wk_ahead = chess.square_rank(wk) > pawn_rank
    print(f'  White king ahead of pawn: {wk_ahead}')
    print()
