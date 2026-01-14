#!/usr/bin/env python3
"""Analyze pawn files in failed KPK positions."""
import chess

fens = [
    '8/8/8/8/8/6K1/6P1/5k2 w - - 0 1',
    '8/8/8/8/8/6K1/6P1/3k4 w - - 0 1', 
    '8/8/8/8/5k2/8/P1K5/8 w - - 0 1',
    '8/8/8/8/2k5/6K1/7P/8 w - - 0 1',
    'k2K4/8/8/8/8/P7/8/8 w - - 0 1',
]

print('Analyzing failed positions:')
print('=' * 50)

rook_pawn_count = 0
for fen in fens:
    board = chess.Board(fen)
    
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:
            pawn_file = chess.square_file(sq)
            pawn_rank = chess.square_rank(sq)
            file_name = chess.FILE_NAMES[pawn_file]
            is_rook_pawn = pawn_file == 0 or pawn_file == 7
            
            if is_rook_pawn:
                rook_pawn_count += 1
                
            print(f'FEN: {fen}')
            print(f'  Pawn: {file_name}{pawn_rank+1} (file {pawn_file})')
            print(f'  Rook pawn: {is_rook_pawn}')
            print()

print(f'Summary: {rook_pawn_count}/{len(fens)} failures were rook pawns (a/h file)')
print(f'Non-rook-pawn failures: {len(fens) - rook_pawn_count}')
