
import sys
import os

path = 'scripts/run_krk_curriculum.py'
with open(path, 'rb') as f:
    content = f.read()

# Normalize to LF
content = content.replace(b'\r\n', b'\n')

old_code = b'''                # DELTA-BASED LEARNING: Collect (before, after) feature transitions for mate moves
                if board.is_checkmate():
                    try:
                        features_before = stem_manager.feature_extractor(pre_move_board)
                        features_after = stem_manager.feature_extractor(board)
                        fen_before = pre_move_board.fen()
                        
                        # Store transition in all active stem cells
                        for cell_id, cell in stem_manager.cells.items():
                            if cell.state in (StemCellState.EXPLORING, StemCellState.TRIAL):
                                cell.observe_transition(
                                    features_before=features_before,
                                    features_after=features_after,
                                    reward=1.0,
                                    fen_before=fen_before,
                                    tick=game_tick
                                )
                    except Exception:
                        pass  # Skip on error'''

new_code = b'''                # DELTA-BASED LEARNING: Collect (before, after) feature transitions for mate moves
                if board.is_checkmate():
                    try:
                        features_before = stem_manager.feature_extractor(pre_move_board)
                        features_after = stem_manager.feature_extractor(board)
                        fen_before = pre_move_board.fen()
                        
                        # Compute goal deltas if possible
                        goal_deltas = None
                        try:
                            from recon_lite_chess.goal_actuators import compute_goal_feature_deltas, DEFAULT_GOAL_FEATURES
                            goal_deltas = compute_goal_feature_deltas(pre_move_board, move, DEFAULT_GOAL_FEATURES)
                        except Exception:
                            pass

                        # Store transition in all active stem cells
                        for cell_id, cell in stem_manager.cells.items():
                            if cell.state in (StemCellState.EXPLORING, StemCellState.TRIAL):
                                cell.observe_transition(
                                    features_before=features_before,
                                    features_after=features_after,
                                    reward=1.0,
                                    fen_before=fen_before,
                                    tick=game_tick,
                                    goal_deltas=goal_deltas
                                )
                    except Exception:
                        pass  # Skip on error'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(path, 'wb') as f:
        f.write(content)
    print("Fix applied successfully to scripts/run_krk_curriculum.py")
else:
    print("Target code block not found exactly as expected. Check whitespace/line endings.")
    # Show what we found around that area
    try:
        idx = content.find(b'DELTA-BASED LEARNING')
        if idx != -1:
            print("Fragment found at index:", idx)
            print(content[idx:idx+200])
    except: pass
