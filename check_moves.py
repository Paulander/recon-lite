import json
d = json.load(open('demos/visualization/sample_data/bridge_trained.json'))
print(f"Total frames: {len(d['frames'])}")
print("\nMoves:")
for i, f in enumerate(d['frames']):
    move = f.get('move_uci', '-')
    fen = f.get('board_fen', '')
    lock = f.get('subgraph_lock', 'none')
    print(f"{i}: {move:6} lock={lock}")
