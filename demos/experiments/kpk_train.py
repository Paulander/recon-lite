#!/usr/bin/env python3
"""Simple trainer for the KPK Subgraph Weight Pack.
Scans KPK datasets (FENs), estimates a push preference from how often
`can_push_pawn_safely` holds, and writes weights/subgraphs/kpk_weight_pack.swp.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Iterable
import chess
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from recon_lite_chess.sensors import structure as struct_sensors
from recon_lite_chess.sensors import tactics as tactic_sensors
DEFAULT_OUTPUT = Path('weights/subgraphs/kpk_weight_pack.swp')

def _load_fens(path: Path) -> Iterable[str]:
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#'):
            continue
        yield line

def train_kpk(dataset_root: Path, output_path: Path) -> dict:
    fens=[]
    for fen_file in sorted(dataset_root.rglob('*.fen')):
        fens.extend(list(_load_fens(fen_file)))
    if not fens:
        raise SystemExit(f'No KPK FENs found under {dataset_root}')
    total=0; push_ok=0
    for fen in fens:
        board=chess.Board(fen)
        if not struct_sensors.summarize_kpk_material(board).get('is_kpk'):
            continue
        total+=1
        if tactic_sensors.can_push_pawn_safely(board):
            push_ok+=1
    ratio = (push_ok/total) if total>0 else 0.5
    cfg={
        'version':'0.1',
        'move_selector':{
            'push_bias': round(0.4+0.4*ratio,3),
            'king_distance_weight': 0.25,
            'safety_weight': round(0.1+0.2*ratio,3)
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cfg, indent=2) + '\n')
    return cfg

def main():
    p=argparse.ArgumentParser(description='Train KPK move selector weights from dataset')
    p.add_argument('dataset_root', type=Path)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args=p.parse_args()
    cfg=train_kpk(args.dataset_root, args.output)
    print('Wrote', args.output)
    print(json.dumps(cfg, indent=2))

if __name__=='__main__':
    main()
