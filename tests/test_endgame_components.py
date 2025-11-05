import sys
from pathlib import Path

import chess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from recon_lite_chess.sensors import structure as struct_sensors  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite_chess.sensors import tactics as tactic_sensors  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite_chess.scripts.kpk import build_kpk_network  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite_chess.scripts.rook_endings import build_rook_techniques_network  # type: ignore  # pylint: disable=wrong-import-position


def test_kpk_sensors_and_move_selector():
    board = chess.Board('8/8/8/8/4k3/4K3/4P3/8 w - - 0 1')
    summary = struct_sensors.summarize_kpk_material(board)
    assert summary['is_kpk'] is True
    assert struct_sensors.pawn_has_clear_path(board)
    assert tactic_sensors.can_push_pawn_safely(board)

    blocked = chess.Board('8/8/8/4k3/4P3/4K3/8/8 w - - 0 1')
    blocked.turn = chess.WHITE
    assert not tactic_sensors.can_push_pawn_safely(blocked, attacker_color=True)

    graph = build_kpk_network()
    node = graph.nodes['kpk_move_selector']
    env = {'board': board}
    done, success = node.predicate(node, env)  # type: ignore[arg-type]
    assert done is True and success is True
    assert env['kpk']['policy']['suggested_move']


def test_rook_techniques_graph_and_detectors():
    graph = build_rook_techniques_network()
    # Basic structure assertions
    assert 'rook_techniques_root' in graph.nodes
    assert graph.parent_of('rook_cutoff_check') == 'rook_cutoff'

    rook_board = chess.Board('4k3/6K1/8/8/3R4/8/8/8 w - - 0 1')
    env = {'board': rook_board}
    cutoff = graph.nodes['rook_cutoff_ready']
    done, ready = cutoff.predicate(cutoff, env)  # type: ignore[arg-type]
    assert done is True
    assert isinstance(ready, bool)
    assert env['rook_techniques']['cutoff']['ready'] == ready

    bridge = graph.nodes['rook_bridge_ready']
    bridge_done, bridge_ready = bridge.predicate(bridge, env)  # type: ignore[arg-type]
    assert bridge_done is True
    assert isinstance(env['rook_techniques']['bridge']['ready'], bool)

    ladder = graph.nodes['rook_ladder_ready']
    ladder_done, ladder_ready = ladder.predicate(ladder, env)  # type: ignore[arg-type]
    assert ladder_done is True
    assert isinstance(ladder_ready, bool)
