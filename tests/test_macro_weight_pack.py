import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for target in (PROJECT_ROOT / "src", PROJECT_ROOT):
    target_str = str(target)
    if target_str not in sys.path:
        sys.path.append(target_str)

from recon_lite import LinkType  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite.macrograph import instantiate_macrograph  # type: ignore  # pylint: disable=wrong-import-position
from demos.experiments.teacher_stockfish import run_phase_teacher, run_teacher  # type: ignore  # pylint: disable=wrong-import-position


def _edge_weight(graph, src: str, dst: str, ltype: LinkType) -> float:
    for edge in graph.edges:
        if edge.src == src and edge.dst == dst and edge.ltype == ltype:
            return float(edge.w)
    raise AssertionError(f"Edge {src}->{dst} ({ltype}) not found")


def test_macrograph_applies_weight_pack():
    graph = instantiate_macrograph('specs/macrograph_v0.json')
    weight = _edge_weight(graph, 'FeatureHub', 'MoveSynth', LinkType.POR)
    assert weight == pytest.approx(1.05, rel=1e-6)

    hub_meta = graph.nodes['PlanHub'].meta
    assert hub_meta.get('por_policy') == 'weighted'
    assert hub_meta.get('por_theta') == pytest.approx(0.75)
    assert getattr(graph, 'macro_weight_pack_version', None) == '0.1'


def test_teacher_updates_weight_pack(tmp_path):
    fen_file = tmp_path / 'positions.fen'
    fen_file.write_text('\n'.join([
        '4k3/6K1/8/8/8/8/R7/8 w - - 0 1',  # KRK
        '8/8/8/4k3/4P3/4K3/8/8 w - - 0 1',  # KPK
        '8/8/8/4k3/3R4/4K3/8/8 w - - 0 1',  # Rook technique
    ]) + '\n')

    output = tmp_path / 'macro_weight_pack.swp'
    payload = run_teacher(fen_file, output, engine_path=None, depth=2)

    assert output.exists()
    data = json.loads(output.read_text())
    assert data['por_edges']['LearningSupervisor->PlanHub'] >= 0.6
    assert data['por_edges']['LearningSupervisor->MoveSynth'] >= 0.4
    teacher_meta = data.get('notes', {}).get('teacher', {})
    assert teacher_meta.get('labels') == payload.get('notes', {}).get('teacher', {}).get('labels')
    assert sum(teacher_meta.get('labels', {}).values()) == 3


def test_phase_teacher_generates_weights(tmp_path):
    fens = [
        '4k3/6K1/8/8/8/8/R7/8 w - - 0 1',
        '4k3/6K1/8/3R4/8/8/8/8 w - - 0 1',
    ]
    output = tmp_path / 'krk_phase_weight_pack.swp'
    payload = run_phase_teacher(fens, output, engine_path=None, depth=2)
    assert output.exists()
    data = json.loads(output.read_text())
    assert set(data['phase_weights']) == {'phase1', 'phase2', 'phase3'}
    assert payload['phase_weights'] == data['phase_weights']
    assert data['notes']['teacher']['positions_considered'] >= 0
