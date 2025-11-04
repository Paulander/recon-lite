import chess

from recon_lite.macrograph import instantiate_macrograph
from recon_lite.macro_engine import MacroEngine


def test_macrograph_instantiation_mounts_krk(monkeypatch):
    from demos.shared.krk_network import build_krk_network

    graph = instantiate_macrograph("specs/macrograph_v0.json", krk_builder=build_krk_network)
    assert "GameControl" in graph.nodes
    assert "KRKSubgraph" in graph.nodes
    # KRK root should exist after mount
    assert "krk_root" in graph.nodes
    assert "phase4_deliver_mate" in graph.nodes

    krk_parent = graph.parent_of("krk_root")
    assert krk_parent == "KRKSubgraph"


def test_macro_engine_auto_activates_krk():
    engine = MacroEngine("specs/macrograph_v0.json")
    board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")
    env = {"board": board}
    engine.step(env)
    assert engine.g.nodes["KRKSubgraph"].state != engine.g.nodes["KRKSubgraph"].state.__class__.INACTIVE
