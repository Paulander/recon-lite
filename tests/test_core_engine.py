from recon_lite import ActivationMode, EngineConfig, Graph, LinkType, Node, NodeState, NodeType, ReConEngine
from recon_lite.binding.manager import BindingInstance, BindingTable


def mk_term(nid, steps=1):
    counter = {"left": steps}

    def pred(_node, _env):
        counter["left"] -= 1
        return counter["left"] <= 0, True

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=pred)


def build_seq():
    graph = Graph()
    for nid in ["ROOT", "A", "B", "C"]:
        graph.add_node(Node(nid, NodeType.SCRIPT))
    for nid in ["A_done", "B_done", "C_done"]:
        graph.add_node(mk_term(nid, 1))
    graph.add_edge("ROOT", "A", LinkType.SUB)
    graph.add_edge("ROOT", "B", LinkType.SUB)
    graph.add_edge("ROOT", "C", LinkType.SUB)
    graph.add_edge("A", "A_done", LinkType.SUB)
    graph.add_edge("B", "B_done", LinkType.SUB)
    graph.add_edge("C", "C_done", LinkType.SUB)
    graph.add_edge("A", "B", LinkType.POR)
    graph.add_edge("B", "C", LinkType.POR)
    return graph


def run_default_sequence(engine):
    engine.g.nodes["ROOT"].state = NodeState.REQUESTED
    for _ in range(20):
        engine.step({})
    return {
        "states": {nid: node.state for nid, node in engine.g.nodes.items()},
        "logs": list(engine.logs),
        "tick": engine.tick,
    }


def test_default_engine_config_matches_legacy_call_sequence():
    legacy = run_default_sequence(ReConEngine(build_seq()))
    configured = run_default_sequence(ReConEngine(build_seq(), config=EngineConfig()))

    assert configured == legacy
    assert all(state == NodeState.CONFIRMED for state in configured["states"].values())


def test_continuous_mode_updates_activations():
    graph = Graph()
    graph.add_node(Node("root", NodeType.SCRIPT))
    graph.add_node(Node("sensor", NodeType.TERMINAL, predicate=lambda _n, _e: (True, True)))
    graph.add_edge("root", "sensor", LinkType.SUB)
    graph.nodes["sensor"].meta["activation"] = 1.0
    graph.nodes["root"].state = NodeState.REQUESTED

    engine = ReConEngine(
        graph,
        config=EngineConfig(
            activation_mode=ActivationMode.CONTINUOUS,
            microtick_steps=4,
            microtick_eta=0.5,
            record_activation_history=True,
        ),
    )
    env = {"capture": True}
    engine.step(env)

    assert graph.nodes["sensor"].activation.value > 0.9
    assert env["activation_history"]


def test_binding_invalidation_by_signature():
    table = BindingTable()
    with table.begin_tick("grid/sense") as session:
        assert session.reserve(BindingInstance("agent", {"cell:0,0"}))

    assert table.snapshot()
    assert table.invalidate_on_signature("agent=(0,0)") is True
    assert table.snapshot() == {}

    with table.begin_tick("grid/sense") as session:
        assert session.reserve(BindingInstance("agent", {"cell:0,0"}))
    assert table.invalidate_on_signature("agent=(0,0)") is False
    assert table.snapshot()
    assert table.invalidate_on_signature("agent=(1,0)") is True
    assert table.snapshot() == {}
