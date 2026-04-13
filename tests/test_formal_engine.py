import json

import pytest

from recon_lite import FormalMessage, FormalReConEngine, Graph, LinkType, Node, NodeState, NodeType
from recon_lite.examples.formal_trace import generate_trace, main


def success_terminal(nid):
    return Node(nid, NodeType.TERMINAL, predicate=lambda _node, _env: (True, True))


def failed_terminal(nid):
    return Node(nid, NodeType.TERMINAL, predicate=lambda _node, _env: (True, False))


def test_paired_hierarchy_helper_creates_sub_sur():
    graph = Graph()
    graph.add_node(Node("parent", NodeType.SCRIPT))
    graph.add_node(success_terminal("child"))

    graph.add_hierarchy_pair("parent", "child")

    assert graph.get_edge("parent", "child", LinkType.SUB) is not None
    assert graph.get_edge("child", "parent", LinkType.SUR) is not None


def test_paired_sequence_helper_creates_por_ret():
    graph = Graph()
    for nid in ["root", "A", "B", "A_done", "B_done"]:
        node_type = NodeType.TERMINAL if nid.endswith("_done") else NodeType.SCRIPT
        graph.add_node(Node(nid, node_type, predicate=(lambda _n, _e: (True, True)) if node_type == NodeType.TERMINAL else None))
    graph.add_hierarchy_pair("root", "A")
    graph.add_hierarchy_pair("root", "B")
    graph.add_hierarchy_pair("A", "A_done")
    graph.add_hierarchy_pair("B", "B_done")

    graph.add_sequence_pair("A", "B")

    assert graph.get_edge("A", "B", LinkType.POR) is not None
    assert graph.get_edge("B", "A", LinkType.RET) is not None


def test_formal_validation_catches_missing_reverse_links():
    graph = Graph()
    graph.add_node(Node("root", NodeType.SCRIPT))
    graph.add_node(success_terminal("sensor"))
    graph.add_edge("root", "sensor", LinkType.SUB)

    with pytest.raises(ValueError, match="requires reverse SUR"):
        FormalReConEngine(graph)


def test_single_terminal_child_confirms_parent_through_sur():
    graph = Graph()
    graph.add_node(Node("root", NodeType.SCRIPT))
    graph.add_node(success_terminal("sensor"))
    graph.add_hierarchy_pair("root", "sensor")

    engine = FormalReConEngine(graph)
    engine.request("root")
    engine.run(max_ticks=12, until=lambda formal: formal.g.nodes["root"].state == NodeState.CONFIRMED)

    assert graph.nodes["sensor"].state == NodeState.CONFIRMED
    assert graph.nodes["root"].state == NodeState.CONFIRMED
    assert _message_seen(engine.trace, "sensor", "root", LinkType.SUR, FormalMessage.CONFIRM)


def test_linear_sequence_uses_por_and_ret_to_block_premature_execution_and_confirmation():
    graph = _sequence_graph()
    engine = FormalReConEngine(graph)
    engine.request("root")
    engine.run(max_ticks=40, until=lambda formal: formal.g.nodes["root"].state == NodeState.CONFIRMED)

    assert graph.nodes["root"].state == NodeState.CONFIRMED
    assert graph.nodes["A"].state == NodeState.TRUE
    assert graph.nodes["B"].state == NodeState.TRUE
    assert graph.nodes["C"].state == NodeState.CONFIRMED

    a_true_tick = _first_tick_with_state(engine.trace, "A", NodeState.TRUE)
    b_active_tick = _first_tick_with_state(engine.trace, "B", NodeState.ACTIVE)
    c_active_tick = _first_tick_with_state(engine.trace, "C", NodeState.ACTIVE)
    assert a_true_tick is not None
    assert b_active_tick is not None
    assert c_active_tick is not None
    assert b_active_tick > a_true_tick
    assert c_active_tick > _first_tick_with_state(engine.trace, "B", NodeState.TRUE)
    assert _message_seen(engine.trace, "A", "B", LinkType.POR, FormalMessage.INHIBIT_REQUEST)
    assert _message_seen(engine.trace, "B", "A", LinkType.RET, FormalMessage.INHIBIT_CONFIRM)


def test_parent_confirmation_happens_only_after_final_sequence_element_confirms():
    graph = _sequence_graph()
    engine = FormalReConEngine(graph)
    engine.request("root")
    engine.run(max_ticks=40, until=lambda formal: formal.g.nodes["root"].state == NodeState.CONFIRMED)

    root_confirm_tick = _first_tick_with_state(engine.trace, "root", NodeState.CONFIRMED)
    c_confirm_tick = _first_tick_with_state(engine.trace, "C", NodeState.CONFIRMED)

    assert root_confirm_tick is not None
    assert c_confirm_tick is not None
    assert root_confirm_tick > c_confirm_tick


def test_parallel_children_can_run_in_parallel_and_confirm_parent():
    graph = Graph()
    for nid in ["root", "A", "B"]:
        graph.add_node(Node(nid, NodeType.SCRIPT))
    graph.add_node(success_terminal("A_done"))
    graph.add_node(success_terminal("B_done"))
    graph.add_hierarchy_pair("root", "A")
    graph.add_hierarchy_pair("root", "B")
    graph.add_hierarchy_pair("A", "A_done")
    graph.add_hierarchy_pair("B", "B_done")

    engine = FormalReConEngine(graph)
    engine.request("root")
    engine.run(max_ticks=20, until=lambda formal: formal.g.nodes["root"].state == NodeState.CONFIRMED)

    assert graph.nodes["root"].state == NodeState.CONFIRMED
    assert graph.nodes["A"].state == NodeState.CONFIRMED
    assert graph.nodes["B"].state == NodeState.CONFIRMED
    assert _first_tick_with_state(engine.trace, "A", NodeState.ACTIVE) == _first_tick_with_state(engine.trace, "B", NodeState.ACTIVE)


def test_failed_terminal_fails_parent():
    graph = Graph()
    graph.add_node(Node("root", NodeType.SCRIPT))
    graph.add_node(failed_terminal("sensor"))
    graph.add_hierarchy_pair("root", "sensor")

    engine = FormalReConEngine(graph)
    engine.request("root")
    engine.run(max_ticks=12, until=lambda formal: formal.g.nodes["root"].state == NodeState.FAILED)

    assert graph.nodes["sensor"].state == NodeState.FAILED
    assert graph.nodes["root"].state == NodeState.FAILED
    assert _message_seen(engine.trace, "sensor", "root", LinkType.SUR, FormalMessage.FAIL)


def test_trace_frames_contain_edge_messages():
    graph = _sequence_graph()
    engine = FormalReConEngine(graph)
    engine.request("root")
    engine.step({})

    assert engine.trace[0]["messages"] == []
    engine.step({})
    assert engine.trace[1]["messages"]
    assert {"states_before", "states_after", "messages", "activations"} <= set(engine.trace[1])


def test_formal_trace_example_writes_json(tmp_path):
    trace_path = tmp_path / "formal.json"

    assert main(["--trace-json", str(trace_path), "--max-ticks", "40"]) == 0
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    assert trace["engine"] == "FormalReConEngine"
    assert trace["graph"]["nodes"]
    assert trace["graph"]["edges"]
    assert trace["frames"]
    assert any(frame["messages"] for frame in trace["frames"])


def test_generated_formal_trace_reaches_confirmed_root():
    trace = generate_trace(max_ticks=40)

    assert trace["metadata"]["final_root_state"] == NodeState.CONFIRMED.name


def _sequence_graph():
    graph = Graph()
    for nid in ["root", "A", "B", "C"]:
        graph.add_node(Node(nid, NodeType.SCRIPT))
    for nid in ["A_done", "B_done", "C_done"]:
        graph.add_node(success_terminal(nid))

    for child in ["A", "B", "C"]:
        graph.add_hierarchy_pair("root", child)
        graph.add_hierarchy_pair(child, f"{child}_done")
    graph.add_sequence_pair("A", "B")
    graph.add_sequence_pair("B", "C")
    return graph


def _first_tick_with_state(trace, nid, state):
    for frame in trace:
        if frame["states_after"][nid] == state.name:
            return frame["tick"]
    return None


def _message_seen(trace, src, dst, link_type, message):
    for frame in trace:
        for edge_message in frame["messages"]:
            if (
                edge_message["src"] == src
                and edge_message["dst"] == dst
                and edge_message["link_type"] == link_type.name
                and edge_message["message"] == message.value
            ):
                return True
    return False
