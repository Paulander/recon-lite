from recon_lite.graph import Graph, Node, NodeType, NodeState, LinkType
from recon_lite.engine import ReConEngine
from recon_lite.logger import RunLogger


def mk_terminal(nid, succeed_after=1):
    counter = {"left": succeed_after}
    def pred(node, env):
        counter["left"] -= 1
        if counter["left"] <= 0:
            return True, True
        return False, False
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=pred)


def build_graph():
    g = Graph()
    # Scripts
    g.add_node(Node("ROOT", NodeType.SCRIPT))
    g.add_node(Node("A", NodeType.SCRIPT))
    g.add_node(Node("B1", NodeType.SCRIPT))
    g.add_node(Node("B2", NodeType.SCRIPT))
    g.add_node(Node("C", NodeType.SCRIPT))

    # Terminals
    g.add_node(mk_terminal("A_done", succeed_after=1))
    g.add_node(mk_terminal("B1_done", succeed_after=2))
    g.add_node(mk_terminal("B2_done", succeed_after=1))
    g.add_node(mk_terminal("C_done", succeed_after=1))

    # Hierarchy
    g.add_edge("ROOT", "A", LinkType.SUB)
    g.add_edge("ROOT", "B1", LinkType.SUB)
    g.add_edge("ROOT", "B2", LinkType.SUB)
    g.add_edge("ROOT", "C", LinkType.SUB)

    g.add_edge("A", "A_done", LinkType.SUB)
    g.add_edge("B1", "B1_done", LinkType.SUB)
    g.add_edge("B2", "B2_done", LinkType.SUB)
    g.add_edge("C", "C_done", LinkType.SUB)

    # Sequence with alternative: A -> (B1 | B2) -> C
    g.add_edge("A", "B1", LinkType.POR)  # gate B1 on A
    g.add_edge("A", "B2", LinkType.POR)  # gate B2 on A
    g.add_edge("B1", "C", LinkType.POR)  # gate C on B1
    g.add_edge("B2", "C", LinkType.POR)  # and/or B2

    return g


def main():
    g = build_graph()
    eng = ReConEngine(g)
    log = RunLogger()

    g.nodes["ROOT"].state = NodeState.REQUESTED
    log.snapshot(eng, note="start")

    for _ in range(20):
        newly_req = eng.step()
        log.snapshot(eng, note=f"tick {eng.tick}, new={list(newly_req.keys())}")
        if all(n.state == NodeState.CONFIRMED for nid, n in g.nodes.items() if n.ntype == NodeType.SCRIPT):
            break

    out = "demos/sequence_log.json"
    log.to_json(out)
    print("Final states:")
    for nid, n in g.nodes.items():
        print(f"{nid}: {n.state.name}")
    print("Log:", out)


if __name__ == "__main__":
    main()
