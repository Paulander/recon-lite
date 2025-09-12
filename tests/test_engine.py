from recon_lite.graph import Graph, Node, NodeType, NodeState, LinkType
from recon_lite.engine import ReConEngine


def mk_term(nid, steps=1):
    c = {"left": steps}
    def pred(node, env):
        c["left"] -= 1
        return (c["left"] <= 0, True)
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=pred)


def build_seq():
    g = Graph()
    for nid in ["ROOT","A","B","C"]:
        g.add_node(Node(nid, NodeType.SCRIPT))
    for nid in ["A_done","B_done","C_done"]:
        g.add_node(mk_term(nid, 1))
    g.add_edge("ROOT","A",LinkType.SUB)
    g.add_edge("ROOT","B",LinkType.SUB)
    g.add_edge("ROOT","C",LinkType.SUB)
    g.add_edge("A","A_done",LinkType.SUB)
    g.add_edge("B","B_done",LinkType.SUB)
    g.add_edge("C","C_done",LinkType.SUB)
    g.add_edge("A","B",LinkType.POR)
    g.add_edge("B","C",LinkType.POR)
    return g


def test_sequence_confirms():
    g = build_seq()
    eng = ReConEngine(g)
    g.nodes["ROOT"].state = NodeState.REQUESTED
    for _ in range(10):
        eng.step()
    assert all(n.state == NodeState.CONFIRMED for nid,n in g.nodes.items() if n.ntype == NodeType.SCRIPT)
