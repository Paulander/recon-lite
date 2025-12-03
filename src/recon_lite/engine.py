# recon_lite/engine.py
from typing import Dict, Any, Optional
from .graph import Graph, NodeType, NodeState, LinkType
from .time.microtick import MicrotickConfig, run_microticks

class ReConEngine:
    """
    Minimal, discrete-time executor:
    - Parent REQUESTS children via SUB if not inhibited by POR predecessors.
    - POR gating: a node is requestable only if all its POR predecessors are TRUE/CONFIRMED.
    - Parent becomes TRUE when the last node of each POR chain under it is CONFIRMED.
    - TERMINAL nodes use predicate(env) -> (done, success) to progress.
    """

    # Initialize the engine with a graph and set initial tick and log storage
    def __init__(self, graph: Graph):
        # Validate article compliance proactively
        try:
            graph.validate_article_compliance()
        except Exception:
            # If validation utility not present or raises, continue; Graph.add_edge enforces during construction
            pass
        self.g = graph
        self.tick = 0
        self.logs: list[Dict[str, Any]] = []

    # Capture the current state of the network for logging. Should obviously be optional. 
    def snapshot(self, note: str = "") -> Dict[str, Any]:
        snap = {
            "tick": self.tick,
            "note": note,
            "nodes": {nid: n.state.name for nid, n in self.g.nodes.items()}
        }
        self.logs.append(snap)
        return snap

    def _all_por_predecessors_true(self, nid: str) -> bool:
        preds = self.g.predecessors(nid)
        if not preds:
            return True
        return all(self.g.nodes[p].state in (NodeState.TRUE, NodeState.CONFIRMED) for p in preds)

    # Helper: fetch edge weight for a specific link (defaults to 1.0 if absent)
    def _edge_weight(self, src: str, dst: str, ltype: LinkType) -> float:
        for e in self.g.edges:
            if e.ltype == ltype and e.src == src and e.dst == dst:
                try:
                    # Edge.w may be ndarray or scalar-like
                    return float(e.w[0]) if hasattr(e.w, "__len__") else float(e.w)
                except Exception:
                    try:
                        return float(e.w)
                    except Exception:
                        return 1.0
        return 1.0

    # Policy-aware POR gating. Defaults to logical AND for full backward compatibility.
    # Configure on the gated node via meta:
    #   node.meta["por_policy"] ∈ {"and","or","xor","k_of_n","weighted"}
    #   node.meta["por_k"]      (int) for k_of_n
    #   node.meta["por_theta"]  (float) for weighted (sum of satisfied predecessor weights)
    def _por_gate_ready(self, nid: str) -> bool:
        preds = self.g.predecessors(nid)
        if not preds:
            return True
        node = self.g.nodes[nid]
        policy = node.meta.get("por_policy", "and")
        satisfied = [self.g.nodes[p].state in (NodeState.TRUE, NodeState.CONFIRMED) for p in preds]

        if policy == "and":
            return all(satisfied)
        if policy == "or":
            return any(satisfied)
        if policy == "xor":
            return sum(1 for v in satisfied if v) == 1
        if policy == "k_of_n":
            k = int(node.meta.get("por_k", len(preds)))
            return sum(1 for v in satisfied if v) >= k
        if policy == "weighted":
            theta = float(node.meta.get("por_theta", 1.0))
            total = 0.0
            for p, ok in zip(preds, satisfied):
                if ok:
                    total += self._edge_weight(p, nid, LinkType.POR)
            return total >= theta

        # Fallback to legacy behavior
        return all(satisfied)

    def _request_child_if_ready(self, child_id: str, now_requested: Dict[str, bool]):
        child = self.g.nodes[child_id]
        # Use policy-aware POR gating (defaults to AND)
        if child.state == NodeState.INACTIVE and self._por_gate_ready(child_id):
            child.state = NodeState.REQUESTED
            child.tick_entered = self.tick
            now_requested[child_id] = True

    def _children_confirmed_sequence_done(self, parent_id: str) -> bool:
        roots = [c for c in self.g.children(parent_id) if not self.g.predecessors(c)]
        if not roots:
            return True

        policy = self.g.nodes[parent_id].meta.get("confirm_policy", "and")
        satisfied_count = 0

        for r in roots:
            cur = r
            last = cur
            visited = set()
            while True:
                succ = self.g.successors(cur)
                visited.add(cur)
                if not succ:
                    last = cur
                    break
                cur = succ[0]
                if cur in visited:
                    break
            # Per-root override: if root child is marked alt, accept the root when
            # the root script itself reaches CONFIRMED/TRUE.
            if self.g.nodes[r].meta.get("alt", False):
                chain_ok = self.g.nodes[r].state in (NodeState.CONFIRMED, NodeState.TRUE)
            else:
                chain_ok = (self.g.nodes[last].state == NodeState.CONFIRMED)

            if chain_ok:
                satisfied_count += 1
            elif policy == "and":
                # Short-circuit for AND policy
                return False

        if policy == "and":
            return satisfied_count == len(roots)
        if policy == "or":
            return satisfied_count >= 1
        if policy == "xor":
            return satisfied_count == 1
        if policy == "k_of_n":
            k = int(self.g.nodes[parent_id].meta.get("confirm_k", len(roots)))
            return satisfied_count >= k

        # Fallback to legacy behavior (AND)
        return satisfied_count == len(roots)

    def _update_terminals(self, env: Dict[str, Any], now_requested: Dict[str, bool]):
        """Handle state transitions for terminal nodes.
        
        Fan-in terminals (sensors with multiple parents):
        - When confirmed, the confirmation is broadcast to ALL parents
        - Each parent independently evaluates if its children are done
        """
        for nid, node in self.g.nodes.items():
            if node.ntype == NodeType.TERMINAL:
                if node.state == NodeState.REQUESTED:
                    node.state = NodeState.WAITING
                    node.tick_entered = self.tick
                elif node.state == NodeState.WAITING:
                    if node.predicate is None:
                        node.state = NodeState.TRUE
                    else:
                        try:
                            done, success = node.predicate(node, env)
                            if done:
                                node.state = NodeState.TRUE if success else NodeState.FAILED
                        except Exception as e:
                            print(f"Predicate error for {node.nid}: {e}")
                            node.state = NodeState.FAILED
                elif node.state == NodeState.TRUE:
                    node.state = NodeState.CONFIRMED
                    # For fan-in terminals: confirmation is automatically visible
                    # to all parents via the node.state; no special broadcast needed
                    # since _children_confirmed_sequence_done checks child states directly

# Process script nodes to request their children when in a requestable state
    def _process_script_requests(self, now_requested: Dict[str, bool]):
        """Request children for script nodes based on readiness."""
        for nid, node in self.g.nodes.items():
            if node.ntype == NodeType.SCRIPT:
                if node.state == NodeState.REQUESTED:
                    node.state = NodeState.WAITING
                    node.tick_entered = self.tick

                if node.state in (NodeState.REQUESTED, NodeState.WAITING):
                    for child_id in self.g.children(nid):
                        self._request_child_if_ready(child_id, now_requested)

    def _confirm_script_completions(self):
        """Confirm script nodes when all children sequences are done."""
        for nid, node in self.g.nodes.items():
            if node.ntype == NodeType.SCRIPT and node.state in (NodeState.REQUESTED, NodeState.WAITING, NodeState.TRUE):
                if self._children_confirmed_sequence_done(nid):
                    node.state = NodeState.TRUE

        for nid, node in self.g.nodes.items():
            if node.ntype == NodeType.SCRIPT and node.state == NodeState.TRUE:
                node.state = NodeState.CONFIRMED

# Core function.Execute one discrete time step, orchestrating terminal updates, script requests, and confirmations
    def step(self, env: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """Execute one discrete time step of the ReCon network."""
        self.tick += 1
        env = env or {}
        micro_cfg = env.get("microticks")
        micro_history = None

        if isinstance(micro_cfg, MicrotickConfig):
            micro_history = run_microticks(micro_cfg)
        elif isinstance(micro_cfg, dict):
            states = micro_cfg.get("states")
            compute_targets = micro_cfg.get("compute_targets")
            steps = micro_cfg.get("steps", 0)
            eta = micro_cfg.get("eta", 0.3)
            history_flag = bool(micro_cfg.get("history", False))
            if states is not None and compute_targets is not None and steps:
                cfg = MicrotickConfig(
                    states=states,
                    compute_targets=compute_targets,
                    steps=int(steps),
                    eta=float(eta),
                    history=history_flag,
                )
                micro_history = run_microticks(cfg)

        if micro_history is not None:
            env["microtick_history"] = micro_history

        now_requested: Dict[str, bool] = {}

        self._update_terminals(env, now_requested)
        self._process_script_requests(now_requested)
        self._confirm_script_completions()

        self.snapshot() # make optional through either global config or parameter. Maybe possible to set resolution/trigger for log? e.g. "every 10 ticks" or "on request". 
                        # not prioritized for now. Just comment out if performance is an issue. 
        return now_requested
