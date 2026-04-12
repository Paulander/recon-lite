import json
from typing import List, Dict, Any, Optional


class RunLogger:
    """
    Collects per-tick frames for replay/visualization.

    Frame schema (all optional except tick/nodes):
      {
        "type": "snapshot",
        "tick": int,
        "note": str,
        "nodes": { node_id: state_name, ... },
        "new_requests": [node_id, ...],
        "env": { "fen": str, ... },
        "thoughts": str,
        "latents": { node_id: [float, ...] }
      }
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self._graph_payload: Optional[Dict[str, Any]] = None
        self._graph_emitted = False
        self._last_nodes: Optional[Dict[str, str]] = None
        self._last_tick: Optional[int] = None

    def attach_graph(self, edges: List[Dict[str, Any]]):
        """Provide a static graph description to emit on the next snapshot."""
        self._graph_payload = {"edges": edges}
        self._graph_emitted = False

    def snapshot(
        self,
        engine,
        note: str = "",
        env: Optional[Dict[str, Any]] = None,
        thoughts: Optional[str] = None,
        latents: Optional[Dict[str, Any]] = None,
        new_requests: Optional[List[str]] = None,
        macro: Optional[Dict[str, Any]] = None,
    ):
        frame: Dict[str, Any] = {
            "type": "snapshot",
            "note": note,
        }

        # Only include engine-dependent fields if engine is provided
        if engine is not None:
            frame["tick"] = engine.tick
            nodes_snapshot = {nid: n.state.name for nid, n in engine.g.nodes.items()}
            frame["nodes"] = nodes_snapshot
            self._last_nodes = dict(nodes_snapshot)
            self._last_tick = engine.tick
            if self._graph_payload and not self._graph_emitted:
                frame["graph"] = self._graph_payload
                self._graph_emitted = True
        elif self._last_nodes is not None:
            frame["nodes"] = dict(self._last_nodes)
            if self._last_tick is not None:
                frame["tick"] = self._last_tick

        if new_requests is not None:
            frame["new_requests"] = list(new_requests)
        if env is not None:
            frame["env"] = env
        if thoughts is not None:
            frame["thoughts"] = thoughts
        if latents is not None:
            frame["latents"] = latents
        if macro is not None:
            frame["macro_frame"] = macro
        self.events.append(frame)
        
    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)
