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

    def snapshot(
        self,
        engine,
        note: str = "",
        env: Optional[Dict[str, Any]] = None,
        thoughts: Optional[str] = None,
        latents: Optional[Dict[str, Any]] = None,
        new_requests: Optional[List[str]] = None,
    ):
        frame: Dict[str, Any] = {
            "type": "snapshot",
            "note": note,
        }

        # Only include engine-dependent fields if engine is provided
        if engine is not None:
            frame["tick"] = engine.tick
            frame["nodes"] = {nid: n.state.name for nid, n in engine.g.nodes.items()}

        if new_requests is not None:
            frame["new_requests"] = list(new_requests)
        if env is not None:
            frame["env"] = env
        if thoughts is not None:
            frame["thoughts"] = thoughts
        if latents is not None:
            frame["latents"] = latents
        self.events.append(frame)
        
    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)
