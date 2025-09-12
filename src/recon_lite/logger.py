import json
from typing import List, Dict, Any


class RunLogger:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log(self, **kwargs):
        self.events.append(kwargs)

    def snapshot(self, engine, note: str = ""):
        snap = engine.snapshot(note=note)
        self.events.append({**snap, "type": "snapshot"})

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)
