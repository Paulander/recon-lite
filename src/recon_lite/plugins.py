from typing import Any, Protocol, Tuple

class TerminalPlugin(Protocol):
    def reset(self): ...
    def step(self, node, env: dict) -> Tuple[bool, bool]:
        """Return (done, success). Called each tick while WAITING."""

class PredicatePlugin:
    """Small adapter for callables that already return (done, success)."""

    def __init__(self, predicate):
        self.predicate = predicate

    def reset(self):
        pass

    def step(self, node: Any, env: dict) -> Tuple[bool, bool]:
        return self.predicate(node, env)
