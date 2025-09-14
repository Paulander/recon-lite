# src/recon_lite/plugins.py
from typing import Protocol, Any, Tuple

class TerminalPlugin(Protocol):
    def reset(self): ...
    def step(self, node, env: dict) -> Tuple[bool, bool]:
        """Return (done, success). Called each tick while WAITING."""

# Example: rule-based chess checkmate detector
class MatePlugin:
    def __init__(self): self.reset()
    def reset(self): pass
    def step(self, node, env):
        board = env["board"]  # python-chess Board
        return True, board.is_checkmate()

# Wiring into a terminal
Node("is_mate", NodeType.TERMINAL, predicate=lambda n, env: MatePlugin().step(n, env))
