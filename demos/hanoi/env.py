"""Tower of Hanoi environment used by the ReCon demo."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Hanoi:
    """Minimal Tower of Hanoi environment with deterministic transitions."""

    n: int
    pegs: List[List[int]] = field(init=False)
    moves: int = field(default=0, init=False)
    history: List[Tuple[int, int]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("Hanoi requires at least one disc")
        # Peg A starts with all discs (largest at bottom, smallest at top).
        self.pegs = [list(reversed(range(1, self.n + 1))), [], []]

    def legal(self, src: int, dst: int) -> bool:
        """Return True iff moving the top disc from src to dst is legal."""
        if src == dst:
            return False
        if not (0 <= src < 3 and 0 <= dst < 3):
            return False
        if not self.pegs[src]:
            return False
        if not self.pegs[dst]:
            return True
        return self.pegs[src][-1] < self.pegs[dst][-1]

    def move(self, src: int, dst: int) -> bool:
        """Attempt to move a disc; returns True on success, False otherwise."""
        if not self.legal(src, dst):
            return False
        disc = self.pegs[src].pop()
        self.pegs[dst].append(disc)
        self.moves += 1
        self.history.append((src, dst))
        return True

    def is_goal(self) -> bool:
        """Check whether all discs have been moved to the final peg."""
        return not self.pegs[0] and not self.pegs[1] and len(self.pegs[2]) == self.n

    def __str__(self) -> str:
        """Render the pegs as ASCII rows (top row first)."""
        levels = []
        for level in range(self.n - 1, -1, -1):
            row = []
            for peg in self.pegs:
                if len(peg) > level:
                    row.append(str(peg[level]).rjust(2))
                else:
                    row.append(" |")
            levels.append("  ".join(row))
        labels = "  ".join(name for name in ("A", "B", "C"))
        return "\n".join(levels + [labels])
