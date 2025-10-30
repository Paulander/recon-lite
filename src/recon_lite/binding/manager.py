"""
Binding management for ReCoN-lite demos.

Bindings associate feature instances with concrete squares/pieces so that
multiple hypotheses do not silently reuse the same resources. Namespaces keep
bindings for separate scripts independent.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Set

import chess


@dataclass(frozen=True)
class BindingInstance:
    feature_id: str
    items: FrozenSet[str]

    def __post_init__(self):
        if not self.feature_id:
            raise ValueError("BindingInstance requires a non-empty feature_id")
        if not self.items:
            raise ValueError("BindingInstance requires at least one item")
        object.__setattr__(self, "items", frozenset(self.items))

    def to_dict(self) -> Dict[str, object]:
        return {"feature": self.feature_id, "items": sorted(self.items)}


class _NamespaceSession:
    def __init__(
        self,
        table: "BindingTable",
        namespace: str,
        base_instances: Iterable[BindingInstance],
    ):
        self._table = table
        self._namespace = namespace
        self._pending: List[BindingInstance] = list(base_instances)

    # Public API ---------------------------------------------------------
    def reserve(self, instance: BindingInstance) -> bool:
        """Attempt to add `instance` to the namespace; returns False on conflicts."""
        if self.conflicts(instance):
            return False
        self._pending.append(instance)
        return True

    def conflicts(self, candidate: BindingInstance) -> List[BindingInstance]:
        overlaps: List[BindingInstance] = []
        for existing in self._pending:
            if existing.items & candidate.items:
                overlaps.append(existing)
        return overlaps

    def snapshot(self) -> List[Dict[str, object]]:
        return [inst.to_dict() for inst in self._pending]

    # Internal helpers ---------------------------------------------------
    def _commit(self) -> None:
        self._table._namespaces[self._namespace] = list(self._pending)


class BindingTable:
    """
    Keeps track of feature bindings per namespace. Each namespace gets its own
    list of BindingInstance objects. A table persists across plies until
    `invalidate_on_board_change` is called.
    """

    def __init__(self):
        self._namespaces: Dict[str, List[BindingInstance]] = defaultdict(list)
        self._board_signature: Optional[str] = None

    def clear(self) -> None:
        self._namespaces.clear()

    @contextmanager
    def begin_tick(self, namespace: str) -> Iterator[_NamespaceSession]:
        base = list(self._namespaces.get(namespace, []))
        session = _NamespaceSession(self, namespace, base)
        try:
            yield session
        finally:
            session._commit()

    def invalidate_on_board_change(self, board: chess.Board) -> bool:
        """
        Reset the table if the observed board changes. Returns True when an
        invalidation occurred.
        """
        signature = board.board_fen()
        if self._board_signature is None:
            self._board_signature = signature
            return False
        if signature != self._board_signature:
            self._board_signature = signature
            self.clear()
            return True
        return False

    def snapshot(self) -> Dict[str, List[Dict[str, object]]]:
        return {ns: [inst.to_dict() for inst in instances] for ns, instances in self._namespaces.items()}

    def namespaces(self) -> Iterable[str]:
        return list(self._namespaces.keys())
