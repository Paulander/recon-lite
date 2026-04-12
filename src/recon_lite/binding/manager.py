"""
Binding management for ReCoN-lite examples.

Bindings associate feature instances with concrete squares/pieces so that
multiple hypotheses do not silently reuse the same resources. Namespaces keep
bindings for separate scripts independent.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class BindingInstance:
    feature_id: str
    items: FrozenSet[str]
    node_id: Optional[str] = None

    def __post_init__(self):
        if not self.feature_id:
            raise ValueError("BindingInstance requires a non-empty feature_id")
        if not self.items:
            raise ValueError("BindingInstance requires at least one item")
        object.__setattr__(self, "items", frozenset(self.items))

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "feature": self.feature_id,
            "items": sorted(self.items),
        }
        if self.node_id:
            payload["id"] = self.node_id
            payload["terminal_id"] = self.node_id
        return payload


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
    list of BindingInstance objects. A table persists across environment states
    until an invalidation method observes a changed signature.
    """

    def __init__(self):
        self._namespaces: Dict[str, List[BindingInstance]] = defaultdict(list)
        self._signature: Optional[str] = None

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

    def invalidate_on_signature(self, signature: str) -> bool:
        """
        Reset the table if the observed signature changes. Returns True when an
        invalidation occurred.
        """
        if self._signature is None:
            changed = bool(self._namespaces)
            self._signature = signature
            if changed:
                self.clear()
            return changed
        if signature != self._signature:
            self._signature = signature
            self.clear()
            return True
        return False

    def invalidate_on_object(self, obj: Any, signature_fn: Callable[[Any], str]) -> bool:
        """Optional convenience wrapper for deriving a signature from an object."""
        return self.invalidate_on_signature(str(signature_fn(obj)))

    def invalidate_on_board_change(self, board: Any) -> bool:
        """
        Backward-compatible convenience wrapper for board-like objects.

        Prefer `invalidate_on_signature` in new code.
        """
        if hasattr(board, "board_fen"):
            signature = board.board_fen()
        elif hasattr(board, "fen"):
            signature = board.fen()
        else:
            signature = str(board)
        return self.invalidate_on_signature(signature)

    def snapshot(self) -> Dict[str, List[Dict[str, object]]]:
        return {ns: [inst.to_dict() for inst in instances] for ns, instances in self._namespaces.items()}

    def namespaces(self) -> Iterable[str]:
        return list(self._namespaces.keys())
