"""M6 Dynamics modules for plan persistence and activation management.

This package provides:
- Persistence: Inertia and decay for plan activation (hysteresis)
- Interrupt: Signals that can override plan persistence
"""

from .persistence import (
    PersistenceConfig,
    PersistenceState,
    update_persistence,
    apply_persistence_to_node,
    create_interrupt_terminal,
    InterruptType,
)

__all__ = [
    "PersistenceConfig",
    "PersistenceState",
    "update_persistence",
    "apply_persistence_to_node",
    "create_interrupt_terminal",
    "InterruptType",
]

