"""
Binding NodeSpace - MicroPsi2-style grouped representations.

Part of Bach-Integrated architecture. NodeSpaces group related nodes
that share bound variables, enabling domain transfer (KPK -> KRK).

Based on Joscha Bach's concept of "Binding" phase where scripts
organize in working memory to simulate a specific scene.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum, auto


class BindingState(Enum):
    """State of a NodeSpace binding."""
    UNBOUND = auto()      # No variables bound
    PARTIAL = auto()      # Some variables bound
    BOUND = auto()        # All required variables bound
    ACTIVE = auto()       # Bound and actively processing


@dataclass
class BoundVariable:
    """A variable bound within a NodeSpace."""
    name: str
    value: Any
    var_type: str = "any"  # Type hint: "piece", "square", "distance", etc.
    source: Optional[str] = None  # Where the binding came from
    confidence: float = 1.0  # How certain is this binding
    
    def matches(self, other: "BoundVariable") -> bool:
        """Check if this variable matches another (for transfer)."""
        if self.var_type != other.var_type:
            return False
        # Type-specific matching could be added here
        return True


@dataclass
class BindingNodeSpace:
    """
    MicroPsi2-style NodeSpace for grouped representations.
    
    A NodeSpace groups nodes that operate on shared bound variables.
    This enables:
    - Domain transfer: Same tactical logic, different pieces
    - Scene representation: Current game state as bound variables
    - Hierarchical binding: Parent spaces inherit child bindings
    
    Example:
        king_space = BindingNodeSpace(
            space_id="king_binding",
            bound_variables={"piece_type": "king", "role": "approach"},
            member_nodes=["krk_king_leg", "universal_king_logic"]
        )
    """
    
    space_id: str
    member_nodes: List[str] = field(default_factory=list)
    bound_variables: Dict[str, BoundVariable] = field(default_factory=dict)
    required_bindings: Set[str] = field(default_factory=set)  # Must be bound
    parent_space: Optional[str] = None  # Inherit bindings from parent
    state: BindingState = BindingState.UNBOUND
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def bind(self, var_name: str, value: Any, var_type: str = "any", 
             source: Optional[str] = None, confidence: float = 1.0) -> None:
        """
        Bind a variable for all member nodes.
        
        Args:
            var_name: Name of the variable
            value: Value to bind
            var_type: Type hint for transfer matching
            source: Where the binding came from
            confidence: Binding confidence (0-1)
        """
        self.bound_variables[var_name] = BoundVariable(
            name=var_name,
            value=value,
            var_type=var_type,
            source=source,
            confidence=confidence
        )
        self._update_state()
    
    def unbind(self, var_name: str) -> bool:
        """Remove a binding."""
        if var_name in self.bound_variables:
            del self.bound_variables[var_name]
            self._update_state()
            return True
        return False
    
    def get_binding(self, var_name: str) -> Optional[Any]:
        """Get a bound variable's value."""
        bound = self.bound_variables.get(var_name)
        return bound.value if bound else None
    
    def get_binding_info(self, var_name: str) -> Optional[BoundVariable]:
        """Get full binding info including confidence."""
        return self.bound_variables.get(var_name)
    
    def is_fully_bound(self) -> bool:
        """Check if all required bindings are present."""
        return all(
            req in self.bound_variables 
            for req in self.required_bindings
        )
    
    def _update_state(self) -> None:
        """Update binding state based on current bindings."""
        if not self.bound_variables:
            self.state = BindingState.UNBOUND
        elif self.is_fully_bound():
            self.state = BindingState.BOUND
        else:
            self.state = BindingState.PARTIAL
    
    def activate(self) -> bool:
        """Activate the space for processing."""
        if self.state == BindingState.BOUND:
            self.state = BindingState.ACTIVE
            return True
        return False
    
    def deactivate(self) -> None:
        """Deactivate the space."""
        if self.state == BindingState.ACTIVE:
            self.state = BindingState.BOUND
    
    def add_member(self, node_id: str) -> None:
        """Add a node to this space."""
        if node_id not in self.member_nodes:
            self.member_nodes.append(node_id)
    
    def remove_member(self, node_id: str) -> bool:
        """Remove a node from this space."""
        if node_id in self.member_nodes:
            self.member_nodes.remove(node_id)
            return True
        return False
    
    def get_binding_context(self) -> Dict[str, Any]:
        """
        Get all bindings as a simple dict for node evaluation.
        
        This is passed to node predicates as part of 'env'.
        """
        return {
            name: bv.value 
            for name, bv in self.bound_variables.items()
        }
    
    def transfer_compatible(self, other: "BindingNodeSpace") -> float:
        """
        Check compatibility for domain transfer.
        
        Returns a score 0-1 indicating how compatible the spaces are.
        Used for KPK -> KRK knowledge transfer.
        """
        if not self.bound_variables or not other.bound_variables:
            return 0.0
        
        # Find matching variable types
        matches = 0
        total = len(self.bound_variables)
        
        for var_name, bound_var in self.bound_variables.items():
            # Check if other space has a variable of same type
            for other_var in other.bound_variables.values():
                if bound_var.matches(other_var):
                    matches += 1
                    break
        
        return matches / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "space_id": self.space_id,
            "member_nodes": self.member_nodes,
            "bound_variables": {
                name: {
                    "name": bv.name,
                    "value": bv.value,
                    "var_type": bv.var_type,
                    "source": bv.source,
                    "confidence": bv.confidence
                }
                for name, bv in self.bound_variables.items()
            },
            "required_bindings": list(self.required_bindings),
            "parent_space": self.parent_space,
            "state": self.state.name,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BindingNodeSpace":
        """Deserialize from dict."""
        space = cls(
            space_id=data["space_id"],
            member_nodes=data.get("member_nodes", []),
            required_bindings=set(data.get("required_bindings", [])),
            parent_space=data.get("parent_space"),
            metadata=data.get("metadata", {})
        )
        
        # Restore bindings
        for name, bv_data in data.get("bound_variables", {}).items():
            space.bound_variables[name] = BoundVariable(
                name=bv_data["name"],
                value=bv_data["value"],
                var_type=bv_data.get("var_type", "any"),
                source=bv_data.get("source"),
                confidence=bv_data.get("confidence", 1.0)
            )
        
        # Restore state
        state_name = data.get("state", "UNBOUND")
        space.state = BindingState[state_name]
        
        return space


class BindingSpaceRegistry:
    """
    Registry for all NodeSpaces in a graph.
    
    Manages creation, lookup, and transfer of binding spaces.
    """
    
    def __init__(self):
        self.spaces: Dict[str, BindingNodeSpace] = {}
        self.node_to_space: Dict[str, str] = {}  # node_id -> space_id
    
    def create_space(
        self,
        space_id: str,
        member_nodes: Optional[List[str]] = None,
        required_bindings: Optional[Set[str]] = None,
        parent_space: Optional[str] = None,
        **metadata
    ) -> BindingNodeSpace:
        """Create and register a new NodeSpace."""
        space = BindingNodeSpace(
            space_id=space_id,
            member_nodes=member_nodes or [],
            required_bindings=required_bindings or set(),
            parent_space=parent_space,
            metadata=metadata
        )
        
        self.spaces[space_id] = space
        
        # Register node memberships
        for node_id in space.member_nodes:
            self.node_to_space[node_id] = space_id
        
        return space
    
    def get_space(self, space_id: str) -> Optional[BindingNodeSpace]:
        """Get a space by ID."""
        return self.spaces.get(space_id)
    
    def get_space_for_node(self, node_id: str) -> Optional[BindingNodeSpace]:
        """Get the space a node belongs to."""
        space_id = self.node_to_space.get(node_id)
        return self.spaces.get(space_id) if space_id else None
    
    def add_node_to_space(self, node_id: str, space_id: str) -> bool:
        """Add a node to a space."""
        space = self.spaces.get(space_id)
        if not space:
            return False
        
        space.add_member(node_id)
        self.node_to_space[node_id] = space_id
        return True
    
    def get_binding_context_for_node(self, node_id: str) -> Dict[str, Any]:
        """
        Get all bindings relevant to a node.
        
        Includes bindings from parent spaces.
        """
        space = self.get_space_for_node(node_id)
        if not space:
            return {}
        
        context = {}
        
        # Add parent space bindings first (can be overridden)
        if space.parent_space:
            parent = self.spaces.get(space.parent_space)
            if parent:
                context.update(parent.get_binding_context())
        
        # Add this space's bindings
        context.update(space.get_binding_context())
        
        return context
    
    def find_transfer_targets(
        self, 
        source_space: BindingNodeSpace,
        min_compatibility: float = 0.5
    ) -> List[tuple]:
        """
        Find spaces compatible for knowledge transfer.
        
        Returns list of (space_id, compatibility_score) tuples.
        """
        targets = []
        for space_id, space in self.spaces.items():
            if space_id == source_space.space_id:
                continue
            
            compat = source_space.transfer_compatible(space)
            if compat >= min_compatibility:
                targets.append((space_id, compat))
        
        return sorted(targets, key=lambda x: x[1], reverse=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry."""
        return {
            "spaces": {
                sid: space.to_dict() 
                for sid, space in self.spaces.items()
            },
            "node_to_space": self.node_to_space
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BindingSpaceRegistry":
        """Deserialize registry."""
        registry = cls()
        
        for space_id, space_data in data.get("spaces", {}).items():
            registry.spaces[space_id] = BindingNodeSpace.from_dict(space_data)
        
        registry.node_to_space = data.get("node_to_space", {})
        
        return registry


# Convenience function for creating domain-specific spaces
def create_chess_endgame_spaces(
    registry: BindingSpaceRegistry,
    endgame_type: str = "krk"
) -> Dict[str, BindingNodeSpace]:
    """
    Create standard binding spaces for a chess endgame.
    
    Args:
        registry: The registry to add spaces to
        endgame_type: "kpk" or "krk"
        
    Returns:
        Dict mapping space names to created spaces
    """
    spaces = {}
    
    # King space (universal across endgames)
    king_space = registry.create_space(
        space_id=f"{endgame_type}_king_binding",
        required_bindings={"piece_type", "target_distance"},
        description="King movement and approach patterns"
    )
    king_space.bind("piece_type", "king", var_type="piece")
    king_space.bind("role", "approach", var_type="role")
    spaces["king"] = king_space
    
    if endgame_type == "krk":
        # Rook space (KRK specific)
        rook_space = registry.create_space(
            space_id="krk_rook_binding",
            required_bindings={"piece_type", "cut_rank", "cut_file"},
            description="Rook cut-off and box shrinking patterns"
        )
        rook_space.bind("piece_type", "rook", var_type="piece")
        rook_space.bind("role", "cut", var_type="role")
        spaces["rook"] = rook_space
        
    elif endgame_type == "kpk":
        # Pawn space (KPK specific)
        pawn_space = registry.create_space(
            space_id="kpk_pawn_binding",
            required_bindings={"piece_type", "promotion_distance"},
            description="Pawn pushing and promotion patterns"
        )
        pawn_space.bind("piece_type", "pawn", var_type="piece")
        pawn_space.bind("role", "promote", var_type="role")
        spaces["pawn"] = pawn_space
    
    return spaces

