"""
Integration layer for FeatureHub with ReCoN graph system.

This module provides functions to:
1. Wire FeatureHub into the unified graph
2. Create sensor nodes that subscribe to hub features
3. Enable feature-based node activation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import chess

from recon_lite import Node, NodeType, Graph

from .hub import FeatureHub, FeatureCategory, create_default_hub


class FeatureHubIntegration:
    """
    Integration layer between FeatureHub and ReCoN graph.
    
    Manages feature computation per tick and routes features to subscribing nodes.
    """
    
    def __init__(self, hub: Optional[FeatureHub] = None):
        self.hub = hub or create_default_hub()
        self._node_subscriptions: Dict[str, List[str]] = {}  # node_id -> [feature_names]
    
    def subscribe_node(self, node_id: str, features: List[str]) -> None:
        """
        Subscribe a node to multiple features.
        
        Args:
            node_id: The node to subscribe
            features: List of feature names to subscribe to
        """
        self._node_subscriptions[node_id] = features
        for feature_name in features:
            self.hub.subscribe(feature_name, node_id)
    
    def get_node_features(self, node_id: str) -> Dict[str, float]:
        """
        Get all subscribed features for a node.
        
        Args:
            node_id: The node to get features for
            
        Returns:
            Dict of feature_name -> value for subscribed features
        """
        subscribed = self._node_subscriptions.get(node_id, [])
        return {name: self.hub.get(name) for name in subscribed}
    
    def compute_for_board(self, board: chess.Board) -> Dict[str, float]:
        """
        Compute all features for current board.
        
        Args:
            board: Current chess position
            
        Returns:
            Dict of all feature values
        """
        return self.hub.compute_all(board)
    
    def update_env(self, env: Dict[str, Any], board: chess.Board) -> None:
        """
        Update environment dict with computed features.
        
        Args:
            env: Environment dict to update
            board: Current board position
        """
        features = self.hub.compute_all(board)
        env["feature_hub"] = features
        
        # Also populate categorical summaries
        env["tactical_features"] = self.hub.get_by_category(FeatureCategory.TACTICAL)
        env["positional_features"] = self.hub.get_by_category(FeatureCategory.POSITIONAL)
        env["phase_features"] = self.hub.get_by_category(FeatureCategory.PHASE)


def create_feature_sensor_node(
    node_id: str,
    feature_name: str,
    threshold: float = 0.5,
    hub: Optional[FeatureHub] = None,
) -> Node:
    """
    Create a terminal node that senses a specific feature.
    
    The node activates when the feature value exceeds the threshold.
    
    Args:
        node_id: ID for the new node
        feature_name: Name of feature to sense
        threshold: Activation threshold
        hub: FeatureHub instance (will be bound via closure)
        
    Returns:
        A new terminal Node
    """
    _hub = hub
    
    def predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        # Try to get feature from env first (computed by integration layer)
        features = env.get("feature_hub", {})
        value = features.get(feature_name, 0.0)
        
        # If not in env and we have a hub reference, compute it
        if not features and _hub is not None:
            board = env.get("board")
            if board:
                value = _hub.get(feature_name)
        
        # Store in node meta for inspection
        node.meta["feature_value"] = value
        node.meta["feature_name"] = feature_name
        
        # Update activation
        node.activation.value = value
        
        # Return (is_done, is_success)
        is_active = value >= threshold
        return True, is_active
    
    return Node(
        nid=node_id,
        ntype=NodeType.TERMINAL,
        predicate=predicate,
        meta={
            "feature_name": feature_name,
            "threshold": threshold,
            "layer": "feature_sensor",
            "fan_in_allowed": True,  # Can be queried by multiple parents
        },
    )


def create_tactical_feature_sensors(hub: Optional[FeatureHub] = None) -> List[Node]:
    """
    Create sensor nodes for all tactical features.
    
    Args:
        hub: Optional FeatureHub to bind to
        
    Returns:
        List of feature sensor nodes
    """
    sensors = []
    
    tactical_features = [
        ("feature_fork_available", "fork_available", 0.3),
        ("feature_pin_present", "pin_present", 0.3),
        ("feature_hanging_piece", "hanging_piece", 0.3),
        ("feature_back_rank_vulnerable", "back_rank_vulnerable", 0.4),
        ("feature_discovered_attack", "discovered_attack", 0.3),
        ("feature_double_check", "double_check", 0.5),
        ("feature_skewer", "skewer", 0.3),
    ]
    
    for node_id, feature_name, threshold in tactical_features:
        sensors.append(create_feature_sensor_node(
            node_id=node_id,
            feature_name=feature_name,
            threshold=threshold,
            hub=hub,
        ))
    
    return sensors


def create_positional_feature_sensors(hub: Optional[FeatureHub] = None) -> List[Node]:
    """
    Create sensor nodes for positional features.
    
    Args:
        hub: Optional FeatureHub to bind to
        
    Returns:
        List of feature sensor nodes
    """
    sensors = []
    
    positional_features = [
        ("feature_king_safety", "king_safety", 0.3),
        ("feature_center_control", "center_control", 0.4),
        ("feature_mobility", "mobility", 0.4),
        ("feature_pawn_structure", "pawn_structure", 0.5),
    ]
    
    for node_id, feature_name, threshold in positional_features:
        sensors.append(create_feature_sensor_node(
            node_id=node_id,
            feature_name=feature_name,
            threshold=threshold,
            hub=hub,
        ))
    
    return sensors


def wire_feature_sensors_to_graph(
    graph: Graph,
    hub: Optional[FeatureHub] = None,
    parent_id: str = "GameRoot",
) -> List[str]:
    """
    Add feature sensor nodes to an existing graph.
    
    Args:
        graph: The graph to add sensors to
        hub: Optional FeatureHub to bind
        parent_id: Parent node to connect sensors under
        
    Returns:
        List of created node IDs
    """
    from recon_lite import LinkType
    
    created = []
    
    # Create tactical sensors
    tactical_sensors = create_tactical_feature_sensors(hub)
    for node in tactical_sensors:
        graph.add_node(node)
        graph.add_edge(parent_id, node.nid, LinkType.SUB)
        created.append(node.nid)
    
    # Create positional sensors
    positional_sensors = create_positional_feature_sensors(hub)
    for node in positional_sensors:
        graph.add_node(node)
        graph.add_edge(parent_id, node.nid, LinkType.SUB)
        created.append(node.nid)
    
    return created


# Global integration instance (singleton pattern)
_GLOBAL_INTEGRATION: Optional[FeatureHubIntegration] = None


def get_global_integration() -> FeatureHubIntegration:
    """Get or create the global FeatureHub integration."""
    global _GLOBAL_INTEGRATION
    if _GLOBAL_INTEGRATION is None:
        _GLOBAL_INTEGRATION = FeatureHubIntegration()
    return _GLOBAL_INTEGRATION


def reset_global_integration() -> None:
    """Reset the global integration (useful for testing)."""
    global _GLOBAL_INTEGRATION
    _GLOBAL_INTEGRATION = None

