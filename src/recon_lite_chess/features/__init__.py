"""
Global FeatureHub for ReCoN Chess.

Provides a centralized feature registry that tactical sensors can subscribe to.
This enables feature hoisting - moving pattern detection from local subgraphs
to a global level where they can be reused across contexts.

Example:
    from recon_lite_chess.features import FeatureHub
    
    hub = FeatureHub()
    hub.compute_all(board)
    
    # Access features
    if hub.get("fork_available") > 0.5:
        # Fork opportunity detected
        pass
    
    # Check subscribers
    for subscriber in hub.get_subscribers("back_rank_vulnerable"):
        # Notify interested nodes
        pass
"""

from .hub import (
    FeatureHub,
    FeatureDefinition,
    FeatureResult,
    FeatureCategory,
    create_default_hub,
)
from .integration import (
    FeatureHubIntegration,
    create_feature_sensor_node,
    create_tactical_feature_sensors,
    create_positional_feature_sensors,
    wire_feature_sensors_to_graph,
    get_global_integration,
    reset_global_integration,
)

__all__ = [
    "FeatureHub",
    "FeatureDefinition",
    "FeatureResult",
    "FeatureCategory",
    "create_default_hub",
    "FeatureHubIntegration",
    "create_feature_sensor_node",
    "create_tactical_feature_sensors",
    "create_positional_feature_sensors",
    "wire_feature_sensors_to_graph",
    "get_global_integration",
    "reset_global_integration",
]

