"""Features subpackage for chess feature extraction."""

from .kpk_features import (
    extract_kpk_features,
    create_feature_extractor,
    get_kpk_pieces,
    FEATURE_NAMES,
    features_to_dict,
)

from .krk_features import (
    extract_krk_features,
    extract_krk_feature_dict,
    KRKFeatures,
    krk_feature_similarity,
    universal_feature_match,
    extract_move_features,
)

__all__ = [
    # KPK features
    "extract_kpk_features",
    "create_feature_extractor",
    "get_kpk_pieces",
    "FEATURE_NAMES",
    "features_to_dict",
    # KRK features
    "extract_krk_features",
    "extract_krk_feature_dict",
    "KRKFeatures",
    "krk_feature_similarity",
    "universal_feature_match",
    "extract_move_features",
]
