#!/usr/bin/env python3
"""Compare KPK and KRK feature extractors."""

from recon_lite_chess.features.kpk_features import extract_kpk_features, FEATURE_NAMES as KPK_FEATURE_NAMES
try:
    from recon_lite_chess.features.krk_features import extract_krk_features, KRK_FEATURE_NAMES
except ImportError:
    from recon_lite_chess.features.krk_features import extract_krk_features
    KRK_FEATURE_NAMES = None

print("KPK Features:", len(KPK_FEATURE_NAMES))
for name in KPK_FEATURE_NAMES[:15]:
    print(f"  {name}")

print()
if KRK_FEATURE_NAMES:
    print("KRK Features:", len(KRK_FEATURE_NAMES))
    for name in KRK_FEATURE_NAMES[:15]:
        print(f"  {name}")
else:
    print("KRK Features: (names not exported)")
    # Try to inspect the function
    import chess
    test_board = chess.Board("8/8/8/3k4/8/8/8/4K2R w - - 0 1")
    features = extract_krk_features(test_board)
    print(f"  Feature type: {type(features)}")
    if isinstance(features, dict):
        print(f"  Keys: {list(features.keys())[:15]}")
    elif isinstance(features, list):
        print(f"  Length: {len(features)}")

# Check overlap if both are available
if KRK_FEATURE_NAMES:
    kpk_set = set(KPK_FEATURE_NAMES)
    krk_set = set(KRK_FEATURE_NAMES)
    overlap = kpk_set & krk_set
else:
    overlap = set()

print()
print("Overlapping features:", len(overlap))
for name in sorted(overlap)[:10]:
    print(f"  {name}")

