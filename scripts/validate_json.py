#!/usr/bin/env python3
import json
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "topologies/krk_legs_topology.json"
try:
    with open(path) as f:
        json.load(f)
    print(f"JSON valid: {path}")
except json.JSONDecodeError as e:
    print(f"JSON ERROR: {e}")
    sys.exit(1)

