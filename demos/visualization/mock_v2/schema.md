# mock_v2 Dataset Schema

This document formalises the JSON payload expected by the mock_v2 generator and viewer.

## Root Object

```
{
  "topology": Topology,
  "frames": Frame[],
  "deltas": Delta[]?   // optional
}
```

- `topology` is optional when reloading frames-only updates. When absent, the viewer infers a minimal topology from the first frame.
- Additional producer-specific keys are ignored by the viewer.

## Topology

```
Topology {
  nodes: Node[],
  edges: Edge[]
}

Node {
  id: string,                    // unique identifier used by frames/deltas
  label?: string,                // optional display label
  type: "SCRIPT" | "TERMINAL",
  pos?: [ number, number, number ] // optional fixed position (x, y, z)
}

Edge {
  from: string,                  // upstream node id
  to: string,                    // downstream node id
  type: "sub" | "por",
  label?: string                 // optional descriptor
}
```

The mock_v2 viewer ignores edges for performance but keeps them for forward compatibility.

## Frames

```
Frame {
  tick: number,
  note?: string,
  thoughts?: string,
  env?: {
    style?: string,
    risk?: number,                // 0.0–1.0 recommended
    phase_weights?: Record<string, number>
  },
  new_requests?: string[],        // node ids transitioning into REQUESTED this tick
  states: Record<string, STATE>   // per-node snapshot
}
```

`states` must include every node id that is visible in the topology; missing nodes default to `INACTIVE` when rendered.

## Deltas (Optional)

```
Delta {
  tick: number,
  note?: string,
  thoughts?: string,
  new_requests?: string[],
  changes: Record<string, STATE>  // only nodes that changed since last frame
}
```

`deltas` provide a compressed stream for future streaming or instanced-mesh updates. The current viewer disregards the array but preserves it when present.

## Runtime States

`STATE` is one of:

- `INACTIVE`
- `REQUESTED`
- `WAITING`
- `TRUE`
- `CONFIRMED`
- `FAILED`

The viewer maps each value to a colour and simple pulse animation (for `new_requests`).

## Example Snippet

```json
{
  "topology": {
    "nodes": [
      { "id": "script.root", "label": "ROOT", "type": "SCRIPT", "pos": [0, 0, 0] },
      { "id": "terminal.tactic.fork.001", "label": "FORK_001", "type": "TERMINAL" }
    ],
    "edges": [
      { "from": "script.root", "to": "terminal.tactic.fork.001", "type": "sub" }
    ]
  },
  "frames": [
    {
      "tick": 0,
      "env": { "style": "BALANCED", "risk": 0.42, "phase_weights": { "opening": 0.4, "endgame": 0.2 } },
      "states": {
        "script.root": "INACTIVE",
        "terminal.tactic.fork.001": "REQUESTED"
      },
      "new_requests": ["terminal.tactic.fork.001"]
    }
  ]
}
```

## Compatibility & Extensions

- The viewer tolerates `frames` supplied directly as an array (`[Frame, …]`) or wrapped in an object (`{ "frames": [...] }`).
- Future extensions may stream `deltas` while keeping a fixed `topology`.
- Only ASCII is required for identifiers; labels can include UTF-8 if needed by downstream tooling.
