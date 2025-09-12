export const frames = [
  {
    "type": "snapshot",
    "tick": 0,
    "note": "start",
    "nodes": {
      "ROOT": "REQUESTED",
      "A": "INACTIVE",
      "B1": "INACTIVE",
      "B2": "INACTIVE",
      "C": "INACTIVE",
      "A_done": "INACTIVE",
      "B1_done": "INACTIVE",
      "B2_done": "INACTIVE",
      "C_done": "INACTIVE"
    },
    "env": {},
    "thoughts": "Initiate ROOT"
  },
  {
    "type": "snapshot",
    "tick": 1,
    "note": "tick 1",
    "nodes": {
      "ROOT": "WAITING",
      "A": "WAITING",
      "B1": "INACTIVE",
      "B2": "INACTIVE",
      "C": "INACTIVE",
      "A_done": "REQUESTED",
      "B1_done": "INACTIVE",
      "B2_done": "INACTIVE",
      "C_done": "INACTIVE"
    },
    "new_requests": [
      "A",
      "A_done"
    ],
    "env": {},
    "thoughts": "tick 1: requested ['A', 'A_done']"
  },
  {
    "type": "snapshot",
    "tick": 2,
    "note": "tick 2",
    "nodes": {
      "ROOT": "WAITING",
      "A": "WAITING",
      "B1": "INACTIVE",
      "B2": "INACTIVE",
      "C": "INACTIVE",
      "A_done": "WAITING",
      "B1_done": "INACTIVE",
      "B2_done": "INACTIVE",
      "C_done": "INACTIVE"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 2: requested []"
  },
  {
    "type": "snapshot",
    "tick": 3,
    "note": "tick 3",
    "nodes": {
      "ROOT": "WAITING",
      "A": "WAITING",
      "B1": "INACTIVE",
      "B2": "INACTIVE",
      "C": "INACTIVE",
      "A_done": "TRUE",
      "B1_done": "INACTIVE",
      "B2_done": "INACTIVE",
      "C_done": "INACTIVE"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 3: requested []"
  },
  {
    "type": "snapshot",
    "tick": 4,
    "note": "tick 4",
    "nodes": {
      "ROOT": "WAITING",
      "A": "CONFIRMED",
      "B1": "INACTIVE",
      "B2": "INACTIVE",
      "C": "INACTIVE",
      "A_done": "CONFIRMED",
      "B1_done": "INACTIVE",
      "B2_done": "INACTIVE",
      "C_done": "INACTIVE"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 4: requested []"
  },
  {
    "type": "snapshot",
    "tick": 5,
    "note": "tick 5",
    "nodes": {
      "ROOT": "WAITING",
      "A": "CONFIRMED",
      "B1": "WAITING",
      "B2": "WAITING",
      "C": "INACTIVE",
      "A_done": "CONFIRMED",
      "B1_done": "REQUESTED",
      "B2_done": "REQUESTED",
      "C_done": "INACTIVE"
    },
    "new_requests": [
      "B1",
      "B2",
      "B1_done",
      "B2_done"
    ],
    "env": {},
    "thoughts": "tick 5: requested ['B1', 'B2', 'B1_done', 'B2_done']"
  },
  {
    "type": "snapshot",
    "tick": 6,
    "note": "tick 6",
    "nodes": {
      "ROOT": "WAITING",
      "A": "CONFIRMED",
      "B1": "WAITING",
      "B2": "WAITING",
      "C": "INACTIVE",
      "A_done": "CONFIRMED",
      "B1_done": "WAITING",
      "B2_done": "WAITING",
      "C_done": "INACTIVE"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 6: requested []"
  },
  {
    "type": "snapshot",
    "tick": 7,
    "note": "tick 7",
    "nodes": {
      "ROOT": "WAITING",
      "A": "CONFIRMED",
      "B1": "WAITING",
      "B2": "WAITING",
      "C": "INACTIVE",
      "A_done": "CONFIRMED",
      "B1_done": "WAITING",
      "B2_done": "TRUE",
      "C_done": "INACTIVE"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 7: requested []"
  },
  {
    "type": "snapshot",
    "tick": 8,
    "note": "tick 8",
    "nodes": {
      "ROOT": "WAITING",
      "A": "CONFIRMED",
      "B1": "WAITING",
      "B2": "CONFIRMED",
      "C": "INACTIVE",
      "A_done": "CONFIRMED",
      "B1_done": "TRUE",
      "B2_done": "CONFIRMED",
      "C_done": "INACTIVE"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 8: requested []"
  },
  {
    "type": "snapshot",
    "tick": 9,
    "note": "tick 9",
    "nodes": {
      "ROOT": "WAITING",
      "A": "CONFIRMED",
      "B1": "CONFIRMED",
      "B2": "CONFIRMED",
      "C": "INACTIVE",
      "A_done": "CONFIRMED",
      "B1_done": "CONFIRMED",
      "B2_done": "CONFIRMED",
      "C_done": "INACTIVE"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 9: requested []"
  },
  {
    "type": "snapshot",
    "tick": 10,
    "note": "tick 10",
    "nodes": {
      "ROOT": "WAITING",
      "A": "CONFIRMED",
      "B1": "CONFIRMED",
      "B2": "CONFIRMED",
      "C": "WAITING",
      "A_done": "CONFIRMED",
      "B1_done": "CONFIRMED",
      "B2_done": "CONFIRMED",
      "C_done": "REQUESTED"
    },
    "new_requests": [
      "C",
      "C_done"
    ],
    "env": {},
    "thoughts": "tick 10: requested ['C', 'C_done']"
  },
  {
    "type": "snapshot",
    "tick": 11,
    "note": "tick 11",
    "nodes": {
      "ROOT": "WAITING",
      "A": "CONFIRMED",
      "B1": "CONFIRMED",
      "B2": "CONFIRMED",
      "C": "WAITING",
      "A_done": "CONFIRMED",
      "B1_done": "CONFIRMED",
      "B2_done": "CONFIRMED",
      "C_done": "WAITING"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 11: requested []"
  },
  {
    "type": "snapshot",
    "tick": 12,
    "note": "tick 12",
    "nodes": {
      "ROOT": "WAITING",
      "A": "CONFIRMED",
      "B1": "CONFIRMED",
      "B2": "CONFIRMED",
      "C": "WAITING",
      "A_done": "CONFIRMED",
      "B1_done": "CONFIRMED",
      "B2_done": "CONFIRMED",
      "C_done": "TRUE"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 12: requested []"
  },
  {
    "type": "snapshot",
    "tick": 13,
    "note": "tick 13",
    "nodes": {
      "ROOT": "WAITING",
      "A": "CONFIRMED",
      "B1": "CONFIRMED",
      "B2": "CONFIRMED",
      "C": "CONFIRMED",
      "A_done": "CONFIRMED",
      "B1_done": "CONFIRMED",
      "B2_done": "CONFIRMED",
      "C_done": "CONFIRMED"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 13: requested []"
  },
  {
    "type": "snapshot",
    "tick": 14,
    "note": "tick 14",
    "nodes": {
      "ROOT": "CONFIRMED",
      "A": "CONFIRMED",
      "B1": "CONFIRMED",
      "B2": "CONFIRMED",
      "C": "CONFIRMED",
      "A_done": "CONFIRMED",
      "B1_done": "CONFIRMED",
      "B2_done": "CONFIRMED",
      "C_done": "CONFIRMED"
    },
    "new_requests": [],
    "env": {},
    "thoughts": "tick 14: requested []"
  }
];

export const stateColors = {
    'INACTIVE': 0x888888,
    'REQUESTED': 0x4169E1,
    'WAITING': 0xFF8C00,
    'TRUE': 0x32CD32,
    'CONFIRMED': 0x228B22,
    'FAILED': 0xFF0000
};

export const nodePositions = {
    'ROOT': [0, 5, 0],
    'A': [0, 3, 0],
    'B1': [-2, 1, 0],
    'B2': [2, 1, 0],
    'C': [0, -1, 0],
    'A_done': [0, 2, -1],
    'B1_done': [-2, 0, -1],
    'B2_done': [2, 0, -1],
    'C_done': [0, -2, -1]
};

export const edges = [
    {from: 'ROOT', to: 'A', type: 'sub', label: 'sub'},
    {from: 'A', to: 'A_done', type: 'sub', label: 'sub'},
    {from: 'B1', to: 'B1_done', type: 'sub', label: 'sub'},
    {from: 'B2', to: 'B2_done', type: 'sub', label: 'sub'},
    {from: 'C', to: 'C_done', type: 'sub', label: 'sub'},
    {from: 'A', to: 'B1', type: 'por', label: 'por'},
    {from: 'A', to: 'B2', type: 'por', label: 'por'},
    {from: 'B1', to: 'C', type: 'por', label: 'por'},
    {from: 'B2', to: 'C', type: 'por', label: 'por'}
];

export const nodeIds = Object.keys(nodePositions);