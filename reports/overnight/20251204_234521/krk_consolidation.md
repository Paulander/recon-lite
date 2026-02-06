# Consolidation Report

Generated: 2025-12-05 00:01:43

**Source file:** `/home/paulander/git/recon-lite/weights/nightly/krk_consol.json`
**Checksum:** `d2b285ef7668`

## Summary

- **Total episodes seen:** 2270
- **Episodes since last apply:** 0
- **Last updated:** 2025-12-05T00:01:42.970829
- **Edges tracked:** 4

## Configuration

- **Learning rate (eta):** 0.01
- **Min episodes:** 10
- **Outcome weight:** 0.5
- **Max base delta:** 0.5
- **Weight bounds:** [0.1, 3.0]

## Edge Weights (w_base)

### Top Changes from Initial

| Edge | w_base | w_init | Δ |
|------|--------|--------|---|
| `p0_move` → `p0_wait` (POR) | 0.1065 | 1.0000 | -0.8935 |
| `p0_check` → `p0_move` (POR) | 0.1108 | 1.0000 | -0.8892 |
| `phase0_establish_cut` → `phase1_drive_to_edge` (POR) | 0.8032 | 1.0000 | -0.1968 |
| `p1_check` → `p1_move` (POR) | 0.8032 | 1.0000 | -0.1968 |

### All Tracked Edges

| Edge | w_base |
|------|--------|
| `p0_check` → `p0_move` (POR) | 0.1108 |
| `p0_move` → `p0_wait` (POR) | 0.1065 |
| `p1_check` → `p1_move` (POR) | 0.8032 |
| `phase0_establish_cut` → `phase1_drive_to_edge` (POR) | 0.8032 |

## Statistics

- **Average w_base:** 0.4559
- **Min w_base:** 0.1065
- **Max w_base:** 0.8032
- **Range:** 0.6967
- **Edges with significant change (>0.01):** 4

### Weight Distribution

```
    [0.11, 0.18) | ████████████████████████████████████████ 2
    [0.18, 0.25) |  0
    [0.25, 0.32) |  0
    [0.32, 0.39) |  0
    [0.39, 0.45) |  0
    [0.45, 0.52) |  0
    [0.52, 0.59) |  0
    [0.59, 0.66) |  0
    [0.66, 0.73) |  0
    [0.73, 0.80) | ████████████████████████████████████████ 2
```

### Delta Distribution (w_base - w_init)

```
  [-0.89, -0.71) | ████████████████████████████████████████ 2
  [-0.71, -0.54) |  0
  [-0.54, -0.36) |  0
  [-0.36, -0.18) | ████████████████████████████████████████ 2
   [-0.18, 0.00) |  0
    [0.00, 0.18) |  0
    [0.18, 0.36) |  0
    [0.36, 0.54) |  0
    [0.54, 0.71) |  0
    [0.71, 0.89) |  0
```
