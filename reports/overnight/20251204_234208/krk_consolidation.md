# Consolidation Report

Generated: 2025-12-04 23:44:13

**Source file:** `/home/paulander/git/recon-lite/weights/nightly/krk_consol.json`
**Checksum:** `a73535baaa84`

## Summary

- **Total episodes seen:** 270
- **Episodes since last apply:** 0
- **Last updated:** 2025-12-04T23:43:44.584154
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
| `p0_check` → `p0_move` (POR) | 0.6619 | 1.0000 | -0.3381 |
| `p0_move` → `p0_wait` (POR) | 0.7872 | 1.0000 | -0.2128 |
| `phase0_establish_cut` → `phase1_drive_to_edge` (POR) | 0.9574 | 1.0000 | -0.0426 |
| `p1_check` → `p1_move` (POR) | 0.9574 | 1.0000 | -0.0426 |

### All Tracked Edges

| Edge | w_base |
|------|--------|
| `p0_check` → `p0_move` (POR) | 0.6619 |
| `p0_move` → `p0_wait` (POR) | 0.7872 |
| `p1_check` → `p1_move` (POR) | 0.9574 |
| `phase0_establish_cut` → `phase1_drive_to_edge` (POR) | 0.9574 |

## Statistics

- **Average w_base:** 0.8410
- **Min w_base:** 0.6619
- **Max w_base:** 0.9574
- **Range:** 0.2955
- **Edges with significant change (>0.01):** 4

### Weight Distribution

```
    [0.66, 0.69) | ████████████████████ 1
    [0.69, 0.72) |  0
    [0.72, 0.75) |  0
    [0.75, 0.78) |  0
    [0.78, 0.81) | ████████████████████ 1
    [0.81, 0.84) |  0
    [0.84, 0.87) |  0
    [0.87, 0.90) |  0
    [0.90, 0.93) |  0
    [0.93, 0.96) | ████████████████████████████████████████ 2
```

### Delta Distribution (w_base - w_init)

```
  [-0.34, -0.27) | ████████████████████ 1
  [-0.27, -0.20) | ████████████████████ 1
  [-0.20, -0.14) |  0
  [-0.14, -0.07) |  0
   [-0.07, 0.00) | ████████████████████████████████████████ 2
    [0.00, 0.07) |  0
    [0.07, 0.14) |  0
    [0.14, 0.20) |  0
    [0.20, 0.27) |  0
    [0.27, 0.34) |  0
```
