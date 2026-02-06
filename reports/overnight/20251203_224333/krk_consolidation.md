# Consolidation Report

Generated: 2025-12-03 22:50:23

**Source file:** `/home/paulander/git/recon-lite/weights/nightly/krk_consol.json`
**Checksum:** `55cedf33808f`

## Summary

- **Total episodes seen:** 260
- **Episodes since last apply:** 0
- **Last updated:** 2025-12-03T22:50:22.911921
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
| `p0_check` → `p0_move` (POR) | 0.6850 | 1.0000 | -0.3150 |
| `p0_move` → `p0_wait` (POR) | 0.8012 | 1.0000 | -0.1988 |
| `phase0_establish_cut` → `phase1_drive_to_edge` (POR) | 0.9574 | 1.0000 | -0.0426 |
| `p1_check` → `p1_move` (POR) | 0.9574 | 1.0000 | -0.0426 |

### All Tracked Edges

| Edge | w_base |
|------|--------|
| `p0_check` → `p0_move` (POR) | 0.6850 |
| `p0_move` → `p0_wait` (POR) | 0.8012 |
| `p1_check` → `p1_move` (POR) | 0.9574 |
| `phase0_establish_cut` → `phase1_drive_to_edge` (POR) | 0.9574 |

## Statistics

- **Average w_base:** 0.8503
- **Min w_base:** 0.6850
- **Max w_base:** 0.9574
- **Range:** 0.2724
- **Edges with significant change (>0.01):** 4

### Weight Distribution

```
    [0.69, 0.71) | ████████████████████ 1
    [0.71, 0.74) |  0
    [0.74, 0.77) |  0
    [0.77, 0.79) |  0
    [0.79, 0.82) | ████████████████████ 1
    [0.82, 0.85) |  0
    [0.85, 0.88) |  0
    [0.88, 0.90) |  0
    [0.90, 0.93) |  0
    [0.93, 0.96) | ████████████████████████████████████████ 2
```

### Delta Distribution (w_base - w_init)

```
  [-0.31, -0.25) | ████████████████████ 1
  [-0.25, -0.19) | ████████████████████ 1
  [-0.19, -0.13) |  0
  [-0.13, -0.06) |  0
   [-0.06, 0.00) | ████████████████████████████████████████ 2
    [0.00, 0.06) |  0
    [0.06, 0.13) |  0
    [0.13, 0.19) |  0
    [0.19, 0.25) |  0
    [0.25, 0.31) |  0
```
