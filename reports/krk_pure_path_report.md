# KRK Pure Path Status (Learned Actuators + Goal Backchaining)

Date: 2026-01-24  
Repo: `/home/paulander/git/recon-lite`

## TL;DR
- The old integration pipeline still hits 100% because it uses a **built-in checkmate actuator** as an oracle.  
- The new baseline-compiled topology is **pure learned**, but it was failing because (a) Stage-0 mate generator was invalid and (b) actuator→sensor mapping assumed indices were IDs.  
- Generator is fixed. Actuator mapping now respects **mature sensor list order**.
- Next: retrain baseline on corrected data, recompile, then backchain Stage‑1 using goal memories.

---

## What “old solution” was doing (why it looked great)
- `krk_integration.py` used **affordance + checkmate actuator** as a hard anchor.
- Bandit + spawn points learned *around* that oracle.
- This is acceptable for demo but **not “pure”** because success is guaranteed once the checkmate feature is read.

---

## What “pure” means here
- No handcoded move selection.
- Actuators are learned from **terminal‑space deltas**.
- Goal memory stores terminal-space snapshots of mate‑in‑1 starts.
- Stage‑1/2/3 backchain by minimizing distance to stored goals.
- Spawn points create new sensors/legs dynamically; no POR between legs.

---

## Code changes made
1) **Valid mate‑in‑1 generator**
- `src/recon_lite_chess/baseline_teacher.py`
- `generate_krk_mate_in_1_position()` now brute‑forces legal KRK boards and verifies mate‑in‑1 using python‑chess.

2) **Actuator mapping fix**
- `scripts/baseline_to_recon.py`
- Actuator `sensor_indices` now map through the **mature sensors list order** (with fallback to IDs).

3) **Save learner pickle**
- `scripts/train_baseline_krk.py` now supports `--save-learner` for clean compilation.

4) **Stage‑0/1 chained training (pure)**
- `scripts/train_baseline_krk_chain.py`
- Stage‑0: learns sensors/actuators from mate‑in‑1.
- Stage‑1: labels transitions by moving **closer to mate‑in‑1 goal memories**.

---

## Recommended run sequence

### A) Train pure baseline (Stage‑0 + Stage‑1 backchain)
```
uv run python3 scripts/train_baseline_krk_chain.py \
  --stage0-cycles 50 \
  --stage1-cycles 50 \
  --samples-per-cycle 50 \
  --save-learner snapshots/baseline_krk_chain/final_learner.pkl
```

### B) Compile to ReCoN topology
```
uv run python3 scripts/baseline_to_recon.py \
  --learner snapshots/baseline_krk_chain/final_learner.pkl \
  --output topologies/krk_entry_topology.json
```

### C) Test runtime
```
uv run python3 scripts/test_krk_entry.py
```

---

## Open questions / next upgrades
- **Goal memory format**: currently stored as sparse `{sensor_id: value}` in chain trainer.
- **Backchain depth**: Stage‑2 can reuse Stage‑1 goal memories the same way (add another loop).
- **Recursive spawn in runtime**: can attach spawn points to legs to grow deeper goals live.
