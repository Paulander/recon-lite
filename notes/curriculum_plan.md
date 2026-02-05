# KRK Curriculum Plan (Living Document)

Last updated: 2026-02-04

This is a **living plan** for a multi-stage KRK curriculum that aims to reach
arbitrary starting positions through recursive goal chaining and self‑organizing
ReCoN growth (no hand‑coded tactics beyond base features + top‑level success).

---

## 0) Principles (non‑negotiable)
- **Pure ReCoN**: hierarchy via SUB/SUR, sequencing via POR/RET inside legs.
- **No hand‑coded tactics** beyond base features + top‑level goal signals.
- **Goal chaining**: “If we can reach this state, we can win” → recursive.
- **Self‑organizing growth**: new hypotheses spawn where evidence shows value.

---

## Progress (checkpoint)
- **Stage‑0 seed**: 100% mate‑in‑1 on 100 samples (corner‑balanced; no aliasing).
- **Stage‑1 evaluator**: added `scripts/test_stage1_backchain.py` to verify goal‑bank backchaining.

---

## 1) Stage map (high‑level)

We will start with a minimal set of stages and expand once chaining is stable.

### Stage 0 — Mate‑in‑1
**Start states:** mate‑in‑1 positions  
**Goal:** deliver mate  
**Notes:** already functional; goal bank is generated here.

### Stage 1 — Reach Stage‑0
**Start states:** 1–3 moves away from mate‑in‑1  
**Goal:** increase similarity to Stage‑0 goal prototypes  
**Notes:** backchaining via goal bank (terminal‑space).

### Stage 2 — Edge‑trap conversion (current focus)
**Start states:** enemy king trapped at edge (not necessarily mate‑in‑2)  
**Goal:** convert edge‑trap into Stage‑1 basin (mate‑in‑2 patterns)

We will start Stage‑2 with three **categories** (5–10 positions each, all White to move):

**2A. Edge‑trap, our king close, correct geometry**  
- Enemy king trapped at edge.  
- Our king close (one move from knight distance).  
- Our king between enemy king and rook.  
- Example: enemy Kd8, our Kb6, rook on 7th rank.  

**2B. Edge‑trap, enemy king between us and rook**  
- Enemy king trapped at edge.  
- Enemy king is between our king and rook.  
- Forces learning of tempo / waiting to avoid the king reaching the rook.  

**2C. Edge‑trap, wrong tempo at knight distance**  
- Enemy king trapped at edge.  
- Our king at knight distance but **wrong side to move** (tempo mismatch).  
- Example: enemy Ke1, our Kd3 (White to move).  

Naming note: these can be labeled **Stage‑2A/2B/2C** (or Stage‑2.1/2.2/2.3) to keep the hierarchy explicit.

### Stage 3 — Force edge‑trap (later)
**Start states:** arbitrary KRK (non‑trivial)  
**Goal:** reach Stage‑2 basin (edge‑trap positions)

> Later stages extend distance and complexity, not new hand‑coded heuristics.

---

## 2) Position sets (to be added later)
For each stage, we will attach **validated** positions:
- All FENs must be legal (no illegal check, both kings present).
- Each stage uses **constraints**, not memorized coordinates.
- When in doubt, use python‑chess to validate.

Placeholder list:
- Stage 0 positions: [TBD]
- Stage 1 positions: [TBD]
- Stage 2 positions: [TBD]
- Stage 3 positions: [TBD]

---

## 3) Goal chaining (core mechanism)
Stage k learns to move toward **goal prototypes of Stage k‑1**.

Key requirements:
- Goal prototypes stored as dicts keyed by **sensor_id**.
- Similarity computed on overlap; require minimum overlap size.
- No boolean affordance terminal required (avoid “hard rules”).
- **Runtime scoring must include goal similarity**, not just training-time extraction.
  (We need logs showing chosen move’s goal_sim vs best candidate.)

---

## 4) AND / OR / LAG exploration
We will implement **composite sensor discovery** to capture higher‑order
relations (tempo/opposition parity, edge‑trap conjunctions).

**Exploration policy (initial):**
- 70% from mature sensors (high XP)
- 30% from exploratory sensors (to avoid missing patterns)

**Why separate pools**
- Base pool: low‑order sensors (1–4 dims)
- Composite pool: AND/OR/LAG built only from sensor pairs/triples

**Hoist criteria**
- Co‑activation improves success or goal progress
- Beats best single sensor by a margin
- Stage‑local by default (do not mix across stages without shared goals)

---

## 5) Attachment policy (self‑organizing)
Composite sensors should attach **under the leg that discovered them**.
Only hoist upward if they prove transferable across legs/stages.

---

## 6) Immediate next tasks
1. Verify Stage‑0/Stage‑1 stability **and runtime chaining**
   (goal similarity logs, overlap counts, chosen vs best candidate).
2. Add SUR/RET in compiled topology (done in baseline_to_recon.py).
3. Define Stage‑2 FEN generator constraints + validation script.
4. Implement composite pool + hoist policy (AND/LAG).
5. Run Stage‑1 evaluator on compiled topology (new).

---

## 7) Open questions
- Minimum overlap threshold for goal similarity (current: 8, now **weighted by sensor XP**).
- Whether to cap max composites per stage or per leg.
- How to visualize composites distinctly in the UI.
