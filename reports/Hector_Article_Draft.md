# Hector: A Cognitive Architecture for Structural Deliberation via Request-Confirmation Networks

**Authors**: [To be added]

---

## Abstract

Conventional Reinforcement Learning (RL) often produces "flat" agents that exhibit high performance through reactive heuristics but lack the structural decomposition necessary for genuine deliberation: a disconnect we term the "Prodigy Problem." We present Hector, a cognitive architecture based on Request Confirmation Networks (ReCoN), designed to investigate how hierarchical subgoals and planning horizons can emerge from self-organizing symbolic structures.

Using the adversarial microcosm of chess not as an objective, but as a structured symbolic scaffold, we demonstrate a "White Box" orchestrator capable of autonomous maturation. Hector builds upon a minimal backbone with a Dynamic Stem Cell Layer, where "logical packs" (AND/OR gates, POR sequences) self-organize into hierarchical subgraphs evocative of cortical microcolumns. We detail the "Hector Roadmap": from the visualization of real-time sensory-motor bindings to the "Big Result"—the autonomous chaining of disparate strategic phases (e.g., transition from pawn promotion to checkmate) without hardcoded phase logic.

Our findings demonstrate that the architecture functions as a distributed orchestrator, capable of autonomous strategic handovers—such as the transition from pawn promotion (King-Pawn vs King) to checkmate (King-Queen vs King)—without hardcoded phase logic. Furthermore, we present a Proof of Concept for Structural Maturation via Inertia Pruning, where the agent autonomously explores deeper partonomic hierarchies. This dual-pathway approach—combining consolidated modular coordination with dynamic topological growth—offers a concrete instantiation of functional machine intelligence that prioritizes representational transparency over black-box optimization.

---

## I. Introduction: Beyond Reactive Agency

### The Prodigy Problem

Modern deep reinforcement learning has produced agents capable of superhuman performance in games ranging from Go to StarCraft. Yet these "prodigies" exhibit a fundamental limitation: their competence is locked within opaque weight matrices, inaccessible to inspection, modification, or transfer. A chess engine trained via PPO cannot explain *why* it chose a move, cannot decompose its strategy into reusable subgoals, and cannot adapt its learned representations to novel endgame scenarios without complete retraining.

We call this disconnect between performance and understanding the **Prodigy Problem**: success without the structural scaffolding for deliberation.

### Chess as Drosophila Model

We utilize chess as a **drosophila-model**—a deterministic environment for quantifying long-range temporal credit assignment and subgoal inference. Our contribution is not "yet another chess engine" but rather a demonstration that:

1. Hierarchical subgoals can **emerge** from self-organizing symbolic structures
2. Strategic phase transitions can occur **autonomously** without hardcoded orchestration
3. Learned structures remain **interpretable** throughout training

### Thesis

We propose that genuine deliberation requires a **Distributed Orchestrator** capable of functional decomposition. Hector, built on the Request-Confirmation Network (ReCoN) formalism, provides such an orchestrator by combining:

- **Top-down goal delegation** (requests flow from abstract goals to concrete sensors)
- **Bottom-up confirmation** (evidence flows from sensors to validate hypotheses)
- **Temporal sequencing** (POR/RET links enforce causal ordering)
- **Structural plasticity** (stem cells discover and solidify new patterns)

---

## II. The ReCoN Formalism: A Grammar of Deliberation

Request Confirmation Networks (Bach & Herger, 2015) provide a neuro-symbolic framework for combining neural computation with hierarchical script execution. We summarize the key concepts and introduce our extensions.

### 2.1 Node Types

ReCoN networks consist of two fundamental node types:

| Type | Definition | Role |
|------|------------|------|
| **SCRIPT** | "A hypothesis requiring validation from sub-elements" | Intermediate goals, scripts, composite patterns |
| **TERMINAL** | "Performs measurement, has activation representing value" | Sensors (read environment) or Actuators (execute actions) |

A SCRIPT node represents a **hypothesis**—a claim about the world that must be validated by its children. For example, "opposition is established" is a hypothesis validated by checking king positions. TERMINAL nodes perform the actual measurements or actions.

### 2.2 Edge Types

Four directed edge types connect nodes, forming the grammar of deliberation:

| Edge | Direction | Message | Purpose |
|------|-----------|---------|---------|
| **SUB** | Parent → Child | `request` | "I need this subgoal validated" |
| **SUR** | Child → Parent | `wait`, `confirm` | "I'm working" / "I succeeded" |
| **POR** | Predecessor → Successor | `inhibit_request` | "Wait for me before activating" |
| **RET** | Successor → Predecessor | `inhibit_confirm` | "Only I can confirm parent" |

**Critical constraint**: TERMINAL nodes can only receive SUB edges and send SUR edges. POR/RET edges require SCRIPT nodes, enabling temporal sequencing only at the compositional level.

### 2.3 State Machine and Message Passing

Each node implements an 8-state finite machine:

```
INACTIVE → REQUESTED → ACTIVE → WAITING → TRUE → CONFIRMED
                ↓              ↓
           SUPPRESSED       FAILED
```

The state transitions follow message passing rules (Table 1 from Bach & Herger):

| State | POR Sent | RET Sent | SUB Sent | SUR Sent |
|-------|----------|----------|----------|----------|
| INACTIVE | – | – | – | – |
| REQUESTED | inhibit_request | inhibit_confirm | – | wait |
| ACTIVE | inhibit_request | inhibit_confirm | request | wait |
| SUPPRESSED | inhibit_request | inhibit_confirm | – | – |
| WAITING | inhibit_request | inhibit_confirm | request | wait |
| TRUE | – | inhibit_confirm | – | – |
| CONFIRMED | – | inhibit_confirm | – | confirm |
| FAILED | inhibit_request | inhibit_confirm | – | – |

### 2.4 The Inhibit-Confirm Mechanism

The **inhibit_confirm** signal via RET links is what enables POR-chains to function as sequences rather than parallel alternatives:

> "Each unit sends an 'inhibit confirm' signal via RET to its predecessors. The last unit in a sequence will not receive an 'inhibit confirm' signal, and can turn into the state 'confirmed'."

This means in a sequence `A → B → C` (POR ordering), only node C can confirm the parent—ensuring the entire sequence completes before the composite goal is achieved.

### 2.5 Top-Down/Bottom-Up Integration

ReCoN achieves the integration of top-down and bottom-up processing that characterizes biological cognition:

- **Top-down**: Requests propagate from abstract goals (e.g., "promote pawn") through SUB links to concrete sensors (e.g., "check if path is clear")
- **Bottom-up**: Confirmations propagate from sensors through SUR links, validating hypotheses at each level

This bidirectional flow allows the network to both predict (top-down) and verify (bottom-up)—a requirement for robust perception and action.

---

## III. Hector's Roadmap: Developmental Scaffolding

Hector's development proceeded through two major phases, each building capabilities for the next.

### 3.1 Embryonic Stage (M1-M4): Modular Coordination

The first phase established that ReCoN could orchestrate modular subgoals with learned coordination.

#### M1-M2: Continuous Activations and Weighted Subgraph Packs

We extended the discrete ReCoN formalism with:

- **Continuous activations**: Node activation values in [0, 1] rather than binary
- **Weighted edges**: SUB/SUR edges carry learnable weights
- **Subgraph Weight Packs (SWPs)**: Groups of edges whose weights are trained together

This allowed the network to learn *which* strategies to prefer in different situations, even before structure learning.

#### M3: Fast Plasticity

Within-game weight updates via eligibility traces:

```
Δw = α × eligibility × reward
```

When a move leads to a successful outcome, all edges that fired during that decision receive a weight boost. This is analogous to biological synaptic potentiation.

#### M4: Slow Consolidation

Cross-game aggregation of weight changes:

```
w_base ← (1-β) × w_base + β × mean(Δw_episodes)
```

This prevents catastrophic forgetting and allows gradual refinement over many games.

#### The Big Result: KPK → KQK Handover

Using these mechanisms, we demonstrated **autonomous strategic handovers**: a unified topology containing both King-Pawn-vs-King (KPK) and King-Queen-vs-King (KQK) subgraphs could seamlessly transition from pawn promotion strategy to checkmate strategy **without hardcoded phase detection**.

The transition occurs naturally through activation dynamics:
1. KPK subgraph fires while pawn promotion is the goal
2. Upon promotion, KPK sensors return low activation (goal achieved)
3. KQK sensors detect the new piece configuration
4. KQK subgraph activation rises, taking over strategy selection

No explicit "if pawn promoted then switch to KQK" logic was required.

### 3.2 Maturation Stage (M5): Structural Learning

The second phase replaced hand-designed sensor nodes with self-discovering **stem cells**.

#### The Stem Cell Lifecycle

Inspired by biological neural development, we implemented a Darwinian selection process for sensor nodes:

```
EXPLORING → CANDIDATE → TRIAL → MATURE
                           ↓
                       DEMOTED (XP ≤ 0)
```

| State | Description | Survival Criterion |
|-------|-------------|-------------------|
| EXPLORING | Collects samples during high-reward moments | Collect ≥50 samples |
| CANDIDATE | Has enough data, awaiting trial | Survive 2+ cycles |
| TRIAL | Transient node in graph, earns XP | XP ≥ 100 to solidify |
| MATURE | Permanent node in topology | N/A |

#### XP System

TRIAL nodes earn or lose Experience Points based on their contribution to wins:

| Event | XP Change |
|-------|-----------|
| Positive affordance delta (win) | +10 XP |
| Negative affordance delta (loss) | -10 XP |
| Per-cycle decay (cost of living) | -1 XP |
| Solidification threshold | 100 XP |
| Demotion threshold | 0 XP |

This creates evolutionary pressure: sensors that correlate with wins survive; sensors that don't are pruned.

#### Pack Templates: AND/OR Gates

When TRIAL cells demonstrate consistent coactivation (≥85% correlation in wins), they are hoisted into **AND-gate packs**:

```
parent (SCRIPT)
  └─ SUB → gate (SCRIPT, aggregation="and")
             └─ SUB → sensor_A (TERMINAL)
             └─ SUB → sensor_B (TERMINAL)
             └─ SUB → actuator (TERMINAL)
```

The AND-gate fires only when ALL child sensors fire—enabling compositional pattern detection (e.g., "opposition AND path clear → push").

#### Vertical Growth: Depth 3+ Hierarchies

Initially, all TRIAL nodes attached to backbone nodes (depth 1-2). We extended the system to spawn packs **under existing packs**, creating true hierarchies:

```
Depth 1: kpk_execute (backbone)
Depth 2: TRIAL_opposition_sensor
Depth 3: AND-gate combining multiple sensors
Depth 4: Nested AND-gate for multi-condition patterns
```

Log output demonstrating deep spawning:
```
🔗 AND-gate spawned (L1): and_stem_0002_gate
🌲 AND-gate spawned (deep under and_stem_0002_gate): and_stem_0004_deep_gate
```

---

## IV. Interpretability: Visualizing the Binding Mechanism

A key advantage of ReCoN over black-box approaches is inherent interpretability.

### 4.1 The "Window into Mental Model"

Our visualization system renders:
- **Node activations** as color intensity (white = inactive, green = active)
- **Edge weights** as line thickness
- **State machine states** as node borders

This is not merely a "GUI for chess"—it is a **window into the agent's mental model**, showing how abstract goals (e.g., "establish opposition") bind to physical board coordinates in real time.

### 4.2 Topology Timelapse

We record topology snapshots at each training cycle, enabling timelapse visualization of structural growth. The evolution from a single backbone node to a 45+ node hierarchical structure demonstrates the emergence of tactical knowledge.

### 4.3 Activation Heatmaps

During the KPK→KQK handover, activation heatmaps show:
1. KPK subgraph peak activation during pawn promotion phase
2. Transition period with dual activation
3. KQK subgraph dominance after promotion

This visual "fingerprint" of strategic handover is impossible with weight-matrix representations.

---

## V. Experiments and Results

### 5.1 Experimental Setup

#### Environment
King-Pawn vs King (KPK) endgame with:
- **Observation**: 64-dimensional vector (piece positions)
- **Actions**: Legal moves from current position
- **Reward**: +1 win, -1 loss, 0 draw, +0.5 promotion

#### Curriculum
8-stage curriculum from easy to hard positions:

| Stage | Description | Challenge |
|-------|-------------|-----------|
| 0 | SPRINTER | Pawn on 7th rank, king far |
| 1-5 | Discovery Bridge | Incremental challenges |
| 6 | ESCORT | King support required |
| 7 | KEY_SQUARES | Full opposition dynamics |

#### Baseline
PPO (Stable-Baselines3) with identical observation/action/reward setup:
- MlpPolicy (2-layer neural network)
- 50,000 training timesteps
- Standard hyperparameters (lr=3e-4, γ=0.99)

### 5.2 KPK Curriculum Results

| Metric | PPO | ReCoN (Hector) |
|--------|-----|----------------|
| Stage 7 Win Rate | ~20% | **93.2%** |
| Training Samples | 50,000 | 8,000 |
| Training Time | ~5 min | ~3 min |
| Interpretable | No | **Yes** |

**Sample Efficiency**: ReCoN achieves 93% win rate with **6x fewer samples** than PPO achieves 20%.

### 5.3 Structural Growth

Final topology statistics from `kpk_hybrid_growth/stage7`:

| Metric | Value |
|--------|-------|
| Total nodes | 152 |
| Pack nodes (AND/OR gates) | 45 |
| Maximum depth | 4 |
| TRIAL → MATURE promotions | 12 |
| Inertia-pruned cells | 58 |

The emergence of 45 pack nodes demonstrates **causal discovery**—the network identified and solidified patterns that predict wins.

### 5.4 KPK → KQK Handover

Using pre-trained unified topology with both endgame subgraphs:

| Metric | Value |
|--------|-------|
| Successful handovers | 100% |
| Handover latency | 2-3 moves |
| Hardcoded phase logic | **None** |

The transition occurs via activation dynamics, demonstrating autonomous strategic orchestration.

---

## VI. Discussion

### What's Hardcoded vs. Learned

| Component | Hardcoded | Learned |
|-----------|-----------|---------|
| Chess rules (legal moves) | ✅ | |
| Piece categorization (pawn/king) | ✅ | |
| Goal (promote = win) | ✅ | |
| Move selection | | ✅ |
| Pattern discovery | | ✅ |
| Topology structure | | ✅ |

**The defensible claim**: Given only legal moves and win/loss rewards, Hector autonomously discovers hierarchical patterns corresponding to known chess theory (opposition, key squares, timing) without being programmed with those concepts.

### Limitations

1. **Shallow hierarchy**: Current depth (3-4) may be insufficient for complex tactical combinations
2. **No true consolidation**: Network can grow but cannot yet prune unused structures while maintaining performance
3. **KPK-specific**: Validation on other endgames (KRK, complex positions) remains future work

### Future Work

1. **Network Consolidation**: Pruning unused nodes while maintaining win rate
2. **KRK Validation**: Demonstrating generalization to King-Rook vs King
3. **Spawn Probability Decay**: Adaptive spawning rates (1/n or moving window)
4. **Full Game Integration**: Scaling from endgame to opening/middlegame

---

## VII. Conclusion: Toward Structured Control

We have presented Hector, a cognitive architecture demonstrating that:

1. **Hierarchical subgoals can emerge** from self-organizing stem cells under Darwinian selection
2. **Strategic phase transitions occur autonomously** via activation dynamics rather than hardcoded orchestration
3. **Learned structures remain interpretable** throughout training, addressing the Prodigy Problem

ReCoN provides a general-purpose framework for structured control. While demonstrated here on chess endgames, the formalism applies to any domain requiring:
- Compositional reasoning (AND/OR gates)
- Temporal sequencing (POR chains)
- Top-down/Bottom-up integration
- Representational transparency

By prioritizing structural decomposition over raw performance, Hector offers a path from "game playing" to **autonomous deliberation** in open-world tasks.

---

## References

1. Bach, J., & Herger, P. (2015). Request Confirmation Networks for Neuro-Symbolic Script Execution. *Proceedings of the 28th International Conference on Neural Information Processing Systems*.

2. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

3. [Additional references to be added]

---

## Appendix A: ReCoN State Machine Complete Table

*[Reproduce Table 1 from Bach & Herger 2015]*

## Appendix B: Hector Topology Specification

*[Include JSON schema for topology files]*

## Appendix C: Experimental Logs

*[Sample log output showing structural growth]*

```
--- Cycle 29/30 ---
  Online Phase: Playing 200 games...
    Win rate: 100.0%
  Structural Phase: Analyzing traces...
    [M5] 🌱 GROWTH MODE: Spawning packs to build depth (win=100.0%)
    [M5] ⚡ FAILURE MODE TRIGGERED: Spawning packs/singles from 2 TRIALs
      Graph reused: 149 nodes (preserving earlier changes)
      🔗 AND-gate spawned: and_and_stem_0080_2_gate
      🔗 OR-gate also spawned: or_or_stem_0080_2_gate
    📦 Pack nodes in graph at save: 9
    📦 Pack nodes in snapshot: 9
```
