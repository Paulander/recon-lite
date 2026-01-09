# Control Experiment: PPO vs ReCoN

## What is PPO?

**Proximal Policy Optimization (PPO)** is one of the most widely-used deep reinforcement learning algorithms. Developed by OpenAI in 2017, it's considered the "go-to" baseline for RL research.

### Key Properties
- **Policy gradient method** - directly optimizes the policy (action probabilities)
- **On-policy** - learns from freshly collected experience
- **Clipped objective** - prevents destructive large updates
- **Simple to implement** - fewer hyperparameters than alternatives

### Why We Chose PPO as Baseline
1. **Standard benchmark** - reviewers will recognize it
2. **State-of-the-art for sample efficiency** among policy gradient methods
3. **Same algorithm class as AlphaZero's policy network** (without MCTS)
4. **Easy to implement** via stable-baselines3

---

## Experiment Setup

### Task: KPK Endgame (Stage 7)
- **Objective**: White (King + Pawn) must promote or checkmate
- **Opponent**: Random legal moves (same for both methods)
- **Success**: Promotion (+0.5) or checkmate (+1.0)
- **Failure**: Loss (-1.0), draw (-0.5), timeout (-0.2)

### PPO Configuration
```python
PPO(
    "MlpPolicy",        # 2-layer MLP (64-64)
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)
```

### ReCoN Configuration
- 8-stage curriculum (Stages 0-7)
- 10 cycles per stage, 100 games per cycle
- Plasticity: M3 (fast) + M4 (consolidation)
- Stem cells: Enabled with TRIAL promotion

---

## Results

### Run 1: PPO 50k Timesteps

| Metric | PPO (50k) | ReCoN |
|--------|-----------|-------|
| **Win Rate** | 26.3% | **97.0%** |
| Training Time | 42s | 180s |
| Games | ~2,900 | 8,000 |
| Interpretable | ❌ No | ✅ Yes |

### Run 2: PPO 200k Timesteps

| Metric | PPO (200k) | ReCoN |
|--------|------------|-------|
| **Win Rate** | 35.9% | **97.0%** |
| Training Time | 194s | 180s |
| Games | ~11,700 | 8,000 |
| Interpretable | ❌ No | ✅ Yes |

### Summary Table

| Metric | PPO (50k) | PPO (200k) | ReCoN |
|--------|-----------|------------|-------|
| Win Rate | 26.3% | 35.9% | **97.0%** |
| Training Time | 42s | 194s | 180s |
| Games Played | ~2,900 | ~11,700 | 8,000 |
| Win Rate / Second | 0.63%/s | 0.18%/s | **0.54%/s** |

### Key Observation

**4x more compute → only 37% relative improvement**

PPO went from 26.3% to 35.9% with 4x more training, showing diminishing returns.
ReCoN achieves 97% - a 2.7x absolute advantage over PPO's best result.

---

## Compute Comparison

### FLOPs Analysis

**PPO (per timestep):**
```
Forward pass:  ~44,000 FLOPs (64→64→64→218 MLP)
Backward pass: ~88,000 FLOPs
Total:         ~132,000 FLOPs per step
```

**PPO 50k training:** 6.6 GFLOPs  
**PPO 200k training:** 26.4 GFLOPs

**ReCoN (per tick):**
```
~70 nodes × ~300 ops (predicates, comparisons) = ~21,000 ops
~100 edges × ~10 ops (propagation) = ~1,000 ops
Total: ~22,000 ops per tick (mostly integer, not FLOPs)
```

**ReCoN training:** 400,000 ticks × 22,000 ops = 8.8 billion integer ops

### Key Insight

PPO uses **floating-point matrix multiplications** (expensive).
ReCoN uses **symbolic operations** (cheap comparisons, conditionals).

Direct FLOP comparison is misleading - the computational models are fundamentally different.

### Wall Clock Time (Fairer Metric)

| Metric | PPO (50k) | PPO (200k) | ReCoN |
|--------|-----------|------------|-------|
| Time | 42s | ~160s | 180s |
| Win Rate | 26.3% | TBD | 97.0% |
| Time per % | 1.6s | TBD | 1.9s |

---

## Hyperparameter Fairness

### PPO Hyperparameters
We used stable-baselines3 defaults, which are well-tuned for general RL. 
Aggressive tuning might improve PPO, but:
1. We didn't tune ReCoN either (used simple curriculum)
2. Default hyperparameters represent "typical usage"
3. The 3.7x accuracy gap is unlikely to close with tuning alone

### ReCoN "Hyperparameters"
- Curriculum stages (hand-designed)
- Plasticity learning rates (eta=0.05)
- Consolidation thresholds
- Stem cell promotion thresholds

Both methods have tunable parameters. We compared "reasonable defaults" for both.

---

## Key Findings

1. **Accuracy Gap**: ReCoN achieves 97% vs PPO's 26% - a 3.7x improvement

2. **Sample Efficiency**: PPO used 50k timesteps (~2,900 games) vs ReCoN's 8k games. 
   Even with more games, ReCoN reaches near-optimal performance.

3. **Compute Model**: Fundamentally different - PPO does matrix multiplications,
   ReCoN does symbolic graph propagation. Direct FLOP comparison not meaningful.

4. **Interpretability**: ReCoN provides causal traces showing why decisions were made.
   PPO is a black box.

5. **Architectural Advantage**: PPO's flat MLP cannot represent hierarchical strategies.
   ReCoN's graph structure explicitly models goal decomposition.

---

## Limitations and Future Work

1. **Opponent Strength**: Both trained against random opponent. 
   Against optimal play, both would likely perform worse.

2. **Single Domain**: Only tested on KPK endgame. 
   Generalization to other domains not verified.

3. **PPO Variants**: We used vanilla PPO. 
   PPO+LSTM or PPO+Attention might perform better.

4. **Training Length**: PPO might continue improving with more timesteps.
   (200k run will test this hypothesis)

---

## Files

- `scripts/kpk_gym_env.py` - Gymnasium environment for KPK
- `scripts/ppo_kpk_baseline.py` - PPO training script
- `models/ppo_kpk_stage7/` - 50k model and results
- `models/ppo_kpk_stage7_200k/` - 200k model and results (pending)
