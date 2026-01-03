"""M5 Structure Learning Module.

Implements the "Dreamer" component that analyzes traces for affordance spikes,
identifies high-impact stem cells, and promotes them to permanent nodes.

Usage:
    from recon_lite.learning.m5_structure import StructureLearner
    
    learner = StructureLearner(registry, trace_db)
    
    # After a batch of games
    stats = learner.apply_structural_phase(
        stem_manager,
        episodes,
        max_promotions=2
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from ..nodes.stem_cell import StemCellTerminal, StemCellManager, StemCellState
    HAS_STEM_CELL = True
except ImportError:
    HAS_STEM_CELL = False

try:
    from ..models.registry import TopologyRegistry
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False


@dataclass
class AffordanceSpike:
    """Record of a high-impact activation moment."""
    tick: int
    episode_id: str
    node_id: str
    reward: float
    fen: str
    features: Optional[List[float]] = None
    move: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "episode_id": self.episode_id,
            "node_id": self.node_id,
            "reward": self.reward,
            "fen": self.fen,
            "features": self.features,
            "move": self.move,
        }


@dataclass
class PromotionResult:
    """Result of a stem cell promotion."""
    cell_id: str
    new_node_id: str
    parent_id: str
    tick: int
    success: bool
    signature_path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class PruningResult:
    """Result of edge pruning analysis."""
    edge_key: str
    old_weight: float
    new_weight: float
    pruned: bool
    games_at_zero: int


class StructureLearner:
    """
    M5 Dreamer: Analyzes traces and proposes structural changes.
    
    The structural phase runs periodically (e.g., every 500 games) to:
    1. Scan traces for affordance spikes (high reward moments)
    2. Identify stem cells that fired right before spikes
    3. Promote promising stem cells to permanent nodes
    4. Prune edges with consistently negative confirmations
    """
    
    def __init__(
        self,
        registry: "TopologyRegistry",
        cooldown_ticks: int = 1000,
        min_spike_reward: float = 0.5,
        decay_rate: float = 0.95,
        prune_threshold_games: int = 100,
        signature_dir: Optional[Path] = None,
    ):
        """
        Args:
            registry: Topology registry for structural changes
            cooldown_ticks: Minimum ticks between promotions for same parent
            min_spike_reward: Minimum reward to count as a spike
            decay_rate: Weight decay rate for negative confirmations
            prune_threshold_games: Games at zero weight before pruning
            signature_dir: Directory for signature PNG files
        """
        self.registry = registry
        self.cooldown_ticks = cooldown_ticks
        self.min_spike_reward = min_spike_reward
        self.decay_rate = decay_rate
        self.prune_threshold_games = prune_threshold_games
        self.signature_dir = signature_dir or Path("signatures")
        
        # Track promotion cooldowns: parent_id -> last_promotion_tick
        self.cooldowns: Dict[str, int] = {}
        
        # Track edge confirmation history: edge_key -> (weight, games_at_zero)
        self.edge_history: Dict[str, Tuple[float, int]] = {}
    
    def scan_for_affordance_spikes(
        self,
        episodes: List[Any],  # List[EpisodeRecord]
        threshold: Optional[float] = None,
    ) -> List[AffordanceSpike]:
        """
        Find high-impact moments in traces.
        
        Scans episode tick records for moments where:
        - reward_tick >= threshold
        - A significant state transition occurred
        """
        threshold = threshold or self.min_spike_reward
        spikes: List[AffordanceSpike] = []
        
        for episode in episodes:
            episode_id = getattr(episode, 'episode_id', str(id(episode)))
            ticks = getattr(episode, 'ticks', [])
            
            for tick in ticks:
                reward = getattr(tick, 'reward_tick', None)
                if reward is None:
                    reward = tick.meta.get('reward_tick', 0) if hasattr(tick, 'meta') else 0
                
                if abs(reward) >= threshold:
                    # Found a spike
                    active_nodes = getattr(tick, 'active_nodes', [])
                    fen = getattr(tick, 'board_fen', '')
                    action = getattr(tick, 'action', None)
                    tick_id = getattr(tick, 'tick_id', 0)
                    
                    # Create spike for each active node
                    for node_id in active_nodes:
                        spike = AffordanceSpike(
                            tick=tick_id,
                            episode_id=episode_id,
                            node_id=node_id,
                            reward=float(reward),
                            fen=fen,
                            move=action,
                        )
                        spikes.append(spike)
        
        return spikes
    
    def find_high_impact_stem_cells(
        self,
        stem_manager: "StemCellManager",
        spikes: List[AffordanceSpike],
        lookback_ticks: int = 5,
    ) -> List["StemCellTerminal"]:
        """
        Identify stem cells that were active near affordance spikes.
        
        A stem cell is "high-impact" if it collected samples within
        `lookback_ticks` of a spike.
        """
        if not HAS_STEM_CELL:
            return []
        
        high_impact: List[StemCellTerminal] = []
        
        # Build a set of (episode_id, tick_range) for faster lookup
        spike_ranges: Dict[str, List[Tuple[int, int]]] = {}
        for spike in spikes:
            if spike.episode_id not in spike_ranges:
                spike_ranges[spike.episode_id] = []
            spike_ranges[spike.episode_id].append(
                (spike.tick - lookback_ticks, spike.tick)
            )
        
        # Check each candidate stem cell
        for cell in stem_manager.cells.values():
            if cell.state not in (StemCellState.CANDIDATE, StemCellState.EXPLORING):
                continue
            
            # Check if any of its samples are near spikes
            impact_score = 0
            for sample in cell.samples:
                # Check sample tick against spike ranges
                sample_tick = sample.tick
                for ep_id, ranges in spike_ranges.items():
                    for start_tick, end_tick in ranges:
                        if start_tick <= sample_tick <= end_tick:
                            impact_score += abs(sample.reward)
            
            if impact_score > 0:
                cell.metadata["impact_score"] = impact_score
                high_impact.append(cell)
        
        # Sort by impact score descending
        high_impact.sort(
            key=lambda c: c.metadata.get("impact_score", 0),
            reverse=True
        )
        
        return high_impact
    
    def promote_stem_cell(
        self,
        cell: "StemCellTerminal",
        parent_id: str,
        current_tick: int,
    ) -> Optional[PromotionResult]:
        """
        Promote a stem cell to a permanent node.
        
        1. Create node entry in topology.json
        2. Inherit weights/filters from the stem cell
        3. Wire to parent with SUB edge
        4. Generate signature.png heatmap
        
        Returns:
            PromotionResult if successful/failed
        """
        if not HAS_STEM_CELL or not HAS_REGISTRY:
            return PromotionResult(
                cell_id=cell.cell_id,
                new_node_id="",
                parent_id=parent_id,
                tick=current_tick,
                success=False,
                error="Missing required modules"
            )
        
        # Check cooldown
        last_promo = self.cooldowns.get(parent_id, 0)
        if current_tick - last_promo < self.cooldown_ticks:
            return PromotionResult(
                cell_id=cell.cell_id,
                new_node_id="",
                parent_id=parent_id,
                tick=current_tick,
                success=False,
                error=f"Parent {parent_id} on cooldown"
            )
        
        # Analyze pattern to get signature
        consistency, signature = cell.analyze_pattern()
        if consistency < 0.5:
            return PromotionResult(
                cell_id=cell.cell_id,
                new_node_id="",
                parent_id=parent_id,
                tick=current_tick,
                success=False,
                error=f"Pattern consistency too low: {consistency:.2f}"
            )
        
        # Generate new node ID
        new_node_id = f"SC_{cell.cell_id}_{current_tick}"
        
        try:
            # Create node spec
            node_spec = {
                "id": new_node_id,
                "type": "TERMINAL",
                "group": "stem_promoted",
                "factory": "recon_lite.nodes.stem_cell:create_pattern_sensor",
                "pattern_signature": signature,
                "weight_source": cell.cell_id,
                "meta": {
                    "promoted_tick": current_tick,
                    "consistency": consistency,
                    "sample_count": len(cell.samples),
                    "avg_reward": sum(s.reward for s in cell.samples) / len(cell.samples) if cell.samples else 0,
                }
            }
            
            # Add to registry
            self.registry.add_node(node_spec, tick=current_tick)
            
            # Wire to parent
            self.registry.add_edge(
                parent_id, new_node_id, "SUB",
                weight=1.0,
                tick=current_tick
            )
            
            # Record promotion
            self.registry.record_promotion(
                cell_id=cell.cell_id,
                new_node_id=new_node_id,
                parent_id=parent_id,
                tick=current_tick,
                pattern_signature=signature,
            )
            
            # Update cooldown
            self.cooldowns[parent_id] = current_tick
            
            # Mark stem cell as specialized
            cell.state = StemCellState.SPECIALIZED
            
            # Generate signature visualization
            signature_path = None
            try:
                signature_path = self._generate_signature(cell, new_node_id)
            except Exception as e:
                print(f"Warning: Could not generate signature: {e}")
            
            return PromotionResult(
                cell_id=cell.cell_id,
                new_node_id=new_node_id,
                parent_id=parent_id,
                tick=current_tick,
                success=True,
                signature_path=signature_path,
            )
            
        except Exception as e:
            return PromotionResult(
                cell_id=cell.cell_id,
                new_node_id=new_node_id,
                parent_id=parent_id,
                tick=current_tick,
                success=False,
                error=str(e),
            )
    
    def _generate_signature(
        self,
        cell: "StemCellTerminal",
        node_id: str,
    ) -> Optional[Path]:
        """Generate signature heatmap for a promoted node."""
        try:
            from recon_lite.viz.signature_viz import generate_signature_heatmap
            
            self.signature_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.signature_dir / f"{node_id}.png"
            
            return generate_signature_heatmap(
                samples=cell.samples,
                output_path=output_path,
            )
        except ImportError:
            return None
    
    def check_edge_for_pruning(
        self,
        src: str,
        dst: str,
        ltype: str,
        confirmation_value: float,
    ) -> PruningResult:
        """
        Apply SUR-based pruning logic.
        
        If confirmation consistently negative: decay weight.
        If weight stays at 0 for N games: mark for removal.
        """
        edge_key = f"{src}->{dst}:{ltype}"
        
        # Get current state
        old_weight, games_at_zero = self.edge_history.get(edge_key, (1.0, 0))
        
        # Update based on confirmation
        if confirmation_value < 0:
            new_weight = old_weight * self.decay_rate
        elif confirmation_value > 0:
            # Positive confirmation: slight recovery
            new_weight = min(1.0, old_weight * 1.05)
        else:
            new_weight = old_weight
        
        # Track games at zero
        if new_weight < 0.01:
            new_weight = 0.0
            games_at_zero += 1
        else:
            games_at_zero = 0
        
        # Check if should prune
        should_prune = games_at_zero >= self.prune_threshold_games
        
        # Update history
        self.edge_history[edge_key] = (new_weight, games_at_zero)
        
        # Update registry
        self.registry.update_edge_weight(src, dst, ltype, new_weight)
        
        return PruningResult(
            edge_key=edge_key,
            old_weight=old_weight,
            new_weight=new_weight,
            pruned=should_prune,
            games_at_zero=games_at_zero,
        )
    
    def apply_structural_phase(
        self,
        stem_manager: "StemCellManager",
        episodes: List[Any],
        max_promotions: int = 2,
        parent_candidates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a full structural phase:
        1. Analyze traces for spikes
        2. Find promising stem cells
        3. Promote top candidates
        4. Check edges for pruning
        
        Args:
            stem_manager: Manager for stem cells
            episodes: List of EpisodeRecord from recent games
            max_promotions: Max nodes to promote this cycle
            parent_candidates: Optional list of eligible parent node IDs
            
        Returns:
            Stats dict with counts and results
        """
        current_tick = 0
        if episodes:
            last_ep = episodes[-1]
            ticks = getattr(last_ep, 'ticks', [])
            if ticks:
                current_tick = getattr(ticks[-1], 'tick_id', 0)
        
        # Step 1: Find spikes
        spikes = self.scan_for_affordance_spikes(episodes)
        
        # Step 2: Find high-impact stem cells
        high_impact = self.find_high_impact_stem_cells(stem_manager, spikes)
        
        # Step 3: Promote CANDIDATE cells to TRIAL tier
        trial_promotions: List[str] = []
        trial_errors: List[str] = []
        
        # Default parent candidates if not specified
        if parent_candidates is None:
            parent_candidates = ["kpk_detect", "kpk_execute"]
        
        for cell in high_impact:
            if len(trial_promotions) >= max_promotions:
                break
            
            # Only promote CANDIDATE cells (not already in TRIAL)
            if cell.state != StemCellState.CANDIDATE:
                continue
            
            # Check if ready for trial
            consistency, _ = cell.analyze_pattern()
            if consistency < 0.35:
                trial_errors.append(f"{cell.cell_id}: consistency {consistency:.2f} < 0.35")
                continue
            
            # Choose parent
            parent_id = parent_candidates[0] if parent_candidates else "kpk_root"
            
            # Promote to TRIAL (not MATURE yet)
            if cell.promote_to_trial(self.registry, parent_id, current_tick):
                trial_promotions.append(cell.cell_id)
            else:
                trial_errors.append(f"{cell.cell_id}: trial promotion failed")
        
        # Step 4: Apply XP decay to all TRIAL cells (-1 XP per cycle)
        xp_decays: List[Tuple[str, int]] = []
        for cell in stem_manager.cells.values():
            if cell.state == StemCellState.TRIAL:
                new_xp = cell.decay_xp()
                xp_decays.append((cell.cell_id, new_xp))
        
        # Step 5: Check for solidification (XP >= 100) or demotion (XP <= 0)
        solidified: List[str] = []
        demoted: List[str] = []
        
        for cell in list(stem_manager.cells.values()):
            if cell.state != StemCellState.TRIAL:
                continue
            
            should_change, new_state = cell.check_solidification()
            if should_change:
                if new_state == "mature":
                    if cell.solidify_to_mature(self.registry, current_tick):
                        solidified.append(cell.cell_id)
                elif new_state == "demoted":
                    if cell.demote_to_exploring(self.registry):
                        demoted.append(cell.cell_id)
        
        # Step 6: Collection confirmation stats for pruning
        pruning_results: List[PruningResult] = []
        
        # Save changes
        self.registry.save()
        
        return {
            "spikes_found": len(spikes),
            "high_impact_cells": len(high_impact),
            # Trial lifecycle stats
            "trial_promotions": len(trial_promotions),
            "trial_promoted": trial_promotions,
            "trial_errors": trial_errors,
            "xp_decays": len(xp_decays),
            "solidified": len(solidified),
            "solidified_cells": solidified,
            "demoted": len(demoted),
            "demoted_cells": demoted,
            # Legacy compat
            "promotions_attempted": len(trial_promotions),
            "promotions_succeeded": len(trial_promotions),
            "promotions": trial_promotions,
            "promotion_errors": trial_errors,
            "pruning_results": [r.edge_key for r in pruning_results if r.pruned],
            "current_tick": current_tick,
        }


def create_pattern_sensor(node_id: str):
    """
    Factory function for creating pattern-matching sensors from promoted stem cells.
    
    This is called when loading a graph from topology.json for nodes with
    factory="recon_lite.nodes.stem_cell:create_pattern_sensor"
    """
    from recon_lite.graph import Node, NodeType
    
    def _pattern_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Pattern matching predicate for promoted stem cells.
        
        Compares current board features against the stored pattern signature.
        """
        signature = node.meta.get("pattern_signature")
        if signature is None:
            return True, True  # No signature, always match
        
        # Get current features
        features = env.get("features")
        if features is None:
            return True, True  # No features to compare, pass through
        
        # Simple cosine similarity match
        try:
            import numpy as np
            sig_arr = np.array(signature)
            feat_arr = np.array(features)
            
            # Normalize
            sig_norm = sig_arr / (np.linalg.norm(sig_arr) + 1e-8)
            feat_norm = feat_arr / (np.linalg.norm(feat_arr) + 1e-8)
            
            similarity = float(np.dot(sig_norm, feat_norm))
            node.meta["last_similarity"] = similarity
            
            # Match if similarity exceeds threshold
            threshold = node.meta.get("threshold", 0.7)
            matched = similarity >= threshold
            
            return matched, matched
            
        except Exception:
            return True, True
    
    return Node(nid=node_id, ntype=NodeType.TERMINAL, predicate=_pattern_predicate)
