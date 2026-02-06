"""M5.1 Hyperparameter Sweep Engine.

Orchestrates multiple evolution_driver.py runs in isolation to systematically
explore learning configurations and find the "Tactical Sweet Spot".

Features:
- Local sweep execution with configurable trials
- Optional Weights & Biases (wandb) integration for distributed sweeps
- Automatic metric logging and report generation

Usage:
    from recon_lite.learning.sweep_engine import HyperSweepEngine, SweepConfig
    
    engine = HyperSweepEngine(base_output_dir=Path("snapshots/sweeps"))
    
    configs = [
        SweepConfig(trial_name="conservative", consistency_threshold=0.50),
        SweepConfig(trial_name="speculative", consistency_threshold=0.30),
    ]
    
    results = engine.run_sweep(configs, stage_id=1)
    report = engine.generate_report(results)
    
    # With W&B logging:
    engine = HyperSweepEngine(use_wandb=True, wandb_project="recon-lite-sweeps")
    results = engine.run_sweep(configs)
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

# Optional W&B support for distributed sweeps
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False


@dataclass
class SweepConfig:
    """Configuration for a single sweep trial."""
    
    trial_name: str
    stage_id: int = 1
    starting_topology: Optional[Path] = None  # If None, use default
    
    # M5 Promotion Thresholds
    consistency_threshold: float = 0.50
    hoist_threshold: float = 0.85
    
    # M5.1 Unblock Features
    enable_success_bypass: bool = False
    enable_speculative_hoisting: bool = False
    enable_stall_recovery: bool = True
    
    # Training Parameters
    spawn_rate: float = 0.05
    max_cycles: int = 20
    games_per_cycle: int = 50
    win_threshold: float = 0.80
    
    # Stall Recovery Settings
    stall_threshold_win_rate: float = 0.10
    stall_threshold_cycles: int = 3
    stall_spawn_multiplier: float = 2.0
    enable_scent_shaping: bool = True  # Scent-based reward for King→Pawn approach
    
    # M5.1 Aggressive Hoisting Settings (for Failure Frontier)
    min_coactivations_for_hoist: int = 50  # Lower = more speculative AND-gates
    
    # M5.1 Emergent Spawning (for stuck situations)
    enable_emergent_spawning: bool = False
    emergent_spawn_threshold_cycles: int = 5  # Cycles below 50% before triggering
    emergent_spawn_count: int = 10  # How many new sensors to spawn
    
    # M5.1 Forced Hierarchy (for "Gauntlet" mode - Stage 8)
    enable_forced_hoisting: bool = False
    forced_hoist_threshold_win_rate: float = 0.20  # Below 20% triggers forced hoisting
    forced_hoist_interval_cycles: int = 5  # Force-create AND-gate every N cycles
    leg_link_xp_multiplier: float = 1.0  # XP multiplier for HOISTED→LEG links
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        if d.get("starting_topology"):
            d["starting_topology"] = str(d["starting_topology"])
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SweepConfig":
        """Create from dictionary."""
        if data.get("starting_topology"):
            data["starting_topology"] = Path(data["starting_topology"])
        return cls(**data)


@dataclass
class SweepResult:
    """Result metrics from a single sweep trial."""
    
    trial_name: str
    config: SweepConfig
    
    # Learning Speed
    cycles_completed: int = 0
    cycles_to_80_percent: Optional[int] = None  # None if never reached
    final_win_rate: float = 0.0
    
    # Structural Maturity
    solid_nodes: int = 0
    hoisted_clusters: int = 0
    por_edges: int = 0
    
    # Topology Metrics
    max_depth: int = 0
    branching_factor: float = 0.0
    non_backbone_scripts: int = 0
    vertical_promotions: int = 0
    
    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    
    # Error tracking
    success: bool = True
    error_message: Optional[str] = None
    
    # Per-cycle history for analysis
    win_rate_history: List[float] = field(default_factory=list)
    depth_history: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SweepResult":
        """Create from dictionary."""
        config_data = data.pop("config")
        config = SweepConfig.from_dict(config_data)
        return cls(config=config, **data)


class HyperSweepEngine:
    """
    Orchestrates hyperparameter sweeps for M5 structural learning.
    
    Each trial runs in isolation with its own output directory, allowing
    systematic comparison of different learning configurations.
    
    Features:
    - Local execution with isolated directories per trial
    - Optional W&B logging for distributed sweeps and visualization
    - Automatic metric computation and report generation
    """
    
    def __init__(
        self,
        base_output_dir: Path = Path("snapshots/sweeps"),
        evolution_driver_path: Path = Path("scripts/evolution_driver.py"),
        use_wandb: bool = False,
        wandb_project: str = "recon-lite-sweeps",
        wandb_entity: Optional[str] = None,
    ):
        """Initialize sweep engine.
        
        Args:
            base_output_dir: Directory for sweep outputs
            evolution_driver_path: Path to evolution driver script
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name
            wandb_entity: W&B entity (username/team)
        """
        self.base_output_dir = Path(base_output_dir)
        self.evolution_driver_path = Path(evolution_driver_path)
        self.results: List[SweepResult] = []
        
        # W&B integration
        self.use_wandb = use_wandb and HAS_WANDB
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        if use_wandb and not HAS_WANDB:
            print("Warning: wandb requested but not installed. Logging disabled.")
    
    def _wandb_init_trial(self, config: "SweepConfig") -> None:
        """Initialize W&B run for a trial."""
        if not self.use_wandb:
            return
        
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=config.trial_name,
            config=config.to_dict(),
            reinit=True,
        )
    
    def _wandb_log_metrics(self, result: "SweepResult", cycle: int = 0) -> None:
        """Log metrics to W&B."""
        if not self.use_wandb:
            return
        
        # Calculate depth_win_score (primary metric for Deep-Pressure plan)
        depth_win_score = result.final_win_rate * result.max_depth
        
        wandb.log({
            "win_rate": result.final_win_rate,
            "max_depth": result.max_depth,
            "depth_win_score": depth_win_score,  # PRIMARY METRIC
            "solid_nodes": result.solid_nodes,
            "hoisted_clusters": result.hoisted_clusters,
            "por_edges": result.por_edges,
            "branching_factor": result.branching_factor,
            "cycles_completed": result.cycles_completed,
        }, step=cycle)
    
    def _wandb_finish_trial(self, result: "SweepResult") -> None:
        """Finish W&B run and log summary."""
        if not self.use_wandb:
            return
        
        # Calculate depth_win_score
        depth_win_score = result.final_win_rate * result.max_depth
        
        wandb.summary.update({
            "final_win_rate": result.final_win_rate,
            "final_max_depth": result.max_depth,
            "final_depth_win_score": depth_win_score,
            "cycles_to_80_percent": result.cycles_to_80_percent,
            "solid_nodes": result.solid_nodes,
            "hoisted_clusters": result.hoisted_clusters,
            "success": result.success,
        })
        
        wandb.finish()
    
    def run_trial(
        self,
        config: SweepConfig,
        verbose: bool = True,
    ) -> SweepResult:
        """
        Run a single sweep trial.
        
        Creates isolated output directory and runs evolution_driver.py with
        the specified configuration.
        
        Args:
            config: Trial configuration
            verbose: Whether to print progress
            
        Returns:
            SweepResult with metrics from the trial
        """
        result = SweepResult(
            trial_name=config.trial_name,
            config=config,
            start_time=datetime.now().isoformat(),
        )
        
        # Initialize W&B run if enabled
        self._wandb_init_trial(config)
        
        # Create trial output directory
        trial_dir = self.base_output_dir / config.trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = trial_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Trial: {config.trial_name}")
            print(f"{'='*60}")
            print(f"  Stage: {config.stage_id}")
            print(f"  Consistency: {config.consistency_threshold}")
            print(f"  Hoist: {config.hoist_threshold}")
            print(f"  Success Bypass: {config.enable_success_bypass}")
            print(f"  Speculative Hoisting: {config.enable_speculative_hoisting}")
            if self.use_wandb:
                print(f"  W&B Logging: ENABLED")
        
        try:
            # Build environment variables for configuration injection
            env_vars = self._build_env_vars(config)
            
            # Run evolution driver
            cycle_results = self._run_evolution_driver(
                config=config,
                output_dir=trial_dir,
                env_vars=env_vars,
                verbose=verbose,
            )
            
            # Extract metrics from results
            result = self._extract_metrics(result, cycle_results, trial_dir)
            result.success = True
            
            # Log metrics to W&B
            self._wandb_log_metrics(result, cycle=result.cycles_completed)
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            if verbose:
                print(f"  ERROR: {e}")
        
        result.end_time = datetime.now().isoformat()
        
        # Calculate duration
        start = datetime.fromisoformat(result.start_time)
        end = datetime.fromisoformat(result.end_time)
        result.duration_seconds = (end - start).total_seconds()
        
        # Save result
        result_path = trial_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        if verbose:
            self._print_trial_summary(result)
        
        # Finish W&B run
        self._wandb_finish_trial(result)
        
        return result
    
    def run_sweep(
        self,
        configs: List[SweepConfig],
        verbose: bool = True,
    ) -> List[SweepResult]:
        """
        Run multiple trials as a sweep.
        
        Args:
            configs: List of trial configurations
            verbose: Whether to print progress
            
        Returns:
            List of SweepResult for each trial
        """
        sweep_start = datetime.now()
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# HYPERPARAMETER SWEEP")
            print(f"# Trials: {len(configs)}")
            print(f"# Started: {sweep_start.isoformat()}")
            print(f"{'#'*60}")
        
        results = []
        for i, config in enumerate(configs):
            if verbose:
                print(f"\n[{i+1}/{len(configs)}] Running trial: {config.trial_name}")
            
            result = self.run_trial(config, verbose=verbose)
            results.append(result)
        
        self.results = results
        
        # Save sweep summary
        self._save_sweep_summary(results)
        
        if verbose:
            sweep_end = datetime.now()
            duration = (sweep_end - sweep_start).total_seconds()
            print(f"\n{'#'*60}")
            print(f"# SWEEP COMPLETE")
            print(f"# Duration: {duration:.1f}s")
            print(f"# Results saved to: {self.base_output_dir}")
            print(f"{'#'*60}")
        
        return results
    
    def _build_env_vars(self, config: SweepConfig) -> Dict[str, str]:
        """Build environment variables to inject configuration into evolution_driver."""
        return {
            "M5_CONSISTENCY_THRESHOLD": str(config.consistency_threshold),
            "M5_HOIST_THRESHOLD": str(config.hoist_threshold),
            "M5_ENABLE_SUCCESS_BYPASS": "1" if config.enable_success_bypass else "0",
            "M5_ENABLE_SPECULATIVE_HOISTING": "1" if config.enable_speculative_hoisting else "0",
            "M5_ENABLE_STALL_RECOVERY": "1" if config.enable_stall_recovery else "0",
            "M5_STALL_THRESHOLD_WIN_RATE": str(config.stall_threshold_win_rate),
            "M5_STALL_THRESHOLD_CYCLES": str(config.stall_threshold_cycles),
            "M5_STALL_SPAWN_MULTIPLIER": str(config.stall_spawn_multiplier),
            "M5_ENABLE_SCENT_SHAPING": "1" if config.enable_scent_shaping else "0",
            "M5_SPAWN_RATE": str(config.spawn_rate),
            # M5.1 Aggressive Hoisting
            "M5_MIN_COACTIVATIONS_FOR_HOIST": str(config.min_coactivations_for_hoist),
            # M5.1 Emergent Spawning
            "M5_ENABLE_EMERGENT_SPAWNING": "1" if config.enable_emergent_spawning else "0",
            "M5_EMERGENT_SPAWN_THRESHOLD_CYCLES": str(config.emergent_spawn_threshold_cycles),
            "M5_EMERGENT_SPAWN_COUNT": str(config.emergent_spawn_count),
            # M5.1 Forced Hierarchy (Gauntlet mode)
            "M5_ENABLE_FORCED_HOISTING": "1" if config.enable_forced_hoisting else "0",
            "M5_FORCED_HOIST_THRESHOLD_WIN_RATE": str(config.forced_hoist_threshold_win_rate),
            "M5_FORCED_HOIST_INTERVAL_CYCLES": str(config.forced_hoist_interval_cycles),
            "M5_LEG_LINK_XP_MULTIPLIER": str(config.leg_link_xp_multiplier),
        }
    
    def _run_evolution_driver(
        self,
        config: SweepConfig,
        output_dir: Path,
        env_vars: Dict[str, str],
        verbose: bool,
    ) -> List[Dict[str, Any]]:
        """
        Run evolution_driver.py as a subprocess or inline.
        
        For now, we'll import and run directly for better integration.
        """
        import os
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Import and run evolution driver directly
        try:
            # Add project root to path if needed
            project_root = Path(__file__).parent.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from scripts.evolution_driver import (
                EvolutionConfig,
                run_evolution_training,
            )
            
            # Build config
            topology_path = config.starting_topology or Path("topologies/kpk_topology.json")
            
            evo_config = EvolutionConfig(
                topology_path=topology_path,
                games_per_cycle=config.games_per_cycle,
                max_cycles=config.max_cycles,
                output_dir=output_dir / "reports",
                snapshot_dir=output_dir / "snapshots",
                trace_dir=output_dir / "traces",
                current_stage_idx=config.stage_id,
                stage_promotion_threshold=config.win_threshold,
                stem_cell_spawn_rate=config.spawn_rate,
            )
            
            # Run training
            results = run_evolution_training(evo_config)
            
            # Convert CycleResult objects to dicts
            return [
                {
                    "cycle": r.cycle,
                    "games_played": r.games_played,
                    "win_rate": r.win_rate,
                    "promotions": r.promotions,
                    "duration": r.duration_seconds,
                }
                for r in results
            ]
            
        finally:
            # Clean up environment variables
            for key in env_vars:
                if key in os.environ:
                    del os.environ[key]
    
    def _extract_metrics(
        self,
        result: SweepResult,
        cycle_results: List[Dict[str, Any]],
        trial_dir: Path,
    ) -> SweepResult:
        """Extract metrics from cycle results and final topology."""
        
        # Win rate history
        result.win_rate_history = [r["win_rate"] for r in cycle_results]
        result.cycles_completed = len(cycle_results)
        result.final_win_rate = result.win_rate_history[-1] if result.win_rate_history else 0.0
        
        # Find cycles to 80%
        for i, wr in enumerate(result.win_rate_history):
            if wr >= 0.80:
                result.cycles_to_80_percent = i + 1
                break
        
        # Load final topology for structural metrics
        snapshot_dir = trial_dir / "snapshots"
        latest_snapshot = self._find_latest_snapshot(snapshot_dir)
        
        if latest_snapshot:
            try:
                with open(latest_snapshot) as f:
                    topology = json.load(f)
                
                result = self._extract_topology_metrics(result, topology)
            except Exception:
                pass
        
        return result
    
    def _find_latest_snapshot(self, snapshot_dir: Path) -> Optional[Path]:
        """Find the most recent topology snapshot."""
        if not snapshot_dir.exists():
            return None
        
        snapshots = sorted(snapshot_dir.glob("cycle_*.json"))
        return snapshots[-1] if snapshots else None
    
    def _extract_topology_metrics(
        self,
        result: SweepResult,
        topology: Dict[str, Any],
    ) -> SweepResult:
        """Extract structural metrics from topology snapshot."""
        nodes = topology.get("nodes", {})
        edges = topology.get("edges", {})
        
        backbone_nodes = {"kpk_root", "kpk_detect", "kpk_execute", "kpk_finish", "kpk_wait"}
        
        # Count node types
        for node_id, node_data in nodes.items():
            group = node_data.get("group", "")
            meta = node_data.get("meta", {})
            
            if group == "mature" or meta.get("tier") == "mature":
                result.solid_nodes += 1
            
            if meta.get("origin") == "hoisted":
                result.hoisted_clusters += 1
        
        # Count POR edges
        for edge_key, edge_data in edges.items():
            if edge_data.get("type") == "POR":
                result.por_edges += 1
        
        # Non-backbone scripts
        result.non_backbone_scripts = sum(
            1 for nid, n in nodes.items()
            if n.get("type") == "SCRIPT" and nid not in backbone_nodes
        )
        
        # Max depth (simplified BFS)
        result.max_depth = self._compute_max_depth(nodes, edges, backbone_nodes)
        
        # Branching factor
        result.branching_factor = self._compute_branching_factor(
            nodes, edges, backbone_nodes
        )
        
        return result
    
    def _compute_max_depth(
        self,
        nodes: Dict[str, Any],
        edges: Dict[str, Any],
        backbone_nodes: set,
    ) -> int:
        """Compute maximum depth from backbone."""
        # Build parent map
        parent_map = {}
        for edge_key, edge_data in edges.items():
            if edge_data.get("type") == "SUB":
                src = edge_data.get("src", "")
                dst = edge_data.get("dst", "")
                parent_map[dst] = src
        
        max_depth = 0
        for node_id in nodes:
            depth = 0
            current = node_id
            visited = set()
            
            while current and current not in backbone_nodes and current not in visited:
                visited.add(current)
                parent = parent_map.get(current)
                if parent:
                    depth += 1
                    current = parent
                else:
                    break
            
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _compute_branching_factor(
        self,
        nodes: Dict[str, Any],
        edges: Dict[str, Any],
        backbone_nodes: set,
    ) -> float:
        """Compute average children per non-backbone SCRIPT node."""
        # Build children map
        children_map: Dict[str, List[str]] = {}
        for edge_key, edge_data in edges.items():
            if edge_data.get("type") == "SUB":
                src = edge_data.get("src", "")
                dst = edge_data.get("dst", "")
                if src not in children_map:
                    children_map[src] = []
                children_map[src].append(dst)
        
        # Find non-backbone SCRIPT nodes
        non_backbone_scripts = [
            nid for nid, n in nodes.items()
            if n.get("type") == "SCRIPT" and nid not in backbone_nodes
        ]
        
        if not non_backbone_scripts:
            return 0.0
        
        total_children = sum(len(children_map.get(nid, [])) for nid in non_backbone_scripts)
        return total_children / len(non_backbone_scripts)
    
    def _save_sweep_summary(self, results: List[SweepResult]) -> None:
        """Save sweep summary to JSON."""
        summary_path = self.base_output_dir / "sweep_summary.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "trials": len(results),
            "results": [r.to_dict() for r in results],
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    
    def _print_trial_summary(self, result: SweepResult) -> None:
        """Print summary of trial results."""
        print(f"\n  --- Trial Summary: {result.trial_name} ---")
        print(f"  Win Rate: {result.final_win_rate:.1%}")
        print(f"  Cycles to 80%: {result.cycles_to_80_percent or 'Never'}")
        print(f"  SOLID Nodes: {result.solid_nodes}")
        print(f"  Hoisted Clusters: {result.hoisted_clusters}")
        print(f"  POR Edges: {result.por_edges}")
        print(f"  Max Depth: {result.max_depth}")
        print(f"  Branching Factor: {result.branching_factor:.2f}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        if not result.success:
            print(f"  ERROR: {result.error_message}")
    
    def generate_report(
        self,
        results: Optional[List[SweepResult]] = None,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate markdown comparison report.
        
        Args:
            results: Results to compare (uses self.results if None)
            output_path: Optional path to save report
            
        Returns:
            Markdown formatted report string
        """
        results = results or self.results
        
        if not results:
            return "No results to report."
        
        lines = [
            "# Hyperparameter Sweep Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Trials:** {len(results)}",
            "",
            "## Summary Table",
            "",
            "| Trial | Win Rate | Cycles to 80% | SOLID | Hoisted | POR | Max Depth | Branching |",
            "|-------|----------|---------------|-------|---------|-----|-----------|-----------|",
        ]
        
        for r in results:
            cycles_80 = str(r.cycles_to_80_percent) if r.cycles_to_80_percent else "N/A"
            lines.append(
                f"| {r.trial_name} | {r.final_win_rate:.1%} | {cycles_80} | "
                f"{r.solid_nodes} | {r.hoisted_clusters} | {r.por_edges} | "
                f"{r.max_depth} | {r.branching_factor:.2f} |"
            )
        
        lines.extend([
            "",
            "## Configuration Comparison",
            "",
            "| Trial | Consistency | Hoist | Success Bypass | Speculative | Stall Recovery |",
            "|-------|-------------|-------|----------------|-------------|----------------|",
        ])
        
        for r in results:
            c = r.config
            lines.append(
                f"| {r.trial_name} | {c.consistency_threshold} | {c.hoist_threshold} | "
                f"{'Yes' if c.enable_success_bypass else 'No'} | "
                f"{'Yes' if c.enable_speculative_hoisting else 'No'} | "
                f"{'Yes' if c.enable_stall_recovery else 'No'} |"
            )
        
        lines.extend([
            "",
            "## Win Rate Progression",
            "",
        ])
        
        for r in results:
            history_str = " → ".join(f"{wr:.0%}" for wr in r.win_rate_history[:10])
            if len(r.win_rate_history) > 10:
                history_str += f" ... ({len(r.win_rate_history)} cycles)"
            lines.append(f"**{r.trial_name}:** {history_str}")
            lines.append("")
        
        report = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
        
        return report


# =============================================================================
# Predefined Sweep Configurations
# =============================================================================

def create_stage1_validation_sweep() -> List[SweepConfig]:
    """
    Create the Stage 1 (Guardian_E) validation sweep configs.
    
    Three learning personas:
    - Conservative: High thresholds, safe learning
    - Speculative: Low thresholds, aggressive discovery
    - Recursive: Success bypass + speculative hoisting enabled
    """
    return [
        SweepConfig(
            trial_name="conservative",
            stage_id=1,
            consistency_threshold=0.50,
            hoist_threshold=0.90,
            enable_success_bypass=False,
            enable_speculative_hoisting=False,
            games_per_cycle=50,
            max_cycles=20,
        ),
        SweepConfig(
            trial_name="speculative",
            stage_id=1,
            consistency_threshold=0.30,
            hoist_threshold=0.75,
            enable_success_bypass=False,
            enable_speculative_hoisting=False,
            games_per_cycle=50,
            max_cycles=20,
        ),
        SweepConfig(
            trial_name="recursive",
            stage_id=1,
            consistency_threshold=0.40,
            hoist_threshold=0.85,
            enable_success_bypass=True,
            enable_speculative_hoisting=True,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            games_per_cycle=50,
            max_cycles=20,
        ),
    ]


def create_opposition_sweep() -> List[SweepConfig]:
    """
    Create the Stage 5 (Opposition_Lite) "Failure Frontier" sweep configs.
    
    This is the TRUE test of M5.1 structural learning. Stage 5 requires
    the King to take specific opposition squares - pure "pawn vibes" won't work.
    
    Three personas designed for STRUGGLE, not instant success:
    - Baseline: Standard settings to establish the "wall"
    - Aggressive: Very low thresholds, early speculation
    - Recursive_Turbo: Maximum aggression with emergent spawning
    """
    return [
        # Baseline: See where Hector naturally struggles
        SweepConfig(
            trial_name="baseline",
            stage_id=5,  # Opposition_Lite
            consistency_threshold=0.40,
            hoist_threshold=0.85,
            enable_success_bypass=False,
            enable_speculative_hoisting=False,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            spawn_rate=0.05,
            games_per_cycle=50,
            max_cycles=25,
        ),
        # Aggressive: Lower bars for hypothesis formation
        SweepConfig(
            trial_name="aggressive",
            stage_id=5,
            consistency_threshold=0.25,  # Very low - let weak patterns through
            hoist_threshold=0.70,  # Hoist at 70% correlation
            enable_success_bypass=True,
            enable_speculative_hoisting=True,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            spawn_rate=0.10,  # Double spawn rate
            games_per_cycle=50,
            max_cycles=25,
            # Lower co-activation threshold for hoisting
            min_coactivations_for_hoist=20,  # Down from 50
        ),
        # Recursive_Turbo: Maximum structural aggression
        SweepConfig(
            trial_name="recursive_turbo",
            stage_id=5,
            consistency_threshold=0.20,  # Accept almost anything
            hoist_threshold=0.65,  # Hoist early and often
            enable_success_bypass=True,
            enable_speculative_hoisting=True,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            spawn_rate=0.15,  # 3x spawn rate
            games_per_cycle=50,
            max_cycles=25,
            min_coactivations_for_hoist=15,  # Very aggressive hoisting
            # NEW: Emergent Spawning trigger
            enable_emergent_spawning=True,
            emergent_spawn_threshold_cycles=5,  # Trigger after 5 cycles below 50%
            emergent_spawn_count=10,  # Spawn 10 new sensors
        ),
    ]


def create_escort_sweep() -> List[SweepConfig]:
    """
    Create the Stage 6 (ESCORT) sweep - the original "Stage 1" difficulty.
    
    This is the classic KPK challenge where King must actively support
    pawn promotion. Good for validating structural growth after Opposition.
    """
    return [
        SweepConfig(
            trial_name="escort_baseline",
            stage_id=6,
            consistency_threshold=0.35,
            hoist_threshold=0.80,
            enable_success_bypass=True,
            enable_speculative_hoisting=True,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            spawn_rate=0.10,
            games_per_cycle=50,
            max_cycles=30,
        ),
        SweepConfig(
            trial_name="escort_turbo",
            stage_id=6,
            consistency_threshold=0.20,
            hoist_threshold=0.65,
            enable_success_bypass=True,
            enable_speculative_hoisting=True,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            spawn_rate=0.15,
            games_per_cycle=50,
            max_cycles=30,
            enable_emergent_spawning=True,
            emergent_spawn_threshold_cycles=5,
            emergent_spawn_count=10,
        ),
    ]


def create_gauntlet_sweep() -> List[SweepConfig]:
    """
    Stage 8 (Full Escort / FRONTAL_BLOCKADE) - The Gauntlet.
    
    This is where FLAT NETWORKS DIE. The enemy King is active,
    and only hierarchical scripts (If Opposition then Shouldering then Push)
    can win consistently.
    
    The "Vibe Ceiling" is broken here - random sensor firing won't work.
    
    Includes "Forced Hierarchy" trial that seeds AND-gates when struggling.
    """
    return [
        # Baseline: See how badly flat network fails
        SweepConfig(
            trial_name="gauntlet_baseline",
            stage_id=8,  # FRONTAL_BLOCKADE
            consistency_threshold=0.40,
            hoist_threshold=0.85,
            enable_success_bypass=False,
            enable_speculative_hoisting=False,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            spawn_rate=0.05,
            games_per_cycle=50,
            max_cycles=30,
        ),
        # Aggressive: All M5.1 features enabled
        SweepConfig(
            trial_name="gauntlet_aggressive",
            stage_id=8,
            consistency_threshold=0.20,
            hoist_threshold=0.60,
            enable_success_bypass=True,
            enable_speculative_hoisting=True,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            spawn_rate=0.15,
            games_per_cycle=50,
            max_cycles=30,
            min_coactivations_for_hoist=15,
            enable_emergent_spawning=True,
            emergent_spawn_threshold_cycles=3,
            emergent_spawn_count=10,
        ),
        # Forced Hierarchy: THE SLEDGEHAMMER
        # Seeds AND-gates when struggling badly (< 20% win rate)
        SweepConfig(
            trial_name="forced_hierarchy",
            stage_id=8,
            consistency_threshold=0.15,  # Accept almost anything
            hoist_threshold=0.50,  # Hoist at 50% correlation
            enable_success_bypass=True,
            enable_speculative_hoisting=True,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            spawn_rate=0.20,  # 4x spawn rate
            games_per_cycle=50,
            max_cycles=30,
            min_coactivations_for_hoist=10,  # Very aggressive
            enable_emergent_spawning=True,
            emergent_spawn_threshold_cycles=3,  # Faster trigger
            emergent_spawn_count=15,
            # FORCED HIERARCHY: Seed the brain with structural guesses
            enable_forced_hoisting=True,
            forced_hoist_threshold_win_rate=0.20,  # Below 20% triggers
            forced_hoist_interval_cycles=5,  # Every 5 cycles
            leg_link_xp_multiplier=5.0,  # 5x XP for HOISTED→LEG links
        ),
    ]


if __name__ == "__main__":
    # Quick test of sweep engine
    engine = HyperSweepEngine()
    configs = create_stage1_validation_sweep()
    
    print("Stage 1 Validation Sweep Configurations:")
    for c in configs:
        print(f"  - {c.trial_name}: consistency={c.consistency_threshold}, hoist={c.hoist_threshold}")

