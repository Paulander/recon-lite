"""M5.1 Hyperparameter Sweep Engine.

Orchestrates multiple evolution_driver.py runs in isolation to systematically
explore learning configurations and find the "Tactical Sweet Spot".

Usage:
    from recon_lite.learning.sweep_engine import HyperSweepEngine, SweepConfig
    
    engine = HyperSweepEngine(base_output_dir=Path("snapshots/sweeps"))
    
    configs = [
        SweepConfig(trial_name="conservative", consistency_threshold=0.50),
        SweepConfig(trial_name="speculative", consistency_threshold=0.30),
    ]
    
    results = engine.run_sweep(configs, stage_id=1)
    report = engine.generate_report(results)
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
    """
    
    def __init__(
        self,
        base_output_dir: Path = Path("snapshots/sweeps"),
        evolution_driver_path: Path = Path("scripts/evolution_driver.py"),
    ):
        self.base_output_dir = Path(base_output_dir)
        self.evolution_driver_path = Path(evolution_driver_path)
        self.results: List[SweepResult] = []
    
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


if __name__ == "__main__":
    # Quick test of sweep engine
    engine = HyperSweepEngine()
    configs = create_stage1_validation_sweep()
    
    print("Stage 1 Validation Sweep Configurations:")
    for c in configs:
        print(f"  - {c.trial_name}: consistency={c.consistency_threshold}, hoist={c.hoist_threshold}")

