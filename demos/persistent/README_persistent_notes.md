Persistent KRK Demo Notes
-------------------------
- Visualization: first frame carries static graph edges; later frames only update node states + the current board/moves. This prevents overlapping layers in the viewer.
- Logging split: `viz_logger` writes lightweight snapshots for the UI (`*_viz.json`), while `debug_logger` captures full validation metrics and fallbacks (`*_debug.json`). Toggle via `--combined-log` when needed.
- Testing matrix:
  * Phase0 rendezvous → `test_krk_phase0_rendezvous.py`
  * Phase1 drive → `test_krk_phase1_drive.py`
  * Phase2 shrink (worst-case safe) → `test_krk_box_minimization.py`, `test_krk_persistent_phase2.py`
  * Phase3 opposition lock-in → `test_krk_phase3_opposition.py`
  * Phase4 mate delivery → `test_krk_phase4_mate.py`
  * Integration harness → `test_krk_persistent_integration.py` exercises deterministic opponent scripts and phase transitions.
- CLI tips: `--skip-opponent`, `--single-phase PHASE2`, and `--seed` help isolate issues during debugging; combine with `--step-mode` to advance one tick at a time.
