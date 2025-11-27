import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for target in (PROJECT_ROOT / "src", PROJECT_ROOT):
    target_str = str(target)
    if target_str not in sys.path:
        sys.path.append(target_str)

from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint  # type: ignore  # pylint: disable=wrong-import-position


def test_trace_db_roundtrip(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    db = TraceDB(path)

    tick = TickRecord(
        tick_id=1,
        phase_estimate="phase1",
        action="e2e4",
        reward_tick=0.3,
        active_nodes=["phase1_drive_to_edge"],
    )
    ep = EpisodeRecord(
        episode_id="game-001",
        result="win",
        ticks=[tick],
        pack_meta=[{"path": "weights/krk_phase_weight_pack.swp", "sha256": "deadbeef"}],
        notes={"engine": "stockfish-depth2"},
    )
    db.add_episode(ep)
    db.flush()

    assert path.exists()
    content = path.read_text().strip().splitlines()
    assert len(content) == 1
    assert '"episode_id": "game-001"' in content[0]
    assert '"result": "win"' in content[0]
    assert '"tick_id": 1' in content[0]


def test_pack_fingerprint(tmp_path: Path):
    file_a = tmp_path / "a.bin"
    file_a.write_text("hello")
    fp = pack_fingerprint([file_a])
    assert fp and fp[0]["path"].endswith("a.bin")
    assert len(fp[0]["sha256"]) == 64
