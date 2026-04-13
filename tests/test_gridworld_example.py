import json

from recon_lite import ActivationMode
from recon_lite.examples.gridworld import main, run_simulation


def test_gridworld_runs_discrete_and_continuous():
    discrete = run_simulation(mode=ActivationMode.DISCRETE, steps=10)
    continuous = run_simulation(
        engine_kind="pragmatic",
        mode=ActivationMode.CONTINUOUS,
        steps=10,
        microticks=3,
    )

    assert discrete[-1].endswith("bindings=1")
    assert "agent=(4, 4)" in discrete[-1]
    assert "engine_ticks=" in discrete[-1]
    assert "engine=formal" in discrete[0]
    assert "mode=continuous" in continuous[0]
    assert "engine=pragmatic" in continuous[0]


def test_gridworld_cli_smoke(capsys):
    assert main(["--mode", "discrete", "--steps", "2"]) == 0
    captured = capsys.readouterr()
    assert "mode=discrete" in captured.out
    assert "engine=formal" in captured.out


def test_gridworld_explain_output_includes_network_state():
    lines = run_simulation(mode=ActivationMode.DISCRETE, steps=1, explain=True)

    assert any(line.strip() == "grid:" for line in lines)
    assert any("activations:" in line for line in lines)
    assert any("node_states:" in line for line in lines)


def test_gridworld_trace_json_contains_visualization_data(tmp_path):
    trace_path = tmp_path / "gridworld.json"

    lines = run_simulation(
        engine_kind="pragmatic",
        mode=ActivationMode.CONTINUOUS,
        steps=3,
        microticks=2,
        trace_json=trace_path,
    )

    assert lines
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["schema_version"] == 1
    assert trace["metadata"]["engine"] == "pragmatic"
    assert trace["metadata"]["mode"] == "continuous"
    assert trace["graph"]["nodes"]
    assert trace["graph"]["edges"]

    first_step = trace["steps"][0]
    assert "world" in first_step
    assert "bindings" in first_step
    assert first_step["ticks"]
    assert first_step["ticks"][0]["nodes"]
    assert first_step["ticks"][0]["activations"]
    assert "activation_history" in first_step["ticks"][0]

    assert any(step["binding_invalidated"] for step in trace["steps"][1:])


def test_gridworld_cli_writes_trace_json(tmp_path):
    trace_path = tmp_path / "cli-trace.json"

    assert main(["--engine", "pragmatic", "--mode", "continuous", "--steps", "2", "--microticks", "2", "--trace-json", str(trace_path)]) == 0

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["metadata"]["microticks"] == 2


def test_gridworld_formal_trace_contains_messages_and_confirms_root_last(tmp_path):
    trace_path = tmp_path / "formal-gridworld.json"

    lines = run_simulation(
        mode=ActivationMode.DISCRETE,
        steps=1,
        trace_json=trace_path,
    )

    assert "engine=formal" in lines[0]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["metadata"]["engine"] == "formal"
    assert any(frame["messages"] for frame in trace["steps"][0]["ticks"])

    frames = trace["steps"][0]["ticks"]
    root_confirm_tick = _first_tick_with_state(frames, "root", "CONFIRMED")
    move_confirm_tick = _first_tick_with_state(frames, "move_agent", "CONFIRMED")
    assert move_confirm_tick is not None
    assert root_confirm_tick is not None
    assert root_confirm_tick > move_confirm_tick


def _first_tick_with_state(frames, nid, state):
    for frame in frames:
        if frame["nodes"][nid] == state:
            return frame["tick"]
    return None
