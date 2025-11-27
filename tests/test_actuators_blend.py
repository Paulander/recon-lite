import chess

from recon_lite_chess import strategy  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite_chess import actuators_blend as blend  # type: ignore  # pylint: disable=wrong-import-position


def test_phase_bias_override_affects_selection(monkeypatch):
    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")

    monkeypatch.setattr(
        blend,
        "PHASE_CHOOSERS",
        {
            "phase1": lambda _board, _env: "e2e6",
            "phase2": lambda _board, _env: "e2e7",
            "phase3": lambda _board, _env: "e2e8",
        },
        raising=False,
    )
    monkeypatch.setattr(blend, "_phase_score", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(blend, "cheap_eval_after", lambda *_args, **_kwargs: 0.0)

    phase_latents = {
        strategy.SCRIPT_BY_PHASE["phase1"]: 0.5,
        strategy.SCRIPT_BY_PHASE["phase2"]: 0.5,
        strategy.SCRIPT_BY_PHASE["phase3"]: 0.5,
    }
    env = {"phase_weight_override": {"phase1": 0.5, "phase2": 2.0, "phase3": 1.0}}

    move, diagnostics = blend.choose_blended_move(board, phase_latents, env)
    assert move == "e2e7"
    assert diagnostics
    assert diagnostics[0]["phase"] == "phase2"
    assert diagnostics[0]["phase_bias"] == 2.0
