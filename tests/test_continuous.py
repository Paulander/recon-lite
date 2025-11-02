import chess

from recon_lite.binding.manager import BindingInstance, BindingTable
from recon_lite.core.activations import ActivationState
from recon_lite.time.microtick import MicrotickConfig, run_microticks
from recon_lite_chess.strategy import compute_phase_logits


def test_binding_conflicts_and_commit():
    table = BindingTable()
    with table.begin_tick("krk/p1/drive") as session:
        assert session.reserve(BindingInstance("rook_anchor", {"square:a4"}))
        conflicts = session.conflicts(BindingInstance("rook_conflict", {"square:a4"}))
        assert conflicts, "Expected conflict on reused square"
        assert not session.reserve(BindingInstance("rook_conflict", {"square:a4"}))

    snapshot = table.snapshot()
    assert "krk/p1/drive" in snapshot
    assert snapshot["krk/p1/drive"][0]["items"] == ["square:a4"]

    # Invalidate on board change clears namespaces
    fresh_board = chess.Board()
    table.invalidate_on_board_change(fresh_board)
    assert table.snapshot() == {}


def test_microtick_convergence_to_target():
    states = {"phase": ActivationState()}

    def compute_targets(_states):
        return {"phase": 1.0}

    cfg = MicrotickConfig(states=states, compute_targets=compute_targets, steps=6, eta=0.4)
    run_microticks(cfg)
    assert states["phase"].value > 0.9


def test_phase_logits_respond_to_features():
    base_board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")
    mate_board = chess.Board("4k3/3R4/2K5/8/8/8/8/8 w - - 0 1")

    base_logits = compute_phase_logits(base_board)
    mate_logits = compute_phase_logits(mate_board)

    assert mate_logits["phase4"] > base_logits["phase4"], "Mate opportunity should boost phase4 logit"

    # Stable cut increases phase1 score
    scattered = chess.Board()
    scattered.clear()
    scattered.set_piece_at(chess.E6, chess.Piece(chess.KING, chess.BLACK))
    scattered.set_piece_at(chess.C4, chess.Piece(chess.KING, chess.WHITE))
    scattered.set_piece_at(chess.B5, chess.Piece(chess.ROOK, chess.WHITE))
    scattered.turn = chess.WHITE

    cut_board = chess.Board()
    cut_board.clear()
    cut_board.set_piece_at(chess.E6, chess.Piece(chess.KING, chess.BLACK))
    cut_board.set_piece_at(chess.C4, chess.Piece(chess.KING, chess.WHITE))
    cut_board.set_piece_at(chess.A4, chess.Piece(chess.ROOK, chess.WHITE))
    cut_board.turn = chess.WHITE

    scatter_logits = compute_phase_logits(scattered)
    cut_logits = compute_phase_logits(cut_board)

    assert cut_logits["phase1"] > scatter_logits["phase1"], "Stable cut should increase phase1 activation"
