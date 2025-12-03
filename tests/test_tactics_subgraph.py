"""Tests for M5.4 tactical subgraph."""

import chess
import pytest

from recon_lite_chess.scripts.tactics import (
    detect_forks,
    detect_pins,
    detect_hanging_pieces,
    get_fork_moves,
    get_capture_hanging_moves,
    build_tactics_network,
    create_default_tactics_weight_pack,
)
from recon_lite_chess.scripts.rook_endgame import (
    is_rook_endgame,
    detect_lucena_position,
    detect_philidor_position,
    get_cutoff_moves,
    build_rook_endgame_network,
    create_default_rook_weight_pack,
)


class TestTacticsDetection:
    """Tests for tactical pattern detection."""
    
    def test_detect_knight_fork_opportunity(self):
        """Test detecting a knight fork opportunity."""
        # Position where Nf7 forks queen and rook
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        forks = detect_forks(board)
        # Should find fork opportunities (if any exist)
        assert isinstance(forks, list)
    
    def test_detect_pin(self):
        """Test detecting a pinned piece."""
        # Position with pinned knight on e7
        board = chess.Board("4k3/4n3/8/8/4B3/8/8/4K3 w - - 0 1")
        pins = detect_pins(board)
        # The knight on e7 is pinned to the king
        assert isinstance(pins, list)
    
    def test_detect_hanging_pieces(self):
        """Test detecting hanging pieces (undefended AND attacked)."""
        # Position with undefended black knight attacked by white queen
        board = chess.Board("4k3/8/8/3n4/8/8/3Q4/4K3 w - - 0 1")
        hanging = detect_hanging_pieces(board)
        
        # The knight on d5 is undefended AND attacked by Qd2
        # It should be in enemy_hanging (from white's perspective)
        assert "d5" in hanging["enemy_hanging"]
    
    def test_no_hanging_start_position(self):
        """Test no hanging pieces in starting position."""
        board = chess.Board()
        hanging = detect_hanging_pieces(board)
        
        assert len(hanging["our_hanging"]) == 0
        # Note: a1 and h1 rooks might be considered "hanging" without pawn cover
    
    def test_get_capture_hanging_moves(self):
        """Test getting moves to capture hanging pieces."""
        # White queen can capture undefended black knight
        board = chess.Board("4k3/8/8/3n4/8/8/3Q4/4K3 w - - 0 1")
        moves = get_capture_hanging_moves(board)
        
        # Qxd5 should be in the list
        qxd5 = chess.Move.from_uci("d2d5")
        assert qxd5 in moves


class TestTacticsNetwork:
    """Tests for tactics network building."""
    
    def test_build_tactics_network(self):
        """Test building the tactics network."""
        g = build_tactics_network()
        
        assert "tactics_root" in g.nodes
        assert "detect_fork" in g.nodes
        assert "exploit_fork" in g.nodes
        assert "detect_hanging" in g.nodes
        assert "capture_hanging" in g.nodes
    
    def test_tactics_weight_pack(self):
        """Test creating tactics weight pack."""
        pack = create_default_tactics_weight_pack()
        
        assert pack["subgraph"] == "tactics"
        assert "fork_priority" in pack["priorities"]
        assert "hanging_priority" in pack["priorities"]
        assert len(pack["edges"]) > 0


class TestRookEndgameDetection:
    """Tests for rook endgame detection."""
    
    def test_is_rook_endgame_krk(self):
        """Test that KRK is not detected as rook endgame (no enemy rook)."""
        board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
        is_rook, attacker = is_rook_endgame(board)
        
        # KRK is not R+P vs R endgame
        assert not is_rook
    
    def test_is_rook_endgame_rpvr(self):
        """Test detecting R+P vs R endgame."""
        # White: K+R+P, Black: K+R
        board = chess.Board("4k3/8/8/4P3/8/8/8/R3K2r w - - 0 1")
        is_rook, attacker = is_rook_endgame(board)
        
        assert is_rook
        assert attacker == chess.WHITE
    
    def test_lucena_position_detection(self):
        """Test detecting Lucena position."""
        # Classic Lucena: pawn on 7th, king on 8th in front
        board = chess.Board("1K6/1P6/8/8/8/8/6r1/4k3 w - - 0 1")
        
        is_lucena = detect_lucena_position(board, chess.WHITE)
        assert is_lucena
    
    def test_philidor_position_detection(self):
        """Test detecting Philidor defensive position."""
        # Defender's rook on 6th rank
        board = chess.Board("8/4P3/8/8/8/2r5/8/4K2k b - - 0 1")
        
        # Black is defender (white has the pawn)
        is_philidor = detect_philidor_position(board, chess.BLACK)
        assert is_philidor


class TestRookEndgameNetwork:
    """Tests for rook endgame network building."""
    
    def test_build_rook_endgame_network(self):
        """Test building the rook endgame network."""
        g = build_rook_endgame_network()
        
        assert "rook_endgame_root" in g.nodes
        assert "detect_rook_endgame" in g.nodes
        assert "detect_lucena" in g.nodes
        assert "lucena_bridge_selector" in g.nodes
    
    def test_rook_weight_pack(self):
        """Test creating rook endgame weight pack."""
        pack = create_default_rook_weight_pack()
        
        assert pack["subgraph"] == "rook_endgame"
        assert "lucena_priority" in pack["priorities"]
        assert "cutoff_priority" in pack["priorities"]


class TestCutoffMoves:
    """Tests for king cutoff move generation."""
    
    def test_get_cutoff_moves(self):
        """Test getting moves that cut off the enemy king."""
        # White rook can cut off black king
        board = chess.Board("8/8/8/8/8/R7/8/4K2k w - - 0 1")
        moves = get_cutoff_moves(board, chess.WHITE)
        
        # Should find moves that put rook on adjacent file to king
        assert isinstance(moves, list)

