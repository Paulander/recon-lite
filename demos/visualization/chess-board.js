// Chess Board Module
// Handles chess board rendering from FEN or move data

class ChessBoard {
    constructor() {
        this.boardElement = null;
        this.chess = null; // Will hold Chess.js instance
        this.lastEnv = null; // Cache the last env that had position info
    }

    init() {
        this.boardElement = document.getElementById('chess-board');
        this.boardElement.innerHTML = '<div class="chess-info">Loading chess position...</div>';
    }

    // Render chess board from FEN or moves
    render(env) {
        const boardElement = this.boardElement;

        const hasPositionData = (candidate) => {
            if (!candidate) return false;
            if (candidate.moves && candidate.moves.length > 0) return true;
            if (typeof candidate.fen === 'string' && candidate.fen.length > 0) return true;
            if (typeof candidate.initial_fen === 'string' && candidate.initial_fen.length > 0) return true;
            return false;
        };

        let renderEnv = env;
        if (hasPositionData(renderEnv)) {
            this.lastEnv = {
                fen: renderEnv.fen,
                initial_fen: renderEnv.initial_fen,
                moves: Array.isArray(renderEnv.moves) ? [...renderEnv.moves] : undefined,
            };
        } else if (this.lastEnv) {
            renderEnv = {
                fen: this.lastEnv.fen,
                initial_fen: this.lastEnv.initial_fen,
                moves: this.lastEnv.moves ? [...this.lastEnv.moves] : undefined,
            };
        }

        if (!hasPositionData(renderEnv)) {
            boardElement.innerHTML = '<div class="chess-info">No position data available</div>';
            return;
        }

        // Create chess instance
        this.chess = null;

        if (renderEnv.moves && renderEnv.moves.length > 0) {
            // Use move-based format for efficiency
            this.chess = new Chess(renderEnv.initial_fen || renderEnv.fen || undefined);
            // Apply all moves to get current position (UCI parsing)
            renderEnv.moves.forEach(moveUci => {
                if (typeof moveUci !== 'string' || moveUci.length < 4) {
                    console.warn(`Invalid move string:`, moveUci);
                    return;
                }
                const from = moveUci.slice(0, 2);
                const to = moveUci.slice(2, 4);
                const promotion = moveUci.length >= 5 ? moveUci.slice(4, 5) : undefined;
                const spec = promotion ? { from, to, promotion } : { from, to };
                const m = this.chess.move(spec);
                if (!m) {
                    console.warn(`Invalid UCI move for chess.js: ${moveUci}`);
                }
            });
        } else if (renderEnv.fen || renderEnv.initial_fen) {
            // Fallback to FEN format
            this.chess = new Chess(renderEnv.fen || renderEnv.initial_fen);
        } else {
            // No position data available
            boardElement.innerHTML = '<div class="chess-info">No position data available</div>';
            return;
        }

        let boardHtml = '';
        const board = this.chess.board();

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const square = board[row][col];
                const isLight = (row + col) % 2 === 0;
                const squareClass = isLight ? 'white' : 'black';

                boardHtml += `<div class="square ${squareClass}">`;

                if (square) {
                    const piece = square.type;
                    const color = square.color === 'w' ? 'white' : 'black';
                    const pieceSymbol = this.getPieceSymbol(piece, color);
                    boardHtml += `<div class="piece ${color}">${pieceSymbol}</div>`;
                }

                boardHtml += '</div>';
            }
        }

        boardElement.innerHTML = boardHtml;
    }

    // Get chess piece symbol
    getPieceSymbol(piece, color) {
        const symbols = {
            'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚',
            'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔'
        };
        return symbols[color === 'white' ? piece.toUpperCase() : piece.toLowerCase()] || '';
    }

    // Get current FEN position
    getCurrentFen() {
        return this.chess ? this.chess.fen() : null;
    }
}

// Export for use in other modules
window.ChessBoard = ChessBoard;
