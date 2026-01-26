// Chess Board Module
// Handles chess board rendering from FEN or move data

class ChessBoard {
    constructor() {
        this.boardElement = null;
        this.chess = null; // Will hold Chess.js instance
        this.lastEnv = null; // Cache the last env that had position info
        this.bindingColors = {};
        this.bindingPalette = ['#0ea5e9', '#f97316', '#22c55e', '#a855f7', '#ef4444', '#14b8a6'];
        this.showBindings = true;
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
                binding: renderEnv.binding,
            };
        } else if (this.lastEnv) {
            renderEnv = {
                fen: this.lastEnv.fen,
                initial_fen: this.lastEnv.initial_fen,
                moves: this.lastEnv.moves ? [...this.lastEnv.moves] : undefined,
                binding: this.lastEnv.binding,
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
        const bindingMap = this.showBindings ? this.computeBindingMap(renderEnv.binding) : {};

        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const square = board[row][col];
                const isLight = (row + col) % 2 === 0;
                const squareClass = isLight ? 'white' : 'black';

                const file = 'abcdefgh'[col];
                const rank = 8 - row;
                const squareName = `${file}${rank}`;
                const bindings = bindingMap[squareName];

                boardHtml += `<div class="square ${squareClass}">`;

                if (square) {
                    const piece = square.type;
                    const color = square.color === 'w' ? 'white' : 'black';
                    const pieceSymbol = this.getPieceSymbol(piece, color);
                    boardHtml += `<div class="piece ${color}">${pieceSymbol}</div>`;
                }

                if (bindings && bindings.length) {
                    boardHtml += '<div class="binding-overlay">';
                    bindings.slice(0, 3).forEach((info) => {
                        const rawLabel = info.terminalId || info.feature || info.namespace || '•';
                        const compact = rawLabel.replace(/[^a-zA-Z0-9_]/g, '');
                        const label = compact.length > 6 ? compact.slice(0, 6) : compact;
                        const title = `${info.namespace} | ${info.terminalId || info.feature || 'binding'}`;
                        boardHtml += `<span class="binding-chip" style="background:${info.color}" title="${title}">${label}</span>`;
                    });
                    boardHtml += '</div>';
                }

                boardHtml += '</div>';
            }
        }

        boardElement.innerHTML = boardHtml;
    }

    setBindingsEnabled(enabled) {
        this.showBindings = Boolean(enabled);
        if (this.lastEnv) {
            this.render(this.lastEnv);
        }
    }

    colorForNamespace(namespace) {
        if (!namespace) return '#64748b';
        if (!this.bindingColors[namespace]) {
            const palette = this.bindingPalette;
            const index = Object.keys(this.bindingColors).length % palette.length;
            this.bindingColors[namespace] = palette[index];
        }
        return this.bindingColors[namespace];
    }

    colorForKey(key) {
        if (!key) return '#64748b';
        if (!this.bindingColors[key]) {
            const palette = this.bindingPalette;
            const index = Object.keys(this.bindingColors).length % palette.length;
            this.bindingColors[key] = palette[index];
        }
        return this.bindingColors[key];
    }

    computeBindingMap(bindingPayload) {
        const mapping = {};
        if (!bindingPayload || typeof bindingPayload !== 'object') {
            return mapping;
        }
        Object.entries(bindingPayload).forEach(([namespace, instances]) => {
            if (!Array.isArray(instances)) return;
            instances.forEach((instance) => {
                if (!instance || !Array.isArray(instance.items)) return;
                const feature = instance.feature || instance.id || namespace;
                const terminalId = instance.id || instance.terminal_id || instance.node || feature;
                const colorKey = `${namespace}:${terminalId}`;
                const color = this.colorForKey(colorKey);
                instance.items.forEach((item) => {
                    if (typeof item !== 'string') return;
                    const [kind, value] = item.split(':');
                    if (kind !== 'square' || !value) return;
                    const square = value.toLowerCase();
                    if (!mapping[square]) {
                        mapping[square] = [];
                    }
                    mapping[square].push({ namespace, feature, color, terminalId });
                });
            });
        });
        return mapping;
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
