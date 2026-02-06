/**
 * Board Overlay System for Chess Visualization
 * 
 * Provides:
 * - Arrow drawing for tactics
 * - Square highlighting
 * - Attack indicators
 * - King safety zones
 */

class BoardOverlays {
    constructor(boardElement) {
        this.board = boardElement;
        this.svgLayer = null;
        this.setupSVGLayer();
    }
    
    setupSVGLayer() {
        // Create SVG layer for arrows
        const existing = this.board.querySelector('.overlay-svg');
        if (existing) {
            this.svgLayer = existing;
            return;
        }
        
        this.svgLayer = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svgLayer.classList.add('overlay-svg');
        this.svgLayer.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 10;
        `;
        
        // Add arrow marker definition
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        
        // Red arrow marker
        const markerRed = this.createArrowMarker('arrow-red', '#f85149');
        defs.appendChild(markerRed);
        
        // Orange arrow marker
        const markerOrange = this.createArrowMarker('arrow-orange', '#d29922');
        defs.appendChild(markerOrange);
        
        // Blue arrow marker
        const markerBlue = this.createArrowMarker('arrow-blue', '#58a6ff');
        defs.appendChild(markerBlue);
        
        // Green arrow marker
        const markerGreen = this.createArrowMarker('arrow-green', '#3fb950');
        defs.appendChild(markerGreen);
        
        this.svgLayer.appendChild(defs);
        
        // Make board position relative if not already
        if (getComputedStyle(this.board).position === 'static') {
            this.board.style.position = 'relative';
        }
        
        this.board.appendChild(this.svgLayer);
    }
    
    createArrowMarker(id, color) {
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', id);
        marker.setAttribute('markerWidth', '10');
        marker.setAttribute('markerHeight', '10');
        marker.setAttribute('refX', '9');
        marker.setAttribute('refY', '5');
        marker.setAttribute('orient', 'auto');
        
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', 'M0,0 L0,10 L10,5 Z');
        path.setAttribute('fill', color);
        
        marker.appendChild(path);
        return marker;
    }
    
    clearArrows() {
        if (!this.svgLayer) return;
        
        // Remove all lines (arrows)
        const lines = this.svgLayer.querySelectorAll('line');
        lines.forEach(line => line.remove());
    }
    
    drawArrow(fromSquare, toSquare, color = 'red', label = '') {
        if (!this.svgLayer) return;
        
        // Get square positions
        const fromEl = this.board.querySelector(`[data-square="${fromSquare}"]`);
        const toEl = this.board.querySelector(`[data-square="${toSquare}"]`);
        
        if (!fromEl || !toEl) return;
        
        const boardRect = this.board.getBoundingClientRect();
        const fromRect = fromEl.getBoundingClientRect();
        const toRect = toEl.getBoundingClientRect();
        
        // Calculate center points relative to board
        const fromX = fromRect.left - boardRect.left + fromRect.width / 2;
        const fromY = fromRect.top - boardRect.top + fromRect.height / 2;
        const toX = toRect.left - boardRect.left + toRect.width / 2;
        const toY = toRect.top - boardRect.top + toRect.height / 2;
        
        // Shorten arrow slightly so arrowhead doesn't overlap square
        const dx = toX - fromX;
        const dy = toY - fromY;
        const len = Math.sqrt(dx * dx + dy * dy);
        const shortenBy = 12;
        const endX = toX - (dx / len) * shortenBy;
        const endY = toY - (dy / len) * shortenBy;
        
        // Create line
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', fromX);
        line.setAttribute('y1', fromY);
        line.setAttribute('x2', endX);
        line.setAttribute('y2', endY);
        
        // Color mapping
        const colorMap = {
            'red': { stroke: '#f85149', marker: 'arrow-red' },
            'orange': { stroke: '#d29922', marker: 'arrow-orange' },
            'blue': { stroke: '#58a6ff', marker: 'arrow-blue' },
            'green': { stroke: '#3fb950', marker: 'arrow-green' },
        };
        
        const colorConfig = colorMap[color] || colorMap['red'];
        
        line.setAttribute('stroke', colorConfig.stroke);
        line.setAttribute('stroke-width', '4');
        line.setAttribute('stroke-opacity', '0.8');
        line.setAttribute('marker-end', `url(#${colorConfig.marker})`);
        
        this.svgLayer.appendChild(line);
        
        // Add label if provided
        if (label) {
            const midX = (fromX + endX) / 2;
            const midY = (fromY + endY) / 2;
            
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', midX);
            text.setAttribute('y', midY - 8);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('fill', colorConfig.stroke);
            text.setAttribute('font-size', '10');
            text.setAttribute('font-weight', 'bold');
            text.textContent = label;
            
            this.svgLayer.appendChild(text);
        }
    }
    
    highlightSquare(square, type) {
        const el = this.board.querySelector(`[data-square="${square}"]`);
        if (!el) return;
        
        // Add highlight class
        el.classList.add(type);
    }
    
    clearSquareHighlights() {
        const squares = this.board.querySelectorAll('.square');
        squares.forEach(sq => {
            sq.classList.remove('attacked', 'hanging', 'center', 'king-zone', 'fork-target', 'pin-target');
        });
    }
    
    applyOverlays(boardTags, options = {}) {
        const {
            showAttacks = true,
            showArrows = true,
            showCenter = false,
            showKingZones = false,
        } = options;
        
        // Clear existing
        this.clearArrows();
        this.clearSquareHighlights();
        
        if (!boardTags) return;
        
        // Apply square tags
        const squares = boardTags.squares || {};
        for (const [sqName, tags] of Object.entries(squares)) {
            if (showAttacks && (tags.includes('attacked_by_white') || tags.includes('attacked_by_black'))) {
                this.highlightSquare(sqName, 'attacked');
            }
            
            if (tags.includes('hanging')) {
                this.highlightSquare(sqName, 'hanging');
            }
            
            if (showCenter && tags.includes('center')) {
                this.highlightSquare(sqName, 'center');
            }
            
            if (showKingZones && (tags.includes('white_king_zone') || tags.includes('black_king_zone'))) {
                this.highlightSquare(sqName, 'king-zone');
            }
        }
        
        // Draw arrows
        if (showArrows && boardTags.arrows) {
            for (const arrow of boardTags.arrows) {
                this.drawArrow(arrow.from, arrow.to, arrow.color || 'red', arrow.label || '');
            }
        }
    }
}

// Global instance
let boardOverlays = null;

// Initialize board overlays
function initBoardOverlays() {
    const board = document.getElementById('chess-board');
    if (!board) return;
    
    boardOverlays = new BoardOverlays(board);
}

// Apply overlays function (called from main visualization)
function applyBoardOverlays(boardTags) {
    if (!boardOverlays) {
        initBoardOverlays();
    }
    
    if (!boardOverlays) return;
    
    // Read toggle states
    const showAttacks = document.getElementById('toggle-attacks')?.checked ?? true;
    const showArrows = document.getElementById('toggle-arrows')?.checked ?? true;
    const showCenter = document.getElementById('toggle-center')?.checked ?? false;
    const showKingZones = document.getElementById('toggle-king-zones')?.checked ?? false;
    
    boardOverlays.applyOverlays(boardTags, {
        showAttacks,
        showArrows,
        showCenter,
        showKingZones,
    });
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initBoardOverlays);

