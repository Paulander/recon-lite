// Network Visualization Module
// Handles the ReCoN network canvas drawing and state management

class NetworkVisualization {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.externalEdges = null; // from JSON graph.edges if provided
        this.lastNodesState = {};  // for transition detection
        this.transitionSet = new Set(); // nodes that changed state this frame
        this.compact = true; // render without wrapper script nodes by default
    }

    normalizeId(id) {
        const map = {
            ROOT: 'krk_root',
            PHASE0: 'phase0_establish_cut',
            CHOOSE_P0: 'choose_phase0',
            PHASE1: 'phase1_drive_to_edge',
            PHASE2: 'phase2_shrink_box',
            PHASE3: 'phase3_take_opposition',
            PHASE4: 'phase4_deliver_mate'
        };
        return map[id] || id;
    }

    normalizeNodes(nodes) {
        const out = {};
        Object.entries(nodes).forEach(([k, v]) => {
            out[this.normalizeId(k)] = v;
        });
        return out;
    }

    // Base layout definition used for computing responsive positions
    static get layoutConfig() {
        return {
            baseWidth: 1280,
            baseHeight: 900,
            nodeRadius: 28,
            positions: {
                // Root and gating
                'krk_root': { x: 640, y: 90 },
                'wait_for_board_change': { x: 640, y: 210 },

                // Phases (scripts)
                'phase0_establish_cut': { x: 320, y: 360 },
                'phase1_drive_to_edge': { x: 640, y: 360 },
                'phase2_shrink_box': { x: 320, y: 520 },
                'phase3_take_opposition': { x: 640, y: 520 },
                'phase4_deliver_mate': { x: 960, y: 520 },

                // Move generators (actuators)
                'choose_phase0': { x: 320, y: 440 },
                'king_drive_moves': { x: 640, y: 440 },
                'box_shrink_moves': { x: 320, y: 620 },
                'opposition_moves': { x: 640, y: 620 },
                'mate_moves': { x: 960, y: 620 },
                'random_legal_moves': { x: 640, y: 720 },

                // Evaluators/sensors
                'box_can_shrink': { x: 320, y: 780 },
                'king_at_edge': { x: 640, y: 780 },
                'can_take_opposition': { x: 640, y: 840 },
                'can_deliver_mate': { x: 960, y: 780 },
                'is_stalemate': { x: 1040, y: 360 },

                // New article-compliant per-phase scripts and wait gates
                'p0_check': { x: 240, y: 320 },
                'p0_move':  { x: 320, y: 380 },
                'p0_wait':  { x: 400, y: 440 },
                'wait_after_p0': { x: 320, y: 500 },
                'cut_established': { x: 200, y: 260 },

                'p1_check': { x: 560, y: 320 },
                'p1_move':  { x: 640, y: 380 },
                'p1_wait':  { x: 720, y: 440 },
                'wait_after_p1': { x: 640, y: 500 },

                'p2_check': { x: 240, y: 560 },
                'p2_move':  { x: 320, y: 600 },
                'p2_wait':  { x: 400, y: 660 },
                'wait_after_p2': { x: 320, y: 700 },

                'p3_check': { x: 560, y: 560 },
                'p3_move':  { x: 640, y: 600 },
                'p3_wait':  { x: 720, y: 660 },
                'wait_after_p3': { x: 640, y: 700 },

                'p4_check': { x: 880, y: 560 },
                'p4_move':  { x: 960, y: 600 },
                'p4_wait':  { x: 1040, y: 660 },
                'wait_after_p4': { x: 960, y: 700 },

                // Root sentinels
                'rook_lost': { x: 1120, y: 200 }
            }
        };
    }

    static computeLayout(width, height) {
        const { baseWidth, baseHeight, nodeRadius, positions } = NetworkVisualization.layoutConfig;
        const scaleX = width / baseWidth;
        const scaleY = height / baseHeight;
        const scale = Math.min(scaleX, scaleY);
        const radius = Math.max(18, Math.round(nodeRadius * scale));
        const labelMargin = Math.max(radius * 0.55, 18 * scale);
        const labelFontSize = Math.round(Math.max(12, 14 * scale));
        const labelLineHeight = Math.round(labelFontSize + Math.max(2, 4 * scale));
        const labelPaddingX = Math.max(6, 10 * scale);
        const labelPaddingY = Math.max(4, 8 * scale);
        const labelMaxWidth = radius * 3.4;
        const edgeFontSize = Math.round(Math.max(9, 11 * scale));
        const edgeLabelOffset = Math.max(6, 12 * scale);
        const baseStrokeWidth = Math.max(1.5, 2.4 * scale);
        const emphasisStrokeWidth = Math.max(baseStrokeWidth + 1.5, 4 * scale);
        const baseEdgeWidth = Math.max(1.5, 2.2 * scale);
        const emphasisEdgeWidth = Math.max(baseEdgeWidth + 1, 3.4 * scale);
        const glowBlur = Math.max(8, 14 * scale);
        const glowLineWidth = Math.max(1.5, 2.4 * scale);
        const glowRadiusOffset = Math.max(4, 6 * scale);

        const scaledPositions = {};
        Object.entries(positions).forEach(([id, pos]) => {
            scaledPositions[id] = {
                x: pos.x * scaleX,
                y: pos.y * scaleY
            };
        });

        return {
            positions: scaledPositions,
            radius,
            scale,
            labelMargin,
            labelFontSize,
            labelLineHeight,
            labelPaddingX,
            labelPaddingY,
            labelMaxWidth,
            edgeFontSize,
            edgeLabelOffset,
            baseStrokeWidth,
            emphasisStrokeWidth,
            baseEdgeWidth,
            emphasisEdgeWidth,
            glowBlur,
            glowLineWidth,
            glowRadiusOffset
        };
    }

    static wrapLabel(ctx, text, maxWidth) {
        const words = text.split(' ');
        if (words.length === 0) return [''];
        const lines = [];
        let current = words.shift();

        words.forEach((word) => {
            const testLine = `${current} ${word}`;
            if (ctx.measureText(testLine).width <= maxWidth) {
                current = testLine;
            } else {
                lines.push(current);
                current = word;
            }
        });

        if (current) {
            lines.push(current);
        }

        return lines;
    }

    static formatLabel(nodeId) {
        return nodeId
            .replace(/_/g, ' ')
            .split(' ')
            .map((segment) => {
                if (segment.length <= 2) {
                    return segment.toUpperCase();
                }
                return segment.charAt(0).toUpperCase() + segment.slice(1).toLowerCase();
            })
            .join(' ');
    }

    // Edges between nodes
    static get edges() {
        return [
            // Root and gating
            ['krk_root', 'wait_for_board_change'],
            ['wait_for_board_change', 'phase0_establish_cut'],
            ['wait_for_board_change', 'phase1_drive_to_edge'],

            // Root subs
            ['krk_root', 'phase2_shrink_box'],
            ['krk_root', 'phase3_take_opposition'],
            ['krk_root', 'phase4_deliver_mate'],

            // Phase internals
            ['phase0_establish_cut', 'choose_phase0'],
            ['phase1_drive_to_edge', 'king_drive_moves'],
            ['phase2_shrink_box', 'box_shrink_moves'],
            ['phase3_take_opposition', 'opposition_moves'],
            ['phase4_deliver_mate', 'mate_moves'],

            // Evaluators
            ['phase1_drive_to_edge', 'king_at_edge'],
            ['phase2_shrink_box', 'box_can_shrink'],
            ['phase3_take_opposition', 'can_take_opposition'],
            ['phase4_deliver_mate', 'can_deliver_mate'],
            ['phase4_deliver_mate', 'is_stalemate']
        ];
    }

    // Node categories
    static get sensorNodes() {
        return new Set([
            'wait_for_board_change',
            'king_at_edge', 'box_can_shrink', 'can_take_opposition', 'can_deliver_mate', 'is_stalemate',
            'cut_established',
            'wait_after_p0','wait_after_p1','wait_after_p2','wait_after_p3','wait_after_p4',
            'rook_lost'
        ]);
    }

    // State colors
    static get stateColors() {
        return {
            'INACTIVE': '#cfd8dc',
            'REQUESTED': '#64b5f6',
            'ACTIVE': '#90caf9',
            'SUPPRESSED': '#b0bec5',
            'WAITING': '#ffd54f',
            'TRUE': '#81c784',
            'CONFIRMED': '#4dd0e1',
            'FAILED': '#e57373'
        };
    }

    static get stateBorderColors() {
        return {
            'INACTIVE': '#90a4ae',
            'REQUESTED': '#1e88e5',
            'ACTIVE': '#1976d2',
            'SUPPRESSED': '#607d8b',
            'WAITING': '#f9a825',
            'TRUE': '#2e7d32',
            'CONFIRMED': '#00838f',
            'FAILED': '#b71c1c'
        };
    }

    static hexToRgb(hex) {
        const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return m ? {
            r: parseInt(m[1], 16),
            g: parseInt(m[2], 16),
            b: parseInt(m[3], 16)
        } : { r: 255, g: 255, b: 255 };
    }

    static luminance(hex) {
        const { r, g, b } = NetworkVisualization.hexToRgb(hex);
        const a = [r, g, b].map(v => {
            v /= 255;
            return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
    }

    static textColorFor(bgHex) {
        return NetworkVisualization.luminance(bgHex) > 0.55 ? '#1f2937' : '#ffffff';
    }

    init() {
        this.canvas = document.getElementById('network-canvas');
        this.ctx = this.canvas.getContext('2d');

        function resizeCanvas() {
            this.canvas.width = this.canvas.offsetWidth;
            this.canvas.height = this.canvas.offsetHeight;
        }

        resizeCanvas.call(this);
        window.addEventListener('resize', resizeCanvas.bind(this));

        // Draw initial network
        this.draw();
    }

    setExternalEdges(edges) {
        this.externalEdges = edges;
    }

    draw(frame = null, newReqSet = new Set()) {
        const canvas = this.ctx.canvas;

        // Clear canvas
        this.ctx.clearRect(0, 0, canvas.width, canvas.height);

        const nodesState = (frame && frame.nodes) ? this.normalizeNodes(frame.nodes) : {};

        // Determine edges to draw: JSON-provided else fallback static
        let edgesToDraw = this.externalEdges ?
            this.externalEdges.map(e => [this.normalizeId(e.src), this.normalizeId(e.dst), e.type]) :
            NetworkVisualization.edges.map(([a,b]) => [a,b,'SUB']);

        // --- Compact mode: hide wrapper script nodes and rewire edges visually ---
        if (this.compact) {
            const hiddenRe = /^(p[0-4]_(check|move|wait)|wait_after_p[0-4])$/;
            const hidden = new Set();
            edgesToDraw.forEach(([src, dst]) => {
                if (hiddenRe.test(src)) hidden.add(src);
                if (hiddenRe.test(dst)) hidden.add(dst);
            });

            // Build SUB adjacency to synthesize phase->terminal edges when wrapper is hidden
            const subOut = new Map();
            const subIn = new Map();
            edgesToDraw.forEach(([src, dst, t]) => {
                if (t !== 'SUB') return;
                if (!subOut.has(src)) subOut.set(src, []);
                subOut.get(src).push(dst);
                if (!subIn.has(dst)) subIn.set(dst, []);
                subIn.get(dst).push(src);
            });

            const isPhase = (id) => (
                id === 'phase0_establish_cut' || id === 'phase1_drive_to_edge' ||
                id === 'phase2_shrink_box'   || id === 'phase3_take_opposition' ||
                id === 'phase4_deliver_mate'
            );

            // Filter out edges touching hidden nodes
            let filtered = edgesToDraw.filter(([src, dst]) => !hidden.has(src) && !hidden.has(dst));

            // For each hidden wrapper, add synthetic edges: (phase -> terminal) for each terminal child
            hidden.forEach((w) => {
                const parents = subIn.get(w) || [];
                const phaseParent = parents.find(isPhase);
                if (!phaseParent) return;
                const children = subOut.get(w) || [];
                children.forEach((c) => {
                    if (hidden.has(c)) return; // don't show hidden wait terminals
                    // Avoid duplicates
                    const exists = filtered.some(([s,d,t]) => s===phaseParent && d===c && t==='SUB');
                    if (!exists) filtered.push([phaseParent, c, 'SUB']);
                });
            });

            edgesToDraw = filtered;
        }

        const layout = NetworkVisualization.computeLayout(canvas.width, canvas.height);
        const positions = layout.positions;

        // Draw edges with labels
        edgesToDraw.forEach(([src, dst, etype]) => {
            const fromPos = positions[src];
            const toPos = positions[dst];
            if (!fromPos || !toPos) return;
            const dstState = nodesState[dst] || 'INACTIVE';
            const isNewReq = newReqSet.has(dst);
            const colorMap = { SUB: '#94a3b8', POR: '#1e88e5', RET: '#8e24aa', SUR: '#90a4ae' };
            this.ctx.strokeStyle = isNewReq ? '#1e88e5' : (dstState === 'TRUE' || dstState === 'CONFIRMED') ? '#2e7d32' : (colorMap[etype] || '#cbd5e1');
            this.ctx.lineWidth = isNewReq ? layout.emphasisEdgeWidth : layout.baseEdgeWidth;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(fromPos.x, fromPos.y);
            this.ctx.lineTo(toPos.x, toPos.y);
            this.ctx.stroke();

            // Label at edge midpoint
            const mx = (fromPos.x + toPos.x) / 2;
            const my = (fromPos.y + toPos.y) / 2;
            this.ctx.fillStyle = '#374151';
            this.ctx.font = `${layout.edgeFontSize}px Segoe UI, Arial`;
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'bottom';
            this.ctx.fillText(etype, mx, my - layout.edgeLabelOffset);
        });

        // Determine which nodes are actually present (avoid drawing unused)
        const presentNodes = new Set();
        Object.keys(nodesState).forEach((id) => presentNodes.add(id));
        edgesToDraw.forEach(([src, dst]) => { presentNodes.add(src); presentNodes.add(dst); });

        // Draw nodes
        presentNodes.forEach((nodeId) => {
            const pos = positions[nodeId];
            if (!pos) return;
            const state = nodesState[nodeId] || 'INACTIVE';
            const fill = NetworkVisualization.stateColors[state] || NetworkVisualization.stateColors['INACTIVE'];
            const border = NetworkVisualization.stateBorderColors[state] || '#607d8b';

            // Node circle
            const radius = layout.radius;
            this.ctx.fillStyle = fill;
            this.ctx.strokeStyle = border;
            this.ctx.lineWidth = (state === 'REQUESTED' || state === 'WAITING' || state === 'ACTIVE') ? layout.emphasisStrokeWidth : layout.baseStrokeWidth;

            if (NetworkVisualization.sensorNodes.has(nodeId)) {
                // Diamond for sensors
                this.ctx.beginPath();
                this.ctx.moveTo(pos.x, pos.y - radius);
                this.ctx.lineTo(pos.x + radius, pos.y);
                this.ctx.lineTo(pos.x, pos.y + radius);
                this.ctx.lineTo(pos.x - radius, pos.y);
                this.ctx.closePath();
                this.ctx.fill();
                this.ctx.stroke();
            } else {
                // Circle for scripts/actuators
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
                this.ctx.fill();
                this.ctx.stroke();
            }

            // Transition glow for nodes that changed state this frame
            if (this.transitionSet.has(nodeId)) {
                this.ctx.save();
                this.ctx.shadowColor = 'rgba(255,255,255,0.8)';
                this.ctx.shadowBlur = layout.glowBlur;
                this.ctx.beginPath();
                if (NetworkVisualization.sensorNodes.has(nodeId)) {
                    this.ctx.moveTo(pos.x, pos.y - (radius + layout.glowRadiusOffset));
                    this.ctx.lineTo(pos.x + (radius + layout.glowRadiusOffset), pos.y);
                    this.ctx.lineTo(pos.x, pos.y + (radius + layout.glowRadiusOffset));
                    this.ctx.lineTo(pos.x - (radius + layout.glowRadiusOffset), pos.y);
                    this.ctx.closePath();
                } else {
                    this.ctx.arc(pos.x, pos.y, radius + layout.glowRadiusOffset, 0, 2 * Math.PI);
                }
                this.ctx.lineWidth = layout.glowLineWidth;
                this.ctx.strokeStyle = '#ffffff';
                this.ctx.stroke();
                this.ctx.restore();
            }

            // Label
            const label = NetworkVisualization.formatLabel(nodeId);
            const labelFont = `${layout.labelFontSize}px Segoe UI, Arial`;
            this.ctx.font = labelFont;
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'top';
            const wrapped = NetworkVisualization.wrapLabel(this.ctx, label, layout.labelMaxWidth);
            const labelHeight = wrapped.length * layout.labelLineHeight;
            let labelTop = pos.y + radius + layout.labelMargin;
            if (labelTop + labelHeight > canvas.height - layout.labelMargin) {
                labelTop = pos.y - radius - layout.labelMargin - labelHeight;
            }
            const labelWidths = wrapped.map((line) => this.ctx.measureText(line).width);
            const labelWidth = labelWidths.length ? Math.max(...labelWidths) : 0;
            const bgLeft = pos.x - (labelWidth / 2) - layout.labelPaddingX;
            const bgTop = labelTop - layout.labelPaddingY;
            const bgWidth = labelWidth + layout.labelPaddingX * 2;
            const bgHeight = labelHeight + layout.labelPaddingY * 2;

            this.ctx.fillStyle = 'rgba(255,255,255,0.88)';
            this.ctx.fillRect(bgLeft, bgTop, bgWidth, bgHeight);
            this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.6)';
            this.ctx.lineWidth = Math.max(1, layout.scale);
            this.ctx.strokeRect(bgLeft, bgTop, bgWidth, bgHeight);

            this.ctx.fillStyle = '#0f172a';
            wrapped.forEach((line, idx) => {
                this.ctx.fillText(line, pos.x, labelTop + (idx * layout.labelLineHeight));
            });
        });
    }

    updateTransitionSet(currentNodes) {
        this.transitionSet = new Set();
        Object.keys(currentNodes).forEach(nid => {
            const prev = this.lastNodesState[nid];
            const cur = currentNodes[nid];
            if (prev !== undefined && prev !== cur) {
                this.transitionSet.add(nid);
            }
        });
        this.lastNodesState = { ...currentNodes };
    }
}

// Export for use in other modules
window.NetworkVisualization = NetworkVisualization;
