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
        this.showLabels = true;
        this.lastFrame = null;
        this.lastNewRequests = new Set();
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

    static cloneDeep(obj) {
        if (obj === null || obj === undefined) return obj;
        if (typeof structuredClone === 'function') {
            try {
                return structuredClone(obj);
            } catch (err) {
                // Fall through to JSON fallback
            }
        }
        return JSON.parse(JSON.stringify(obj));
    }

    // Base layout definition used for computing responsive positions
    static get layoutConfig() {
        const profile = (typeof window !== 'undefined' && window.__RECON_PROFILE__) || 'generic';

        const base = {
            baseWidth: 1400,
            baseHeight: 900,
            nodeRadius: 28
        };

        const genericPositions = {
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
            'confinement_moves': { x: 600, y: 492 },
            'barrier_placement_moves': { x: 680, y: 492 },
            'box_shrink_moves': { x: 320, y: 620 },
            'opposition_moves': { x: 640, y: 620 },
            'mate_moves': { x: 960, y: 620 },
            'random_legal_moves': { x: 640, y: 720 },

            // Evaluators/sensors
            'box_can_shrink': { x: 320, y: 780 },
            'king_at_edge': { x: 640, y: 780 },
            'king_confined': { x: 560, y: 830 },
            'barrier_ready': { x: 720, y: 830 },
            'can_take_opposition': { x: 640, y: 840 },
            'can_deliver_mate': { x: 960, y: 780 },
            'is_stalemate': { x: 1040, y: 360 },

            // Per-phase scripts and wait gates
            'p0_check': { x: 240, y: 320 },
            'p0_move': { x: 320, y: 380 },
            'p0_wait': { x: 400, y: 440 },
            'wait_after_p0': { x: 320, y: 500 },
            'cut_established': { x: 200, y: 260 },

            'p1_check': { x: 560, y: 320 },
            'p1_move': { x: 640, y: 380 },
            'p1_wait': { x: 720, y: 440 },
            'wait_after_p1': { x: 640, y: 500 },

            'p2_check': { x: 240, y: 560 },
            'p2_move': { x: 320, y: 600 },
            'p2_wait': { x: 400, y: 660 },
            'wait_after_p2': { x: 320, y: 700 },

            'p3_check': { x: 560, y: 560 },
            'p3_move': { x: 640, y: 600 },
            'p3_wait': { x: 720, y: 660 },
            'wait_after_p3': { x: 640, y: 700 },

            'p4_check': { x: 880, y: 560 },
            'p4_move': { x: 960, y: 600 },
            'p4_wait': { x: 1040, y: 660 },
            'wait_after_p4': { x: 960, y: 700 },

            // Root sentinels / monitors
            'rook_lost': { x: 1120, y: 200 },
            'no_progress_watch': { x: 1120, y: 260 }
        };

        const krkOverrides = {
            'krk_root': { x: 700, y: 90 },
            'wait_for_board_change': { x: 700, y: 200 },

            'phase0_establish_cut': { x: 300, y: 340 },
            'phase1_drive_to_edge': { x: 700, y: 340 },
            'phase2_shrink_box': { x: 300, y: 520 },
            'phase3_take_opposition': { x: 700, y: 520 },
            'phase4_deliver_mate': { x: 1080, y: 520 },

            'p0_check': { x: 220, y: 300 },
            'p0_move': { x: 300, y: 380 },
            'p0_wait': { x: 380, y: 440 },
            'wait_after_p0': { x: 300, y: 500 },
            'cut_established': { x: 180, y: 250 },
            'choose_phase0': { x: 300, y: 430 },

            'p1_check': { x: 620, y: 300 },
            'p1_move': { x: 700, y: 380 },
            'p1_wait': { x: 780, y: 440 },
            'wait_after_p1': { x: 700, y: 500 },
            'king_drive_moves': { x: 700, y: 430 },
            'confinement_moves': { x: 640, y: 430 },
            'barrier_placement_moves': { x: 760, y: 430 },

            'p2_check': { x: 220, y: 560 },
            'p2_move': { x: 300, y: 600 },
            'p2_wait': { x: 380, y: 660 },
            'wait_after_p2': { x: 300, y: 700 },
            'box_shrink_moves': { x: 300, y: 630 },

            'p3_check': { x: 620, y: 560 },
            'p3_move': { x: 700, y: 600 },
            'p3_wait': { x: 780, y: 660 },
            'wait_after_p3': { x: 700, y: 700 },
            'opposition_moves': { x: 700, y: 630 },

            'p4_check': { x: 1000, y: 560 },
            'p4_move': { x: 1080, y: 600 },
            'p4_wait': { x: 1160, y: 660 },
            'wait_after_p4': { x: 1080, y: 700 },
            'mate_moves': { x: 1080, y: 630 },

            'box_can_shrink': { x: 300, y: 780 },
            'king_at_edge': { x: 700, y: 780 },
            'king_confined': { x: 620, y: 820 },
            'barrier_ready': { x: 780, y: 820 },
            'can_take_opposition': { x: 700, y: 860 },
            'can_deliver_mate': { x: 1080, y: 780 },
            'is_stalemate': { x: 1180, y: 380 },
            'random_legal_moves': { x: 700, y: 740 },
            'no_progress_watch': { x: 1240, y: 260 },
            'rook_lost': { x: 1240, y: 200 }
        };

        const positions = profile === 'krk'
            ? { ...genericPositions, ...krkOverrides }
            : genericPositions;

        return {
            ...base,
            positions
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
            // Root hierarchy
            ['krk_root', 'phase0_establish_cut', 'SUB'],
            ['krk_root', 'phase1_drive_to_edge', 'SUB'],
            ['krk_root', 'phase2_shrink_box', 'SUB'],
            ['krk_root', 'phase3_take_opposition', 'SUB'],
            ['krk_root', 'phase4_deliver_mate', 'SUB'],
            ['krk_root', 'is_stalemate', 'SUB'],
            ['krk_root', 'no_progress_watch', 'SUB'],
            ['krk_root', 'rook_lost', 'SUB'],

            // Phase sequencing
            ['phase0_establish_cut', 'phase1_drive_to_edge', 'POR'],
            ['phase1_drive_to_edge', 'phase2_shrink_box', 'POR'],
            ['phase2_shrink_box', 'phase3_take_opposition', 'POR'],
            ['phase3_take_opposition', 'phase4_deliver_mate', 'POR'],

            // Phase 0 internals
            ['phase0_establish_cut', 'p0_check', 'SUB'],
            ['phase0_establish_cut', 'p0_move', 'SUB'],
            ['phase0_establish_cut', 'p0_wait', 'SUB'],
            ['p0_check', 'p0_move', 'POR'],
            ['p0_move', 'p0_wait', 'POR'],
            ['p0_check', 'cut_established', 'SUB'],
            ['p0_move', 'choose_phase0', 'SUB'],
            ['p0_wait', 'wait_after_p0', 'SUB'],

            // Phase 1 internals
            ['phase1_drive_to_edge', 'p1_check', 'SUB'],
            ['phase1_drive_to_edge', 'p1_move', 'SUB'],
            ['phase1_drive_to_edge', 'p1_wait', 'SUB'],
            ['p1_check', 'p1_move', 'POR'],
            ['p1_move', 'p1_wait', 'POR'],
            ['p1_check', 'king_at_edge', 'SUB'],
            ['p1_move', 'king_drive_moves', 'SUB'],
            ['p1_move', 'confinement_moves', 'SUB'],
            ['p1_move', 'barrier_placement_moves', 'SUB'],
            ['p1_wait', 'wait_after_p1', 'SUB'],

            // Phase 2 internals
            ['phase2_shrink_box', 'p2_check', 'SUB'],
            ['phase2_shrink_box', 'p2_move', 'SUB'],
            ['phase2_shrink_box', 'p2_wait', 'SUB'],
            ['p2_check', 'p2_move', 'POR'],
            ['p2_move', 'p2_wait', 'POR'],
            ['p2_check', 'box_can_shrink', 'SUB'],
            ['p2_move', 'box_shrink_moves', 'SUB'],
            ['p2_wait', 'wait_after_p2', 'SUB'],

            // Phase 3 internals
            ['phase3_take_opposition', 'p3_check', 'SUB'],
            ['phase3_take_opposition', 'p3_move', 'SUB'],
            ['phase3_take_opposition', 'p3_wait', 'SUB'],
            ['p3_check', 'p3_move', 'POR'],
            ['p3_move', 'p3_wait', 'POR'],
            ['p3_check', 'can_take_opposition', 'SUB'],
            ['p3_move', 'opposition_moves', 'SUB'],
            ['p3_wait', 'wait_after_p3', 'SUB'],

            // Phase 4 internals
            ['phase4_deliver_mate', 'p4_check', 'SUB'],
            ['phase4_deliver_mate', 'p4_move', 'SUB'],
            ['phase4_deliver_mate', 'p4_wait', 'SUB'],
            ['p4_check', 'p4_move', 'POR'],
            ['p4_move', 'p4_wait', 'POR'],
            ['p4_check', 'can_deliver_mate', 'SUB'],
            ['p4_move', 'mate_moves', 'SUB'],
            ['p4_wait', 'wait_after_p4', 'SUB']
        ];
    }

    // Node categories
    static get sensorNodes() {
        return new Set([
            'wait_for_board_change',
            'king_at_edge', 'box_can_shrink', 'can_take_opposition', 'can_deliver_mate', 'is_stalemate',
            'king_confined', 'barrier_ready',
            'cut_established',
            'wait_after_p0','wait_after_p1','wait_after_p2','wait_after_p3','wait_after_p4',
            'rook_lost',
            'no_progress_watch'
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
        let drawFrame = frame;
        let requestSet = newReqSet;
        const hasFramePayload = drawFrame && typeof drawFrame === 'object';
        const hasRequestPayload = arguments.length > 1;

        if (!hasFramePayload) {
            drawFrame = this.lastFrame ? NetworkVisualization.cloneDeep(this.lastFrame) : { nodes: {} };
        }

        if (hasRequestPayload) {
            this.lastNewRequests = new Set(requestSet || []);
        } else if (this.lastNewRequests) {
            requestSet = new Set(this.lastNewRequests);
        } else {
            requestSet = new Set();
        }

        const rawNodes = (drawFrame && drawFrame.nodes) ? drawFrame.nodes : {};
        let nodesState = this.normalizeNodes(rawNodes);
        const nodeCount = Object.keys(nodesState).length;

        if (nodeCount > 1) {
            this.lastFrame = NetworkVisualization.cloneDeep(drawFrame);
        } else if (this.lastFrame) {
            drawFrame = NetworkVisualization.cloneDeep(this.lastFrame);
            nodesState = this.normalizeNodes(drawFrame.nodes || {});
        }

        const canvas = this.ctx.canvas;

        // Clear canvas
        this.ctx.clearRect(0, 0, canvas.width, canvas.height);

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
            const isNewReq = requestSet.has(dst);
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
            if (this.showLabels) {
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
            }
        });
    }

    setShowLabels(show) {
        this.showLabels = !!show;
        if (this.lastFrame) {
            this.draw();
        }
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
