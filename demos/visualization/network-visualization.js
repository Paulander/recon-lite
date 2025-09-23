// Network Visualization Module
// Handles the ReCoN network canvas drawing and state management

class NetworkVisualization {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.externalEdges = null; // from JSON graph.edges if provided
        this.lastNodesState = {};  // for transition detection
        this.transitionSet = new Set(); // nodes that changed state this frame
        this.compact = false; // render full network with wrapper script nodes by default
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

    // Node positions for network visualization (simplified)
    static get nodePositions() {
        const baseX = 160;
        const colSpacing = 150;
        const colX = (idx) => baseX + idx * colSpacing;
        const levels = {
            root: 80,
            watchers: 150,
            check: 230,
            phase: 320,
            move: 400,
            actuator: 480,
            waitGate: 530,
            sensor: 580,
            sentinelLow: 640
        };

        const positions = {
            // Root and global sentinels
            'krk_root': { x: colX(2), y: levels.root },
            'wait_for_board_change': { x: colX(2), y: levels.watchers },
            'no_progress_watch': { x: colX(1), y: levels.watchers },
            'is_stalemate': { x: colX(4.6), y: levels.watchers },
            'rook_lost': { x: colX(4.6), y: levels.sentinelLow },

            // Fallback actuator
            'random_legal_moves': { x: colX(2) + 80, y: levels.actuator + 50 }
        };

        const phaseConfigs = [
            {
                idx: 0,
                phase: 'phase0_establish_cut',
                prefix: 'p0',
                actuators: [
                    { id: 'choose_phase0', dx: -40 }
                ],
                sensors: [
                    { id: 'cut_established', dx: -50 }
                ]
            },
            {
                idx: 1,
                phase: 'phase1_drive_to_edge',
                prefix: 'p1',
                actuators: [
                    { id: 'king_drive_moves', dx: -60 },
                    { id: 'confinement_moves', dx: 0 },
                    { id: 'barrier_placement_moves', dx: 60 }
                ],
                sensors: [
                    { id: 'king_at_edge', dx: -70 },
                    { id: 'king_confined', dx: -10 },
                    { id: 'barrier_ready', dx: 60 }
                ]
            },
            {
                idx: 2,
                phase: 'phase2_shrink_box',
                prefix: 'p2',
                actuators: [
                    { id: 'box_shrink_moves', dx: 0 }
                ],
                sensors: [
                    { id: 'box_can_shrink', dx: 0 }
                ]
            },
            {
                idx: 3,
                phase: 'phase3_take_opposition',
                prefix: 'p3',
                actuators: [
                    { id: 'opposition_moves', dx: 0 }
                ],
                sensors: [
                    { id: 'can_take_opposition', dx: 0 }
                ]
            },
            {
                idx: 4,
                phase: 'phase4_deliver_mate',
                prefix: 'p4',
                actuators: [
                    { id: 'mate_moves', dx: 0 }
                ],
                sensors: [
                    { id: 'can_deliver_mate', dx: 0 }
                ]
            }
        ];

        phaseConfigs.forEach(({ idx, phase, prefix, actuators = [], sensors = [] }) => {
            const base = colX(idx);
            positions[phase] = { x: base, y: levels.phase };
            positions[`${prefix}_check`] = { x: base, y: levels.check };
            positions[`${prefix}_move`] = { x: base, y: levels.move };
            positions[`${prefix}_wait`] = { x: base + 70, y: levels.actuator };
            positions[`wait_after_${prefix}`] = { x: base + 70, y: levels.waitGate };

            actuators.forEach(({ id, dx = 0 }) => {
                positions[id] = { x: base + dx, y: levels.actuator };
            });

            sensors.forEach(({ id, dx = 0, level = 'sensor' }) => {
                const targetLevel = level === 'sensor' ? levels.sensor : levels[level] || levels.sensor;
                positions[id] = { x: base + dx, y: targetLevel };
            });
        });

        return positions;
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
            'wait_for_board_change', 'no_progress_watch',
            'king_at_edge', 'box_can_shrink', 'can_take_opposition', 'can_deliver_mate', 'is_stalemate',
            'cut_established', 'king_confined', 'barrier_ready',
            'wait_after_p0','wait_after_p1','wait_after_p2','wait_after_p3','wait_after_p4',
            'rook_lost'
        ]);
    }

    // State colors
    static get stateColors() {
        return {
            'INACTIVE': '#d8dee6',
            'REQUESTED': '#2563eb',
            'ACTIVE': '#0ea5e9',
            'SUPPRESSED': '#94a3b8',
            'WAITING': '#f59e0b',
            'TRUE': '#22c55e',
            'CONFIRMED': '#0d9488',
            'FAILED': '#ef4444'
        };
    }

    static get stateBorderColors() {
        return {
            'INACTIVE': '#94a3b8',
            'REQUESTED': '#1d4ed8',
            'ACTIVE': '#0284c7',
            'SUPPRESSED': '#64748b',
            'WAITING': '#b45309',
            'TRUE': '#15803d',
            'CONFIRMED': '#0f766e',
            'FAILED': '#b91c1c'
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
        return NetworkVisualization.luminance(bgHex) > 0.48 ? '#0f172a' : '#ffffff';
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

        let hiddenNodes = new Set();

        // --- Compact mode: hide wrapper script nodes and rewire edges visually ---
        if (this.compact) {
            const hiddenRe = /^(p[0-4]_(check|move|wait)|wait_after_p[0-4])$/;
            hiddenNodes = new Set();
            edgesToDraw.forEach(([src, dst]) => {
                if (hiddenRe.test(src)) hiddenNodes.add(src);
                if (hiddenRe.test(dst)) hiddenNodes.add(dst);
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
            let filtered = edgesToDraw.filter(([src, dst]) => !hiddenNodes.has(src) && !hiddenNodes.has(dst));

            // For each hidden wrapper, add synthetic edges: (phase -> terminal) for each terminal child
            hiddenNodes.forEach((w) => {
                const parents = subIn.get(w) || [];
                const phaseParent = parents.find(isPhase);
                if (!phaseParent) return;
                const children = subOut.get(w) || [];
                children.forEach((c) => {
                    if (hiddenNodes.has(c)) return; // don't show hidden wait terminals
                    // Avoid duplicates
                    const exists = filtered.some(([s,d,t]) => s===phaseParent && d===c && t==='SUB');
                    if (!exists) filtered.push([phaseParent, c, 'SUB']);
                });
            });

            edgesToDraw = filtered;
        }

        // Draw edges with labels
        edgesToDraw.forEach(([src, dst, etype]) => {
            const fromPos = NetworkVisualization.nodePositions[src];
            const toPos = NetworkVisualization.nodePositions[dst];
            if (!fromPos || !toPos) return;
            const dstState = nodesState[dst] || 'INACTIVE';
            const isNewReq = newReqSet.has(dst);
            const colorMap = { SUB: '#94a3b8', POR: '#1e88e5', RET: '#8e24aa', SUR: '#90a4ae' };
            this.ctx.strokeStyle = isNewReq ? '#1e88e5' : (dstState === 'TRUE' || dstState === 'CONFIRMED') ? '#2e7d32' : (colorMap[etype] || '#cbd5e1');
            this.ctx.lineWidth = isNewReq ? 3 : 2;
            this.ctx.beginPath();
            this.ctx.moveTo(fromPos.x, fromPos.y);
            this.ctx.lineTo(toPos.x, toPos.y);
            this.ctx.stroke();

            // Label at edge midpoint
            const mx = (fromPos.x + toPos.x) / 2;
            const my = (fromPos.y + toPos.y) / 2;
            this.ctx.fillStyle = '#374151';
            this.ctx.font = '10px Segoe UI, Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'bottom';
            this.ctx.fillText(etype, mx, my - 4);
        });

        // Determine which nodes are actually present (avoid drawing unused)
        const presentNodes = new Set();
        Object.keys(nodesState).forEach((id) => {
            if (!hiddenNodes.has(id)) presentNodes.add(id);
        });
        edgesToDraw.forEach(([src, dst]) => {
            if (!hiddenNodes.has(src)) presentNodes.add(src);
            if (!hiddenNodes.has(dst)) presentNodes.add(dst);
        });

        // Draw nodes
        presentNodes.forEach((nodeId) => {
            if (hiddenNodes.has(nodeId)) return;
            const pos = NetworkVisualization.nodePositions[nodeId];
            if (!pos) return;
            const state = nodesState[nodeId] || 'INACTIVE';
            const fill = NetworkVisualization.stateColors[state] || NetworkVisualization.stateColors['INACTIVE'];
            const border = NetworkVisualization.stateBorderColors[state] || '#607d8b';

            // Node circle
            const radius = 22;
            this.ctx.fillStyle = fill;
            this.ctx.strokeStyle = border;
            this.ctx.lineWidth = (state === 'REQUESTED' || state === 'WAITING' || state === 'ACTIVE') ? 4 : 2;

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
                this.ctx.shadowBlur = 12;
                this.ctx.beginPath();
                if (NetworkVisualization.sensorNodes.has(nodeId)) {
                    this.ctx.moveTo(pos.x, pos.y - (radius + 4));
                    this.ctx.lineTo(pos.x + (radius + 4), pos.y);
                    this.ctx.lineTo(pos.x, pos.y + (radius + 4));
                    this.ctx.lineTo(pos.x - (radius + 4), pos.y);
                    this.ctx.closePath();
                } else {
                    this.ctx.arc(pos.x, pos.y, radius + 4, 0, 2 * Math.PI);
                }
                this.ctx.lineWidth = 2;
                this.ctx.strokeStyle = '#ffffff';
                this.ctx.stroke();
                this.ctx.restore();
            }

            // Label (with contrasting backdrop for readability)
            const labelColor = NetworkVisualization.textColorFor(fill);
            const label = nodeId.replace(/_/g, ' ').toUpperCase();
            const labelFont = '12px Segoe UI, Arial';
            const labelY = pos.y + radius + 6;

            this.ctx.font = labelFont;
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'top';

            const metrics = this.ctx.measureText(label);
            const textHeight = (metrics.actualBoundingBoxAscent !== undefined && metrics.actualBoundingBoxDescent !== undefined)
                ? metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent
                : 12;
            const paddingX = 6;
            const paddingY = 3;
            const bgWidth = metrics.width + paddingX * 2;
            const bgHeight = textHeight + paddingY * 2;
            const bgX = pos.x - bgWidth / 2;
            const bgY = labelY - paddingY;
            const bgColor = labelColor === '#ffffff' ? 'rgba(15,23,42,0.65)' : 'rgba(255,255,255,0.85)';

            this.ctx.fillStyle = bgColor;
            this.ctx.fillRect(bgX, bgY, bgWidth, bgHeight);

            this.ctx.fillStyle = labelColor;
            this.ctx.fillText(label, pos.x, labelY);
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
