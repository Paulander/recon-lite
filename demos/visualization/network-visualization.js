// Network Visualization Module
// Handles the ReCoN network canvas drawing and state management

class NetworkVisualization {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.externalEdges = null; // from JSON graph.edges if provided
        this.lastNodesState = {};  // for transition detection
        this.transitionSet = new Set(); // nodes that changed state this frame
    }

    // Node positions for network visualization (simplified)
    static get nodePositions() {
        return {
            // Root and gating
            'krk_root': { x: 400, y: 70 },
            'wait_for_board_change': { x: 400, y: 150 },

            // Phases (scripts)
            'phase0_establish_cut': { x: 220, y: 240 },
            'phase1_drive_to_edge': { x: 580, y: 240 },
            'phase2_shrink_box': { x: 220, y: 330 },
            'phase3_take_opposition': { x: 400, y: 330 },
            'phase4_deliver_mate': { x: 580, y: 330 },

            // Move generators (actuators)
            'choose_phase0': { x: 220, y: 290 },
            'king_drive_moves': { x: 580, y: 290 },
            'box_shrink_moves': { x: 220, y: 380 },
            'opposition_moves': { x: 400, y: 380 },
            'mate_moves': { x: 580, y: 380 },
            'random_legal_moves': { x: 400, y: 430 },

            // Evaluators/sensors
            'king_at_edge': { x: 120, y: 480 },
            'box_can_shrink': { x: 300, y: 480 },
            'can_take_opposition': { x: 500, y: 480 },
            'can_deliver_mate': { x: 680, y: 480 },
            'is_stalemate': { x: 680, y: 540 }
        };
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
            'king_at_edge', 'box_can_shrink', 'can_take_opposition', 'can_deliver_mate', 'is_stalemate'
        ]);
    }

    // State colors
    static get stateColors() {
        return {
            'INACTIVE': '#cfd8dc',
            'REQUESTED': '#64b5f6',
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

        const nodesState = (frame && frame.nodes) ? frame.nodes : {};

        // Determine edges to draw: JSON-provided else fallback static
        const edgesToDraw = this.externalEdges ?
            this.externalEdges.map(e => [e.src, e.dst, e.type]) :
            NetworkVisualization.edges.map(([a,b]) => [a,b,'SUB']);

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

        // Draw nodes
        Object.entries(NetworkVisualization.nodePositions).forEach(([nodeId, pos]) => {
            const state = nodesState[nodeId] || 'INACTIVE';
            const fill = NetworkVisualization.stateColors[state] || NetworkVisualization.stateColors['INACTIVE'];
            const border = NetworkVisualization.stateBorderColors[state] || '#607d8b';

            // Node circle
            const radius = 22;
            this.ctx.fillStyle = fill;
            this.ctx.strokeStyle = border;
            this.ctx.lineWidth = (state === 'REQUESTED' || state === 'WAITING') ? 4 : 2;

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

            // Label
            this.ctx.fillStyle = NetworkVisualization.textColorFor(fill);
            this.ctx.font = '12px Segoe UI, Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'top';
            const label = nodeId.replace(/_/g, ' ').toUpperCase();
            this.ctx.fillText(label, pos.x, pos.y + 26);
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
