// Data Manager Module
// Handles loading and managing visualization data

class DataManager {
    constructor() {
        this.visualizationData = [];
        this.externalEdges = null;
        const params = new URLSearchParams(window.location.search);
        this.seedFen = params.get('fen') || '';
    }

    // Load visualization data from JSON
    async loadVisualizationData() {
        try {
            const params = new URLSearchParams(window.location.search);
            const fileParam = params.get('file');
            const file = fileParam || '../outputs/krk_persistent_visualization.json';
            const url = `${file}${file.includes('?') ? '&' : '?'}t=${Date.now()}`;

            const response = await fetch(url, { cache: 'no-store' });
            if (!response.ok) throw new Error('Failed to load visualization data');

            this.visualizationData = await response.json();
            this.fillMissingNodeStates();
            this.applySeedFen();
            console.log('Loaded visualization data from', url, this.visualizationData);

            // Read graph edges if present on first frame
            if (this.visualizationData.length > 0 && this.visualizationData[0].graph && this.visualizationData[0].graph.edges) {
                this.externalEdges = this.visualizationData[0].graph.edges;
            }

            // Update header with data type info
            this.updateDataTypeDisplay();

            return this.visualizationData;

        } catch (error) {
            console.error('Error loading visualization data:', error);
            const params = new URLSearchParams(window.location.search);
            if (params.get('file')) {
                console.warn('File parameter provided; not falling back to mock data.');
                this.visualizationData = [];
                this.updateDataTypeDisplay();
                return this.visualizationData;
            }
            console.log('Falling back to mock data for testing');

            // Create mock data for testing
            this.visualizationData = this.createMockData();
            this.fillMissingNodeStates();
            this.applySeedFen();
            this.updateDataTypeDisplay();

            return this.visualizationData;
        }
    }

    // Update header with data type information
    updateDataTypeDisplay() {
        if (!this.visualizationData || this.visualizationData.length === 0) {
            document.getElementById('game-info').textContent = 'No data loaded';
            return;
        }

        const firstFrame = this.visualizationData[0];
        let infoText = '';

        // Check if it's an interactive game (has move_number)
        const hasGameMoves = this.visualizationData.some(frame =>
            frame.env && (frame.env.move_number !== undefined ||
                         frame.env.recons_move ||
                         frame.env.opponents_move)
        );

        // Check if it's evaluation data (has ReCoN node states)
        const hasReconNodes = this.visualizationData.some(frame =>
            frame.nodes && Object.keys(frame.nodes).length > 0
        );

        if (hasGameMoves && hasReconNodes) {
            infoText = '🎮 Interactive Game + ReCoN Evaluation - Full Experience!';
        } else if (hasGameMoves) {
            infoText = '🎮 Interactive Chess Game - Move by move';
        } else if (hasReconNodes) {
            infoText = '🧠 ReCoN Network Evaluation - Thinking process';
        } else {
            infoText = '📊 Static Data Visualization';
        }

        infoText += ` | ${this.visualizationData.length} frames loaded`;

        document.getElementById('game-info').textContent = infoText;
    }

    // Load JSON file manually
    async loadFromFile(file) {
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            this.visualizationData = data;
            this.fillMissingNodeStates();
            this.applySeedFen();
            this.externalEdges = (data.length > 0 && data[0].graph && data[0].graph.edges) ? data[0].graph.edges : null;
            this.updateDataTypeDisplay();
            return data;
        } catch (e) {
            console.error('Failed to load JSON file:', e);
            throw e;
        }
    }

    // Create mock data for testing
    createMockData() {
        return [
            {
                type: "snapshot",
                tick: 0,
                note: "Initial state",
                nodes: {
                    "krk_root": "INACTIVE",
                    "phase1_drive_to_edge": "INACTIVE",
                    "phase2_shrink_box": "INACTIVE",
                    "phase3_take_opposition": "INACTIVE",
                    "phase4_deliver_mate": "INACTIVE",
                    "king_at_edge": "INACTIVE",
                    "box_can_shrink": "INACTIVE",
                    "can_take_opposition": "INACTIVE",
                    "can_deliver_mate": "INACTIVE",
                    "is_stalemate": "INACTIVE"
                },
                env: {
                    "initial_fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
                    "moves": [],
                    "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
                },
                thoughts: "Initializing ReCoN network...",
                new_requests: []
            },
            {
                type: "snapshot",
                tick: 1,
                note: "Step 1",
                nodes: {
                    "krk_root": "REQUESTED",
                    "phase1_drive_to_edge": "REQUESTED",
                    "phase2_shrink_box": "REQUESTED",
                    "king_at_edge": "REQUESTED",
                    "box_can_shrink": "REQUESTED"
                },
                env: {
                    "initial_fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
                    "moves": [],
                    "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
                    "move_number": 1,
                    "evaluation_tick": 5
                },
                thoughts: "Evaluating KRK position...",
                new_requests: ["phase1_drive_to_edge", "phase2_shrink_box"]
            },
            {
                type: "snapshot",
                tick: 2,
                note: "ReCoN move 1: a1a2",
                nodes: {
                    "krk_root": "CONFIRMED",
                    "phase1_drive_to_edge": "CONFIRMED",
                    "king_drive_moves": "TRUE"
                },
                env: {
                    "initial_fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
                    "moves": ["a1a2"],
                    "fen": "4k3/8/8/8/8/8/R7/3K3 b - - 0 1",
                    "move_number": 1,
                    "recons_move": "a1a2"
                },
                thoughts: "Chose a1a2 on move 1",
                new_requests: []
            }
        ];
    }

    getData() {
        return this.visualizationData;
    }

    getExternalEdges() {
        return this.externalEdges;
    }

    applySeedFen() {
        if (!this.seedFen || !this.visualizationData || this.visualizationData.length === 0) {
            return;
        }
        const first = this.visualizationData[0];
        if (!first.env) {
            first.env = {};
        }
        if (!first.env.initial_fen && !first.env.fen) {
            first.env.initial_fen = this.seedFen;
            first.env.fen = this.seedFen;
        }
    }

    fillMissingNodeStates() {
        let lastNodes = null;
        this.visualizationData = (this.visualizationData || []).map((frame) => {
            if (!frame || typeof frame !== 'object') {
                return frame;
            }
            const nodes = frame.nodes;
            if (nodes && Object.keys(nodes).length > 1) {
                lastNodes = { ...nodes };
                return frame;
            }
            if (lastNodes) {
                frame.nodes = { ...lastNodes };
            }
            return frame;
        });
    }
}

// Export for use in other modules
window.DataManager = DataManager;
