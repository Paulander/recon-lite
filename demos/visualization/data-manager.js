// Data Manager Module
// Handles loading and managing visualization data

class DataManager {
    constructor() {
        this.visualizationData = [];
        this.externalEdges = null;
    }

    // Load visualization data from JSON
    async loadVisualizationData() {
        try {
            const response = await fetch('../krk_visualization_data.json');
            if (!response.ok) {
                throw new Error('Failed to load visualization data');
            }
            this.visualizationData = await response.json();
            console.log('Loaded visualization data:', this.visualizationData);

            // Read graph edges if present on first frame
            if (this.visualizationData.length > 0 && this.visualizationData[0].graph && this.visualizationData[0].graph.edges) {
                this.externalEdges = this.visualizationData[0].graph.edges;
            }

            // Update header with data type info
            this.updateDataTypeDisplay();

            return this.visualizationData;

        } catch (error) {
            console.error('Error loading visualization data:', error);
            console.log('Falling back to mock data for testing');

            // Create mock data for testing
            this.visualizationData = this.createMockData();
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
            frame.nodes && (frame.nodes.krk_root || frame.nodes.phase1_drive_to_edge)
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
            this.externalEdges = (data.length > 0 && data[0].graph && data[0].graph.edges) ? data[0].graph.edges : null;
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
}

// Export for use in other modules
window.DataManager = DataManager;
