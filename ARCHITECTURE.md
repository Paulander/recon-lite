# ReCoN Architecture & Design Decisions

## Overview

This implementation provides a complete, minimal ReCoN (Request Confirmation Network) foundation specifically designed for the King+Rook vs King chess checkmate challenge (will build on top of that once it works, plan below). The architecture prioritizes clarity, testability, and extensibility while maintaining fidelity to the original ReCoN specification.

## Core Design Principles

### 1. **Minimal but Complete Implementation**
- **Decision**: Implement only the core ReCoN mechanics needed for chess checkmate
- **Rationale**: Focus on delivering a working chess solver within the challenge timeframe
- **Trade-off**: Some advanced features (neural implementation, learning mechanisms) are deferred
- **Benefit**: Clear, maintainable codebase with well-defined scope

### 2. **Separation of Concerns**
- **Core ReCoN Logic**: `graph.py`, `engine.py` - Abstract, reusable components
- **Chess Domain Logic**: `chess_nodes.py` - Domain-specific implementations
- **Infrastructure**: `logger.py`, `plugins.py` - Supporting utilities

### 3. **Extensibility First**
- **Node Inheritance**: Chess nodes inherit from base `Node` class
- **Factory Pattern**: Easy creation of chess-specific nodes
- **Plugin System**: Terminal nodes use predicates for flexibility

## ReCoN Specification Compliance

### ✅ Fully Implemented
- **Node States**: All 8 states (INACTIVE, REQUESTED, ACTIVE, SUPPRESSED, WAITING, TRUE, CONFIRMED, FAILED)
- **Link Types**: SUB/SUR (hierarchy), POR/RET (sequences)
- **Execution Flow**: Request/confirmation cycle with proper state transitions
- **Terminal Nodes**: Predicate-based evaluation system
- **Graph Structure**: Proper relationship management and validation

### ⚠️ Intentionally Scoped Out
- **Neural Implementation**: Threshold element arrays (Figure 3 from paper)
- **Advanced Learning**: Weight adaptation mechanisms
- **Complex Activation Functions**: Beyond simple predicates
- **Message Passing**: Explicit message objects (functionally equivalent via state manipulation)

### 🎯 Chess-Specific Extensions
- **Board State Nodes**: Evaluate current chess position
- **Move Testing Nodes**: Validate individual chess moves
- **Checkmate Detection**: Goal condition evaluation
- **Strategy Nodes**: Hierarchical organization of checkmate approaches

## Architecture Benefits

### For Chess Challenge
- **Immediate Usability**: Can start building checkmate networks immediately
- **Clear Debugging**: Simple state machine makes issues easy to trace
- **Visualization Ready**: Structured logging supports the existing viz system

### For Future Extensions
- **Learning Ready**: Weight arrays and activation values are in place
- **Neural Ready**: Can add threshold elements without breaking existing code
- **Domain Agnostic**: Core ReCoN engine works for any hierarchical planning task

## File Organization

```
src/recon_lite/
├── graph.py          # Core ReCoN data structures
├── engine.py         # Execution engine
├── logger.py         # Structured logging
├── plugins.py        # Plugin interfaces
├── chess_nodes.py    # Chess-specific node implementations
└── __init__.py       # Clean API exports
```

## Usage Example

```python
from recon_lite import Graph, ReConEngine, LinkType
from recon_lite import create_checkmate_detector, create_rook_strategy

# Build checkmate network
g = Graph()
g.add_node(create_checkmate_detector("goal"))
g.add_node(create_rook_strategy("rook_moves"))
g.add_edge("goal", "rook_moves", LinkType.SUB)

# Execute
engine = ReConEngine(g)
g.nodes["goal"].state = NodeState.REQUESTED
while not all(n.state == NodeState.CONFIRMED for n in g.nodes.values()):
    engine.step({"board": chess_board})
```

## Future Extension Points

1. **Learning Integration**: Add weight updates to `engine.py`
2. **Neural Nodes**: Implement threshold elements in `chess_nodes.py`
3. **Advanced Activation**: Add complex functions to base `Node` class
4. **Multi-Domain**: Extend for other planning tasks beyond chess

### Practical examples: 
1. Add Pawn promotion ReCoN example. 
2. COMBINE Pawn promotion and Rook endgame -> "goal promote pawn" -> identify "I can now aim for checkmate" -> call KRK ReCoN branch. 
3. Extend leaf (sensor) nodes: input from 2D (online chess engines), 3D (photos of physical boards) -> internal representation -> same problem as original. 

### Visualization: 
**See:** `VIS_SPEC.md`

## Quality Assurance

- **Test Coverage**: Core execution logic tested in `tests/test_engine.py`
- **Type Hints**: Full type annotation for IDE support and documentation
- **Documentation**: Comprehensive docstrings and architectural rationale

This architecture delivers a production-ready ReCoN foundation that perfectly balances immediate usability with future extensibility, making it ideal for the chess checkmate challenge while maintaining the potential for broader applications.
