# Changelog

## 0.1.0

- Extracted the dependency-light ReCoN core package.
- Added discrete and continuous activation execution modes.
- Added generic binding invalidation by signature.
- Added a grid-world example with optional trace JSON export.
- Added `FormalReConEngine` for explicit symbolic SUB/SUR/POR/RET message passing.
- Added a formal trace generator and static HTML trace viewer.
- Made the grid-world example use the formal executor by default, with
  `--engine pragmatic` available for the legacy high-level executor.
- Added standalone package metadata, MIT license, and GitHub CI configuration.
