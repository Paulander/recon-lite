# Endgame Dataset Layout

This folder holds curated FEN batches that drive the macrograph teacher/training loop.

```
data/endgames/
  kpk/    # King + pawn vs king conversions
  krk/    # King + rook vs king technique refresher
  rook/   # Generic rook ending motifs (bridge, cut-off, ladder)
```

Each directory can contain any number of `*.fen` text files. The `train_and_refresh.py`
utility will walk the tree, load every FEN (ignoring blank lines and `#` comments), and
aggregate teacher statistics before refreshing the macro weights.

You can drop additional scenario files anywhere under these folders. A few seeds are
included so the loop can run out of the box; extend/replace them as needed.
