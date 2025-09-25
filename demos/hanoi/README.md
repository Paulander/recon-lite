# Tower of Hanoi Demo

This miniature demo shows how a ReCon graph can drive a strictly ordered
sequence using only SCRIPT and TERMINAL nodes linked by SUB edges and a POR
chain. The root SCRIPT node fans out to one SCRIPT per planned move, and each
move SCRIPT owns a single TERMINAL that performs the physical action by calling
into the demos.hanoi.env.Hanoi environment. With POR edges between the move
scripts, the engine requests them one at a time, ensuring the classic Tower of
Hanoi sequence plays out deterministically.

## Mapping Hanoi to ReCon

* Environment – demos.hanoi.env.Hanoi keeps the three pegs as stacks, exposes
  legal, move, is_goal, and renders an ASCII board so every tick prints the peg
  configuration.
* Planning – demos.hanoi.build.plan_hanoi generates the recursive (src, dst)
  move list. demos.hanoi.build.build_graph converts it into a ReCon graph with
  one SCRIPT/TERMINAL pair per move and a linear POR chain from the first to the
  last move.
* Execution – demos.hanoi.build.make_move_predicate wraps each move as a
  TERMINAL predicate. When requested, it moves a disc exactly once, stores the
  result in node.meta, and hands control back to the engine.

## Running the demo

    python -m demos.hanoi.run --n 3

Add --log-every 5 to capture periodic ReConEngine.snapshot notes alongside the
console trace. The runner prints the peg layout after every tick and ends with
a summary including whether the goal state was reached and how many moves were
executed (should match 2**n - 1). Try --n 4 to watch a longer run finish with 15
moves.
