# mip_morph

Read a MIP in MPS format, apply a randomized sequence of **equivalence-preserving** transformations,
and write it out under a different name. The result is the same optimization problem — same optimal
objective value, same feasibility status — but structurally unrecognizable.

Each run also writes a **key**, which recovers the original model, and translates a solution of the
hidden model back to the original variables. Without the key the hiding is not reversible.

## Usage

```bash
python -m mipmorph model.mps -o hidden.mps --seed 42 --rounds 40
```

```
positional:
  input                 input model (.mps, .mps.gz, .lp, ...)

options:
  -o, --output PATH     output model (format follows the extension)
  --seed N              RNG seed; the same seed reproduces the same output byte for byte
  --rounds N            how many transformations to apply (default 40)
  --enable  a,b,c       only draw from these transformations
  --disable a,b,c       exclude these transformations
  --sloppy-scaling      use arbitrary factors in [0.1, 10] instead of exact powers of two
  --max-coef-ratio F    reject transforms that push the coefficient range past F (default 1e6)
  --mix-max-nnz N       skip equality donors denser than this when mixing (default 25)
  --dispatch-fanout N   how many rows one equality is scattered across by 'dispatch'
                        (default 4); higher hides the donor row better but grows
                        coefficients faster
  --preserve-objective  never alter the cost vector (default for zero-objective models)
  --no-preserve-objective
                        allow objective mixing even on a zero-objective model
  --no-rename           keep the original names
  --no-shuffle          keep the original row/column order
  --key PATH            where to write the recovery key (default <output>.key)
  --no-key              write no key, making the hiding one-way
  --key-password PW     encrypt the key; also read from $MIPMORPH_KEY_PASSWORD
  --sol PATH            solve the original first and write its solution, carried onto
                        the hidden model, to PATH
  --time-limit SECONDS  time limit for that solve; the best solution found is used
  -q, --quiet           only report errors
```

Every run also writes a **key** next to the output, which is what makes hiding reversible:

```bash
python -m mipmorph unhide     hidden.mps -o recovered.mps   # -> the original model
python -m mipmorph unhide-sol hidden.sol -o original.sol    # -> the original variables
```

Both read `<input>.key` unless `--key` says otherwise. See [The key](#the-key) below.

`--sol` goes the other way: it solves the original, then carries that solution through every
transformation as it is applied, so the hidden model comes with a known feasible point of the same
objective value (useful as a MIP start, or to check a solver on the hidden model). The point is
carried during the run rather than rebuilt from the key, because a `slack` value depends on its row
as it stood at that step. Fresh XOR binaries are set to `(x, 0, 0)` and a split free variable to
`(max(x, 0), max(-x, 0))`; any of the equivalent choices would do.

### Checking that a run preserves the objective

```bash
python -m mipmorph check model.mps --rounds 1000 --disable xor-binary
python -m mipmorph check model.mps --sol model.sol --no-solve   # no Gurobi solve at all
python -m mipmorph check model.mps --runs 100 --seed 0          # seeds 0..99, one line each
python -m mipmorph check model.mps --runs 1000 -x               # stop at the first failure
```

`check` runs the ordinary driver (default 1000 rounds, and the same transformation options as
hiding) while carrying a point of the original through every step. After each step it recomputes the
point's objective, offset included, and its worst bound, integrality and row violation on the model
as it stands, and reports any step that moves either past `--tol` (default 1e-6) by its position and
description. The point is the original's optimum, or the solution in `--sol`. It then solves the
hidden model and compares the two optima within `--opt-tol` (default 1e-4, Gurobi's default
`MIPGap`); `--no-solve` skips that. `--runs N` repeats the whole check with N seeds — `S, S+1, ...`
under `--seed S`, random ones otherwise — reusing the one reference point, and ends with the seeds
that failed; every seed is printed, so any run can be replayed alone, and `-x` / `--stop-on-failure`
stops at the first failing run.

A failing run saves what reproduces it under `--failure-dir` (default `mipmorph-check-failures/`):
`seed<S>.mps` is the hidden model — byte for byte what `mipmorph model.mps -o … --seed S` with the
same options writes — with its key `seed<S>.mps.key`, and `seed<S>.sol` is the expected solution of
it: the reference point as carried through every step, labelled with the objective it should have.
`original.sol` holds the reference point itself, and `seed<S>.txt` the command that replays the run
and every step's objective and violation, marking the steps that made things worse rather than every
step that merely inherited the damage. The exit status is 0 when everything holds, 1 when a check
fails, and 2 on an error.

The per-step check shows the hidden model has a point as good as the original's optimum; only the
final solve shows it has none better. `check` is developer-only and is not part of the release
build.

As a library:

```python
from mip_morph import morph_model, read_problem, unmorph_problem, write_key, write_problem

problem = read_problem("model.mps")
hidden, log = morph_model(problem, rounds=40, seed=42)
write_problem(hidden, "hidden.mps")
write_key(log.key, "hidden.mps.key")

assert unmorph_problem(hidden, log.key) == problem

# Carry a solution (values in column order) onto the hidden model:
hidden, log = morph_model(problem, rounds=40, seed=42, solution=values)
log.solution  # {hidden column name: value}
```

## Transformations

| name           | what it does                                                                                       |
| -------------- | -------------------------------------------------------------------------------------------------- |
| `complement`   | substitutes `x = l + u - x'` on a bounded integer or binary                                        |
| `shift`        | substitutes `x = y + δ`, translating a variable's box (never a binary)                             |
| `shift-binary` | opt-in: shifts a binary off [0, 1] and retypes it integer, trading solver strength for hiding      |
| `xor-binary`   | replaces a binary by `x = u + v - 2w` with `w = u ∧ v`, making integrality _implied_               |
| `scale-var`    | substitutes `x = a·y` with `a > 0` (continuous columns only)                                       |
| `split-free`   | replaces a free variable by `x⁺ - x⁻`, both non-negative                                           |
| `scale-row`    | multiplies a row through by a nonzero factor, flipping the sense if negative                       |
| `invert-row`   | turns a `<=` row into a `>=` row by negating it                                                    |
| `slack`        | turns an inequality into an equality with an explicit non-negative slack, folded into a second row |
| `mix-row`      | adds a multiple of an equality row to another row                                                  |
| `dispatch`     | scatters **one** equality row into several rows at once, a different multiple into each            |
| `mix-obj`      | adds a multiple of an equality row to the objective                                                |

Names are mangled to `C0001…`/`R0001…` and row/column order is permuted at the end, unless
`--no-rename` / `--no-shuffle` are given.

### Why row mixing is the interesting one

Adding a _redundant_ row is easy to detect: a redundant row is a non-negative combination of the
others, which an LP will find. `mix-row` instead performs Gaussian-elimination-style row operations
**in place**. If `a'x = b` holds at every feasible point then `a'x - b == 0` there, so adding
`λ·(a'x - b)` to any other row leaves the feasible set exactly unchanged — for either sign of `λ`.
That rewrites existing coefficients and changes the sparsity pattern without adding any structure to
find, and the row count does not change.

It composes with `slack`: turning inequalities into equalities manufactures the donors that
`mix-row` and `mix-obj` then consume.

### Dispatching one row into many

`dispatch` takes a single equality and adds a _different_ multiple of it to each of several rows in
one go. Correctness does not depend on the fanout — `a'x - b == 0` at every feasible point, so each
addition is independently exact — so the fanout only controls fill-in and coefficient growth
(`--dispatch-fanout`, default 4).

It is worth having as its own transformation rather than just running `mix-row` more. `mix-row`
re-picks a random donor every round, so its edits scatter across many donors and no single row's
contribution grows large. Dispatching commits to one donor and pushes it through the matrix at once,
which is what actually dissolves that row's identity: afterwards its coefficient pattern lives in
its targets instead of sitting in one recognizable row.

It is also the only row operation here measured to leave the model **harder to reduce than the
original**. On `pret60_25` at 40 rounds, medians over 8 hiding seeds:

| configuration        | presolved cols | presolved rows | presolved nz | settled in presolve |
| -------------------- | -------------- | -------------- | ------------ | ------------------- |
| original             | 24             | 88             | 336          | —                   |
| `slack` only         | 24             | 80             | 306          | 0/8                 |
| `slack,mix-row`      | 24             | 80             | 296          | **3/8**             |
| `slack,dispatch`     | **35**         | **100**        | **373**      | 0/8                 |
| everything (default) | 32             | 96             | 367          | 0/8                 |

Two things to read off that. Dispatching leaves presolve with more model than it started with, which
is the goal. And `mix-row` on its own let presolve _prove infeasibility outright_ on 3 of 8 seeds —
a hiding failure, since the instance then solves in no nodes at all. Diluted among the other eleven
transformations in a default run it does not happen (0/8), but `--enable slack,mix-row` is not a
good configuration on this instance.

The price of dispatching is coefficient growth: ratios of 32–168 against 1–16 for the milder
operations. That is far below the 1e6 rollback threshold, so the trade is cheap.

### The cancelling variant, and why it was removed

The obvious refinement is to choose the multiplier instead of drawing it: `λ = -g_j/e_j` cancels
column `j` out of target row `g`. Apply it to every row holding `j` and the column is confined to
its donor — textbook Gaussian elimination, nominally the _inverse_ of the substitution presolve
performs, with much milder growth than a random multiplier.

It was implemented, measured, and deleted. Confining a column to one row makes it a column
singleton, and every slack this package creates is continuous, costless and free — so the result is
precisely the free-column-singleton reduction that `slack`'s fold exists to defeat. On `pret60_25`,
whose only continuous columns _are_ the slacks, 14 and 20 elimination steps produced 14 and 20 free
singleton columns, and presolve then deleted each column together with its donor row: **15 columns
and 56 rows survived, against 24 and 88 for the untouched original**, and one seed in eight
collapsed entirely.

The general lesson is worth stating, because it is easy to get backwards: making a column sparser is
the direction presolve itself travels. A hiding transformation must add structure to a column's
support, never remove it. `tests/test_dispatch.py` asserts that `dispatch` never leaves a free
singleton and never shrinks any column's support.

### A bare slack is not an obfuscation

A slack added on its own lands as a **continuous, costless, free singleton column** — which is
precisely the free-column-singleton reduction every presolve implements. Measured on `pret60_25`, 20
slack-only applications left **20 of 20 slacks as singletons, and presolve returned the model at
exactly its original size, (24, 88)**. The transformation was a no-op against any solver.

Two plausible-looking fixes were measured and **neither removed a single singleton**: tightening the
slack's upper bound from the row's activity range, and typing it `INTEGER` when the row's
coefficients and rhs justify it. Presolve does not care how well-described a singleton is; it cares
that the column appears once.

What works is folding the fresh equality into a second row immediately, so the new column has at
least two nonzeros and no reduction can eliminate it in isolation. That takes singletons to **0 of
20**, and the slacked model no longer collapses back to its original size. The fold obeys the same
`--mix-max-nnz` cap as `mix-row`, so a dense row is not smeared across the matrix; when the cap or a
one-row model blocks the fold, the log says the slack was left as a singleton rather than pretending
otherwise.

So `slack`'s real value is manufacturing equality donors for `mix-row`/`mix-obj`. Doing one mix at
birth is what makes the added column pay for itself.

## Design notes

Two behaviours of Gurobi shaped the implementation:

- **Ranged rows never reach the IR.** An MPS `RANGES` entry is desugared at read time into an
  equality row plus a bounded auxiliary column, so the IR only needs the senses `<`, `>`, `=`. Those
  manufactured equalities are free mixing donors.
- **`BINARY` bounds are clamped to [0, 1]** regardless of what `lb`/`ub` say, with no error. A
  transformation that moved a binary's bounds off [0, 1] would therefore have them silently
  discarded, changing what the model means. No transformation does, and `model_from_problem` refuses
  such a column outright rather than letting it through.

### Binaries keep their type

Being declared 0/1 is worth real solver strength: clique cuts, cover cuts, probing and implied-bound
reasoning all key off it. So `shift` **excludes binaries** — translating one off [0, 1] would force
it to a general integer and throw that away, for no gain in obfuscation. `complement` is the only
nontrivial affine map of {0, 1} onto itself, so that is the cheap way to disguise a binary.

If you do want binaries translated, `--enable shift-binary,...` turns on `shift-binary`. Shifting by
an integer `δ` is exact — the column ranges over `{-δ, 1-δ}` — but it cannot stay BINARY (Gurobi
clamps a BINARY off [0, 1] to `LB = UB = 0`), so it is retyped as a general integer on the shifted
box and the key records its original type. It is never drawn in a default run, because it gives up
exactly the strength described above.

`xor-binary` is the thorough way. It substitutes `x = u + v - 2w` for fresh binaries `u, v, w`, with
`w = u ∧ v` linearized as `w ≤ u`, `w ≤ v`, `w ≥ u + v - 1`. Over the integers those rows admit
exactly `(0,0,0)`, `(0,1,0)`, `(1,0,0)`, `(1,1,1)`, so `u + v - 2w` takes exactly the values {0, 1};
the LP relaxation of the system projects onto exactly `[0, 1]`, so nothing tightens or loosens
either. The original column stays, keeping its bounds, cost and every row entry, but is **relaxed to
continuous** — its integrality is now implied by the new binaries rather than declared, so the
original 0/1 structure no longer appears in the variable types at all.

The catch is size: each application adds three columns and four rows. In a mixed run the driver caps
it at 25% of the rounds (`DEFAULT_GROWTH_FRACTION`) so it cannot dominate and bloat the model;
requesting it alone with `--enable xor-binary` is not throttled.

### Feasibility problems keep an empty objective

A model whose objective is identically zero is a _pure feasibility_ problem, and
`--preserve-objective` is on by default for one. Objective mixing would preserve the optimal value —
zero stays zero — while still giving the model a dense cost vector, and that changes the kind of
problem the solver sees: presolve treats feasibility problems differently, and a model that reported
a clean `INFEASIBLE` can come back as `INF_OR_UNBD`. Pass `--no-preserve-objective` to override; the
CLI warns if the output gains an objective the input did not have.

### Judging difficulty needs several solver seeds

Do not compare node counts between an original and a hidden model from one run each. Row and column
ordering alone moves branching and cut selection, and on a hard instance the effect dwarfs
everything the transformations do.

Measured on `pret60_25` (infeasible, 60 columns), node counts as medians over 3 solver seeds,
sampled across 12 hiding seeds:

| configuration                        | min | q1  | median  | q3  | max   |
| ------------------------------------ | --- | --- | ------- | --- | ----- |
| original, no reformulation           |     |     | 6530    |     |       |
| `--rounds 0` (rename + permute only) | 0   | 0   | **1**   | 76  | 13442 |
| `--rounds 40` (everything)           | 1   | 97  | **332** | 651 | 8064  |

`--rounds 0` produces a model with the identical multiset of rows and coefficients — only the names
and the ordering differ — and it spans 0 to 13442 nodes, _wider_ than the fully transformed model
and straddling the original in both directions. One permutation solves `pret60_25` in **0 nodes**
versus 6530 for the original file. So a hidden model that looks dramatically easier is the expected
behaviour of this instance class under reordering, not evidence of a simplification.

Holding the ordering fixed with `--no-shuffle --no-rename` isolates the reformulations themselves.
Across the same 12 hiding seeds that gives
`[104, 160, 445, 447, 451, 876, 950, 1516, 2174, 5990, 6568, 6790]` — a median of 913 against the
original's 6530, but a range that reaches it. So the honest summary is that the reformulations do
shift the distribution downward on this instance while individual seeds still land at original
difficulty; the effect is a shift in a broad distribution, not a simplification of the problem.
Feasibility status is preserved throughout (`INFEASIBLE` in every run above).

`tests/test_hardness.py` compares medians over several `Seed` values for this reason, and asserts
that `--rounds 0` leaves the mathematics untouched.

### Bit-exactness, verified against exact arithmetic

Scale factors and mixing multipliers are dyadic (powers of two) by default. Multiplying by a power
of two is always exact, but that alone does not make the package exact: row mixing _adds_ dyadic
multiples, and a sum of two dyadics rounds as soon as it needs more than 53 mantissa bits.

Solver output cannot settle whether that happens. Two formulations with _identical_ mathematics
still report objective values differing by ~1e-12, because the simplex takes a different path
through them — measured that way `dispatch` showed 2e-12 and the cancelling variant 7e-11, numbers
that reveal nothing about the coefficients.

Replaying the same transformation chain over `fractions.Fraction` does settle it: same choices,
exact arithmetic, so any difference in the resulting coefficients is a rounding error in the float
path. **Zero of 1793 coefficients differ, at chains up to 120 rounds deep** — bounds, costs and
right-hand sides included. So the float path is genuinely bit-exact here, and the earlier 1e-11
discrepancies were the solver, not the data.

This is a measured property of the default settings rather than a theorem: what keeps the sums
representable is the coefficient-range guard holding the exponent spread narrow.
`tests/test_exactness.py` pins it, and asserts the contrast — `--sloppy-scaling` really does round,
which is why it is opt-in. On a matrix that is entirely ±1, like the `pret*` instances, sloppy
factors also show up plainly in Gurobi's `Matrix range`.

Repeated mixing grows coefficients exponentially, which is both numerically dangerous and an obvious
tell, so the driver rolls back any transformation that would push `max|coef| / min|coef|` past
`--max-coef-ratio`.

Models with quadratic objectives or constraints, SOS constraints, general constraints,
piecewise-linear objectives, or semi-continuous variables are refused outright rather than silently
mishandled.

## The key

Hiding used to be one-way. It is now reversible, through a key written alongside the hidden model
that records the exact edit list: `key + hidden model → the original model`, and
`key + a solution of the hidden model → values of the original variables`. Pass `--no-key` to get
the old one-way behaviour back.

Solutions come back because the row operations are identity on the variable space — `mix-row`,
`dispatch`, `scale-row`, `invert-row` and `mix-obj` rewrite the constraints a point satisfies, not
the point — so only the four variable substitutions have to be inverted, and each has a closed form
(`x = total - x'`, `x = y + δ`, `x = a·y`, `x = x⁺ - x⁻`).

### Why the printed log cannot be the key

The human-readable log is lossy in five separate ways, which is why the key records parameters
separately from the description:

| loss                                                                    | consequence                              |
| ----------------------------------------------------------------------- | ---------------------------------------- |
| every float is `%g`-formatted, i.e. six significant digits              | `--sloppy-scaling` factors unrecoverable |
| `dispatch` names neither its target rows nor its per-target multipliers | the fanout cannot be undone              |
| `slack` reports where it folded but not the multiplier                  | the fold cannot be undone                |
| `xor-binary` overwrites the original `vtype` and never reports it       | BINARY vs INTEGER lost                   |
| `rename_all` discards the name map, `shuffle_all` both permutations     | nothing can be lined up again            |

Replaying the driver from its seed is not an alternative either. The coefficient-range guard rolls a
candidate back **after** it has already drawn from the RNG, and only accepted steps are logged, so
the seed and the list of accepted steps do not determine one another.

### What makes the inverse exact

Two properties the forward transformations already had:

- **Indices mean the same thing all run.** No transformation deletes a column or a row —
  `split-free` _retires_ its column rather than removing it — and every addition is appended. So the
  original columns keep indices `0…n₀-1` for the whole run, and whatever a step appended is still at
  the tail when that step comes to be undone.
- **Steps are undone in reverse.** Walking backwards, by the time step _k_ is reached the model is
  exactly in its post-step-_k_ state, so every index it recorded still points where it did and the
  rows a fold read are unmodified since. No inverse has to reason about what came after it.

Only the final shuffle defeats indices, so recovery starts by re-indexing the hidden model onto the
_pre-shuffle_ names stored in the key. That one step undoes the permutation and absorbs any
reordering the writer or reader introduced, so nothing downstream has to trust file order.

Two places need care. `scale-var` must **divide by the recorded factor**, never multiply by a stored
reciprocal — `1/0.1` is not representable. And a fold has to restore the target's _sparsity_, not
just its numbers, since the forward pass drops entries that cancel to zero; the key records which
columns each fold introduced rather than letting the inverse guess from the values, because under
`--sloppy-scaling` two folds touching one entry leave a rounding residue where a clean zero would
have been.

With that, recovery of a default run is **exact, not approximate** — `tests/test_key.py` asserts
coefficient-for-coefficient equality over 20 random models at chain depths of 1, 5, 40 and 120, and
through a write/read round trip in both MPS and LP format.

### Three honest limits

1. **The key recovers the model as read, not the original file's bytes.** Gurobi's reader desugars
   an MPS `RANGES` entry into an equality plus an auxiliary column, and drops explicitly stored
   zeros, both _before_ the IR exists. Those happen upstream of anything a key can observe, so a
   model with `RANGES` recovers to its desugared equivalent — the same mathematics, not the same
   MPS.
2. **`--sloppy-scaling` recovers only to floating-point tolerance**, because the forward pass itself
   rounds there. The key still stores the exact factor, which is the best available.
3. **A key without a password is obfuscated, not encrypted.** On disk it is canonical JSON, then
   zlib, then base64 behind a magic line, so it is not human-readable and will not be skimmed by
   accident — but anyone who wants the contents can inflate them in two lines. Pass `--key-password`
   (or `$MIPMORPH_KEY_PASSWORD`, which keeps it out of the process list) for actual confidentiality:
   scrypt for the KDF, an HMAC-SHA256 counter-mode keystream, and encrypt-then-MAC so a damaged key
   is refused rather than half-decoded. This matters, because a leaked key undoes the hiding
   completely.

## Tests

```bash
python -m pytest tests/ -q
```

Equivalence is asserted through the solver rather than through a mapping, since hiding deliberately
changes names, column counts and row order. The suite covers exact per-transformation arithmetic on
hand-built models, a property test over 20 random MIPs × 30 transformations, chains up to 120 rounds
deep, preservation of infeasible and unbounded status, MPS round-tripping including a `RANGES`
fixture, and the CLI.

Two suites check things the solver cannot show directly: `tests/test_exactness.py` replays
transformation chains in exact rational arithmetic to verify the float path does not round, and
`tests/test_dispatch.py` asserts the structural invariant that separates hiding from presolve — no
transformation may shrink a column's support or leave a free column singleton behind.

Recovery is the one place equality _is_ asserted directly, since a key makes the mapping explicit:
`tests/test_key.py` compares the recovered model to the original coefficient by coefficient, inverts
each transformation on its own, maps solutions back and evaluates them on the original model, and
checks that a key from another run, another model, or a wrong password is refused rather than
half-applied.

`tests/test_check.py` covers `mipmorph check`, including that it pins a deliberately broken forward
map on the step that broke it.

Test models stay well under 2000 variables/constraints so they run on the size-limited Gurobi trial
license.
