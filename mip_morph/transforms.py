"""Equivalence-preserving transformations.

Every transformation keeps the optimal objective value and the feasibility
status of the model intact, while changing its structure. Each one takes
``(problem, rng, ctx)`` and returns a :class:`~mip_morph.key.Step` describing
what it did, or ``None`` when the model offers no candidate to work on.

A ``Step`` prints as the short human-readable sentence these functions have
always returned, and additionally carries the exact parameters of the choice it
made, which is what makes a run invertible from a key. Those parameters are
column and row *indices*, never names: ``rename_all`` rewrites every name at the
end of a run, and a transformation that adds a column names it after the current
column count, so names are not stable identifiers here. Indices are -- no
transformation ever deletes a column or a row (``split-free`` retires its column
instead), and every addition is appended, so an index means the same thing for
the whole run.

Scale factors are dyadic (powers of two) by default so that every
multiplication is exact in IEEE double arithmetic and equivalence is bit-exact.
``ctx.sloppy_scaling`` trades that for arbitrary factors in [0.1, 10].
"""

from .ir import BINARY, CONTINUOUS, FLIP, INF, INTEGER, drop_zeros, negate
from .key import Step

DYADIC = (0.125, 0.25, 0.5, 2.0, 4.0, 8.0)
MIX_LAMBDAS = (-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0)
SHIFTS = (-5, -3, -2, -1, 1, 2, 3, 5)


class Context:
    """Tuning knobs shared by the transformations."""

    def __init__(
        self,
        sloppy_scaling=False,
        mix_max_nnz=25,
        preserve_objective=False,
        dispatch_fanout=4,
    ):
        self.sloppy_scaling = sloppy_scaling
        self.mix_max_nnz = mix_max_nnz
        self.preserve_objective = preserve_objective
        self.dispatch_fanout = dispatch_fanout

    def scale_factor(self, rng, signed=True):
        if self.sloppy_scaling:
            mag = rng.uniform(0.1, 10.0)
        else:
            mag = rng.choice(DYADIC)
        return -mag if signed and rng.random() < 0.5 else mag

    def positive_scale_factor(self, rng):
        if self.sloppy_scaling:
            return rng.uniform(0.1, 10.0)
        return rng.choice(DYADIC)

    def mix_lambda(self, rng):
        if self.sloppy_scaling:
            lam = rng.uniform(0.1, 4.0)
            return -lam if rng.random() < 0.5 else lam
        return rng.choice(MIX_LAMBDAS)


def _row_indices(problem):
    """All row indices.

    Picking an index rather than a row object costs nothing: ``rng.choice`` and
    ``rng.sample`` consume generator state as a function of ``len`` alone, so
    choosing over indices leaves every existing seed producing exactly the
    output it produced before.
    """
    return range(len(problem.rows))


# --------------------------------------------------------------------------
# variable substitutions
# --------------------------------------------------------------------------


def complement_variable(problem, rng, ctx):
    """Substitute ``x = l + u - x'`` for a bounded integer variable.

    Bounds are unchanged ([l, u] maps onto itself) and integrality survives
    because l and u are integral. Binaries stay binary: l + u == 1.
    """
    candidates = [
        j for j, v in enumerate(problem.vars) if v.is_integral and v.is_bounded
    ]
    if not candidates:
        return None

    j = rng.choice(candidates)
    var = problem.vars[j]
    total = var.lb + var.ub

    for row in problem.rows_with(j):
        coef = row.coeffs[j]
        row.rhs -= coef * total
        row.coeffs[j] = -coef

    problem.obj_offset += var.obj * total
    var.obj = negate(var.obj)
    # ``total`` is recorded rather than re-derived: the substitution leaves the
    # bounds alone, but a later ``shift`` or ``scale-var`` on the same column
    # moves them, so l + u read off the hidden model is the wrong number.
    return Step(
        "complement",
        {"var": j, "total": total},
        f"complement {var.name} (x -> {total:g} - x)",
    )


def shift_variable(problem, rng, ctx):
    """Substitute ``x = y + delta``, translating the variable's box.

    0/1 variables are deliberately excluded. Translating one off [0, 1] throws
    away the binary designation that clique, cover and implied-bound reasoning
    all depend on -- a real loss of solver strength for no gain in obfuscation.
    Use ``complement`` on one instead: it is the only nontrivial affine map of
    {0, 1} onto itself.

    The test is ``is_zero_one``, not ``vtype == BINARY``: MPS files routinely
    carry 0/1 columns in an INTEGER section, and those are just as much binaries
    to presolve as declared ones.
    """
    candidates = [
        j for j, v in enumerate(problem.vars) if v.is_bounded and not v.is_zero_one
    ]
    if not candidates:
        return None

    j = rng.choice(candidates)
    var = problem.vars[j]
    delta = float(rng.choice(SHIFTS))

    var.lb -= delta
    var.ub -= delta

    for row in problem.rows_with(j):
        row.rhs -= row.coeffs[j] * delta

    problem.obj_offset += var.obj * delta
    return Step("shift", {"var": j, "delta": delta}, f"shift {var.name} by {delta:g}")


def shift_binary(problem, rng, ctx):
    """Substitute ``x = y + delta`` for a 0/1 column, retyping it INTEGER.

    The opt-in counterpart of ``shift``. Exact: ``y`` ranges over
    ``{-delta, 1 - delta}``, which is integral because ``delta`` is. The column
    cannot stay BINARY -- Gurobi clamps a BINARY with bounds off [0, 1] to
    LB=UB=0 -- so it becomes a general INTEGER on the shifted box. That costs
    the clique, cover and implied-bound reasoning ``shift`` refuses to give up,
    which is why this is not in the default set; presolve may still recover the
    two-valued column, but that is up to the solver.

    The declared type is recorded: a 0/1 column may have been BINARY or an
    INTEGER pinned to [0, 1], and the hidden model no longer says which.
    """
    candidates = [j for j, v in enumerate(problem.vars) if v.is_zero_one]
    if not candidates:
        return None

    j = rng.choice(candidates)
    var = problem.vars[j]
    delta = float(rng.choice(SHIFTS))

    var.lb -= delta
    var.ub -= delta
    vtype = var.vtype
    var.vtype = INTEGER

    for row in problem.rows_with(j):
        row.rhs -= row.coeffs[j] * delta

    problem.obj_offset += var.obj * delta
    return Step(
        "shift-binary",
        {"var": j, "delta": delta, "vtype": vtype},
        f"shift binary {var.name} by {delta:g} (now integer in "
        f"[{var.lb:g}, {var.ub:g}])",
    )


def scale_variable(problem, rng, ctx):
    """Substitute ``x = a * y`` with a > 0, for continuous variables only.

    Restricted to continuous columns: rescaling an integer variable would make
    its feasible values a non-unit lattice, which integrality cannot express.
    """
    candidates = [j for j, v in enumerate(problem.vars) if not v.is_integral]
    if not candidates:
        return None

    j = rng.choice(candidates)
    var = problem.vars[j]
    factor = ctx.positive_scale_factor(rng)

    if var.lb > -INF:
        var.lb /= factor
    if var.ub < INF:
        var.ub /= factor
    var.obj *= factor

    for row in problem.rows_with(j):
        row.coeffs[j] *= factor

    return Step(
        "scale-var",
        {"var": j, "factor": factor},
        f"scale {var.name} (x -> {factor:g} y)",
    )


def encode_binary_as_xor(problem, rng, ctx):
    """Replace a binary ``x`` by the XOR of two fresh binaries.

    Substitutes ``x = u + v - 2w`` where ``w = u AND v`` is linearized by

        w <= u,   w <= v,   w >= u + v - 1

    Over integers the three AND rows admit exactly (0,0,0), (0,1,0), (1,0,0) and
    (1,1,1), so ``u + v - 2w`` takes exactly the values {0, 1} -- and the LP
    relaxation of that system projects onto exactly [0, 1], so the relaxation is
    unchanged too.

    Unlike ``complement``, this genuinely hides the variable: ``x`` is kept as a
    continuous column whose integrality is *implied* by the new binaries rather
    than declared, so the original 0/1 structure no longer appears in the
    model's variable types. Each application grows the model by three columns
    and four rows, so the driver applies it sparingly.

    Applies to any 0/1 column, whether declared BINARY or an INTEGER pinned to
    [0, 1].
    """
    candidates = [j for j, v in enumerate(problem.vars) if v.is_zero_one]
    if not candidates:
        return None

    j = rng.choice(candidates)
    var = problem.vars[j]
    tag = len(problem.vars)
    u = problem.add_var(f"_u{tag}", lb=0.0, ub=1.0, obj=0.0, vtype=BINARY)
    v = problem.add_var(f"_v{tag}", lb=0.0, ub=1.0, obj=0.0, vtype=BINARY)
    w = problem.add_var(f"_w{tag}", lb=0.0, ub=1.0, obj=0.0, vtype=BINARY)

    rows = [
        problem.add_row(f"_and{tag}a", {w: 1.0, u: -1.0}, "<", 0.0),
        problem.add_row(f"_and{tag}b", {w: 1.0, v: -1.0}, "<", 0.0),
        problem.add_row(f"_and{tag}c", {w: 1.0, u: -1.0, v: -1.0}, ">", -1.0),
        problem.add_row(f"_xor{tag}", {j: 1.0, u: -1.0, v: -1.0, w: 2.0}, "=", 0.0),
    ]

    # x keeps its bounds, cost and every existing row entry; only its declared
    # type relaxes, because integrality now follows from the linking row. That
    # type is the one thing here a key cannot re-derive from the hidden model --
    # a 0/1 column may have been declared BINARY or INTEGER -- so record it.
    vtype = var.vtype
    var.vtype = CONTINUOUS
    return Step(
        "xor-binary",
        {"var": j, "u": u, "v": v, "w": w, "rows": rows, "vtype": vtype},
        f"encode binary {var.name} as XOR of {problem.vars[u].name}, {problem.vars[v].name}",
    )


def split_free_variable(problem, rng, ctx):
    """Replace a free variable by the difference of two non-negative ones."""
    candidates = [j for j, v in enumerate(problem.vars) if v.lb <= -INF and v.ub >= INF]
    if not candidates:
        return None

    j = rng.choice(candidates)
    var = problem.vars[j]
    pos = problem.add_var(f"{var.name}_p", lb=0.0, ub=INF, obj=var.obj, vtype=var.vtype)
    neg = problem.add_var(
        f"{var.name}_n", lb=0.0, ub=INF, obj=negate(var.obj), vtype=var.vtype
    )

    for row in problem.rows_with(j):
        coef = row.coeffs.pop(j)
        row.coeffs[pos] = row.coeffs.get(pos, 0.0) + coef
        row.coeffs[neg] = row.coeffs.get(neg, 0.0) - coef
        drop_zeros(row)

    # Retire the original column rather than deleting it, so that the indices
    # held by every other row stay valid. The three values overwritten here are
    # recorded, because reading them back off the new columns is only correct
    # while nothing later touches them.
    lb, ub, obj = var.lb, var.ub, var.obj
    var.lb = 0.0
    var.ub = 0.0
    var.obj = 0.0
    return Step(
        "split-free",
        {"var": j, "pos": pos, "neg": neg, "lb": lb, "ub": ub, "obj": obj},
        f"split free {var.name} into {var.name}_p - {var.name}_n",
    )


# --------------------------------------------------------------------------
# row operations
# --------------------------------------------------------------------------


def scale_row(problem, rng, ctx):
    """Multiply a row through by a nonzero factor, flipping the sense if < 0."""
    if not problem.rows:
        return None

    i = rng.choice(_row_indices(problem))
    row = problem.rows[i]
    factor = ctx.scale_factor(rng)

    for j in row.coeffs:
        row.coeffs[j] *= factor
    row.rhs *= factor
    if factor < 0:
        row.sense = FLIP[row.sense]

    return Step(
        "scale-row",
        {"row": i, "factor": factor},
        f"scale row {row.name} by {factor:g}",
    )


def invert_row(problem, rng, ctx):
    """Turn a <= row into a >= row (or vice versa) by negating it."""
    candidates = [i for i, r in enumerate(problem.rows) if r.sense != "="]
    if not candidates:
        return None

    i = rng.choice(candidates)
    row = problem.rows[i]
    for j in row.coeffs:
        row.coeffs[j] = -row.coeffs[j]
    row.rhs = -row.rhs
    row.sense = FLIP[row.sense]
    return Step("invert-row", {"row": i}, f"invert row {row.name}")


def _fold_into(problem, donor_index, target_index, lam):
    """Add ``lam`` times row ``donor_index`` to row ``target_index``.

    Returns the fold's record: the multiplier, the target, and the columns the
    fold *introduced* into the target. That last list is what lets the fold be
    undone exactly. Subtracting the multiple back restores the numbers, but not
    the sparsity: the forward pass drops entries that cancel to zero, and an
    inverse working from the numbers alone cannot always tell an entry the fold
    created from one it merely changed. Under ``--sloppy-scaling`` two folds
    touching the same entry leave a rounding residue where a clean zero would
    have been, and guessing from that residue would leave a stray coefficient
    behind -- on a column that is about to be removed, which would be a dangling
    reference rather than a rounding error. Recording the answer costs a few
    integers and removes the guess.
    """
    donor = problem.rows[donor_index]
    target = problem.rows[target_index]
    created = sorted(j for j in donor.coeffs if j not in target.coeffs)
    for j, coef in donor.coeffs.items():
        target.coeffs[j] = target.coeffs.get(j, 0.0) + lam * coef
    target.rhs += lam * donor.rhs
    drop_zeros(target)
    return {"target": target_index, "lam": lam, "created": created}


def _spread_row_into_another(problem, rng, ctx, donor_index):
    """Fold equality row ``donor_index`` into some other row.

    Returns the fold record on success and ``None`` when there is no eligible
    target. Shared by ``slack``, which must not leave its new column as a
    singleton, so both callers obey the same density cap: a dense donor would
    otherwise smear across the matrix and show up as a fill-in tell.
    """
    donor = problem.rows[donor_index]
    if not 0 < donor.nnz <= ctx.mix_max_nnz:
        return None

    targets = [i for i in _row_indices(problem) if i != donor_index]
    if not targets:
        return None

    target_index = rng.choice(targets)
    lam = ctx.mix_lambda(rng)
    return _fold_into(problem, donor_index, target_index, lam)


def add_explicit_slack(problem, rng, ctx):
    """Turn an inequality into an equality with a non-negative slack column.

    The new column is immediately folded into a second row. On its own a slack
    is *not* an obfuscation: it lands as a continuous, costless, free singleton
    column, which is the textbook free-column-singleton presolve reduction.
    Measured on ``pret60_25``, 20 slack-only applications left 20 singleton
    columns and presolve returned the model at exactly its original size --
    the transformation was a no-op against any solver.

    Tightening the slack's bounds from the row activity, or typing it INTEGER
    when the row's data justifies it, does *not* fix that: both were measured
    and neither removed a single singleton. Only spreading the column over a
    second row does, because then no reduction can eliminate it in isolation.

    What slack is really for is manufacturing equality donors for ``mix-row``
    and ``mix-obj``, which is where the hard-to-detect work happens. Doing one
    mix here is what makes the new column pay for itself.
    """
    candidates = [i for i, r in enumerate(problem.rows) if r.sense != "="]
    if not candidates:
        return None

    i = rng.choice(candidates)
    row = problem.rows[i]
    slack = problem.add_var(f"_s{len(problem.vars)}", lb=0.0, ub=INF, obj=0.0)
    row.coeffs[slack] = 1.0 if row.sense == "<" else -1.0
    # The sign of the slack coefficient does encode the old sense, but only
    # until a later ``scale-row`` or ``invert-row`` touches the row, so record
    # the sense itself.
    old_sense = row.sense
    row.sense = "="

    fold = _spread_row_into_another(problem, rng, ctx, i)
    if fold is None:
        note = " (left as a singleton: nothing eligible to fold into)"
    else:
        note = f", folded into {problem.rows[fold['target']].name}"
    return Step(
        "slack",
        {"row": i, "slack": slack, "old_sense": old_sense, "fold": fold},
        f"add slack to row {row.name} ({old_sense}= -> =){note}",
    )


def _equality_donors(problem, ctx):
    return [
        i
        for i, r in enumerate(problem.rows)
        if r.sense == "=" and 0 < r.nnz <= ctx.mix_max_nnz
    ]


def dispatch_equality(problem, rng, ctx):
    """Scatter one equality row across several other rows at once.

    The natural extension of ``mix-row``: rather than one donor into one target
    per round, take a single equality ``a'x = b`` and add a *different* multiple
    of it to each of several rows. Since ``a'x - b == 0`` at every feasible
    point, each addition is independently exact, so the fanout is irrelevant to
    correctness -- only to fill-in and coefficient growth.

    Why it is worth having as its own transform rather than just running
    ``mix-row`` repeatedly: ``mix-row`` re-picks a random donor each round, so
    its edits are scattered over many donors and any single row's contribution
    stays small. Dispatching commits to one donor and pushes it through the
    matrix in one go, which is what actually dissolves that row's identity --
    afterwards the donor's coefficient pattern is spread over its targets rather
    than sitting in one recognizable row.

    It is also the one row operation here measured to leave the model *harder*
    to reduce than the original. On ``pret60_25`` at 40 rounds, medians over 8
    hiding seeds: presolve leaves 35 columns and 100 rows against the original's
    24 and 88, and never settles the model outright. ``mix-row`` under the same
    budget leaves 24 columns and proved infeasibility in presolve on 3 of 8
    seeds. The price is coefficient growth -- ratio 32 to 168 against 1 to 16
    for milder operations -- which ``--max-coef-ratio`` is there to bound, so
    this trades growth for structural change deliberately.
    """
    donors = _equality_donors(problem, ctx)
    if not donors:
        return None

    donor_index = rng.choice(donors)
    donor = problem.rows[donor_index]
    targets = [i for i in _row_indices(problem) if i != donor_index]
    if not targets:
        return None

    fanout = min(max(2, ctx.dispatch_fanout), len(targets))
    chosen = rng.sample(targets, fanout)
    folds = []
    for target_index in chosen:
        lam = ctx.mix_lambda(rng)
        folds.append(_fold_into(problem, donor_index, target_index, lam))

    return Step(
        "dispatch",
        {"donor": donor_index, "folds": folds},
        f"dispatch {donor.name} into {fanout} rows",
    )


# A cancelling variant of ``dispatch`` was implemented and removed. Choosing
# ``lam = -g_j / e_j`` clears variable ``j`` from target row ``g``, which looks
# appealing -- textbook Gaussian elimination, milder coefficient growth than a
# random multiplier, and nominally the inverse of presolve's substitution. It is
# counterproductive. Applied to every row holding ``j``, it confines the column
# to its donor, which *is* a column singleton, and if that column is continuous
# and free -- every slack this package creates is -- presolve deletes it and the
# donor row with it. Measured on ``pret60_25``, 14 and 20 elimination steps
# produced 14 and 20 free singleton columns, and the presolved model came out
# *smaller than the original* (15 columns against 24; ``dispatch`` leaves 35).
# Making a column sparser is the direction presolve itself travels, so a hiding
# transformation must not do it. See the README for the full numbers.


def mix_equality_into_row(problem, rng, ctx):
    """Add a multiple of an equality row to another row.

    If ``a'x = b`` holds at every feasible point then ``a'x - b == 0`` there, so
    adding ``lam * (a'x - b)`` to any other row leaves the feasible set exactly
    unchanged. This rewrites existing coefficients in place rather than adding
    structure, which makes it considerably harder to spot than a redundant row.
    """
    donors = _equality_donors(problem, ctx)
    if not donors:
        return None

    donor_index = rng.choice(donors)
    fold = _spread_row_into_another(problem, rng, ctx, donor_index)
    if fold is None:
        return None
    donor = problem.rows[donor_index]
    target = problem.rows[fold["target"]]
    return Step(
        "mix-row",
        {"donor": donor_index, "fold": fold},
        f"mix {fold['lam']:g} x {donor.name} into {target.name}",
    )


def mix_equality_into_objective(problem, rng, ctx):
    """Add a multiple of an equality row to the objective.

    Same argument as above: the objective changes by ``lam * (a'x - b)``, which
    is zero on the feasible set, so optimal values are preserved while the cost
    vector itself is scrambled.

    Refused when ``ctx.preserve_objective`` is set. The driver turns that on by
    default for a model whose objective is identically zero: such a model is a
    pure feasibility problem, and giving it a nonzero cost vector -- even one
    whose optimal value is still zero -- changes the kind of problem the solver
    sees. Presolve treats feasibility problems differently, and a model that
    reported a clean INFEASIBLE can come back as INF_OR_UNBD instead.
    """
    if ctx.preserve_objective:
        return None

    donors = _equality_donors(problem, ctx)
    if not donors:
        return None

    donor_index = rng.choice(donors)
    donor = problem.rows[donor_index]
    lam = ctx.mix_lambda(rng)

    for j, coef in donor.coeffs.items():
        problem.vars[j].obj += lam * coef
    problem.obj_offset -= lam * donor.rhs

    return Step(
        "mix-obj",
        {"donor": donor_index, "lam": lam},
        f"mix {lam:g} x {donor.name} into objective",
    )


TRANSFORMS = {
    "complement": complement_variable,
    "shift": shift_variable,
    "shift-binary": shift_binary,
    "xor-binary": encode_binary_as_xor,
    "scale-var": scale_variable,
    "split-free": split_free_variable,
    "scale-row": scale_row,
    "invert-row": invert_row,
    "slack": add_explicit_slack,
    "mix-row": mix_equality_into_row,
    "dispatch": dispatch_equality,
    "mix-obj": mix_equality_into_objective,
}

ALL_NAMES = tuple(TRANSFORMS)

# Drawn from only when asked for by name. ``shift-binary`` demotes a 0/1 column to
# a general integer, which is a deliberate trade of solver strength for hiding,
# so a default run must not make it silently.
OPT_IN_NAMES = ("shift-binary",)
DEFAULT_NAMES = tuple(n for n in ALL_NAMES if n not in OPT_IN_NAMES)
