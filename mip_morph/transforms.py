"""Equivalence-preserving transformations.

Every transformation keeps the optimal objective value and the feasibility
status of the model intact, while changing its structure. Each one takes
``(problem, rng, ctx)`` and returns a short description of what it did, or
``None`` when the model offers no candidate to work on.

Scale factors are dyadic (powers of two) by default so that every
multiplication is exact in IEEE double arithmetic and equivalence is bit-exact.
``ctx.sloppy_scaling`` trades that for arbitrary factors in [0.1, 10].
"""

from .ir import BINARY, CONTINUOUS, FLIP, INF, drop_zeros, negate

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
    return f"complement {var.name} (x -> {total:g} - x)"


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
    return f"shift {var.name} by {delta:g}"


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

    return f"scale {var.name} (x -> {factor:g} y)"


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

    problem.add_row(f"_and{tag}a", {w: 1.0, u: -1.0}, "<", 0.0)
    problem.add_row(f"_and{tag}b", {w: 1.0, v: -1.0}, "<", 0.0)
    problem.add_row(f"_and{tag}c", {w: 1.0, u: -1.0, v: -1.0}, ">", -1.0)
    problem.add_row(f"_xor{tag}", {j: 1.0, u: -1.0, v: -1.0, w: 2.0}, "=", 0.0)

    # x keeps its bounds, cost and every existing row entry; only its declared
    # type relaxes, because integrality now follows from the linking row.
    var.vtype = CONTINUOUS
    return f"encode binary {var.name} as XOR of {problem.vars[u].name}, {problem.vars[v].name}"


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
    # held by every other row stay valid.
    var.lb = 0.0
    var.ub = 0.0
    var.obj = 0.0
    return f"split free {var.name} into {var.name}_p - {var.name}_n"


# --------------------------------------------------------------------------
# row operations
# --------------------------------------------------------------------------


def scale_row(problem, rng, ctx):
    """Multiply a row through by a nonzero factor, flipping the sense if < 0."""
    if not problem.rows:
        return None

    row = rng.choice(problem.rows)
    factor = ctx.scale_factor(rng)

    for j in row.coeffs:
        row.coeffs[j] *= factor
    row.rhs *= factor
    if factor < 0:
        row.sense = FLIP[row.sense]

    return f"scale row {row.name} by {factor:g}"


def invert_row(problem, rng, ctx):
    """Turn a <= row into a >= row (or vice versa) by negating it."""
    candidates = [r for r in problem.rows if r.sense != "="]
    if not candidates:
        return None

    row = rng.choice(candidates)
    for j in row.coeffs:
        row.coeffs[j] = -row.coeffs[j]
    row.rhs = -row.rhs
    row.sense = FLIP[row.sense]
    return f"invert row {row.name}"


def _spread_row_into_another(problem, rng, ctx, donor):
    """Fold equality ``donor`` into some other row.

    Returns ``(lam, target)`` on success and ``None`` when there is no eligible
    target. Shared by ``slack``, which must not leave its new column as a
    singleton, so both callers obey the same density cap: a dense donor would
    otherwise smear across the matrix and show up as a fill-in tell.
    """
    if not 0 < donor.nnz <= ctx.mix_max_nnz:
        return None

    targets = [r for r in problem.rows if r is not donor]
    if not targets:
        return None

    target = rng.choice(targets)
    lam = ctx.mix_lambda(rng)
    for j, coef in donor.coeffs.items():
        target.coeffs[j] = target.coeffs.get(j, 0.0) + lam * coef
    target.rhs += lam * donor.rhs
    drop_zeros(target)
    return lam, target


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
    candidates = [r for r in problem.rows if r.sense != "="]
    if not candidates:
        return None

    row = rng.choice(candidates)
    slack = problem.add_var(f"_s{len(problem.vars)}", lb=0.0, ub=INF, obj=0.0)
    row.coeffs[slack] = 1.0 if row.sense == "<" else -1.0
    old_sense = row.sense
    row.sense = "="

    spread = _spread_row_into_another(problem, rng, ctx, row)
    if spread is None:
        note = " (left as a singleton: nothing eligible to fold into)"
    else:
        note = f", folded into {spread[1].name}"
    return f"add slack to row {row.name} ({old_sense}= -> =){note}"


def _equality_donors(problem, ctx):
    return [r for r in problem.rows if r.sense == "=" and 0 < r.nnz <= ctx.mix_max_nnz]


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

    donor = rng.choice(donors)
    targets = [r for r in problem.rows if r is not donor]
    if not targets:
        return None

    fanout = min(max(2, ctx.dispatch_fanout), len(targets))
    chosen = rng.sample(targets, fanout)
    for target in chosen:
        lam = ctx.mix_lambda(rng)
        for j, coef in donor.coeffs.items():
            target.coeffs[j] = target.coeffs.get(j, 0.0) + lam * coef
        target.rhs += lam * donor.rhs
        drop_zeros(target)

    return f"dispatch {donor.name} into {fanout} rows"


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

    donor = rng.choice(donors)
    spread = _spread_row_into_another(problem, rng, ctx, donor)
    if spread is None:
        return None
    lam, target = spread
    return f"mix {lam:g} x {donor.name} into {target.name}"


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

    donor = rng.choice(donors)
    lam = ctx.mix_lambda(rng)

    for j, coef in donor.coeffs.items():
        problem.vars[j].obj += lam * coef
    problem.obj_offset -= lam * donor.rhs

    return f"mix {lam:g} x {donor.name} into objective"


TRANSFORMS = {
    "complement": complement_variable,
    "shift": shift_variable,
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
