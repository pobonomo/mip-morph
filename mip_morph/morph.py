"""Driver that applies a randomized sequence of transformations."""

import random

from .ir import coef_range
from .key import Key, structure_fingerprint, values_fingerprint
from .transforms import ALL_NAMES, DEFAULT_NAMES, TRANSFORMS, Context

DEFAULT_ROUNDS = 40
DEFAULT_MAX_COEF_RATIO = 1e6

# Each xor-binary application adds three columns and four rows, so left
# unchecked it would dominate a long run and bloat the model. Cap it at a
# fraction of the rounds.
GROWTH_TRANSFORMS = {"xor-binary"}
DEFAULT_GROWTH_FRACTION = 0.25

# How many rows one equality is scattered across by ``dispatch``. Higher hides
# the donor row better but grows coefficients faster; ``max_coef_ratio`` is what
# stops it running away.
DEFAULT_DISPATCH_FANOUT = 4


def resolve_transforms(enable=None, disable=None):
    """Work out which transformations to draw from.

    Without ``enable`` this is ``DEFAULT_NAMES``: opt-in transformations only run
    when listed explicitly.
    """
    names = list(enable) if enable else list(DEFAULT_NAMES)
    unknown = [n for n in names if n not in TRANSFORMS]
    if unknown:
        raise ValueError(
            f"unknown transformation(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(ALL_NAMES)}"
        )
    if disable:
        unknown = [n for n in disable if n not in TRANSFORMS]
        if unknown:
            raise ValueError(
                f"unknown transformation(s): {', '.join(sorted(unknown))}; "
                f"available: {', '.join(ALL_NAMES)}"
            )
        names = [n for n in names if n not in disable]
    if not names:
        raise ValueError("no transformations left enabled")
    return names


def rename_all(problem):
    """Replace every name with an opaque sequential one.

    Returns the names it overwrote, as ``(var_names, row_names)`` in index order,
    which is the only record of them: renaming is otherwise irreversible.
    """
    old_var_names = [var.name for var in problem.vars]
    old_row_names = [row.name for row in problem.rows]
    width = max(4, len(str(len(problem.vars))))
    for i, var in enumerate(problem.vars, start=1):
        var.name = f"C{i:0{width}d}"
    width = max(4, len(str(len(problem.rows))))
    for i, row in enumerate(problem.rows, start=1):
        row.name = f"R{i:0{width}d}"
    return old_var_names, old_row_names


def shuffle_all(problem, rng):
    """Permute row and column order, remapping the coefficient indices.

    Returns ``(col_order, row_order)``, each mapping a new position to the index
    it came from. The row permutation is materialized rather than done with
    ``rng.shuffle(problem.rows)`` so that it can be returned and inverted;
    shuffling a list of integers of the same length consumes exactly the same
    generator state, so this does not change what any seed produces.
    """
    col_order = list(range(len(problem.vars)))
    rng.shuffle(col_order)
    remap = {old: new for new, old in enumerate(col_order)}
    problem.vars = [problem.vars[old] for old in col_order]
    for row in problem.rows:
        row.coeffs = {remap[j]: c for j, c in row.coeffs.items()}

    row_order = list(range(len(problem.rows)))
    rng.shuffle(row_order)
    problem.rows = [problem.rows[old] for old in row_order]
    return col_order, row_order


# --------------------------------------------------------------------------
# carrying a solution forward
# --------------------------------------------------------------------------
#
# Each map takes the point ``x`` (values by column index), the step just
# accepted, and the model *after* that step, and rewrites ``x`` into the step's
# variable space. These run inside the hiding loop rather than being replayed
# from a key afterwards because ``slack`` needs the row as it stood at that
# step, which a key does not record. They mirror ``unmorph.SOLUTION_MAPS``.


def _append(x, index, value, step):
    if index != len(x):
        raise AssertionError(
            f"step {step.op!r} added column {index}, but the point has {len(x)} values"
        )
    x.append(value)


def _fwd_complement(x, step, problem):
    x[step.params["var"]] = step.params["total"] - x[step.params["var"]]


def _fwd_shift(x, step, problem):
    x[step.params["var"]] -= step.params["delta"]


def _fwd_scale_var(x, step, problem):
    x[step.params["var"]] /= step.params["factor"]


def _fwd_xor_binary(x, step, problem):
    # Any of (u, v, w) with u + v - 2w == x works; (x, 0, 0) is the simplest.
    # Rounding keeps the fresh binaries exactly integral when the solver returned
    # x within its integrality tolerance of 0 or 1.
    value = float(round(x[step.params["var"]]))
    _append(x, step.params["u"], value, step)
    _append(x, step.params["v"], 0.0, step)
    _append(x, step.params["w"], 0.0, step)


def _fwd_split_free(x, step, problem):
    j = step.params["var"]
    value = x[j]
    x[j] = 0.0  # the retired column is fixed at zero
    _append(x, step.params["pos"], max(value, 0.0), step)
    _append(x, step.params["neg"], max(-value, 0.0), step)


def _fwd_slack(x, step, problem):
    # The fold went *out of* this row into another one, so the row still reads
    # ``a'x + c * s = rhs`` and ``s`` follows from it. Clamped at zero so that a
    # point satisfying the inequality only to tolerance does not break the
    # slack's bound instead.
    row = problem.rows[step.params["row"]]
    slack = step.params["slack"]
    activity = sum(c * x[j] for j, c in row.coeffs.items() if j != slack)
    value = (row.rhs - activity) / row.coeffs[slack]
    _append(x, slack, max(value, 0.0), step)


def _fwd_identity(x, step, problem):
    """Row operations and objective mixing do not move the variable space."""


FORWARD_MAPS = {
    "complement": _fwd_complement,
    "shift": _fwd_shift,
    "shift-binary": _fwd_shift,
    "scale-var": _fwd_scale_var,
    "xor-binary": _fwd_xor_binary,
    "split-free": _fwd_split_free,
    "scale-row": _fwd_identity,
    "invert-row": _fwd_identity,
    "slack": _fwd_slack,
    "mix-row": _fwd_identity,
    "dispatch": _fwd_identity,
    "mix-obj": _fwd_identity,
}


class MorphLog(list):
    """The applied steps, plus the :class:`~mip_morph.key.Key` that inverts them.

    A plain ``list`` of :class:`~mip_morph.key.Step` as far as every existing
    caller is concerned -- it prints, iterates, counts and compares exactly as
    the old list of strings did -- with the key hanging off ``.key`` for callers
    that want to write one, and ``.solution`` holding the hidden point when
    ``morph_model`` was given one to carry.
    """

    def __init__(self, iterable=(), key=None, solution=None):
        super().__init__(iterable)
        self.key = key
        self.solution = solution


def morph_model(
    problem,
    rounds=DEFAULT_ROUNDS,
    seed=None,
    enable=None,
    disable=None,
    sloppy_scaling=False,
    max_coef_ratio=DEFAULT_MAX_COEF_RATIO,
    mix_max_nnz=25,
    rename=True,
    shuffle=True,
    preserve_objective=None,
    growth_fraction=DEFAULT_GROWTH_FRACTION,
    dispatch_fanout=DEFAULT_DISPATCH_FANOUT,
    solution=None,
    on_step=None,
):
    """Return a transformed copy of ``problem`` plus a log of what was applied.

    Each round picks an enabled transformation at random. A transformation that
    finds nothing to work on, or that would push the matrix's coefficient range
    past ``max_coef_ratio``, is rolled back and another is tried.

    ``preserve_objective`` keeps the cost vector untouched. It defaults to True
    for a model whose objective is identically zero, so that a pure feasibility
    problem is not silently turned into an optimization problem.

    The log is a :class:`MorphLog`, which is a list of the applied
    :class:`~mip_morph.key.Step` objects with a recovery
    :class:`~mip_morph.key.Key` on ``log.key``. Pass that key and the hidden
    model to :func:`mip_morph.unmorph.unmorph_problem` to get this ``problem``
    back.

    ``on_step``, if given, is called as ``on_step(step, model, point)`` after
    every accepted step, with the model as it stands after that step and the
    carried ``solution`` (or ``None``) already moved into its variable space.
    It is for checking a run as it goes; it must not modify either argument.
    """
    names = resolve_transforms(enable, disable)
    rng = random.Random(seed)
    if preserve_objective is None:
        preserve_objective = problem.objective_is_zero()
    ctx = Context(
        sloppy_scaling=sloppy_scaling,
        mix_max_nnz=mix_max_nnz,
        preserve_objective=preserve_objective,
        dispatch_fanout=dispatch_fanout,
    )

    key = Key(
        problem_name=problem.name,
        obj_sense=problem.obj_sense,
        orig_var_names=[var.name for var in problem.vars],
        orig_row_names=[row.name for row in problem.rows],
    )

    current = problem.copy()
    log = MorphLog(key=key)

    point = None
    if solution is not None:
        point = [float(value) for value in solution]
        if len(point) != len(problem.vars):
            raise ValueError(
                f"the solution has {len(point)} values but the model has "
                f"{len(problem.vars)} columns"
            )

    # Only cap growth transforms when they are sharing the run with others;
    # asking for them explicitly should not be silently throttled.
    growth_budget = None
    if any(n not in GROWTH_TRANSFORMS for n in names):
        growth_budget = max(1, int(rounds * growth_fraction))
    growth_used = 0

    for _ in range(rounds):
        # Try each transformation in a random order until one applies, so a
        # round is only wasted when nothing at all is applicable.
        for name in rng.sample(names, len(names)):
            if (
                growth_budget is not None
                and name in GROWTH_TRANSFORMS
                and growth_used >= growth_budget
            ):
                continue
            candidate = current.copy()
            step = TRANSFORMS[name](candidate, rng, ctx)
            if step is None:
                continue
            if max_coef_ratio and coef_range(candidate) > max_coef_ratio:
                continue  # too much numerical damage; leave the model alone
            current = candidate
            log.append(step)
            if point is not None:
                FORWARD_MAPS[step.op](point, step, current)
            if on_step is not None:
                on_step(step, current, point)
            if name in GROWTH_TRANSFORMS:
                growth_used += 1
            break

    key.steps = list(log)

    if rename:
        rename_all(current)

    # Snapshot the hidden names here, after renaming but before shuffling: an
    # inverse that re-indexes a hidden model onto this list undoes the shuffle
    # and any reordering a writer or reader introduced in the same step. The
    # fingerprints likewise describe this pre-shuffle state, so checking them
    # after that re-indexing is order-independent.
    key.hidden_var_names = [var.name for var in current.vars]
    key.hidden_row_names = [row.name for row in current.rows]
    key.structure_fingerprint = structure_fingerprint(current)
    key.values_fingerprint = values_fingerprint(current)

    if shuffle:
        col_order, _ = shuffle_all(current, rng)
        if point is not None:
            point = [point[old] for old in col_order]

    if point is not None:
        log.solution = {var.name: value for var, value in zip(current.vars, point)}

    return current, log
