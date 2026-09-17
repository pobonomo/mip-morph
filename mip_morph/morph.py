"""Driver that applies a randomized sequence of transformations."""

import random

from .ir import coef_range
from .transforms import ALL_NAMES, TRANSFORMS, Context

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
    """Work out which transformations to draw from."""
    names = list(enable) if enable else list(ALL_NAMES)
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
    """Replace every name with an opaque sequential one."""
    width = max(4, len(str(len(problem.vars))))
    for i, var in enumerate(problem.vars, start=1):
        var.name = f"C{i:0{width}d}"
    width = max(4, len(str(len(problem.rows))))
    for i, row in enumerate(problem.rows, start=1):
        row.name = f"R{i:0{width}d}"


def shuffle_all(problem, rng):
    """Permute row and column order, remapping the coefficient indices."""
    order = list(range(len(problem.vars)))
    rng.shuffle(order)
    remap = {old: new for new, old in enumerate(order)}
    problem.vars = [problem.vars[old] for old in order]
    for row in problem.rows:
        row.coeffs = {remap[j]: c for j, c in row.coeffs.items()}
    rng.shuffle(problem.rows)


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
):
    """Return a transformed copy of ``problem`` plus a log of what was applied.

    Each round picks an enabled transformation at random. A transformation that
    finds nothing to work on, or that would push the matrix's coefficient range
    past ``max_coef_ratio``, is rolled back and another is tried.

    ``preserve_objective`` keeps the cost vector untouched. It defaults to True
    for a model whose objective is identically zero, so that a pure feasibility
    problem is not silently turned into an optimization problem.
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

    current = problem.copy()
    log = []

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
            description = TRANSFORMS[name](candidate, rng, ctx)
            if description is None:
                continue
            if max_coef_ratio and coef_range(candidate) > max_coef_ratio:
                continue  # too much numerical damage; leave the model alone
            current = candidate
            log.append(description)
            if name in GROWTH_TRANSFORMS:
                growth_used += 1
            break

    if rename:
        rename_all(current)
    if shuffle:
        shuffle_all(current, rng)

    return current, log
