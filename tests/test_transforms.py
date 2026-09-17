"""Per-transformation unit tests with exact expected numbers."""

import random

import pytest
from conftest import solve_problem
from gurobipy import GRB

from mip_morph.ir import BINARY, INF, INTEGER
from mip_morph.transforms import (
    TRANSFORMS,
    Context,
    add_explicit_slack,
    complement_variable,
    invert_row,
    mix_equality_into_objective,
    mix_equality_into_row,
    scale_row,
    scale_variable,
    shift_variable,
    split_free_variable,
)


@pytest.fixture
def ctx():
    return Context()


def test_complement_binary_keeps_type_and_bounds(tiny_problem, ctx):
    # Force the choice onto 'a' by making it the only integral candidate.
    tiny_problem.vars[1].vtype = "C"
    rng = random.Random(0)
    assert complement_variable(tiny_problem, rng, ctx) is not None

    a = tiny_problem.vars[0]
    assert a.vtype == BINARY  # l + u == 1, so it is still a binary
    assert (a.lb, a.ub) == (0.0, 1.0)
    assert a.obj == -2.0
    assert tiny_problem.obj_offset == 12.0  # 10 + 2 * 1

    # r1: a + b <= 5  becomes  -a + b <= 4
    assert tiny_problem.rows[0].coeffs[0] == -1.0
    assert tiny_problem.rows[0].rhs == 4.0
    # r2: 2a - b >= -1  becomes  -2a - b >= -3
    assert tiny_problem.rows[1].coeffs[0] == -2.0
    assert tiny_problem.rows[1].rhs == -3.0


def test_complement_bounded_integer(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("y", lb=2.0, ub=6.0, obj=5.0, vtype=INTEGER)
    problem.add_row("r", {0: 3.0}, "<", 15.0)
    complement_variable(problem, random.Random(0), ctx)

    y = problem.vars[0]
    assert (y.lb, y.ub) == (2.0, 6.0)  # [l, u] maps onto itself
    assert y.vtype == INTEGER
    assert y.obj == -5.0
    assert problem.obj_offset == 40.0  # 5 * (2 + 6)
    assert problem.rows[0].coeffs[0] == -3.0
    assert problem.rows[0].rhs == 15.0 - 3.0 * 8.0


def test_shift_leaves_binaries_alone(tiny_problem, ctx):
    """Shifting a binary would cost the binary designation, so it is refused.

    Translating a binary off [0, 1] forces it to a general integer, losing the
    clique/cover/implied-bound reasoning that depends on knowing a variable is
    0/1. ``complement`` is the right transformation for a binary. See
    ``tests/test_binary.py`` for the full contract.
    """
    binary_index = 0
    assert tiny_problem.vars[binary_index].vtype == BINARY
    before = (
        tiny_problem.vars[binary_index].lb,
        tiny_problem.vars[binary_index].ub,
        tiny_problem.vars[binary_index].vtype,
    )

    rng = random.Random(1)
    for _ in range(40):
        shift_variable(tiny_problem, rng, ctx)

    var = tiny_problem.vars[binary_index]
    assert (var.lb, var.ub, var.vtype) == before, "a binary must never be shifted"


def test_shift_preserves_objective_value(tiny_problem, ctx):
    status0, obj0 = solve_problem(tiny_problem)
    assert status0 == GRB.OPTIMAL

    problem = tiny_problem.copy()
    rng = random.Random(7)
    for _ in range(5):
        shift_variable(problem, rng, ctx)

    status1, obj1 = solve_problem(problem)
    assert status1 == GRB.OPTIMAL
    assert obj1 == pytest.approx(obj0)


def test_scale_variable_skips_integers(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("i", lb=0.0, ub=4.0, obj=1.0, vtype=INTEGER)
    problem.add_var("b", lb=0.0, ub=1.0, obj=1.0, vtype=BINARY)
    assert scale_variable(problem, random.Random(0), ctx) is None


def test_scale_variable_updates_bounds_and_objective(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("c", lb=1.0, ub=8.0, obj=3.0)
    problem.add_row("r", {0: 2.0}, "<", 6.0)

    ctx_fixed = Context()
    rng = random.Random(0)
    # Pin the factor so the assertion is exact.
    ctx_fixed.positive_scale_factor = lambda _rng: 2.0
    scale_variable(problem, rng, ctx_fixed)

    c = problem.vars[0]
    assert (c.lb, c.ub) == (0.5, 4.0)  # x = 2y  =>  y in [lb/2, ub/2]
    assert c.obj == 6.0
    assert problem.rows[0].coeffs[0] == 4.0
    assert problem.rows[0].rhs == 6.0  # rhs untouched by a column scaling


def test_scale_variable_preserves_infinite_bounds(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("c", lb=-INF, ub=INF, obj=1.0)
    problem.add_row("r", {0: 1.0}, "<", 1.0)
    scale_variable(problem, random.Random(3), ctx)
    assert problem.vars[0].lb == -INF
    assert problem.vars[0].ub == INF


def test_split_free_variable(tiny_problem, ctx):
    n_before = len(tiny_problem.vars)
    assert split_free_variable(tiny_problem, random.Random(0), ctx) is not None
    assert len(tiny_problem.vars) == n_before + 2

    original = tiny_problem.vars[2]
    assert (original.lb, original.ub, original.obj) == (0.0, 0.0, 0.0)

    pos, neg = tiny_problem.vars[-2], tiny_problem.vars[-1]
    assert pos.obj == 1.0 and neg.obj == -1.0
    assert (pos.lb, neg.lb) == (0.0, 0.0)

    # r3 held z with coefficient 1; it now holds +1 on pos and -1 on neg.
    r3 = tiny_problem.rows[2]
    assert 2 not in r3.coeffs
    assert r3.coeffs[len(tiny_problem.vars) - 2] == 1.0
    assert r3.coeffs[len(tiny_problem.vars) - 1] == -1.0


def test_split_free_variable_no_candidates(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x", lb=0.0, ub=1.0)
    assert split_free_variable(problem, random.Random(0), ctx) is None


def test_scale_row_negative_flips_sense(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x")
    problem.add_row("r", {0: 3.0}, "<", 6.0)

    ctx_fixed = Context()
    ctx_fixed.scale_factor = lambda _rng, signed=True: -2.0
    scale_row(problem, random.Random(0), ctx_fixed)

    row = problem.rows[0]
    assert row.coeffs[0] == -6.0
    assert row.rhs == -12.0
    assert row.sense == ">"


def test_scale_row_leaves_equality_sense(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x")
    problem.add_row("r", {0: 3.0}, "=", 6.0)

    ctx_fixed = Context()
    ctx_fixed.scale_factor = lambda _rng, signed=True: -2.0
    scale_row(problem, random.Random(0), ctx_fixed)
    assert problem.rows[0].sense == "="


def test_invert_row(tiny_problem, ctx):
    # r1 is the <= row; invert whichever inequality is picked and check both.
    rng = random.Random(0)
    assert invert_row(tiny_problem, rng, ctx) is not None
    senses = {r.name: r.sense for r in tiny_problem.rows}
    assert senses["r3"] == "="  # equalities are never chosen


def test_invert_row_negates_everything(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x")
    problem.add_row("r", {0: 3.0}, "<", 6.0)
    invert_row(problem, random.Random(0), ctx)
    assert problem.rows[0].coeffs[0] == -3.0
    assert problem.rows[0].rhs == -6.0
    assert problem.rows[0].sense == ">"


def test_explicit_slack_signs(ctx):
    from mip_morph.ir import Problem

    for sense, expected in (("<", 1.0), (">", -1.0)):
        problem = Problem()
        problem.add_var("x", ub=10.0)
        problem.add_row("r", {0: 1.0}, sense, 5.0)
        assert add_explicit_slack(problem, random.Random(0), ctx) is not None
        row = problem.rows[0]
        assert row.sense == "="
        slack_index = len(problem.vars) - 1
        assert row.coeffs[slack_index] == expected
        slack = problem.vars[slack_index]
        assert (slack.lb, slack.ub, slack.obj) == (0.0, INF, 0.0)


def test_explicit_slack_is_not_left_a_singleton(ctx):
    """A slack with one nonzero is free presolve food, not an obfuscation.

    A continuous, costless, free singleton column is the textbook
    free-column-singleton reduction: measured on ``pret60_25``, 20 slack-only
    applications produced 20 singletons and presolve returned the model at its
    original size. So the new column must land in a second row.
    """
    from collections import Counter

    from mip_morph.ir import Problem

    problem = Problem()
    for i in range(4):
        problem.add_var(f"x{i}", ub=10.0)
    problem.add_row("r0", {0: 1.0, 1: 1.0}, "<", 5.0)
    problem.add_row("r1", {1: 1.0, 2: 1.0}, "<", 5.0)
    problem.add_row("r2", {2: 1.0, 3: 1.0}, ">", 1.0)

    assert add_explicit_slack(problem, random.Random(0), ctx) is not None

    slack_index = len(problem.vars) - 1
    occurrences = Counter(j for r in problem.rows for j in r.coeffs)
    assert occurrences[slack_index] >= 2, "slack must appear in more than one row"


def test_explicit_slack_reports_where_it_folded(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x", ub=10.0)
    problem.add_var("y", ub=10.0)
    problem.add_row("r0", {0: 1.0}, "<", 5.0)
    problem.add_row("r1", {1: 1.0}, "<", 5.0)

    description = add_explicit_slack(problem, random.Random(0), ctx)
    assert "folded into" in description


def test_explicit_slack_respects_the_density_cap():
    """A dense row must not be smeared across the matrix by the fold."""
    from collections import Counter

    from mip_morph.ir import Problem

    problem = Problem()
    for i in range(8):
        problem.add_var(f"x{i}", ub=10.0)
    problem.add_row("dense", {j: 1.0 for j in range(8)}, "<", 5.0)
    # 'other' is already an equality so it is not a slack candidate, leaving
    # 'dense' as the only row that can be picked -- and a valid fold target.
    problem.add_row("other", {0: 1.0}, "=", 2.0)

    # The slacked row then has 9 nonzeros, over the cap, so no fold may happen.
    tight = Context(mix_max_nnz=3)
    description = add_explicit_slack(problem, random.Random(0), tight)
    assert description is not None
    assert "singleton" in description
    slack_index = len(problem.vars) - 1
    occurrences = Counter(j for r in problem.rows for j in r.coeffs)
    assert occurrences[slack_index] == 1


def test_explicit_slack_no_inequalities(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x")
    problem.add_row("r", {0: 1.0}, "=", 1.0)
    assert add_explicit_slack(problem, random.Random(0), ctx) is None


def test_mix_row_arithmetic(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x")
    problem.add_var("y")
    # donor: x + y = 4  ; target: 2x <= 6
    problem.add_row("donor", {0: 1.0, 1: 1.0}, "=", 4.0)
    problem.add_row("target", {0: 2.0}, "<", 6.0)

    ctx_fixed = Context()
    ctx_fixed.mix_lambda = lambda _rng: 3.0
    mix_equality_into_row(problem, random.Random(0), ctx_fixed)

    target = next(r for r in problem.rows if r.name == "target")
    assert target.coeffs[0] == 5.0  # 2 + 3*1
    assert target.coeffs[1] == 3.0  # 0 + 3*1
    assert target.rhs == 18.0  # 6 + 3*4
    assert target.sense == "<"


def test_mix_row_drops_cancelled_coefficients(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x")
    problem.add_var("y")
    problem.add_row("donor", {0: 1.0, 1: 1.0}, "=", 4.0)
    problem.add_row("target", {0: 1.0, 1: 2.0}, "<", 6.0)

    ctx_fixed = Context()
    ctx_fixed.mix_lambda = lambda _rng: -1.0
    mix_equality_into_row(problem, random.Random(0), ctx_fixed)

    target = next(r for r in problem.rows if r.name == "target")
    assert 0 not in target.coeffs, "cancelled coefficient must be dropped"
    assert target.coeffs[1] == 1.0


def test_mix_row_never_targets_the_donor(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x")
    problem.add_row("only", {0: 1.0}, "=", 1.0)
    # A single equality row has no other row to mix into.
    assert mix_equality_into_row(problem, random.Random(0), ctx) is None


def test_mix_row_respects_density_cap():
    from mip_morph.ir import Problem

    problem = Problem()
    for i in range(5):
        problem.add_var(f"x{i}")
    problem.add_row("donor", {j: 1.0 for j in range(5)}, "=", 4.0)
    problem.add_row("target", {0: 1.0}, "<", 6.0)

    tight = Context(mix_max_nnz=3)
    assert mix_equality_into_row(problem, random.Random(0), tight) is None
    loose = Context(mix_max_nnz=10)
    assert mix_equality_into_row(problem, random.Random(0), loose) is not None


def test_mix_objective_arithmetic(ctx):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x", obj=1.0)
    problem.add_var("y", obj=0.0)
    problem.add_row("donor", {0: 2.0, 1: 3.0}, "=", 5.0)
    problem.obj_offset = 1.0

    ctx_fixed = Context()
    ctx_fixed.mix_lambda = lambda _rng: 2.0
    mix_equality_into_objective(problem, random.Random(0), ctx_fixed)

    assert problem.vars[0].obj == 5.0  # 1 + 2*2
    assert problem.vars[1].obj == 6.0  # 0 + 2*3
    assert problem.obj_offset == 1.0 - 2.0 * 5.0


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_each_transform_preserves_optimum(name, tiny_problem, ctx):
    """Applied repeatedly on its own, every transform keeps the optimum."""
    status0, obj0 = solve_problem(tiny_problem)
    assert status0 == GRB.OPTIMAL

    problem = tiny_problem.copy()
    rng = random.Random(11)
    applied = 0
    for _ in range(6):
        if TRANSFORMS[name](problem, rng, ctx) is not None:
            applied += 1

    status1, obj1 = solve_problem(problem)
    assert status1 == GRB.OPTIMAL, f"{name} broke feasibility"
    assert obj1 == pytest.approx(obj0), f"{name} changed the optimal value"


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transform_returns_none_on_empty_problem(name, ctx):
    """No transform may crash on a model with nothing to work on."""
    from mip_morph.ir import Problem

    assert TRANSFORMS[name](Problem(), random.Random(0), ctx) is None
