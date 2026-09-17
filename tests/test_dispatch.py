"""Dispatching an equality row across many others.

``mix-row`` moves one donor into one target per application. Dispatching commits
to a single equality and pushes it through several rows at once, which is what
actually dissolves that row's identity: afterwards its coefficient pattern lives
in its targets rather than sitting in one recognizable row.

Correctness does not depend on the fanout. ``a'x - b == 0`` at every feasible
point, so each of the k additions is independently exact; k only controls fill-in
and coefficient growth.
"""

import random
from collections import Counter

import pytest
from gurobipy import GRB

from mip_morph.io import model_from_problem
from mip_morph.ir import CONTINUOUS, INF, INTEGER, Problem, coef_range
from mip_morph.morph import DEFAULT_DISPATCH_FANOUT, morph_model
from mip_morph.transforms import (
    TRANSFORMS,
    Context,
    add_explicit_slack,
    dispatch_equality,
)


def _mixed_model(seed=0, n_int=20, n_con=15, m=25):
    """A mix of integer and continuous columns."""
    rng = random.Random(seed)
    problem = Problem(name="mx")
    for i in range(n_int):
        problem.add_var(f"x{i}", 0.0, 1.0, float(rng.randint(1, 9)), INTEGER)
    for i in range(n_con):
        problem.add_var(f"z{i}", 0.0, 10.0, float(rng.randint(1, 5)), CONTINUOUS)
    n = n_int + n_con
    for r in range(m):
        support = rng.sample(range(n), 5)
        problem.add_row(
            f"c{r}", {j: float(rng.choice([1, 2])) for j in support}, "<", 12.0
        )
    for r in range(6):
        problem.add_row(f"d{r}", {j: 1.0 for j in rng.sample(range(n), 6)}, ">", 2.0)
    problem.obj_sense = -1
    return problem


def _solve(problem):
    model = model_from_problem(problem)
    try:
        model.setParam("OutputFlag", 0)
        model.optimize()
        return model.Status, (model.ObjVal if model.Status == GRB.OPTIMAL else None)
    finally:
        model.dispose()


def _with_donors(seed=0, count=8):
    """A model that has equality rows to dispatch."""
    problem = _mixed_model()
    rng = random.Random(seed)
    ctx = Context()
    for _ in range(count):
        add_explicit_slack(problem, rng, ctx)
    return problem


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_dispatch_needs_an_equality_donor():
    problem = Problem()
    problem.add_var("x")
    problem.add_row("r", {0: 1.0}, "<", 1.0)
    assert dispatch_equality(problem, random.Random(0), Context()) is None


def test_dispatch_hits_the_requested_number_of_rows():
    problem = Problem()
    for i in range(4):
        problem.add_var(f"x{i}", ub=10.0)
    problem.add_row("donor", {0: 1.0, 1: 1.0}, "=", 4.0)
    for r in range(5):
        problem.add_row(f"t{r}", {2: 1.0}, "<", 9.0)

    before = [dict(r.coeffs) for r in problem.rows]
    ctx = Context(dispatch_fanout=3)
    description = dispatch_equality(problem, random.Random(0), ctx)
    assert description is not None
    assert "into 3 rows" in description

    changed = sum(1 for old, r in zip(before, problem.rows) if old != r.coeffs)
    assert changed == 3, "exactly the fanout many rows should have moved"


def test_dispatch_never_targets_its_own_donor():
    problem = Problem()
    problem.add_var("x")
    problem.add_var("y")
    problem.add_row("donor", {0: 1.0, 1: 2.0}, "=", 4.0)
    problem.add_row("t0", {0: 1.0}, "<", 9.0)
    problem.add_row("t1", {1: 1.0}, "<", 9.0)

    donor_before = dict(problem.rows[0].coeffs), problem.rows[0].rhs
    dispatch_equality(problem, random.Random(0), Context(dispatch_fanout=2))
    assert (dict(problem.rows[0].coeffs), problem.rows[0].rhs) == donor_before


def test_dispatch_arithmetic_is_the_same_as_repeated_mixing():
    """Each target gets its own lambda; the donor is added, not replaced."""
    problem = Problem()
    problem.add_var("x")
    problem.add_var("y")
    problem.add_row("donor", {0: 1.0, 1: 1.0}, "=", 4.0)
    problem.add_row("t0", {0: 2.0}, "<", 6.0)
    problem.add_row("t1", {0: 2.0}, "<", 6.0)

    ctx = Context(dispatch_fanout=2)
    ctx.mix_lambda = lambda _rng: 3.0
    dispatch_equality(problem, random.Random(0), ctx)

    for name in ("t0", "t1"):
        row = next(r for r in problem.rows if r.name == name)
        assert row.coeffs[0] == 5.0  # 2 + 3*1
        assert row.coeffs[1] == 3.0  # 0 + 3*1
        assert row.rhs == 18.0  # 6 + 3*4


def test_dispatch_respects_the_density_cap():
    problem = Problem()
    for i in range(6):
        problem.add_var(f"x{i}")
    problem.add_row("dense", {j: 1.0 for j in range(6)}, "=", 4.0)
    problem.add_row("t", {0: 1.0}, "<", 6.0)
    assert dispatch_equality(problem, random.Random(0), Context(mix_max_nnz=3)) is None


def test_dispatch_preserves_the_optimum_at_every_fanout():
    base = _with_donors()
    status0, want = _solve(base)
    assert status0 == GRB.OPTIMAL

    for fanout in (2, 3, 5, 10):
        for seed in range(4):
            problem = base.copy()
            rng = random.Random(seed)
            ctx = Context(dispatch_fanout=fanout)
            for _ in range(8):
                dispatch_equality(problem, rng, ctx)
            status, obj = _solve(problem)
            assert status == GRB.OPTIMAL, f"fanout {fanout} seed {seed}"
            assert obj == pytest.approx(want, abs=1e-6), f"fanout {fanout} seed {seed}"


def test_dispatch_fanout_trades_growth_for_spread():
    """Higher fanout spreads the donor further and grows coefficients faster."""
    base = _with_donors()
    nnz = {}
    ratio = {}
    for fanout in (2, 10):
        problem = base.copy()
        rng = random.Random(1)
        ctx = Context(dispatch_fanout=fanout)
        for _ in range(8):
            dispatch_equality(problem, rng, ctx)
        nnz[fanout] = sum(r.nnz for r in problem.rows)
        ratio[fanout] = coef_range(problem)

    assert nnz[10] > nnz[2], "more targets should mean more fill-in"
    assert ratio[10] > ratio[2], "more targets should mean more coefficient growth"


def test_dispatch_fanout_is_clamped_to_available_targets():
    problem = Problem()
    problem.add_var("x")
    problem.add_row("donor", {0: 1.0}, "=", 1.0)
    problem.add_row("t", {0: 1.0}, "<", 5.0)
    # Only one possible target even though the fanout asks for 10.
    description = dispatch_equality(
        problem, random.Random(0), Context(dispatch_fanout=10)
    )
    assert description is not None
    assert "into 1 rows" in description


# ---------------------------------------------------------------------------
# why the cancelling variant is not here
# ---------------------------------------------------------------------------


def test_dispatch_never_confines_a_column_to_one_row():
    """The property that got the cancelling variant of dispatch removed.

    Choosing ``lam = -g_j/e_j`` clears column ``j`` from target ``g``. Doing that
    to every row holding ``j`` leaves it in its donor alone -- a column singleton,
    and for the continuous free slacks this package creates, exactly the
    free-column-singleton reduction ``slack``'s fold exists to defeat. Presolve
    then deletes the column *and* the donor row, so the presolved model came out
    smaller than the original.

    Random multipliers cannot do this: ``lam`` is drawn without reference to the
    target, so a cancellation would need ``lam`` to hit ``-g_j/e_j`` exactly.
    """
    base = _with_donors()
    for seed in range(6):
        problem = base.copy()
        rng = random.Random(seed)
        ctx = Context(dispatch_fanout=4)
        for _ in range(10):
            dispatch_equality(problem, rng, ctx)

        counts = Counter(j for r in problem.rows for j in r.coeffs)
        free_singletons = [
            problem.vars[j].name
            for j, n in counts.items()
            if n == 1
            and not problem.vars[j].is_integral
            and problem.vars[j].obj == 0.0
            and problem.vars[j].lb == 0.0
            and problem.vars[j].ub >= INF
        ]
        assert not free_singletons, f"seed {seed}: {free_singletons}"


def test_eliminate_is_not_a_registered_transform():
    """It was implemented, measured to help presolve, and removed."""
    assert "eliminate" not in TRANSFORMS


def test_dispatch_does_not_shrink_the_column_support():
    """Dispatching adds entries; it must never sparsify, which presolve rewards."""
    base = _with_donors()
    before = Counter(j for r in base.rows for j in r.coeffs)

    problem = base.copy()
    rng = random.Random(3)
    for _ in range(10):
        dispatch_equality(problem, rng, Context(dispatch_fanout=4))
    after = Counter(j for r in problem.rows for j in r.coeffs)

    shrunk = {
        problem.vars[j].name: (before[j], after[j])
        for j in before
        if after[j] < before[j]
    }
    assert not shrunk, f"columns lost support: {shrunk}"


# ---------------------------------------------------------------------------
# through the driver
# ---------------------------------------------------------------------------


def test_dispatch_is_reachable_from_the_driver():
    problem = _mixed_model()
    hidden, log = morph_model(
        problem,
        rounds=30,
        seed=1,
        enable=["slack", "dispatch"],
        rename=False,
        shuffle=False,
    )
    assert any("dispatch" in entry for entry in log), f"never applied: {log[:5]}"


def test_driver_threads_the_fanout_through_to_the_transform():
    problem = _mixed_model()
    for fanout in (2, 6):
        _, log = morph_model(
            problem,
            rounds=30,
            seed=1,
            enable=["slack", "dispatch"],
            dispatch_fanout=fanout,
            rename=False,
            shuffle=False,
        )
        dispatches = [e for e in log if e.startswith("dispatch")]
        assert dispatches, f"fanout {fanout}: nothing dispatched"
        assert all(f"into {fanout} rows" in e for e in dispatches), dispatches


def test_default_fanout_is_used_when_unspecified():
    ctx = Context()
    assert ctx.dispatch_fanout == DEFAULT_DISPATCH_FANOUT


def test_full_pipeline_still_preserves_the_optimum_with_the_new_transforms():
    base = _mixed_model()
    status0, want = _solve(base)
    assert status0 == GRB.OPTIMAL

    for seed in range(8):
        hidden, _ = morph_model(base, rounds=40, seed=seed)
        status, obj = _solve(hidden)
        assert status == GRB.OPTIMAL, f"seed {seed}: status {status}"
        assert obj == pytest.approx(want, abs=1e-6), f"seed {seed}"
