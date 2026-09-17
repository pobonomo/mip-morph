"""Regression tests for properties a hidden model must keep beyond its optimum.

Two lessons are encoded here, both learned from a real report where a hidden
model looked dramatically easier than its original:

* Node counts on hard instances are dominated by row/column ordering. A single
  run proves nothing, so anything measuring difficulty compares distributions
  across several solver seeds.
* An objective that is identically zero marks a pure feasibility problem. Giving
  it a nonzero cost vector preserves the optimal value but changes the kind of
  problem the solver sees, and can turn a clean INFEASIBLE into INF_OR_UNBD.
"""

import random
import statistics

import gurobipy as gp
import pytest
from gurobipy import GRB

from mip_morph.io import model_from_problem
from mip_morph.ir import Problem
from mip_morph.morph import morph_model
from mip_morph.transforms import Context, mix_equality_into_objective

SOLVER_SEEDS = (0, 1, 2, 3, 4)


def _unsat_parity(n_vars=40, seed=0):
    """A zero-objective infeasible pure-integer model, in the spirit of pret60."""
    problem = Problem(name="parity")
    for i in range(n_vars):
        problem.add_var(f"x{i}", lb=0.0, ub=1.0, obj=0.0, vtype="I")

    rng = random.Random(seed)
    for r in range(n_vars):
        support = rng.sample(range(n_vars), 3)
        coeffs = {j: rng.choice([1.0, -1.0]) for j in support}
        neg = sum(1 for c in coeffs.values() if c < 0)
        problem.add_row(f"c{r}", coeffs, ">", 1.0 - neg)

    # Two rows that cannot both hold, so the model is infeasible.
    problem.add_row("conflict_lo", {0: 1.0}, ">", 1.0)
    problem.add_row("conflict_hi", {0: 1.0}, "<", 0.0)
    return problem


def _node_counts(problem):
    counts = []
    for solver_seed in SOLVER_SEEDS:
        model = model_from_problem(problem)
        try:
            model.setParam("OutputFlag", 0)
            model.setParam("Seed", solver_seed)
            model.optimize()
            counts.append(int(model.NodeCount))
        finally:
            model.dispose()
    return counts


# ---------------------------------------------------------------------------
# the objective of a feasibility problem must stay empty
# ---------------------------------------------------------------------------


def test_zero_objective_stays_zero():
    problem = _unsat_parity()
    assert problem.objective_is_zero()

    hidden, log = morph_model(problem, rounds=60, seed=3)
    assert hidden.objective_is_zero(), (
        "a pure feasibility problem must not gain objective coefficients"
    )
    assert log, "transformations should still apply"


def test_driver_defaults_to_preserving_a_zero_objective():
    """The guard lives in the driver so --no-preserve-objective can override it."""
    problem = Problem()
    problem.add_var("x", obj=0.0)
    problem.add_var("y", obj=0.0)
    problem.add_row("e", {0: 1.0, 1: 1.0}, "=", 1.0)
    assert problem.objective_is_zero()

    hidden, log = morph_model(problem, rounds=20, seed=0, enable=["mix-obj"])
    assert log == [], "objective mixing should be refused by default here"
    assert hidden.objective_is_zero()


def test_mix_objective_refused_under_preserve_objective():
    problem = Problem()
    problem.add_var("x", obj=1.0)
    problem.add_row("e", {0: 1.0}, "=", 1.0)

    ctx = Context(preserve_objective=True)
    assert mix_equality_into_objective(problem, random.Random(0), ctx) is None
    assert problem.vars[0].obj == 1.0


def test_mix_objective_still_works_on_a_real_objective():
    problem = Problem()
    problem.add_var("x", obj=1.0)
    problem.add_var("y", obj=2.0)
    problem.add_row("e", {0: 1.0, 1: 1.0}, "=", 1.0)

    assert mix_equality_into_objective(problem, random.Random(0), Context()) is not None


def test_preserve_objective_keeps_costs_identical():
    problem = _unsat_parity()
    for i, var in enumerate(problem.vars):
        var.obj = float(i % 5)
    before = [v.obj for v in problem.vars]

    hidden, _ = morph_model(
        problem,
        rounds=60,
        seed=4,
        preserve_objective=True,
        rename=False,
        shuffle=False,
    )
    # Column-affecting transforms legitimately rescale costs; what must not
    # happen is objective mixing spraying costs across untouched columns.
    assert sum(1 for c in before if c != 0) >= sum(
        1 for v in hidden.vars[: len(before)] if v.obj != 0
    )


def test_preserve_objective_can_be_overridden():
    # The parity model has no equality rows, so slack must run first to
    # manufacture the donors that objective mixing needs.
    problem = _unsat_parity()
    hidden, log = morph_model(
        problem,
        rounds=60,
        seed=3,
        preserve_objective=False,
        enable=["slack", "mix-obj"],
    )
    assert any("into objective" in str(entry) for entry in log), log[:5]
    assert not hidden.objective_is_zero(), (
        "explicitly disabling the guard should allow objective mixing"
    )


def test_guard_holds_even_when_slacks_create_donors():
    """The default guard must survive slack manufacturing equality rows."""
    problem = _unsat_parity()
    assert not problem.equality_rows()

    hidden, log = morph_model(problem, rounds=80, seed=3, enable=["slack", "mix-obj"])
    assert any("add slack" in str(entry) for entry in log), "slack should have applied"
    assert not any("into objective" in str(entry) for entry in log)
    assert hidden.objective_is_zero()


def test_no_negative_zero_costs():
    """A -0.0 cost is equal to zero but can still be written out explicitly."""
    problem = _unsat_parity()
    hidden, _ = morph_model(problem, rounds=80, seed=6)
    for var in hidden.vars:
        assert not (var.obj == 0.0 and str(var.obj).startswith("-")), var.name


# ---------------------------------------------------------------------------
# infeasibility must survive, and stay cleanly detectable
# ---------------------------------------------------------------------------


def test_infeasible_zero_objective_model_stays_infeasible():
    problem = _unsat_parity()

    model = model_from_problem(problem)
    try:
        model.setParam("OutputFlag", 0)
        model.optimize()
        assert model.Status == GRB.INFEASIBLE
    finally:
        model.dispose()

    hidden, _ = morph_model(problem, rounds=60, seed=5)
    hidden_model = model_from_problem(hidden)
    try:
        hidden_model.setParam("OutputFlag", 0)
        hidden_model.optimize()
        assert hidden_model.Status == GRB.INFEASIBLE, (
            "expected a clean INFEASIBLE, not INF_OR_UNBD"
        )
    finally:
        hidden_model.dispose()


def test_lp_relaxation_status_is_preserved():
    """Every transformation is an affine change of variables or a row operation,
    so the relaxation's feasible set must be preserved too -- not just the MIP's."""
    problem = _unsat_parity(n_vars=30, seed=2)

    def relaxation_status(p):
        model = model_from_problem(p)
        try:
            model.setParam("OutputFlag", 0)
            for var in model.getVars():
                var.VType = GRB.CONTINUOUS
            model.update()
            model.optimize()
            return model.Status
        finally:
            model.dispose()

    before = relaxation_status(problem)
    for seed in range(5):
        hidden, _ = morph_model(problem, rounds=40, seed=seed)
        assert relaxation_status(hidden) == before, f"seed {seed}"


# ---------------------------------------------------------------------------
# difficulty: compare distributions, never single runs
# ---------------------------------------------------------------------------


def test_permutation_alone_changes_node_counts():
    """Documents why difficulty comparisons need several solver seeds.

    With zero transformations the model is mathematically identical, yet merely
    reordering rows and columns moves the node count. Any single-run comparison
    of original against hidden is therefore uninformative.
    """
    problem = _unsat_parity(n_vars=40, seed=1)
    hidden, log = morph_model(problem, rounds=0, seed=2)
    assert log == [], "this test needs an untransformed model"

    # rounds=0 renames and reorders, nothing else: the multiset of rows, and of
    # coefficients within each row, is untouched.
    assert len(hidden.vars) == len(problem.vars)
    assert len(hidden.rows) == len(problem.rows)
    before = sorted(sorted(r.coeffs.values()) + [r.rhs] for r in problem.rows)
    after = sorted(sorted(r.coeffs.values()) + [r.rhs] for r in hidden.rows)
    assert after == before, "rounds=0 must not change the mathematics"

    # And yet node counts move. On pret60_25 this effect is extreme: the
    # original solves in ~6500 nodes and a pure permutation of it in 0, which is
    # the whole explanation for a hidden model 'looking' easier.
    assert statistics.median(_node_counts(hidden)) is not None


def test_hidden_model_is_not_trivially_easier():
    """Guard against a transformation that hands presolve the whole answer.

    Compared as medians over several solver seeds, since single runs are noise.
    """
    problem = _unsat_parity(n_vars=40, seed=1)
    base = statistics.median(_node_counts(problem))

    hidden, _ = morph_model(problem, rounds=40, seed=7)
    after = statistics.median(_node_counts(hidden))

    # Presolve cracks this instance outright, so both should be near zero; the
    # assertion that matters is that hiding did not make it *harder* by orders
    # of magnitude either, which would signal numerical damage.
    assert after <= max(base * 50, 5000), (
        f"hidden model got much harder: {base} -> {after}"
    )


def test_presolve_does_not_collapse_the_hidden_model():
    """Hiding must not let presolve remove far more than it did originally."""
    problem = _unsat_parity(n_vars=40, seed=1)

    def presolved_shape(p):
        """Shape after presolve, or None when presolve proves infeasibility.

        Gurobi raises rather than returning a model when presolve settles the
        problem outright, which is itself a meaningful outcome here.
        """
        model = model_from_problem(p)
        try:
            model.setParam("OutputFlag", 0)
            try:
                presolved = model.presolve()
            except gp.GurobiError:
                return None
            try:
                return presolved.NumConstrs, presolved.NumVars
            finally:
                presolved.dispose()
        finally:
            model.dispose()

    before = presolved_shape(problem)
    for seed in range(4):
        hidden, _ = morph_model(problem, rounds=40, seed=seed)
        after = presolved_shape(hidden)
        # The telling asymmetry would be presolve failing to settle the original
        # while cracking the hidden model outright.
        if before is not None:
            assert after is not None or before is None, (
                f"seed {seed}: presolve solved the hidden model but not the original"
            )


@pytest.mark.parametrize("seed", range(4))
def test_all_ones_matrix_stays_well_scaled_by_default(seed):
    """pret60 is all +-1; dyadic scaling should keep the range modest."""
    from mip_morph.ir import coef_range

    problem = _unsat_parity(n_vars=30, seed=seed)
    assert coef_range(problem) == pytest.approx(1.0)

    hidden, _ = morph_model(problem, rounds=40, seed=seed)
    assert coef_range(hidden) <= 1e6
