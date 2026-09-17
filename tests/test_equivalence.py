"""The central property: a hidden model has the same optimum as the original.

Equivalence is asserted through the solver rather than through a mapping,
because hiding deliberately changes names, column counts and row order. What
must not change is the optimal objective value and the feasibility status.
"""

import pytest
from conftest import feasible_random_mips, random_mip, solve_problem
from gurobipy import GRB

from mip_morph.ir import INF, Problem, coef_range
from mip_morph.morph import morph_model
from mip_morph.transforms import ALL_NAMES

TOL = 1e-6
SLOPPY_TOL = 1e-6

FEASIBLE = feasible_random_mips(20)


def test_generator_produces_enough_feasible_models():
    assert len(FEASIBLE) == 20


@pytest.mark.parametrize("seed,problem,expected", FEASIBLE)
def test_hidden_model_matches_original(seed, problem, expected):
    hidden, log = morph_model(problem, rounds=30, seed=seed)
    assert log, "expected at least one transformation to apply"

    status, obj = solve_problem(hidden)
    assert status == GRB.OPTIMAL, f"seed {seed}: hidden model status {status}"
    assert obj == pytest.approx(expected, abs=TOL), f"seed {seed}"


@pytest.mark.parametrize("seed,problem,expected", FEASIBLE[:8])
def test_hidden_model_matches_with_sloppy_scaling(seed, problem, expected):
    """Non-dyadic factors introduce rounding, so allow a relative tolerance."""
    hidden, _ = morph_model(problem, rounds=30, seed=seed, sloppy_scaling=True)
    status, obj = solve_problem(hidden)
    assert status == GRB.OPTIMAL, f"seed {seed}: hidden model status {status}"
    assert obj == pytest.approx(expected, rel=1e-6, abs=SLOPPY_TOL), f"seed {seed}"


@pytest.mark.parametrize("name", sorted(ALL_NAMES))
def test_each_transform_alone_preserves_optimum_over_many_models(name):
    """Isolate one transformation and run it across every feasible model."""
    for seed, problem, expected in FEASIBLE[:10]:
        hidden, _ = morph_model(problem, rounds=10, seed=seed, enable=[name])
        status, obj = solve_problem(hidden)
        assert status == GRB.OPTIMAL, f"{name} seed {seed}: status {status}"
        assert obj == pytest.approx(expected, abs=TOL), f"{name} seed {seed}"


@pytest.mark.parametrize("rounds", [1, 5, 20, 60, 120])
def test_deep_transformation_chains_stay_equivalent(rounds):
    seed, problem, expected = FEASIBLE[0]
    hidden, _ = morph_model(problem, rounds=rounds, seed=99)
    status, obj = solve_problem(hidden)
    assert status == GRB.OPTIMAL
    assert obj == pytest.approx(expected, abs=TOL)


def test_infeasible_model_stays_infeasible():
    problem = Problem(name="infeasible")
    problem.add_var("x", lb=0.0, ub=1.0, obj=1.0, vtype="B")
    problem.add_row("lo", {0: 1.0}, ">", 1.0)
    problem.add_row("hi", {0: 1.0}, "<", 0.0)

    status0, _ = solve_problem(problem)
    assert status0 in (GRB.INFEASIBLE, GRB.INF_OR_UNBD)

    hidden, _ = morph_model(problem, rounds=25, seed=3)
    status1, _ = solve_problem(hidden)
    assert status1 in (GRB.INFEASIBLE, GRB.INF_OR_UNBD)


def test_unbounded_model_stays_unbounded():
    problem = Problem(name="unbounded")
    problem.add_var("x", lb=0.0, ub=INF, obj=-1.0)
    problem.add_row("r", {0: 1.0}, ">", 0.0)

    status0, _ = solve_problem(problem)
    assert status0 in (GRB.UNBOUNDED, GRB.INF_OR_UNBD)

    hidden, _ = morph_model(problem, rounds=25, seed=4)
    status1, _ = solve_problem(hidden)
    assert status1 in (GRB.UNBOUNDED, GRB.INF_OR_UNBD)


def test_maximization_is_preserved():
    # Flipping the sense can make a model unbounded, so search for one that
    # still has a finite maximum.
    problem = None
    for seed in range(40):
        candidate = random_mip(seed)
        candidate.obj_sense = -1
        status, expected = solve_problem(candidate)
        if status == GRB.OPTIMAL:
            problem = candidate
            break
    assert problem is not None, "no bounded maximization model found"
    status0, expected = solve_problem(problem)
    assert status0 == GRB.OPTIMAL

    hidden, _ = morph_model(problem, rounds=30, seed=5)
    assert hidden.obj_sense == -1
    status1, obj = solve_problem(hidden)
    assert status1 == GRB.OPTIMAL
    assert obj == pytest.approx(expected, abs=TOL)


def test_hiding_actually_changes_the_model():
    """A tool that returns the input unchanged would pass every test above."""
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(problem, rounds=40, seed=seed)

    assert len(log) >= 20
    original_names = {v.name for v in problem.vars}
    assert not (original_names & {v.name for v in hidden.vars}), "names not mangled"

    original_matrix = sorted(
        (r.sense, r.rhs, tuple(sorted(r.coeffs.items()))) for r in problem.rows
    )
    hidden_matrix = sorted(
        (r.sense, r.rhs, tuple(sorted(r.coeffs.items()))) for r in hidden.rows
    )
    assert original_matrix != hidden_matrix, "matrix is unchanged"


def test_original_problem_is_not_mutated():
    seed, problem, expected = FEASIBLE[0]
    before = (
        [(v.name, v.lb, v.ub, v.obj, v.vtype) for v in problem.vars],
        [(r.name, dict(r.coeffs), r.sense, r.rhs) for r in problem.rows],
        problem.obj_offset,
        problem.obj_sense,
    )
    morph_model(problem, rounds=30, seed=seed)
    after = (
        [(v.name, v.lb, v.ub, v.obj, v.vtype) for v in problem.vars],
        [(r.name, dict(r.coeffs), r.sense, r.rhs) for r in problem.rows],
        problem.obj_offset,
        problem.obj_sense,
    )
    assert before == after, "morph_model must not touch the input problem"


def test_same_seed_is_reproducible():
    seed, problem, _ = FEASIBLE[0]
    a, log_a = morph_model(problem, rounds=30, seed=1234)
    b, log_b = morph_model(problem, rounds=30, seed=1234)
    assert log_a == log_b
    assert [(v.name, v.lb, v.ub, v.obj, v.vtype) for v in a.vars] == [
        (v.name, v.lb, v.ub, v.obj, v.vtype) for v in b.vars
    ]


def test_different_seeds_differ():
    seed, problem, _ = FEASIBLE[0]
    _, log_a = morph_model(problem, rounds=30, seed=1)
    _, log_b = morph_model(problem, rounds=30, seed=2)
    assert log_a != log_b


def test_coefficient_guard_caps_growth():
    """Without the guard, repeated mixing blows the coefficient range up."""
    seed, problem, expected = FEASIBLE[0]

    capped, _ = morph_model(problem, rounds=200, seed=8, max_coef_ratio=1e4)
    assert coef_range(capped) <= 1e4

    status, obj = solve_problem(capped)
    assert status == GRB.OPTIMAL
    assert obj == pytest.approx(expected, abs=TOL)


def test_rename_and_shuffle_can_be_disabled():
    seed, problem, _ = FEASIBLE[0]
    hidden, _ = morph_model(
        problem,
        rounds=10,
        seed=seed,
        rename=False,
        shuffle=False,
        enable=["scale-row"],
    )
    # scale-row touches neither names nor ordering.
    assert [r.name for r in hidden.rows] == [r.name for r in problem.rows]
    assert [v.name for v in hidden.vars] == [v.name for v in problem.vars]


def test_unknown_transform_name_is_rejected():
    problem = random_mip(0)
    with pytest.raises(ValueError, match="unknown transformation"):
        morph_model(problem, enable=["no-such-transform"])
    with pytest.raises(ValueError, match="unknown transformation"):
        morph_model(problem, disable=["no-such-transform"])


def test_disabling_everything_is_rejected():
    problem = random_mip(0)
    with pytest.raises(ValueError, match="no transformations left"):
        morph_model(problem, disable=list(ALL_NAMES))
