"""Carrying a solution of the original model onto the hidden one.

The contract: the carried point is feasible for the hidden model, has the same
objective value, and maps back through the key to the point it started from.
"""

import pytest
from conftest import feasible_random_mips
from gurobipy import GRB

from mip_morph.cli import main, read_solution
from mip_morph.io import read_problem, solve_problem, write_problem
from mip_morph.ir import INF
from mip_morph.morph import FORWARD_MAPS, morph_model
from mip_morph.transforms import ALL_NAMES
from mip_morph.unmorph import SOLUTION_MAPS, unmorph_solution

FEASIBLE = feasible_random_mips(12)

TOL = 1e-6


def _solve(problem):
    status, objective, values = solve_problem(problem)
    assert status == GRB.OPTIMAL
    return objective, values


def _violation(problem, values):
    """Largest bound, row or integrality violation of ``values`` (by name)."""
    x = [values[var.name] for var in problem.vars]
    worst = 0.0
    for var, value in zip(problem.vars, x):
        if var.lb > -INF:
            worst = max(worst, var.lb - value)
        if var.ub < INF:
            worst = max(worst, value - var.ub)
        if var.is_integral:
            worst = max(worst, abs(value - round(value)))
    for row in problem.rows:
        activity = sum(c * x[j] for j, c in row.coeffs.items())
        scale = max(1.0, abs(row.rhs), *(abs(c * x[j]) for j, c in row.coeffs.items()))
        if row.sense == "<":
            gap = activity - row.rhs
        elif row.sense == ">":
            gap = row.rhs - activity
        else:
            gap = abs(activity - row.rhs)
        worst = max(worst, gap / scale)
    return worst


def _objective(problem, values):
    return problem.obj_offset + sum(v.obj * values[v.name] for v in problem.vars)


def _check_carried(problem, hidden, log, objective, values):
    assert log.solution is not None
    assert set(log.solution) == {v.name for v in hidden.vars}
    assert _violation(hidden, log.solution) <= TOL
    assert _objective(hidden, log.solution) == pytest.approx(objective, abs=1e-6)

    back = unmorph_solution(log.solution, log.key)
    for var, value in zip(problem.vars, values):
        assert back[var.name] == pytest.approx(value, abs=1e-6)


def test_every_transform_has_a_forward_map():
    assert set(FORWARD_MAPS) == set(ALL_NAMES) == set(SOLUTION_MAPS)


@pytest.mark.parametrize("name", sorted(ALL_NAMES))
def test_each_transform_carries_the_solution(name):
    """Each transformation on its own, so a failure names the culprit."""
    applied = 0
    for seed, problem, _ in FEASIBLE[:8]:
        objective, values = _solve(problem)
        hidden, log = morph_model(
            problem,
            rounds=12,
            seed=seed,
            enable=[name],
            preserve_objective=False,
            solution=values,
        )
        applied += len(log)
        _check_carried(problem, hidden, log, objective, values)
    assert applied, f"{name} never applied, so the test proved nothing"


@pytest.mark.parametrize("rounds", [5, 40, 120])
def test_full_runs_carry_the_solution(rounds):
    for seed, problem, _ in FEASIBLE:
        objective, values = _solve(problem)
        hidden, log = morph_model(problem, rounds=rounds, seed=seed, solution=values)
        _check_carried(problem, hidden, log, objective, values)


def test_carried_solution_is_optimal_for_the_hidden_model():
    seed, problem, _ = FEASIBLE[0]
    objective, values = _solve(problem)
    hidden, log = morph_model(problem, rounds=40, seed=seed, solution=values)
    hidden_objective, _ = _solve(hidden)
    assert _objective(hidden, log.solution) == pytest.approx(hidden_objective, abs=1e-6)


def test_carrying_a_solution_does_not_change_the_hidden_model():
    seed, problem, _ = FEASIBLE[1]
    _, values = _solve(problem)
    plain, plain_log = morph_model(problem, rounds=40, seed=seed)
    carried, _ = morph_model(problem, rounds=40, seed=seed, solution=values)
    assert plain == carried
    assert plain_log.solution is None


def test_wrong_length_solution_is_refused():
    _, problem, _ = FEASIBLE[0]
    with pytest.raises(ValueError, match="values but the model has"):
        morph_model(problem, rounds=5, seed=0, solution=[0.0])


def test_cli_sol_writes_a_solution_of_the_hidden_model(tmp_path, capsys):
    _, problem, _ = FEASIBLE[2]
    objective, _ = _solve(problem)
    path = tmp_path / "model.mps"
    write_problem(problem, path)
    out = tmp_path / "hidden.mps"
    sol = tmp_path / "hidden.sol"

    assert main([str(path), "-o", str(out), "--seed", "3", "--sol", str(sol)]) == 0
    assert str(sol) in capsys.readouterr().out

    comments, values = read_solution(sol)
    assert comments == [f"# Objective value = {objective!r}"]
    hidden = read_problem(out)
    assert _violation(hidden, values) <= TOL
    assert _objective(hidden, values) == pytest.approx(objective, abs=1e-6)


def test_cli_sol_fails_cleanly_on_an_infeasible_model(tmp_path, capsys):
    from mip_morph.ir import Problem

    problem = Problem()
    problem.add_var("x", lb=0.0, ub=1.0)
    problem.add_row("r", {0: 1.0}, ">", 2.0)
    path = tmp_path / "bad.mps"
    write_problem(problem, path)
    out = tmp_path / "hidden.mps"

    assert main([str(path), "-o", str(out), "--sol", str(tmp_path / "x.sol")]) == 2
    assert "found no solution" in capsys.readouterr().err
    assert not out.exists()
