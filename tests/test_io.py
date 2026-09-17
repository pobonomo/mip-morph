"""Round-tripping through MPS, and refusal of unsupported model features."""

import gurobipy as gp
import pytest
from conftest import feasible_mip, random_mip, solve_problem
from gurobipy import GRB

from mip_morph.io import (
    UnsupportedModelError,
    model_from_problem,
    problem_from_model,
    read_problem,
    write_problem,
)
from mip_morph.ir import BINARY, INTEGER
from mip_morph.morph import morph_model

RANGES_MPS = """NAME          RTEST
ROWS
 N  COST
 L  lim1
 G  lim2
 E  eq1
COLUMNS
    xa        COST      1.0        lim1      1.0
    xa        lim2      1.0        eq1       1.0
    xb        COST      2.0        lim1      1.0
    xb        eq1       1.0
RHS
    RHS       lim1      10.0       lim2      1.0
    RHS       eq1       5.0
RANGES
    RNG       lim1      4.0        lim2      3.0
    RNG       eq1       2.0
BOUNDS
ENDATA
"""


def test_write_then_read_preserves_objective(tmp_path):
    problem, expected = feasible_mip()

    path = tmp_path / "m.mps"
    write_problem(problem, path)
    reloaded = read_problem(path)

    status1, obj = solve_problem(reloaded)
    assert status1 == GRB.OPTIMAL
    assert obj == pytest.approx(expected, abs=1e-6)


def test_round_trip_preserves_structure(tmp_path):
    problem = random_mip(4)
    path = tmp_path / "m.mps"
    write_problem(problem, path)
    reloaded = read_problem(path)

    assert len(reloaded.vars) == len(problem.vars)
    assert len(reloaded.rows) == len(problem.rows)
    assert reloaded.obj_offset == pytest.approx(problem.obj_offset)
    assert reloaded.obj_sense == problem.obj_sense
    assert {v.vtype for v in reloaded.vars} == {v.vtype for v in problem.vars}


def test_objective_offset_and_sense_survive_mps(tmp_path):
    problem, _ = feasible_mip()
    problem.obj_offset = 7.5
    problem.obj_sense = -1
    path = tmp_path / "oc.mps"
    write_problem(problem, path)
    reloaded = read_problem(path)
    assert reloaded.obj_offset == pytest.approx(7.5)
    assert reloaded.obj_sense == -1


def test_gz_output_is_supported(tmp_path):
    problem, _ = feasible_mip()
    path = tmp_path / "m.mps.gz"
    write_problem(problem, path)
    assert path.exists() and path.stat().st_size > 0
    reloaded = read_problem(path)
    assert len(reloaded.vars) == len(problem.vars)


def test_ranged_rows_arrive_as_equalities(tmp_path):
    """Gurobi desugars MPS RANGES into an equality plus a bounded slack column,
    so the IR never has to model a range."""
    path = tmp_path / "rng.mps"
    path.write_text(RANGES_MPS)
    problem = read_problem(path)

    assert all(r.sense == "=" for r in problem.rows), "ranges should be equalities"
    # Two structural columns plus one auxiliary column per ranged row.
    assert len(problem.vars) == 5
    aux = [v for v in problem.vars if v.name.startswith("MPS_Rg")]
    assert len(aux) == 3
    assert all(v.is_bounded for v in aux)


def test_ranged_model_can_be_hidden(tmp_path):
    path = tmp_path / "rng.mps"
    path.write_text(RANGES_MPS)
    problem = read_problem(path)
    status0, expected = solve_problem(problem)
    assert status0 == GRB.OPTIMAL

    hidden, _ = morph_model(problem, rounds=30, seed=1)
    out = tmp_path / "hidden.mps"
    write_problem(hidden, out)

    status1, obj = solve_problem(read_problem(out))
    assert status1 == GRB.OPTIMAL
    assert obj == pytest.approx(expected, abs=1e-6)


def test_integrality_survives_round_trip(tmp_path):
    problem, _ = feasible_mip()
    path = tmp_path / "m.mps"
    write_problem(problem, path)
    reloaded = read_problem(path)

    n_int_before = sum(1 for v in problem.vars if v.vtype in (BINARY, INTEGER))
    n_int_after = sum(1 for v in reloaded.vars if v.vtype in (BINARY, INTEGER))
    assert n_int_before == n_int_after


def test_quadratic_objective_is_refused():
    model = gp.Model("q")
    model.setParam("OutputFlag", 0)
    x = model.addVar(name="x")
    model.setObjective(x * x)
    model.update()
    try:
        with pytest.raises(UnsupportedModelError, match="quadratic"):
            problem_from_model(model)
    finally:
        model.dispose()


def test_quadratic_constraint_is_refused():
    model = gp.Model("qc")
    model.setParam("OutputFlag", 0)
    x = model.addVar(name="x", ub=10)
    y = model.addVar(name="y", ub=10)
    model.addQConstr(x * y <= 4)
    model.update()
    try:
        with pytest.raises(UnsupportedModelError, match="quadratic"):
            problem_from_model(model)
    finally:
        model.dispose()


def test_sos_constraint_is_refused():
    model = gp.Model("sos")
    model.setParam("OutputFlag", 0)
    x = model.addVar(name="x", ub=5)
    y = model.addVar(name="y", ub=5)
    model.addSOS(GRB.SOS_TYPE1, [x, y])
    model.update()
    try:
        with pytest.raises(UnsupportedModelError, match="SOS"):
            problem_from_model(model)
    finally:
        model.dispose()


def test_general_constraint_is_refused():
    model = gp.Model("gen")
    model.setParam("OutputFlag", 0)
    x = model.addVar(name="x", ub=5)
    y = model.addVar(name="y", ub=5)
    z = model.addVar(name="z", ub=5)
    model.addGenConstrMax(z, [x, y])
    model.update()
    try:
        with pytest.raises(UnsupportedModelError, match="general constraints"):
            problem_from_model(model)
    finally:
        model.dispose()


def test_semicontinuous_variable_is_refused():
    model = gp.Model("sc")
    model.setParam("OutputFlag", 0)
    model.addVar(lb=2, ub=8, vtype=GRB.SEMICONT, obj=1.0, name="s")
    model.update()
    try:
        with pytest.raises(UnsupportedModelError, match="unsupported type"):
            problem_from_model(model)
    finally:
        model.dispose()


def test_model_from_problem_matches_ir():
    problem, _ = feasible_mip()
    model = model_from_problem(problem)
    try:
        assert model.NumVars == len(problem.vars)
        assert model.NumConstrs == len(problem.rows)
        assert model.ObjCon == pytest.approx(problem.obj_offset)
    finally:
        model.dispose()


def test_empty_constraint_matrix_round_trips(tmp_path):
    """A model with variables but no rows must not trip up getA()."""
    from mip_morph.ir import Problem

    problem = Problem(name="norows")
    problem.add_var("x", lb=0.0, ub=3.0, obj=1.0)
    path = tmp_path / "norows.mps"
    write_problem(problem, path)
    reloaded = read_problem(path)
    assert len(reloaded.vars) == 1
    assert reloaded.rows == []
