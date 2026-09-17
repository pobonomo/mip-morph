"""Shared fixtures: model factories and a solve helper.

Models are generated programmatically and kept well under the 2000 variable /
2000 constraint ceiling of the size-limited Gurobi trial license.
"""

import random

import gurobipy as gp
import pytest
from gurobipy import GRB

from mip_morph.io import model_from_problem
from mip_morph.ir import BINARY, CONTINUOUS, INF, INTEGER, Problem


@pytest.fixture(scope="session", autouse=True)
def _quiet_gurobi():
    """Keep the solver banner out of the test output."""
    gp.setParam("OutputFlag", 0)


def solve_problem(problem):
    """Return ``(status, objective)``; objective is None unless optimal."""
    model = model_from_problem(problem)
    try:
        model.setParam("OutputFlag", 0)
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            return model.Status, model.ObjVal
        return model.Status, None
    finally:
        model.dispose()


def random_mip(seed, n_bin=6, n_int=3, n_cont=2, n_rows=6):
    """A small mixed model: binaries, bounded integers, continuous, one free."""
    rng = random.Random(seed)
    problem = Problem(name=f"rand{seed}")

    for i in range(n_bin):
        problem.add_var(f"b{i}", lb=0.0, ub=1.0, obj=rng.randint(-5, 5), vtype=BINARY)
    for i in range(n_int):
        problem.add_var(
            f"i{i}", lb=0.0, ub=rng.randint(3, 9), obj=rng.randint(-3, 3), vtype=INTEGER
        )
    for i in range(n_cont):
        problem.add_var(
            f"c{i}",
            lb=0.0,
            ub=rng.randint(5, 20),
            obj=rng.randint(-4, 4),
            vtype=CONTINUOUS,
        )
    free = problem.add_var("zfree", lb=-INF, ub=INF, obj=1.0, vtype=CONTINUOUS)

    n_structural = len(problem.vars) - 1
    for k in range(n_rows):
        coeffs = {
            j: rng.randint(-4, 4) for j in range(n_structural) if rng.random() < 0.7
        }
        coeffs = {j: c for j, c in coeffs.items() if c != 0}
        if not coeffs:
            coeffs = {0: 1.0}
        problem.add_row(
            f"r{k}", coeffs, rng.choice(["<", ">", "="]), rng.randint(-3, 12)
        )

    # Pins the free variable so the model stays bounded.
    problem.add_row("zrow", {free: 1.0, 0: 1.0}, "=", 4.0)
    problem.obj_offset = 3.5
    problem.obj_sense = 1 if seed % 2 == 0 else -1
    return problem


def feasible_mip(start=0):
    """The first random model at or after ``start`` that solves to optimality."""
    for seed in range(start, start + 200):
        problem = random_mip(seed)
        status, obj = solve_problem(problem)
        if status == GRB.OPTIMAL:
            return problem, obj
    raise AssertionError("no feasible random model found")


def feasible_random_mips(count, start=0):
    """Yield ``(seed, problem, objective)`` for models that solve to optimality."""
    found = []
    seed = start
    while len(found) < count and seed < start + count * 20:
        problem = random_mip(seed)
        status, obj = solve_problem(problem)
        if status == GRB.OPTIMAL:
            found.append((seed, problem, obj))
        seed += 1
    return found


@pytest.fixture
def tiny_problem():
    """A hand-built model with known numbers, for exact assertions.

    min  2a + 3b + z + 10
    s.t. r1:  a +  b      <= 5
         r2: 2a -  b      >= -1
         r3:  a +       z  = 4
    a in {0,1}, b integer in [0,4], z free
    """
    problem = Problem(name="tiny")
    problem.add_var("a", lb=0.0, ub=1.0, obj=2.0, vtype=BINARY)
    problem.add_var("b", lb=0.0, ub=4.0, obj=3.0, vtype=INTEGER)
    problem.add_var("z", lb=-INF, ub=INF, obj=1.0, vtype=CONTINUOUS)
    problem.add_row("r1", {0: 1.0, 1: 1.0}, "<", 5.0)
    problem.add_row("r2", {0: 2.0, 1: -1.0}, ">", -1.0)
    problem.add_row("r3", {0: 1.0, 2: 1.0}, "=", 4.0)
    problem.obj_offset = 10.0
    return problem
