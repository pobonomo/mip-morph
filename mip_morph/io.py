"""Reading and writing models via gurobipy.

Only linear MIPs are supported. Ranged rows need no special handling here:
Gurobi desugars an MPS ``RANGES`` entry at read time into an equality row plus a
bounded auxiliary variable, so a ranged row reaches the IR as a plain equality.
"""

import gurobipy as gp
from gurobipy import GRB

from .ir import (
    BINARY,
    CONTINUOUS,
    INTEGER,
    Problem,
    Var,
    binary_bounds_are_sane,
)


class UnsupportedModelError(Exception):
    """Raised for models with features the transformations cannot preserve."""


_VTYPE_TO_GRB = {
    CONTINUOUS: GRB.CONTINUOUS,
    BINARY: GRB.BINARY,
    INTEGER: GRB.INTEGER,
}

_SENSE_TO_GRB = {"<": GRB.LESS_EQUAL, ">": GRB.GREATER_EQUAL, "=": GRB.EQUAL}


def _check_supported(model):
    unsupported = [
        ("quadratic objective terms", model.NumQNZs),
        ("quadratic constraints", model.NumQConstrs),
        ("SOS constraints", model.NumSOS),
        ("general constraints", model.NumGenConstrs),
        ("piecewise-linear objectives", model.NumPWLObjVars),
    ]
    found = [name for name, count in unsupported if count]
    if found:
        raise UnsupportedModelError(
            "cannot hide a model containing " + ", ".join(found)
        )

    for var in model.getVars():
        if var.VType not in _VTYPE_TO_GRB:
            raise UnsupportedModelError(
                f"variable {var.VarName!r} has unsupported type {var.VType!r} "
                "(semi-continuous and semi-integer are not supported)"
            )


def read_problem(path, env=None):
    """Read an MPS/LP file into a :class:`Problem`."""
    model = gp.read(str(path), env=env) if env is not None else gp.read(str(path))
    try:
        return problem_from_model(model)
    finally:
        model.dispose()


def problem_from_model(model):
    """Extract a :class:`Problem` from a live gurobipy model."""
    model.update()
    _check_supported(model)

    gvars = model.getVars()
    problem = Problem(
        name=model.ModelName or "model",
        vars=[
            Var(name=v.VarName, lb=v.LB, ub=v.UB, obj=v.Obj, vtype=v.VType)
            for v in gvars
        ],
        obj_offset=model.ObjCon,
        obj_sense=int(model.ModelSense),
    )

    # One sparse-matrix fetch beats getRow() per constraint on large models.
    matrix = model.getA().tocsr()
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    for i, con in enumerate(model.getConstrs()):
        coeffs = {
            int(indices[k]): float(data[k])
            for k in range(indptr[i], indptr[i + 1])
            if data[k] != 0.0
        }
        problem.add_row(con.ConstrName, coeffs, con.Sense, con.RHS)

    return problem


def model_from_problem(problem, env=None):
    """Build a gurobipy model from a :class:`Problem`."""
    model = (
        gp.Model(problem.name, env=env) if env is not None else gp.Model(problem.name)
    )
    model.setParam("OutputFlag", 0)

    # Gurobi would clamp these to [0, 1] without complaining, so a binary with
    # moved bounds must be caught here rather than turning into a model that
    # quietly means something else.
    bad = [v.name for v in problem.vars if not binary_bounds_are_sane(v)]
    if bad:
        raise UnsupportedModelError(
            "binary variable(s) with bounds outside [0, 1]: "
            + ", ".join(bad[:5])
            + (f" (and {len(bad) - 5} more)" if len(bad) > 5 else "")
            + "; Gurobi would discard these bounds silently"
        )

    gvars = [
        model.addVar(
            lb=v.lb,
            ub=v.ub,
            obj=v.obj,
            vtype=_VTYPE_TO_GRB[v.vtype],
            name=v.name,
        )
        for v in problem.vars
    ]

    for row in problem.rows:
        expr = gp.quicksum(coef * gvars[j] for j, coef in row.coeffs.items())
        model.addLConstr(expr, _SENSE_TO_GRB[row.sense], row.rhs, name=row.name)

    model.ObjCon = problem.obj_offset
    model.ModelSense = problem.obj_sense
    model.update()
    return model


def write_problem(problem, path, env=None):
    """Write a :class:`Problem` out; format follows the file extension."""
    model = model_from_problem(problem, env=env)
    try:
        model.write(str(path))
    finally:
        model.dispose()
