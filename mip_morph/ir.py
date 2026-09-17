"""Solver-independent intermediate representation of a linear/mixed-integer model.

Keeping the model in plain Python objects means the transformations in
``transforms.py`` are pure data manipulation: they can be unit tested without
building a Gurobi model, and they never have to worry about gurobipy's
attribute-update semantics.
"""

from dataclasses import dataclass, field, replace

INF = float("inf")

CONTINUOUS = "C"
BINARY = "B"
INTEGER = "I"

SENSES = ("<", ">", "=")
FLIP = {"<": ">", ">": "<", "=": "="}


@dataclass
class Var:
    name: str
    lb: float = 0.0
    ub: float = INF
    obj: float = 0.0
    vtype: str = CONTINUOUS

    @property
    def is_integral(self):
        return self.vtype in (BINARY, INTEGER)

    @property
    def is_bounded(self):
        return self.lb > -INF and self.ub < INF

    @property
    def is_zero_one(self):
        """True for any variable restricted to {0, 1}, however it is declared.

        MPS files often carry 0/1 columns in an ``INTEGER`` section rather than a
        ``BINARY`` one -- the ``pret*`` instances do -- and Gurobi's presolve
        recognizes them as binaries regardless. What earns the clique, cover and
        implied-bound reasoning is being 0/1, not the declaration, so any rule
        about protecting binaries has to test this rather than ``vtype``.
        """
        return self.is_integral and self.lb == 0.0 and self.ub == 1.0


@dataclass
class Row:
    name: str
    coeffs: dict = field(default_factory=dict)  # var index -> coefficient
    sense: str = "<"
    rhs: float = 0.0

    @property
    def nnz(self):
        return len(self.coeffs)


@dataclass
class Problem:
    name: str = "model"
    vars: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    obj_offset: float = 0.0
    obj_sense: int = 1  # 1 = minimize, -1 = maximize

    # -- construction helpers ------------------------------------------------

    def add_var(self, name, lb=0.0, ub=INF, obj=0.0, vtype=CONTINUOUS):
        """Append a variable and return its index."""
        self.vars.append(Var(name=name, lb=lb, ub=ub, obj=obj, vtype=vtype))
        return len(self.vars) - 1

    def add_row(self, name, coeffs, sense, rhs):
        self.rows.append(Row(name=name, coeffs=dict(coeffs), sense=sense, rhs=rhs))
        return len(self.rows) - 1

    def copy(self):
        """Deep enough copy that transforms cannot alias the original."""
        return Problem(
            name=self.name,
            vars=[replace(v) for v in self.vars],
            rows=[replace(r, coeffs=dict(r.coeffs)) for r in self.rows],
            obj_offset=self.obj_offset,
            obj_sense=self.obj_sense,
        )

    # -- queries -------------------------------------------------------------

    def rows_with(self, j):
        """Rows that have a nonzero on variable index ``j``."""
        return [r for r in self.rows if j in r.coeffs]

    def equality_rows(self):
        return [r for r in self.rows if r.sense == "="]

    def objective_is_zero(self):
        """True when no variable carries a cost: a pure feasibility problem."""
        return all(v.obj == 0.0 for v in self.vars)


def negate(value):
    """Negate without producing ``-0.0``.

    A ``-0.0`` cost compares equal to zero but can still be written out as an
    explicit objective entry, which would turn a pure feasibility problem into
    an optimization problem on paper.
    """
    return -value if value else 0.0


def drop_zeros(row, tol=0.0):
    """Remove coefficients that cancelled out, keeping the matrix sparse."""
    for j in [j for j, c in row.coeffs.items() if abs(c) <= tol]:
        del row.coeffs[j]


def binary_bounds_are_sane(var):
    """False when a binary carries bounds Gurobi would silently discard.

    Gurobi clamps the bounds of a ``BINARY`` variable to [0, 1] regardless of
    what ``lb``/``ub`` say. A transformation that moves a binary's bounds off
    [0, 1] therefore loses them without any error, quietly changing what the
    model means -- so :mod:`mip_morph.io` checks this before building a model
    rather than letting the loss happen. No transformation should ever move a
    binary's bounds: ``complement`` maps {0, 1} onto itself and ``shift``
    excludes 0/1 columns outright.

    Scoped to declared binaries on purpose. An ``INTEGER`` column translated off
    [0, 1] writes out perfectly well -- it is merely weaker, which is why
    ``shift`` avoids it via ``is_zero_one`` -- so it is not an error here.
    """
    return var.vtype != BINARY or (var.lb == 0.0 and var.ub == 1.0)


def coef_stats(problem):
    """(min |coef|, max |coef|) over the constraint matrix; (0, 0) if empty."""
    lo = INF
    hi = 0.0
    for row in problem.rows:
        for c in row.coeffs.values():
            a = abs(c)
            if a == 0.0:
                continue
            lo = min(lo, a)
            hi = max(hi, a)
    if hi == 0.0:
        return 0.0, 0.0
    return lo, hi


def coef_range(problem):
    """max|coef| / min|coef| -- a proxy for how nasty the matrix has become."""
    lo, hi = coef_stats(problem)
    if lo == 0.0:
        return 1.0
    return hi / lo
