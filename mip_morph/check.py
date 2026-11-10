"""Check that a run of transformations really preserves the objective.

Every transformation is meant to preserve the objective value pointwise: a
point of the original model, carried forward through each step, keeps both its
feasibility and its objective value. :func:`check_morph` runs the ordinary
driver with a known point and, after every accepted step, recomputes the
point's objective -- offset included -- and its worst constraint violation on
the model as it stands at that step. A step that breaks either is reported by
its position in the run, so a faulty transformation is caught where it happens
rather than as a wrong optimum at the end.
"""

from dataclasses import dataclass, field

from .ir import INF
from .morph import morph_model

DEFAULT_TOL = 1e-6


def objective_value(problem, x):
    """Objective of the point ``x`` (values by column index), offset included."""
    return problem.obj_offset + sum(
        var.obj * x[j] for j, var in enumerate(problem.vars)
    )


def max_violation(problem, x):
    """Largest bound, integrality or row violation of ``x`` (by column index).

    Row violations are relative to the magnitude of the row's terms, so that a
    row scaled up by a transformation is not held to a tighter standard.
    """
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


@dataclass
class StepCheck:
    """The carried point as it stood after one step."""

    index: int  # 1-based position in the run
    step: object
    objective: float
    error: float  # |objective - reference|, relative to max(1, |reference|)
    violation: float


@dataclass
class CheckResult:
    reference: float  # objective of the point on the original model
    start_violation: float
    tol: float
    steps: list = field(default_factory=list)
    hidden: object = None
    log: object = None

    @property
    def failures(self):
        return [s for s in self.steps if s.error > self.tol or s.violation > self.tol]

    @property
    def culprits(self):
        """The failing steps that made things worse than the step before them.

        Once a step breaks the point every later step inherits the damage;
        these are the ones that caused it.
        """
        found = []
        error = violation = 0.0
        for s in self.steps:
            if (s.error > self.tol and s.error > error + self.tol) or (
                s.violation > self.tol and s.violation > violation + self.tol
            ):
                found.append(s)
            error, violation = s.error, s.violation
        return found

    @property
    def ok(self):
        return not self.failures

    @property
    def worst_error(self):
        return max((s.error for s in self.steps), default=0.0)

    @property
    def worst_violation(self):
        return max((s.violation for s in self.steps), default=0.0)


def check_morph(problem, solution, tol=DEFAULT_TOL, **morph_kwargs):
    """Run :func:`morph_model` carrying ``solution`` and check every step.

    ``solution`` lists a point of ``problem`` by column index; it should be
    feasible, and is most informative when optimal. ``morph_kwargs`` go to
    :func:`morph_model` unchanged. Returns a :class:`CheckResult`; the hidden
    model and log are on ``.hidden`` and ``.log``.
    """
    point = [float(value) for value in solution]
    reference = objective_value(problem, point)
    result = CheckResult(
        reference=reference, start_violation=max_violation(problem, point), tol=tol
    )
    scale = max(1.0, abs(reference))

    def on_step(step, model, x):
        value = objective_value(model, x)
        result.steps.append(
            StepCheck(
                index=len(result.steps) + 1,
                step=step,
                objective=value,
                error=abs(value - reference) / scale,
                violation=max_violation(model, x),
            )
        )

    result.hidden, result.log = morph_model(
        problem, solution=point, on_step=on_step, **morph_kwargs
    )
    return result
