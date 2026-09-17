"""How binaries are handled.

Two rules, both about not throwing away the binary designation:

* ``shift`` never touches a binary. Translating one off [0, 1] forces it to a
  general integer, losing the clique/cover/implied-bound reasoning that depends
  on knowing a variable is 0/1 -- a real loss of solver strength for no gain in
  obfuscation. ``complement`` is the only nontrivial affine map of {0, 1} onto
  itself, so that is what a binary gets.
* ``xor-binary`` goes further: it makes the 0/1 nature *implied* by fresh
  binaries rather than declared on the original column.
"""

import itertools
import random

import pytest
from gurobipy import GRB

from mip_morph.io import model_from_problem
from mip_morph.ir import BINARY, CONTINUOUS, INTEGER, Problem
from mip_morph.morph import GROWTH_TRANSFORMS, morph_model
from mip_morph.transforms import (
    Context,
    encode_binary_as_xor,
    shift_variable,
)


def _binary_model(n=8, seed=0):
    """A small model whose columns are all binary."""
    rng = random.Random(seed)
    problem = Problem(name="bin")
    for i in range(n):
        problem.add_var(
            f"x{i}", lb=0.0, ub=1.0, obj=float(rng.randint(1, 9)), vtype=BINARY
        )
    for r in range(3):
        support = rng.sample(range(n), 4)
        problem.add_row(f"c{r}", {j: 1.0 for j in support}, "<", 2.0)
    problem.add_row("lb", {j: 1.0 for j in range(n)}, ">", 3.0)
    return problem


def _solve(problem):
    model = model_from_problem(problem)
    try:
        model.setParam("OutputFlag", 0)
        model.optimize()
        return model.Status, (model.ObjVal if model.Status == GRB.OPTIMAL else None)
    finally:
        model.dispose()


def _projection(problem, names):
    """0/1 patterns on ``names`` that extend to a feasible point."""
    feasible = set()
    for point in itertools.product([0, 1], repeat=len(names)):
        model = model_from_problem(problem)
        try:
            model.setParam("OutputFlag", 0)
            by_name = {v.VarName: v for v in model.getVars()}
            for name, value in zip(names, point):
                var = by_name[name]
                var.LB = value
                var.UB = value
            model.update()
            model.optimize()
            if model.Status == GRB.OPTIMAL:
                feasible.add(point)
        finally:
            model.dispose()
    return feasible


# ---------------------------------------------------------------------------
# shift must leave binaries alone
# ---------------------------------------------------------------------------


def test_shift_declines_an_all_binary_model():
    problem = _binary_model()
    assert shift_variable(problem, random.Random(0), Context()) is None


def test_shift_never_retypes_a_binary():
    problem = _binary_model()
    hidden, log = morph_model(
        problem, rounds=40, seed=3, enable=["shift"], rename=False, shuffle=False
    )
    assert log == [], "shift should find no candidates in an all-binary model"
    assert all(v.vtype == BINARY for v in hidden.vars)


def test_shift_declines_an_undeclared_binary():
    """A 0/1 column in an INTEGER section is a binary in all but name.

    The ``pret*`` instances are exactly this: 60 columns declared INTEGER with
    bounds [0, 1], which presolve turns into binaries. Keying the rule off
    ``vtype`` alone would let shift strip them.
    """
    problem = Problem()
    problem.add_var("x", lb=0.0, ub=1.0, obj=1.0, vtype=INTEGER)
    problem.add_row("r", {0: 1.0}, "<", 1.0)

    assert shift_variable(problem, random.Random(0), Context()) is None
    assert (problem.vars[0].lb, problem.vars[0].ub) == (0.0, 1.0)


def test_undeclared_binaries_survive_a_full_run():
    """Presolve must still find as many binaries as it did in the original."""
    problem = Problem(name="pretlike")
    n = 12
    for i in range(n):
        problem.add_var(f"x{i}", lb=0.0, ub=1.0, obj=0.0, vtype=INTEGER)
    rng = random.Random(0)
    for r in range(6):
        support = rng.sample(range(n), 3)
        problem.add_row(f"c{r}", {j: 1.0 for j in support}, "<", 2.0)

    def zero_one_columns(p):
        return sum(1 for v in p.vars if v.is_zero_one)

    before = zero_one_columns(problem)
    for seed in range(6):
        hidden, _ = morph_model(problem, rounds=40, seed=seed, disable=["xor-binary"])
        assert zero_one_columns(hidden) >= before, f"seed {seed}: lost 0/1 columns"


def test_xor_encoding_accepts_an_undeclared_binary():
    problem = Problem()
    problem.add_var("x", lb=0.0, ub=1.0, obj=1.0, vtype=INTEGER)
    problem.add_row("r", {0: 1.0}, ">", 1.0)

    assert encode_binary_as_xor(problem, random.Random(0), Context()) is not None
    assert problem.vars[0].vtype == CONTINUOUS


def test_shift_still_works_on_general_integers():
    problem = Problem()
    problem.add_var("y", lb=0.0, ub=6.0, obj=1.0, vtype=INTEGER)
    problem.add_row("r", {0: 1.0}, "<", 4.0)

    assert shift_variable(problem, random.Random(0), Context()) is not None
    assert problem.vars[0].vtype == INTEGER


def test_shift_still_works_on_continuous():
    problem = Problem()
    problem.add_var("c", lb=0.0, ub=6.0, obj=1.0, vtype=CONTINUOUS)
    problem.add_row("r", {0: 1.0}, "<", 4.0)

    assert shift_variable(problem, random.Random(0), Context()) is not None
    assert problem.vars[0].vtype == CONTINUOUS


def test_full_pipeline_keeps_every_binary_binary():
    """No transformation may quietly turn a binary into a general integer."""
    problem = _binary_model(n=10)
    n_binaries = sum(1 for v in problem.vars if v.vtype == BINARY)

    for seed in range(6):
        hidden, _ = morph_model(problem, rounds=40, seed=seed, disable=["xor-binary"])
        assert not any(v.vtype == INTEGER for v in hidden.vars), (
            f"seed {seed}: a binary was demoted to a general integer"
        )
        still_binary = sum(1 for v in hidden.vars if v.vtype == BINARY)
        assert still_binary >= n_binaries, f"seed {seed}: lost binaries"


def test_binaries_survive_presolve_as_binaries():
    problem = _binary_model(n=12)

    def presolved_binaries(p):
        model = model_from_problem(p)
        try:
            model.setParam("OutputFlag", 0)
            return sum(1 for v in model.getVars() if v.VType == GRB.BINARY)
        finally:
            model.dispose()

    before = presolved_binaries(problem)
    hidden, _ = morph_model(problem, rounds=40, seed=1, disable=["xor-binary"])
    assert presolved_binaries(hidden) >= before


# ---------------------------------------------------------------------------
# xor-binary: integrality becomes implied rather than declared
# ---------------------------------------------------------------------------


def test_xor_encoding_relaxes_the_original_column():
    problem = Problem()
    problem.add_var("x", lb=0.0, ub=1.0, obj=1.0, vtype=BINARY)
    problem.add_row("r", {0: 1.0}, ">", 1.0)

    assert encode_binary_as_xor(problem, random.Random(0), Context()) is not None

    assert problem.vars[0].vtype == CONTINUOUS, "x should no longer be declared binary"
    assert (problem.vars[0].lb, problem.vars[0].ub) == (0.0, 1.0)
    assert problem.vars[0].obj == 1.0, "cost must be untouched"
    # three new binaries and four new rows
    assert len(problem.vars) == 4
    assert sum(1 for v in problem.vars if v.vtype == BINARY) == 3
    assert len(problem.rows) == 5


def test_xor_encoding_admits_exactly_zero_and_one():
    """The AND rows must let u+v-2w reach 0 and 1, and nothing else."""
    problem = Problem()
    problem.add_var("x", lb=0.0, ub=1.0, obj=0.0, vtype=BINARY)
    encode_binary_as_xor(problem, random.Random(0), Context())

    for target in (0.0, 1.0):
        p = problem.copy()
        p.vars[0].lb = target
        p.vars[0].ub = target
        status, _ = _solve(p)
        assert status == GRB.OPTIMAL, f"x = {target} should stay feasible"


def test_xor_encoding_preserves_the_lp_relaxation():
    """u+v-2w ranges over exactly [0,1] in the relaxation, so nothing tightens."""
    problem = Problem()
    problem.add_var("x", lb=0.0, ub=1.0, obj=1.0, vtype=BINARY)
    problem.add_row("keep", {0: 1.0}, ">", 0.0)
    encode_binary_as_xor(problem, random.Random(0), Context())

    for sense, expected in ((1, 0.0), (-1, 1.0)):
        model = model_from_problem(problem)
        try:
            model.setParam("OutputFlag", 0)
            for var in model.getVars():
                var.VType = GRB.CONTINUOUS
            model.ModelSense = sense
            model.update()
            model.optimize()
            assert model.Status == GRB.OPTIMAL
            assert model.ObjVal == pytest.approx(expected), (
                "the relaxation of x must still be exactly [0, 1]"
            )
        finally:
            model.dispose()


def test_xor_encoding_preserves_optimum():
    problem = _binary_model(n=8, seed=1)
    status0, expected = _solve(problem)
    assert status0 == GRB.OPTIMAL

    for seed in range(5):
        hidden, log = morph_model(
            problem,
            rounds=20,
            seed=seed,
            enable=["xor-binary"],
            rename=False,
            shuffle=False,
        )
        assert log, f"seed {seed}: expected applications"
        status, obj = _solve(hidden)
        assert status == GRB.OPTIMAL, f"seed {seed}"
        assert obj == pytest.approx(expected, abs=1e-6), f"seed {seed}"


def test_xor_encoding_preserves_the_projected_feasible_set():
    """Solution counts multiply (x=1 has two encodings), so compare projections."""
    problem = _binary_model(n=6, seed=2)
    names = [f"x{i}" for i in range(6)]
    before = _projection(problem, names)

    hidden, _ = morph_model(
        problem, rounds=12, seed=1, enable=["xor-binary"], rename=False, shuffle=False
    )
    assert _projection(hidden, names) == before


def test_xor_encoding_hides_declared_binaries():
    """Each application relaxes one more originally-declared binary.

    Candidates include the fresh u/v/w columns, so a run of N rounds does not
    necessarily cover all N original columns; what must hold is that the set of
    relaxed originals only grows, and that enough rounds relax all of them.
    """
    n = 6
    problem = _binary_model(n=n, seed=0)
    original_names = {f"x{i}" for i in range(n)}

    def relaxed(p):
        return {
            v.name for v in p.vars if v.name in original_names and v.vtype == CONTINUOUS
        }

    previous = set()
    for rounds in (1, 3, 6, 20, 60):
        hidden, log = morph_model(
            problem,
            rounds=rounds,
            seed=1,
            enable=["xor-binary"],
            rename=False,
            shuffle=False,
        )
        assert len(log) == rounds
        current = relaxed(hidden)
        assert current >= previous, "relaxed set should only grow with more rounds"
        previous = current

    # With plenty of rounds every original column ends up implied-integral.
    hidden, _ = morph_model(
        problem,
        rounds=200,
        seed=1,
        enable=["xor-binary"],
        rename=False,
        shuffle=False,
    )
    assert relaxed(hidden) == original_names


def test_xor_growth_is_capped_in_mixed_runs():
    """The growth transform must not dominate a run and bloat the model."""
    problem = _binary_model(n=10)
    rounds = 40
    hidden, log = morph_model(problem, rounds=rounds, seed=1)
    applications = sum(1 for entry in log if "as XOR of" in str(entry))
    assert applications <= max(1, int(rounds * 0.25)), applications
    assert len(hidden.vars) < len(problem.vars) + 3 * rounds


def test_xor_growth_uncapped_when_requested_alone():
    """Asking for it explicitly should not be silently throttled."""
    problem = _binary_model(n=20)
    hidden, log = morph_model(problem, rounds=15, seed=1, enable=["xor-binary"])
    assert len(log) == 15


def test_moved_binary_bounds_are_refused_not_silently_clamped():
    """The hazard that made shifting binaries wrong in the first place.

    Gurobi accepts ``addVar(lb=-1, ub=0, vtype=BINARY)`` and reports LB=UB=0,
    discarding the bounds. Nothing in the package should produce such a column,
    so building a model from one is an error rather than a silent change of
    meaning.
    """
    from mip_morph.io import UnsupportedModelError

    problem = Problem()
    problem.add_var("b", lb=-1.0, ub=0.0, vtype=BINARY)
    with pytest.raises(UnsupportedModelError, match="outside"):
        model_from_problem(problem)


def test_every_transform_leaves_binary_bounds_writable():
    """A full run must never produce a model the io guard would reject."""
    problem = _binary_model(n=10)
    for seed in range(8):
        hidden, _ = morph_model(problem, rounds=40, seed=seed)
        model = model_from_problem(hidden)  # raises if a binary moved
        model.dispose()


def test_growth_transform_registry_is_consistent():
    from mip_morph.transforms import TRANSFORMS

    for name in GROWTH_TRANSFORMS:
        assert name in TRANSFORMS


def test_xor_encoding_on_a_model_without_binaries():
    problem = Problem()
    problem.add_var("y", lb=0.0, ub=5.0, vtype=INTEGER)
    problem.add_var("c", lb=0.0, ub=5.0, vtype=CONTINUOUS)
    assert encode_binary_as_xor(problem, random.Random(0), Context()) is None
