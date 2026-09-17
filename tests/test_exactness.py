"""Is the dyadic arithmetic actually bit-exact?

The README claims that with power-of-two factors every transformation is exact in
IEEE double arithmetic. Solver output cannot verify that: two formulations with
*identical* mathematics still report objectives differing by ~1e-12, because the
simplex takes a different path through them. Measured that way, ``dispatch``
showed 2e-12 and a Gaussian-elimination variant 7e-11 -- numbers that say nothing
about whether the coefficients themselves rounded.

Replaying the same transformation chain over :class:`fractions.Fraction` does
verify it. The choices are identical, only the arithmetic differs, so any
discrepancy in the resulting coefficients is a rounding error in the float path.

Exactness is not automatic. Multiplying by a power of two is always exact, but
row mixing *adds* dyadic multiples, and a sum of two dyadics rounds once it needs
more than 53 mantissa bits. What prevents that here is the coefficient-range
guard: it keeps the exponent spread narrow enough that the sums stay
representable. So this is a measured property of the package's default settings,
not a theorem -- which is exactly why it is worth a regression test.
"""

import random
from fractions import Fraction

import pytest

from mip_morph import transforms as T
from mip_morph.ir import CONTINUOUS, INTEGER, Problem

# Transformations whose multipliers come from the dyadic sets. ``split-free``,
# ``complement`` and ``shift`` use only integers and are exact trivially;
# ``xor-binary`` adds fixed 0/1/2 coefficients.
DYADIC_NAMES = ["slack", "mix-row", "dispatch", "scale-row", "scale-var", "invert-row"]


class RationalContext(T.Context):
    """Makes the same random choices, but returns them as exact rationals.

    Every value in ``DYADIC`` and ``MIX_LAMBDAS`` is exactly representable, so
    ``Fraction(float)`` is a lossless conversion -- the two contexts really do
    differ only in arithmetic, not in the numbers they pick.
    """

    def scale_factor(self, rng, signed=True):
        return Fraction(super().scale_factor(rng, signed))

    def positive_scale_factor(self, rng):
        return Fraction(super().positive_scale_factor(rng))

    def mix_lambda(self, rng):
        return Fraction(super().mix_lambda(rng))


def _build(cast, seed=7):
    rng = random.Random(seed)
    problem = Problem(name="dy")
    for i in range(8):
        problem.add_var(f"x{i}", cast(0), cast(1), cast(rng.randint(1, 9)), INTEGER)
    for i in range(6):
        problem.add_var(f"z{i}", cast(0), cast(10), cast(rng.randint(1, 5)), CONTINUOUS)
    for r in range(12):
        support = rng.sample(range(14), 4)
        problem.add_row(
            f"c{r}", {j: cast(rng.choice([1, 2, 4])) for j in support}, "<", cast(12)
        )
    problem.obj_sense = -1
    return problem


def _run(problem, ctx, seed, rounds):
    """Apply the same round structure the driver uses, minus the rollback guard."""
    rng = random.Random(seed)
    applied = []
    for _ in range(rounds):
        for name in rng.sample(DYADIC_NAMES, len(DYADIC_NAMES)):
            description = T.TRANSFORMS[name](problem, rng, ctx)
            if description is not None:
                applied.append(name)
                break
    return applied


@pytest.mark.parametrize("rounds", [10, 40, 120])
def test_dyadic_transforms_match_exact_rational_arithmetic(rounds):
    for seed in range(6):
        floats = _build(float)
        chain_f = _run(floats, T.Context(dispatch_fanout=4), seed, rounds)

        exact = _build(Fraction)
        chain_r = _run(exact, RationalContext(dispatch_fanout=4), seed, rounds)

        assert chain_f == chain_r, "the two runs took different transformations"
        assert len(floats.rows) == len(exact.rows)
        assert len(floats.vars) == len(exact.vars)

        for row_f, row_r in zip(floats.rows, exact.rows):
            assert set(row_f.coeffs) == set(row_r.coeffs), (
                f"seed {seed}: support diverged in {row_f.name}"
            )
            for j, coef in row_f.coeffs.items():
                assert coef == float(row_r.coeffs[j]), (
                    f"seed {seed}: {row_f.name} column {j} rounded"
                )
            assert row_f.rhs == float(row_r.rhs), f"seed {seed}: {row_f.name} rhs"

        for var_f, var_r in zip(floats.vars, exact.vars):
            assert var_f.obj == float(var_r.obj), f"seed {seed}: {var_f.name} cost"
            assert var_f.lb == float(var_r.lb), f"seed {seed}: {var_f.name} lb"
            assert var_f.ub == float(var_r.ub), f"seed {seed}: {var_f.name} ub"


def test_sloppy_scaling_is_not_exact():
    """The contrast that makes the dyadic claim meaningful.

    With arbitrary factors the coefficients do round, which is the whole reason
    ``--sloppy-scaling`` is opt-in.
    """
    rounds = 40
    inexact = 0
    for seed in range(6):
        floats = _build(float)
        _run(floats, T.Context(sloppy_scaling=True, dispatch_fanout=4), seed, rounds)
        for row in floats.rows:
            for coef in row.coeffs.values():
                # A dyadic chain over dyadic input can only produce values whose
                # binary expansion is short; sloppy factors produce full mantissas.
                if coef and Fraction(coef).limit_denominator(4096) != Fraction(coef):
                    inexact += 1
    assert inexact > 0, "sloppy scaling unexpectedly produced only dyadic values"
