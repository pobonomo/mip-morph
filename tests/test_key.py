"""Recovery: a key plus a hidden model must give the original back.

The central claim is *exact* equality, not equality to a tolerance. A default
hiding run uses dyadic scale factors and undoes every fold by subtracting the
same multiple it added, so every inverse is exact in IEEE double arithmetic and
the recovered model should be identical to the original coefficient for
coefficient. Asserting a tolerance here would hide precisely the bugs worth
catching -- a reciprocal stored instead of a divisor, a fold undone in the wrong
order -- so these tests compare snapshots with ``==``.
"""

import random

import pytest
from conftest import feasible_random_mips, random_mip, solve_problem
from gurobipy import GRB

from mip_morph.io import read_problem, write_problem
from mip_morph.key import (
    KeyFormatError,
    decode_key,
    encode_key,
    read_key,
    write_key,
)
from mip_morph.morph import morph_model
from mip_morph.transforms import ALL_NAMES
from mip_morph.unmorph import KeyMismatchError, unmorph_problem, unmorph_solution

FEASIBLE = feasible_random_mips(20)


def snapshot(problem):
    """The canonical structural fingerprint used across the suite."""
    return (
        [(v.name, v.lb, v.ub, v.obj, v.vtype) for v in problem.vars],
        [(r.name, dict(r.coeffs), r.sense, r.rhs) for r in problem.rows],
        problem.obj_offset,
        problem.obj_sense,
    )


# --------------------------------------------------------------------------
# model recovery
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rounds", [1, 5, 40, 120])
def test_key_recovers_the_original_exactly(rounds):
    """Over every feasible random model, at chain depths up to 120."""
    for seed, problem, _ in FEASIBLE:
        hidden, log = morph_model(problem, rounds=rounds, seed=seed)
        recovered = unmorph_problem(hidden, log.key)
        assert snapshot(recovered) == snapshot(problem), (
            f"seed {seed}, {rounds} rounds, {len(log)} steps applied"
        )


def test_recovery_is_exact_on_the_hand_built_model(tiny_problem):
    hidden, log = morph_model(tiny_problem, rounds=60, seed=7)
    assert len(log) > 0
    recovered = unmorph_problem(hidden, log.key)
    assert snapshot(recovered) == snapshot(tiny_problem)


@pytest.mark.parametrize("name", sorted(ALL_NAMES))
def test_each_transform_alone_is_invertible(name):
    """Each transformation on its own, so a failure names the culprit."""
    for seed, problem, _ in FEASIBLE[:8]:
        hidden, log = morph_model(
            problem,
            rounds=12,
            seed=seed,
            enable=[name],
            preserve_objective=False,
        )
        if not log:
            continue  # this model offers that transform no candidate
        recovered = unmorph_problem(hidden, log.key)
        assert snapshot(recovered) == snapshot(problem), f"{name}, seed {seed}"


def test_recovery_works_without_renaming_or_shuffling():
    """The alignment step must be a no-op when there was nothing to align."""
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(
        problem, rounds=40, seed=seed, rename=False, shuffle=False
    )
    recovered = unmorph_problem(hidden, log.key)
    assert snapshot(recovered) == snapshot(problem)


def test_recovery_survives_a_file_round_trip(tmp_path):
    """The point of aligning by name: file order is not to be trusted.

    This is also what proves the retired column ``split-free`` leaves behind --
    all-zero bounds, zero cost, no row entries -- is not dropped by the writer.
    """
    for suffix in (".mps", ".lp"):
        seed, problem, _ = FEASIBLE[1]
        hidden, log = morph_model(problem, rounds=40, seed=seed)
        path = tmp_path / f"hidden{suffix}"
        write_problem(hidden, str(path))
        reloaded = read_problem(str(path))
        recovered = unmorph_problem(reloaded, log.key)
        assert snapshot(recovered) == snapshot(problem), suffix


def test_sloppy_scaling_recovers_approximately():
    """The forward pass itself rounds here, so exactness is not on offer."""
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(problem, rounds=40, seed=seed, sloppy_scaling=True)
    recovered = unmorph_problem(hidden, log.key)

    assert [v.name for v in recovered.vars] == [v.name for v in problem.vars]
    assert [r.name for r in recovered.rows] == [r.name for r in problem.rows]
    for got, want in zip(recovered.rows, problem.rows):
        assert got.sense == want.sense
        assert set(got.coeffs) == set(want.coeffs)
        assert got.rhs == pytest.approx(want.rhs, rel=1e-9, abs=1e-9)
        for j, c in want.coeffs.items():
            assert got.coeffs[j] == pytest.approx(c, rel=1e-9, abs=1e-9)


def test_the_hidden_model_is_not_mutated_by_recovery():
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(problem, rounds=30, seed=seed)
    before = snapshot(hidden)
    unmorph_problem(hidden, log.key)
    assert snapshot(hidden) == before


# --------------------------------------------------------------------------
# solution recovery
# --------------------------------------------------------------------------


def solve_values(problem):
    """``(status, objective, {name: value})`` -- values empty unless optimal."""
    from mip_morph.io import model_from_problem

    model = model_from_problem(problem)
    try:
        model.setParam("OutputFlag", 0)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            return model.Status, None, {}
        return (
            model.Status,
            model.ObjVal,
            {v.VarName: v.X for v in model.getVars()},
        )
    finally:
        model.dispose()


def check_point(problem, values, tol=1e-6):
    """Every way the original model can reject a point. Returns the violations.

    Checked directly rather than by pinning the values as bounds and re-solving,
    because pinning a binary to 0 or 1 means giving it bounds Gurobi refuses --
    ``model_from_problem`` rejects a BINARY column whose bounds are not exactly
    [0, 1], on purpose. Evaluating the constraints here also says more: a solver
    would only report "infeasible", while this names what failed.
    """
    bad = []
    x = [values[v.name] for v in problem.vars]
    for var, value in zip(problem.vars, x):
        if value < var.lb - tol or value > var.ub + tol:
            bad.append(f"{var.name}={value!r} outside [{var.lb}, {var.ub}]")
        if var.is_integral and abs(value - round(value)) > tol:
            bad.append(f"{var.name}={value!r} is not integral")
    for row in problem.rows:
        activity = sum(coef * x[j] for j, coef in row.coeffs.items())
        if row.sense == "<" and activity > row.rhs + tol:
            bad.append(f"{row.name}: {activity!r} > {row.rhs}")
        elif row.sense == ">" and activity < row.rhs - tol:
            bad.append(f"{row.name}: {activity!r} < {row.rhs}")
        elif row.sense == "=" and abs(activity - row.rhs) > tol:
            bad.append(f"{row.name}: {activity!r} != {row.rhs}")
    return bad


def objective_at(problem, values):
    return problem.obj_offset + sum(
        v.obj * values[v.name] for v in problem.vars
    )


def test_solution_maps_back_to_the_original_variables():
    """Solve the hidden model, map the point back, check the original accepts it.

    Feasibility in the *original* model is the assertion that matters: it says
    every constraint, bound and integrality restriction of the model the user
    started with holds at the recovered point. Attaining the original's known
    optimum on top of that rules out a point that is merely feasible.
    """
    for seed, problem, expected in FEASIBLE[:10]:
        hidden, log = morph_model(problem, rounds=40, seed=seed)
        status, hidden_obj, values = solve_values(hidden)
        assert status == GRB.OPTIMAL, seed
        assert hidden_obj == pytest.approx(expected, rel=1e-7, abs=1e-7)

        recovered = unmorph_solution(values, log.key)
        assert set(recovered) == {v.name for v in problem.vars}

        violations = check_point(problem, recovered)
        assert not violations, f"seed {seed}: recovered point violates {violations}"
        assert objective_at(problem, recovered) == pytest.approx(
            expected, rel=1e-6, abs=1e-6
        ), seed


def test_solution_recovery_rejects_a_foreign_solution():
    seed, problem, _ = FEASIBLE[0]
    _, log = morph_model(problem, rounds=20, seed=seed)
    with pytest.raises(KeyMismatchError):
        unmorph_solution({"nosuchcolumn": 1.0}, log.key)


# --------------------------------------------------------------------------
# mismatch detection
# --------------------------------------------------------------------------


def test_a_key_from_another_run_is_rejected():
    seed, problem, _ = FEASIBLE[0]
    hidden, _ = morph_model(problem, rounds=40, seed=seed)
    _, other_log = morph_model(problem, rounds=40, seed=seed + 1000)
    with pytest.raises(KeyMismatchError):
        unmorph_problem(hidden, other_log.key)


def test_a_key_from_another_model_is_rejected():
    hidden, _ = morph_model(FEASIBLE[0][1], rounds=30, seed=1)
    _, other_log = morph_model(random_mip(999), rounds=30, seed=1)
    with pytest.raises(KeyMismatchError):
        unmorph_problem(hidden, other_log.key)


def test_a_perturbed_model_is_rejected():
    """Same names and shape, one coefficient changed: structure still catches it."""
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(problem, rounds=30, seed=seed)
    damaged = hidden.copy()
    # Give some row an entry it did not have, which changes the pattern.
    row = damaged.rows[0]
    absent = next(j for j in range(len(damaged.vars)) if j not in row.coeffs)
    row.coeffs[absent] = 1.0
    with pytest.raises(KeyMismatchError):
        unmorph_problem(damaged, log.key)


def test_value_drift_warns_but_still_recovers():
    """A number that moved is a lossy round trip, not a wrong key."""
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(problem, rounds=30, seed=seed)
    drifted = hidden.copy()
    row = next(r for r in drifted.rows if r.coeffs)
    j = next(iter(row.coeffs))
    row.coeffs[j] += 1e-11
    with pytest.warns(UserWarning, match="lost precision"):
        unmorph_problem(drifted, log.key)


def test_duplicate_hidden_names_are_reported_clearly():
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(problem, rounds=20, seed=seed)
    hidden.vars[1].name = hidden.vars[0].name
    log.key.hidden_var_names[1] = log.key.hidden_var_names[0]
    with pytest.raises(KeyMismatchError, match="two columns called"):
        unmorph_problem(hidden, log.key)


# --------------------------------------------------------------------------
# the key container
# --------------------------------------------------------------------------


def test_key_round_trips_through_a_file(tmp_path):
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(problem, rounds=40, seed=seed)
    path = tmp_path / "k.key"
    write_key(log.key, str(path))
    loaded = read_key(str(path))

    assert loaded.orig_var_names == log.key.orig_var_names
    assert loaded.hidden_row_names == log.key.hidden_row_names
    assert [s.op for s in loaded.steps] == [s.op for s in log.key.steps]
    assert [s.params for s in loaded.steps] == [s.params for s in log.key.steps]
    recovered = unmorph_problem(hidden, loaded)
    assert snapshot(recovered) == snapshot(problem)


def test_key_round_trips_with_a_password(tmp_path):
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(problem, rounds=40, seed=seed)
    path = tmp_path / "k.key"
    write_key(log.key, str(path), password="hunter2")
    recovered = unmorph_problem(hidden, read_key(str(path), password="hunter2"))
    assert snapshot(recovered) == snapshot(problem)


def test_key_bytes_are_exact_for_sloppy_factors():
    """The reason the key is JSON and not the human-readable log.

    ``%g`` keeps six significant digits, which is not enough to invert an
    arbitrary factor; ``repr`` round-trips a float exactly.
    """
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(problem, rounds=40, seed=seed, sloppy_scaling=True)
    reloaded = decode_key(encode_key(log.key))
    assert [s.params for s in reloaded.steps] == [s.params for s in log.key.steps]
    assert snapshot(unmorph_problem(hidden, reloaded)) == snapshot(
        unmorph_problem(hidden, log.key)
    )


def test_key_is_opaque():
    """Not human-readable: nothing meaningful survives as legible text.

    Only strings of four characters or more are worth asserting on. A two-letter
    name like ``b0`` turns up in a base64 blob of this length by coincidence, so
    checking for it would make the test fail at random without saying anything
    about legibility.
    """
    seed, problem, _ = FEASIBLE[0]
    _, log = morph_model(problem, rounds=40, seed=seed)
    blob = encode_key(log.key)

    body = blob.split(b"\n", 1)[1]
    names = [problem.name] + [v.name for v in problem.vars]
    names += [r.name for r in problem.rows] + list(log.key.hidden_var_names)
    for name in names:
        if len(name) >= 4:
            assert name.encode() not in body, name
    for text in (b"complement", b"scale-row", b"dispatch", b"params", b"steps"):
        assert text not in body, text


def test_a_password_actually_conceals_the_key():
    seed, problem, _ = FEASIBLE[0]
    _, log = morph_model(problem, rounds=40, seed=seed)
    plain = encode_key(log.key)
    secret = encode_key(log.key, password="pw")
    assert plain != secret
    # Two encryptions differ: the salt is fresh each time.
    assert secret != encode_key(log.key, password="pw")
    with pytest.raises(KeyFormatError, match="password is required"):
        decode_key(secret)
    with pytest.raises(KeyFormatError):
        decode_key(secret, password="wrong")


def test_a_damaged_key_is_refused_rather_than_misread():
    seed, problem, _ = FEASIBLE[0]
    _, log = morph_model(problem, rounds=40, seed=seed)

    with pytest.raises(KeyFormatError, match="bad magic"):
        decode_key(b"just some other file\n")

    blob = encode_key(log.key)
    with pytest.raises(KeyFormatError):
        decode_key(blob[: len(blob) // 2])

    for password in (None, "pw"):
        blob = encode_key(log.key, password=password)
        body = bytearray(blob)
        # Flip a bit in the middle of the armoured payload.
        middle = len(blob) // 2
        body[middle] = ord("A") if body[middle] != ord("A") else ord("B")
        with pytest.raises(KeyFormatError):
            decode_key(bytes(body), password=password)


def test_a_plain_key_read_with_a_password_is_refused():
    seed, problem, _ = FEASIBLE[0]
    _, log = morph_model(problem, rounds=10, seed=seed)
    with pytest.raises(KeyFormatError, match="not encrypted"):
        decode_key(encode_key(log.key), password="pw")


# --------------------------------------------------------------------------
# the log still behaves like a list of strings
# --------------------------------------------------------------------------


def test_log_is_still_a_printable_list():
    seed, problem, _ = FEASIBLE[0]
    _, log = morph_model(problem, rounds=20, seed=seed)
    assert isinstance(log, list)
    assert len(log) == len(log.key.steps)
    for entry in log:
        assert isinstance(str(entry), str)
        assert str(entry)
        assert f"{entry}" == entry.description


def test_slack_undoes_its_fold_before_removing_its_column():
    """The one step whose two halves must be undone in the right order.

    ``slack`` adds its column to the row *and then* folds that row into another,
    so the fold carried the slack coefficient with it. Un-folding has to happen
    while the row still holds that coefficient. Doing it the other way round
    subtracts a row that is missing an entry, which leaves the target wrong -- so
    this pins the order on a model built to force a fold.
    """
    problem = random_mip(3)
    hidden, log = morph_model(problem, rounds=25, seed=11, enable=["slack"])
    folded = [s for s in log if s.op == "slack" and s.params["fold"] is not None]
    assert folded, "no slack fold happened, so this test proves nothing"
    recovered = unmorph_problem(hidden, log.key)
    assert snapshot(recovered) == snapshot(problem)


def test_transform_params_never_name_a_column():
    """Names are not stable identifiers mid-run; indices are."""
    seed, problem, _ = FEASIBLE[0]
    _, log = morph_model(problem, rounds=60, seed=seed)
    names = {v.name for v in problem.vars} | {r.name for r in problem.rows}
    for step in log:
        for value in step.params.values():
            assert not (isinstance(value, str) and value in names), step


def test_every_transform_records_an_op_the_inverse_engine_knows():
    from mip_morph.unmorph import INVERSES, SOLUTION_MAPS

    assert set(INVERSES) == set(ALL_NAMES)
    assert set(SOLUTION_MAPS) == set(ALL_NAMES)


def test_rounds_zero_still_produces_a_usable_key():
    seed, problem, _ = FEASIBLE[0]
    hidden, log = morph_model(problem, rounds=0, seed=seed)
    assert len(log) == 0
    recovered = unmorph_problem(hidden, log.key)
    assert snapshot(recovered) == snapshot(problem)


def test_rename_all_reports_the_names_it_overwrote():
    from mip_morph.morph import rename_all

    problem = random_mip(5)
    expected = [v.name for v in problem.vars]
    var_names, row_names = rename_all(problem)
    assert var_names == expected
    assert row_names[0] == "r0"
    assert [v.name for v in problem.vars] != expected


def test_shuffle_all_reports_permutations_that_invert_it():
    from mip_morph.morph import shuffle_all

    problem = random_mip(6)
    before = [v.name for v in problem.vars]
    rows_before = [r.name for r in problem.rows]
    col_order, row_order = shuffle_all(problem, random.Random(2))
    assert sorted(col_order) == list(range(len(before)))
    assert [before[old] for old in col_order] == [v.name for v in problem.vars]
    assert [rows_before[old] for old in row_order] == [r.name for r in problem.rows]
