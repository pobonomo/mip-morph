"""End-to-end tests driving the command line interface."""

import pytest
from conftest import feasible_mip, solve_problem
from gurobipy import GRB

from mip_morph.cli import main
from mip_morph.io import read_problem, write_problem


@pytest.fixture
def model_file(tmp_path):
    problem, obj = feasible_mip()
    path = tmp_path / "model.mps"
    write_problem(problem, path)
    return path, obj


def test_cli_writes_an_equivalent_model(model_file, tmp_path, capsys):
    path, expected = model_file
    out = tmp_path / "hidden.mps"

    assert main([str(path), "-o", str(out), "--seed", "42", "--rounds", "30"]) == 0
    assert out.exists()

    status, obj = solve_problem(read_problem(out))
    assert status == GRB.OPTIMAL
    assert obj == pytest.approx(expected, abs=1e-6)

    captured = capsys.readouterr()
    assert "transformations applied" in captured.out


def test_cli_is_deterministic_for_a_given_seed(model_file, tmp_path):
    path, _ = model_file
    first = tmp_path / "a.mps"
    second = tmp_path / "b.mps"

    args = ["--seed", "7", "--rounds", "25"]
    assert main([str(path), "-o", str(first), *args]) == 0
    assert main([str(path), "-o", str(second), *args]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_cli_different_seeds_give_different_files(model_file, tmp_path):
    path, _ = model_file
    first = tmp_path / "a.mps"
    second = tmp_path / "b.mps"

    assert main([str(path), "-o", str(first), "--seed", "1", "--rounds", "25"]) == 0
    assert main([str(path), "-o", str(second), "--seed", "2", "--rounds", "25"]) == 0
    assert first.read_bytes() != second.read_bytes()


def test_cli_quiet_suppresses_output(model_file, tmp_path, capsys):
    path, _ = model_file
    out = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(out), "--seed", "1", "-q"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_enable_restricts_transformations(model_file, tmp_path, capsys):
    path, expected = model_file
    out = tmp_path / "hidden.mps"
    assert (
        main([str(path), "-o", str(out), "--seed", "1", "--enable", "scale-row"]) == 0
    )
    lines = [
        line for line in capsys.readouterr().out.splitlines() if "scale row" in line
    ]
    assert lines, "expected scale-row entries in the log"

    status, obj = solve_problem(read_problem(out))
    assert obj == pytest.approx(expected, abs=1e-6)


def test_cli_rejects_unknown_transformation(model_file, tmp_path, capsys):
    path, _ = model_file
    out = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(out), "--enable", "bogus"]) == 2
    assert "unknown transformation" in capsys.readouterr().err


def test_cli_no_rename_keeps_names(model_file, tmp_path):
    path, _ = model_file
    out = tmp_path / "hidden.mps"
    assert (
        main(
            [
                str(path),
                "-o",
                str(out),
                "--seed",
                "1",
                "--rounds",
                "10",
                "--enable",
                "scale-row",
                "--no-rename",
                "--no-shuffle",
            ]
        )
        == 0
    )
    original = {v.name for v in read_problem(path).vars}
    assert {v.name for v in read_problem(out).vars} == original


def test_cli_renames_by_default(model_file, tmp_path):
    path, _ = model_file
    out = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(out), "--seed", "1", "-q"]) == 0
    hidden_names = {v.name for v in read_problem(out).vars}
    assert not (hidden_names & {v.name for v in read_problem(path).vars})


def test_cli_refuses_quadratic_model(tmp_path, capsys):
    import gurobipy as gp

    model = gp.Model("q")
    model.setParam("OutputFlag", 0)
    x = model.addVar(name="x", ub=4)
    model.addConstr(x >= 1)
    model.setObjective(x * x)
    model.update()
    path = tmp_path / "q.mps"
    model.write(str(path))
    model.dispose()

    out = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(out)]) == 2
    assert "quadratic" in capsys.readouterr().err
    assert not out.exists()


def _zero_objective_mps(tmp_path):
    """A pure feasibility model, in the spirit of pret60."""
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("feas")
    model.setParam("OutputFlag", 0)
    xs = [
        model.addVar(vtype=GRB.INTEGER, lb=0, ub=1, obj=0.0, name=f"x{i}")
        for i in range(8)
    ]
    model.addConstr(gp.quicksum(xs[:4]) == 2, name="e1")
    model.addConstr(gp.quicksum(xs[4:]) == 3, name="e2")
    model.addConstr(xs[0] + xs[7] <= 1, name="i1")
    model.ModelSense = GRB.MINIMIZE
    model.update()
    path = tmp_path / "feas.mps"
    model.write(str(path))
    model.dispose()
    return path


def test_cli_preserves_a_zero_objective_by_default(tmp_path, capsys):
    path = _zero_objective_mps(tmp_path)
    out = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(out), "--seed", "2", "--rounds", "40"]) == 0

    hidden = read_problem(out)
    assert all(v.obj == 0.0 for v in hidden.vars), (
        "a feasibility problem must not gain objective coefficients"
    )
    captured = capsys.readouterr()
    assert "0 objective nonzeros" in captured.out
    assert "warning" not in captured.err


def test_cli_warns_when_a_feasibility_problem_gains_an_objective(tmp_path, capsys):
    path = _zero_objective_mps(tmp_path)
    out = tmp_path / "hidden.mps"
    assert (
        main(
            [
                str(path),
                "-o",
                str(out),
                "--seed",
                "2",
                "--rounds",
                "40",
                "--no-preserve-objective",
                "--enable",
                "mix-obj",
            ]
        )
        == 0
    )
    hidden = read_problem(out)
    assert any(v.obj != 0.0 for v in hidden.vars)
    assert "pure feasibility problem" in capsys.readouterr().err


def test_cli_reports_objective_nonzeros(model_file, tmp_path, capsys):
    path, _ = model_file
    out = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(out), "--seed", "1", "--rounds", "10"]) == 0
    stdout = capsys.readouterr().out
    assert "objective nonzeros" in stdout


def test_cli_sloppy_scaling(model_file, tmp_path):
    path, expected = model_file
    out = tmp_path / "hidden.mps"
    assert (
        main([str(path), "-o", str(out), "--seed", "3", "--sloppy-scaling", "-q"]) == 0
    )
    status, obj = solve_problem(read_problem(out))
    assert status == GRB.OPTIMAL
    assert obj == pytest.approx(expected, rel=1e-6, abs=1e-6)


# --------------------------------------------------------------------------
# the key, and unhiding
# --------------------------------------------------------------------------


def snapshot(problem):
    return (
        [(v.name, v.lb, v.ub, v.obj, v.vtype) for v in problem.vars],
        [(r.name, dict(r.coeffs), r.sense, r.rhs) for r in problem.rows],
        problem.obj_offset,
        problem.obj_sense,
        problem.name,
    )


def test_cli_writes_a_key_next_to_the_output_by_default(model_file, tmp_path, capsys):
    path, _ = model_file
    out = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(out), "--seed", "5"]) == 0

    key_path = tmp_path / "hidden.mps.key"
    assert key_path.exists()
    assert key_path.read_bytes().startswith(b"MIPMORPH-KEY-1\n")
    assert "recovery key" in capsys.readouterr().out


def test_cli_no_key_writes_none(model_file, tmp_path, capsys):
    path, _ = model_file
    out = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(out), "--seed", "5", "--no-key"]) == 0
    assert not (tmp_path / "hidden.mps.key").exists()
    assert "one-way and irreversible" in capsys.readouterr().out


def test_cli_key_path_can_be_chosen(model_file, tmp_path):
    path, _ = model_file
    out = tmp_path / "hidden.mps"
    key_path = tmp_path / "elsewhere.k"
    assert main([str(path), "-o", str(out), "--seed", "5", "--key", str(key_path)]) == 0
    assert key_path.exists()
    assert not (tmp_path / "hidden.mps.key").exists()


def test_cli_quiet_is_still_silent_with_a_key(model_file, tmp_path, capsys):
    path, _ = model_file
    out = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(out), "--seed", "1", "-q"]) == 0
    assert capsys.readouterr().out == ""
    assert (tmp_path / "hidden.mps.key").exists()


def test_cli_unhide_recovers_the_original_model(model_file, tmp_path, capsys):
    path, _ = model_file
    hidden = tmp_path / "hidden.mps"
    back = tmp_path / "back.mps"

    assert main([str(path), "-o", str(hidden), "--seed", "9", "--rounds", "40"]) == 0
    assert main(["unhide", str(hidden), "-o", str(back)]) == 0

    assert snapshot(read_problem(back)) == snapshot(read_problem(path))
    assert "steps to undo" in capsys.readouterr().out


def test_cli_unhide_finds_a_key_given_explicitly(model_file, tmp_path):
    path, _ = model_file
    hidden = tmp_path / "hidden.mps"
    key_path = tmp_path / "k"
    back = tmp_path / "back.mps"

    assert main([str(path), "-o", str(hidden), "--seed", "9", "--key", str(key_path)]) == 0
    assert main(["unhide", str(hidden), "-o", str(back), "--key", str(key_path)]) == 0
    assert snapshot(read_problem(back)) == snapshot(read_problem(path))


def test_cli_unhide_without_a_key_fails_clearly(model_file, tmp_path, capsys):
    path, _ = model_file
    hidden = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(hidden), "--seed", "9", "--no-key"]) == 0
    assert main(["unhide", str(hidden), "-o", str(tmp_path / "back.mps")]) == 2
    assert "no key at" in capsys.readouterr().err


def test_cli_unhide_rejects_a_key_from_another_run(model_file, tmp_path, capsys):
    path, _ = model_file
    first = tmp_path / "a.mps"
    second = tmp_path / "b.mps"
    assert main([str(path), "-o", str(first), "--seed", "1", "--rounds", "40"]) == 0
    assert main([str(path), "-o", str(second), "--seed", "2", "--rounds", "40"]) == 0

    assert (
        main(
            [
                "unhide",
                str(first),
                "-o",
                str(tmp_path / "back.mps"),
                "--key",
                str(tmp_path / "b.mps.key"),
            ]
        )
        == 2
    )
    assert "the key describes" in capsys.readouterr().err


def test_cli_key_password_round_trip(model_file, tmp_path, capsys):
    path, _ = model_file
    hidden = tmp_path / "hidden.mps"
    back = tmp_path / "back.mps"

    assert (
        main([str(path), "-o", str(hidden), "--seed", "9", "--key-password", "pw"]) == 0
    )
    assert "obfuscated" not in capsys.readouterr().out

    # Without the password the key cannot be read at all.
    assert main(["unhide", str(hidden), "-o", str(back)]) == 2
    assert "password is required" in capsys.readouterr().err

    assert (
        main(["unhide", str(hidden), "-o", str(back), "--key-password", "pw"]) == 0
    )
    assert snapshot(read_problem(back)) == snapshot(read_problem(path))


def test_cli_key_password_from_the_environment(model_file, tmp_path, monkeypatch):
    path, _ = model_file
    hidden = tmp_path / "hidden.mps"
    back = tmp_path / "back.mps"

    monkeypatch.setenv("MIPMORPH_KEY_PASSWORD", "from-env")
    assert main([str(path), "-o", str(hidden), "--seed", "9", "-q"]) == 0
    assert main(["unhide", str(hidden), "-o", str(back), "-q"]) == 0
    assert snapshot(read_problem(back)) == snapshot(read_problem(path))

    monkeypatch.delenv("MIPMORPH_KEY_PASSWORD")
    assert main(["unhide", str(hidden), "-o", str(back), "-q"]) == 2


def _write_solution_of(model_path, sol_path):
    """Solve a model file and write its solution, the way a user would."""
    from mip_morph.io import model_from_problem

    model = model_from_problem(read_problem(model_path))
    try:
        model.setParam("OutputFlag", 0)
        model.optimize()
        assert model.Status == GRB.OPTIMAL
        model.write(str(sol_path))
        return model.ObjVal
    finally:
        model.dispose()


def test_cli_unhide_sol_maps_a_solution_back(model_file, tmp_path, capsys):
    path, expected = model_file
    hidden = tmp_path / "hidden.mps"
    hidden_sol = tmp_path / "hidden.sol"
    original_sol = tmp_path / "original.sol"

    assert main([str(path), "-o", str(hidden), "--seed", "4", "--rounds", "40"]) == 0
    obj = _write_solution_of(hidden, hidden_sol)
    assert obj == pytest.approx(expected, abs=1e-6)

    assert (
        main(
            [
                "unhide-sol",
                str(hidden_sol),
                "-o",
                str(original_sol),
                "--key",
                str(tmp_path / "hidden.mps.key"),
            ]
        )
        == 0
    )
    assert "original variables" in capsys.readouterr().out

    text = original_sol.read_text()
    original = read_problem(path)
    values = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, value = line.split()
        values[name] = float(value)

    assert set(values) == {v.name for v in original.vars}
    # The objective comment carries through, and is the original's value.
    assert "Objective value" in text
    recomputed = original.obj_offset + sum(
        v.obj * values[v.name] for v in original.vars
    )
    assert recomputed == pytest.approx(expected, abs=1e-6)


def test_cli_unhide_sol_rejects_a_foreign_solution(model_file, tmp_path, capsys):
    path, _ = model_file
    hidden = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(hidden), "--seed", "4", "-q"]) == 0

    bogus = tmp_path / "bogus.sol"
    bogus.write_text("# Objective value = 0\nnosuchvariable 1\n")
    assert (
        main(
            [
                "unhide-sol",
                str(bogus),
                "-o",
                str(tmp_path / "out.sol"),
                "--key",
                str(tmp_path / "hidden.mps.key"),
            ]
        )
        == 2
    )
    assert "does not appear to be a solution" in capsys.readouterr().err


def test_cli_unhide_sol_reports_a_malformed_file(model_file, tmp_path, capsys):
    path, _ = model_file
    hidden = tmp_path / "hidden.mps"
    assert main([str(path), "-o", str(hidden), "--seed", "4", "-q"]) == 0

    bad = tmp_path / "bad.sol"
    bad.write_text("this line has three fields\n")
    assert (
        main(
            [
                "unhide-sol",
                str(bad),
                "-o",
                str(tmp_path / "out.sol"),
                "--key",
                str(tmp_path / "hidden.mps.key"),
            ]
        )
        == 2
    )
    assert "expected 'name value'" in capsys.readouterr().err
