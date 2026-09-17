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
