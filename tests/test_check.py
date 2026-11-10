"""The per-step objective check, as a library and as ``mipmorph check``."""

import pytest
from conftest import feasible_mip, feasible_random_mips

from mip_morph import morph
from mip_morph.check import check_morph, objective_value
from mip_morph.cli import _replay_command, main, read_solution, write_solution
from mip_morph.io import read_problem, solve_problem, write_problem
from mip_morph.morph import morph_model

FEASIBLE = feasible_random_mips(6)


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    # A failing run saves files under the working directory by default.
    monkeypatch.chdir(tmp_path)


def _point(problem):
    _, _, values = solve_problem(problem)
    return values


@pytest.mark.parametrize("rounds", [40, 300])
def test_check_passes_on_full_runs(rounds):
    for seed, problem, objective in FEASIBLE:
        result = check_morph(problem, _point(problem), rounds=rounds, seed=seed)
        assert result.ok, [str(s.step) for s in result.failures]
        assert result.reference == pytest.approx(objective, abs=1e-6)
        assert len(result.steps) == len(result.log)
        assert [s.step for s in result.steps] == list(result.log)


def test_check_does_not_change_the_hidden_model():
    seed, problem, _ = FEASIBLE[0]
    plain, _ = morph_model(problem, rounds=60, seed=seed)
    result = check_morph(problem, _point(problem), rounds=60, seed=seed)
    assert result.hidden == plain


def test_check_names_a_step_that_breaks_the_objective(monkeypatch):
    # A forward map that forgets to move the point is exactly the kind of bug
    # the check is for: the model is shifted but the point is not.
    monkeypatch.setitem(morph.FORWARD_MAPS, "shift", lambda x, step, problem: None)
    for seed, problem, _ in FEASIBLE:
        result = check_morph(
            problem, _point(problem), rounds=30, seed=seed, enable=["shift"]
        )
        if result.failures:
            assert {s.step.op for s in result.failures} == {"shift"}
            return
    pytest.fail("no shift ever moved the objective, so the test proved nothing")


def test_objective_value_includes_the_offset(tiny_problem):
    # a=1, b=0, z=3 on: min 2a + 3b + z + 10
    assert objective_value(tiny_problem, [1.0, 0.0, 3.0]) == 15.0


@pytest.fixture
def model_file(tmp_path):
    problem, objective = feasible_mip()
    path = tmp_path / "model.mps"
    write_problem(problem, path)
    return path, problem, objective


def test_cli_check_passes(model_file, capsys):
    path, _, _ = model_file
    assert main(["check", str(path), "--rounds", "200", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "applied 200 transformations" in out
    assert "hidden optimum" in out
    assert out.rstrip().endswith("OK")


def test_cli_check_takes_the_point_from_a_solution_file(model_file, tmp_path, capsys):
    path, problem, objective = model_file
    sol = tmp_path / "model.sol"
    values = dict(zip((v.name for v in problem.vars), _point(problem)))
    write_solution(sol, [], values)

    argv = ["check", str(path), "--sol", str(sol), "--rounds", "50", "--no-solve"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert f"objective {objective:.10g}" in out
    assert "hidden optimum" not in out


def test_cli_check_fails_on_a_broken_transform(model_file, monkeypatch, capsys):
    path, _, _ = model_file
    monkeypatch.setitem(morph.FORWARD_MAPS, "shift", lambda x, step, problem: None)
    argv = ["check", str(path), "--enable", "shift", "--rounds", "20", "--no-solve"]
    assert main(argv) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "shift" in out


def test_cli_check_rejects_an_incomplete_solution_file(model_file, tmp_path, capsys):
    path, _, _ = model_file
    sol = tmp_path / "partial.sol"
    write_solution(sol, [], {"nope": 1.0})
    assert main(["check", str(path), "--sol", str(sol)]) == 2
    assert "has no value for" in capsys.readouterr().err


def test_cli_check_runs_several_seeds(model_file, capsys):
    path, _, _ = model_file
    argv = ["check", str(path), "--runs", "4", "--seed", "20", "--rounds", "30"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    for i, seed in enumerate(range(20, 24), start=1):
        assert f"run {i}/4 seed {seed}: OK" in out
    assert "4 run(s), 0 failed" in out
    assert out.rstrip().endswith("OK")


def test_cli_check_runs_report_the_failing_seeds(model_file, monkeypatch, capsys):
    path, _, _ = model_file
    monkeypatch.setitem(morph.FORWARD_MAPS, "shift", lambda x, step, problem: None)
    argv = ["check", str(path), "--enable", "shift", "--runs", "3", "--seed", "5"]
    assert main([*argv, "--rounds", "20", "--no-solve"]) == 1
    out = capsys.readouterr().out
    assert "3 run(s), 3 failed: seeds 5, 6, 7" in out
    assert out.rstrip().endswith("FAIL")


def test_cli_check_rejects_zero_runs(model_file, capsys):
    path, _, _ = model_file
    assert main(["check", str(path), "--runs", "0"]) == 2
    assert "--runs must be at least 1" in capsys.readouterr().err


def _break_shift(monkeypatch):
    monkeypatch.setitem(morph.FORWARD_MAPS, "shift", lambda x, step, problem: None)


def test_cli_check_saves_what_reproduces_a_failure(
    model_file, monkeypatch, tmp_path, capsys
):
    path, _, objective = model_file
    _break_shift(monkeypatch)
    out_dir = tmp_path / "failures"
    argv = ["check", str(path), "--enable", "shift", "--rounds", "20", "--seed", "5"]
    assert main([*argv, "--no-solve", "--failure-dir", str(out_dir)]) == 1
    capsys.readouterr()

    hidden = read_problem(out_dir / "seed5.mps")
    _, expected = read_solution(out_dir / "seed5.sol")
    assert set(expected) == {v.name for v in hidden.vars}
    comments, _ = read_solution(out_dir / "seed5.sol")
    assert comments[0] == f"# Objective value = {objective!r}"
    assert (out_dir / "seed5.mps.key").exists()
    assert (out_dir / "original.sol").exists()

    report = (out_dir / "seed5.txt").read_text()
    assert "first failing step:" in report
    assert "--seed 5" in report
    assert f"--sol {out_dir / 'original.sol'}" in report


def test_cli_check_saved_model_matches_a_plain_hide(model_file, monkeypatch, tmp_path):
    path, _, _ = model_file
    _break_shift(monkeypatch)
    out_dir = tmp_path / "failures"
    argv = ["check", str(path), "--enable", "shift", "--rounds", "20", "--seed", "9"]
    assert main([*argv, "--no-solve", "--failure-dir", str(out_dir)]) == 1

    plain = tmp_path / "plain.mps"
    hide = [str(path), "-o", str(plain), "--enable", "shift", "--rounds", "20"]
    assert main([*hide, "--seed", "9", "-q"]) == 0
    assert plain.read_bytes() == (out_dir / "seed9.mps").read_bytes()


def test_cli_check_stops_on_the_first_failure(
    model_file, monkeypatch, tmp_path, capsys
):
    path, _, _ = model_file
    _break_shift(monkeypatch)
    argv = ["check", str(path), "--enable", "shift", "--runs", "3", "--seed", "5", "-x"]
    failure_dir = ["--failure-dir", str(tmp_path / "f")]
    assert main([*argv, "--rounds", "20", "--no-solve", *failure_dir]) == 1
    out = capsys.readouterr().out
    assert "1 run(s) (stopped after 1 of 3), 1 failed: seeds 5" in out
    assert "seed 6" not in out


def test_cli_check_writes_nothing_when_all_runs_pass(model_file, tmp_path):
    path, _, _ = model_file
    out_dir = tmp_path / "failures"
    argv = ["check", str(path), "--rounds", "30", "--runs", "2", "--seed", "1"]
    assert main([*argv, "--failure-dir", str(out_dir)]) == 0
    assert not out_dir.exists()


def test_replay_command_keeps_options_but_pins_seed_and_point():
    argv = ["m.mps", "--runs=50", "--seed", "3", "-x", "--rounds", "200", "--sol", "a"]
    assert _replay_command(argv, 7, "dir/original.sol") == [
        "mipmorph", "check", "m.mps", "--rounds", "200",
        "--seed", "7", "--sol", "dir/original.sol",
    ]  # fmt: skip
