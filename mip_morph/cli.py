"""Command line interface.

Four invocations, dispatched on the first argument::

    python -m mipmorph model.mps -o hidden.mps          # hide, and write a key
    python -m mipmorph unhide hidden.mps -o back.mps    # key + model -> original
    python -m mipmorph unhide-sol hidden.sol -o back.sol
    python -m mipmorph check model.mps                  # does a run keep the objective?

``--sol`` on the hide form also solves the original and writes that solution
carried onto the hidden model, which gives the hidden model a known point.

Hiding is the bare form rather than a ``hide`` subcommand so that every existing
command line keeps working unchanged.
"""

import argparse
import os
import random
import sys

from gurobipy import GRB, GurobiError

from .check import DEFAULT_TOL, check_morph, max_violation, objective_value
from .io import UnsupportedModelError, read_problem, solve_problem, write_problem
from .ir import coef_range
from .key import KeyFormatError, read_key, write_key
from .morph import (
    DEFAULT_DISPATCH_FANOUT,
    DEFAULT_MAX_COEF_RATIO,
    DEFAULT_ROUNDS,
    morph_model,
)
from .transforms import DEFAULT_NAMES, OPT_IN_NAMES
from .unmorph import KeyMismatchError, unmorph_problem, unmorph_solution

WARN_COEF_RATIO = 1e6

# ``check`` runs far longer than a hiding run by default: it is looking for the
# rare step that breaks the objective, which a 40-step run can easily miss.
DEFAULT_CHECK_ROUNDS = 1000
DEFAULT_OPT_TOL = 1e-4
MAX_REPORTED_FAILURES = 10
DEFAULT_FAILURE_DIR = "mipmorph-check-failures"

PASSWORD_ENV = "MIPMORPH_KEY_PASSWORD"

KEY_HELP = (
    "recovery key; with the hidden model it reproduces the original, and it "
    "translates a solution of the hidden model back to the original variables"
)


def _name_list(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def _add_password_option(parser):
    parser.add_argument(
        "--key-password",
        default=None,
        metavar="PW",
        help=(
            f"encrypt/decrypt the key with this password; also read from "
            f"${PASSWORD_ENV}, which is preferable since a command line "
            f"argument is visible to other processes. Without a password the "
            f"key is only obfuscated, not encrypted"
        ),
    )


def _password(args):
    if args.key_password:
        return args.key_password
    return os.environ.get(PASSWORD_ENV) or None


def _add_transform_options(parser, rounds=DEFAULT_ROUNDS):
    """The options that steer which transformations a run applies, and how."""
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--rounds",
        type=int,
        default=rounds,
        help=f"number of transformations to apply (default {rounds})",
    )
    parser.add_argument(
        "--enable", type=_name_list, default=None, help="comma-separated allowlist"
    )
    parser.add_argument(
        "--disable", type=_name_list, default=None, help="comma-separated denylist"
    )
    parser.add_argument(
        "--sloppy-scaling",
        action="store_true",
        help="use arbitrary factors in [0.1, 10] instead of exact powers of two",
    )
    parser.add_argument(
        "--max-coef-ratio",
        type=float,
        default=DEFAULT_MAX_COEF_RATIO,
        help=f"reject transforms exceeding this coefficient range (default {DEFAULT_MAX_COEF_RATIO:g})",
    )
    parser.add_argument(
        "--mix-max-nnz",
        type=int,
        default=25,
        help="skip equality donors denser than this when mixing (default 25)",
    )
    parser.add_argument(
        "--dispatch-fanout",
        type=int,
        default=DEFAULT_DISPATCH_FANOUT,
        help=(
            "how many rows one equality is scattered across by 'dispatch' "
            f"(default {DEFAULT_DISPATCH_FANOUT}); higher hides the donor row "
            "better but grows coefficients faster"
        ),
    )
    objective = parser.add_mutually_exclusive_group()
    objective.add_argument(
        "--preserve-objective",
        dest="preserve_objective",
        action="store_true",
        default=None,
        help=(
            "never alter the cost vector; on by default when the objective is "
            "identically zero, so feasibility problems stay feasibility problems"
        ),
    )
    objective.add_argument(
        "--no-preserve-objective",
        dest="preserve_objective",
        action="store_false",
        help="allow objective mixing even on a zero-objective model",
    )


def _morph_options(args):
    """``morph_model`` keyword arguments for the options above."""
    return {
        "rounds": args.rounds,
        "seed": args.seed,
        "enable": args.enable,
        "disable": args.disable,
        "sloppy_scaling": args.sloppy_scaling,
        "max_coef_ratio": args.max_coef_ratio,
        "mix_max_nnz": args.mix_max_nnz,
        "dispatch_fanout": args.dispatch_fanout,
        "preserve_objective": args.preserve_objective,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mipmorph",
        description=(
            "Read a MIP, apply randomized equivalence-preserving "
            "transformations, and write it out under a new name."
        ),
        epilog=(
            "transformations: "
            + ", ".join(DEFAULT_NAMES)
            + "; opt-in via --enable: "
            + ", ".join(OPT_IN_NAMES)
            + ". Subcommands: 'unhide' recovers the original model from a hidden "
            "one and its key, 'unhide-sol' recovers a solution, and 'check' "
            "verifies that a run preserves the objective."
        ),
    )
    parser.add_argument("input", help="input model file (.mps, .mps.gz, .lp, ...)")
    parser.add_argument("-o", "--output", required=True, help="output model file")
    _add_transform_options(parser)
    parser.add_argument(
        "--no-rename", action="store_true", help="keep the original names"
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", help="keep the original row/column order"
    )
    key_group = parser.add_mutually_exclusive_group()
    key_group.add_argument(
        "--key",
        default=None,
        metavar="PATH",
        help=f"where to write the {KEY_HELP} (default: <output>.key)",
    )
    key_group.add_argument(
        "--no-key",
        action="store_true",
        help="do not write a key, making the hiding one-way and irreversible",
    )
    _add_password_option(parser)
    parser.add_argument(
        "--sol",
        default=None,
        metavar="PATH",
        help=(
            "solve the original model first, and write its solution, transformed "
            "onto the hidden model, to PATH (.sol)"
        ),
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        metavar="SECONDS",
        help="time limit for the --sol solve; the best solution found is used",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    return parser


def build_unhide_parser():
    parser = argparse.ArgumentParser(
        prog="mipmorph unhide",
        description="Recover the original model from a hidden one and its key.",
    )
    parser.add_argument("input", help="hidden model file")
    parser.add_argument(
        "-o", "--output", required=True, help="where to write the recovered model"
    )
    parser.add_argument(
        "--key",
        default=None,
        metavar="PATH",
        help="the key written when the model was hidden (default: <input>.key)",
    )
    _add_password_option(parser)
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    return parser


def build_unhide_sol_parser():
    parser = argparse.ArgumentParser(
        prog="mipmorph unhide-sol",
        description=(
            "Translate a solution of a hidden model back to the original "
            "variable names."
        ),
    )
    parser.add_argument("input", help="solution file for the hidden model (.sol)")
    parser.add_argument(
        "-o", "--output", required=True, help="where to write the translated solution"
    )
    parser.add_argument(
        "--key",
        default=None,
        metavar="PATH",
        help="the key written when the model was hidden",
    )
    _add_password_option(parser)
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    return parser


def build_check_parser():
    parser = argparse.ArgumentParser(
        prog="mipmorph check",
        description=(
            "Apply transformations to a model while carrying a known point "
            "through them, and check after every step that the point keeps its "
            "objective value and stays feasible. Then solve the hidden model and "
            "compare its optimum with the original's."
        ),
    )
    parser.add_argument("input", help="input model file (.mps, .mps.gz, .lp, ...)")
    _add_transform_options(parser, rounds=DEFAULT_CHECK_ROUNDS)
    parser.add_argument(
        "--sol",
        default=None,
        metavar="PATH",
        help=(
            "take the point from this solution of the original model (.sol) "
            "instead of solving it; it is then also the reference optimum"
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=(
            "largest relative objective change or constraint violation a step "
            f"may introduce (default {DEFAULT_TOL:g})"
        ),
    )
    parser.add_argument(
        "--opt-tol",
        type=float,
        default=DEFAULT_OPT_TOL,
        help=(
            "largest relative difference between the two optima "
            f"(default {DEFAULT_OPT_TOL:g}, Gurobi's default MIPGap)"
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "repeat the check N times with different seeds: --seed S gives "
            "S, S+1, ..., otherwise each run draws a random seed (default 1)"
        ),
    )
    parser.add_argument(
        "-x",
        "--stop-on-failure",
        action="store_true",
        help="stop after the first run that fails",
    )
    parser.add_argument(
        "--failure-dir",
        default=DEFAULT_FAILURE_DIR,
        metavar="DIR",
        help=(
            "where a failing run saves what reproduces it: the hidden model, its "
            "key, the expected solution of the hidden model, and a report "
            f"(default {DEFAULT_FAILURE_DIR})"
        ),
    )
    parser.add_argument(
        "--no-solve",
        action="store_true",
        help="only run the per-step check; do not solve the hidden model",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        metavar="SECONDS",
        help="time limit for each solve",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print every step as it is checked"
    )
    return parser


# --------------------------------------------------------------------------
# solution files
# --------------------------------------------------------------------------


def read_solution(path):
    """Parse Gurobi's ``.sol`` format into ``(comments, {name: value})``."""
    comments = []
    values = {}
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                comments.append(line)
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{number}: expected 'name value', found {line!r}"
                )
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                raise ValueError(
                    f"{path}:{number}: {parts[1]!r} is not a number"
                ) from None
    return comments, values


def write_solution(path, comments, values):
    with open(path, "w") as handle:
        for comment in comments:
            handle.write(comment + "\n")
        for name, value in values.items():
            handle.write(f"{name} {value!r}\n")


# --------------------------------------------------------------------------
# hide
# --------------------------------------------------------------------------


def hide_main(argv):
    args = build_parser().parse_args(argv)

    try:
        problem = read_problem(args.input)
    except UnsupportedModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Solved before hiding so that a model with no solution fails before
    # anything is written.
    solution = None
    if args.sol:
        try:
            status, objective, solution = solve_problem(
                problem, time_limit=args.time_limit
            )
        except GurobiError as exc:
            print(
                f"error: could not solve {args.input} for --sol: {exc}", file=sys.stderr
            )
            return 2
        if solution is None:
            print(
                f"error: found no solution of {args.input} to write to {args.sol} "
                f"(Gurobi status {status})",
                file=sys.stderr,
            )
            return 2
        if status != GRB.OPTIMAL:
            print(
                f"warning: the solve stopped before optimality (Gurobi status "
                f"{status}); writing the best solution found",
                file=sys.stderr,
            )

    try:
        hidden, log = morph_model(
            problem,
            rename=not args.no_rename,
            shuffle=not args.no_shuffle,
            solution=solution,
            **_morph_options(args),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_problem(hidden, args.output)

    if args.sol:
        # Every transformation preserves the objective value pointwise, so the
        # original's value is also the hidden point's.
        write_solution(args.sol, [f"# Objective value = {objective!r}"], log.solution)

    # After the model, so that a failed model write does not leave an orphan key.
    key_path = None
    if not args.no_key:
        key_path = args.key or args.output + ".key"
        write_key(log.key, key_path, password=_password(args))

    obj_nz_before = sum(1 for v in problem.vars if v.obj != 0.0)
    obj_nz_after = sum(1 for v in hidden.vars if v.obj != 0.0)

    if not args.quiet:
        print(
            f"read {args.input}: {len(problem.vars)} vars, {len(problem.rows)} rows, "
            f"{obj_nz_before} objective nonzeros"
        )
        for i, entry in enumerate(log, start=1):
            print(f"  {i:3d}. {entry}")
        print(
            f"wrote {args.output}: {len(hidden.vars)} vars, {len(hidden.rows)} rows, "
            f"{obj_nz_after} objective nonzeros, {len(log)} transformations applied"
        )
        if key_path is None:
            print("wrote no key: hiding this model is one-way and irreversible")
        else:
            print(
                f"wrote {key_path}: recovery key, {len(log.key.steps)} steps"
                + ("" if _password(args) else " (obfuscated, not encrypted)")
            )
        if args.sol:
            print(
                f"wrote {args.sol}: solution of the hidden model, objective "
                f"{objective:.10g}"
            )

    if obj_nz_before == 0 and obj_nz_after > 0:
        print(
            "warning: the input was a pure feasibility problem but the output has "
            f"{obj_nz_after} objective nonzeros; this can change INFEASIBLE into "
            "INF_OR_UNBD and weaken presolve. Use --preserve-objective.",
            file=sys.stderr,
        )

    ratio = coef_range(hidden)
    if ratio > WARN_COEF_RATIO:
        print(
            f"warning: coefficient range is {ratio:.3g}; the hidden model may be "
            "numerically harder than the original",
            file=sys.stderr,
        )

    return 0


# --------------------------------------------------------------------------
# unhide
# --------------------------------------------------------------------------


def _load_key(path, password):
    try:
        return read_key(path, password=password)
    except FileNotFoundError:
        print(
            f"error: no key at {path}; pass --key, or re-hide the model without "
            f"--no-key",
            file=sys.stderr,
        )
    except KeyFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
    return None


def unhide_main(argv):
    args = build_unhide_parser().parse_args(argv)
    key_path = args.key or args.input + ".key"

    key = _load_key(key_path, _password(args))
    if key is None:
        return 2

    try:
        hidden = read_problem(args.input)
    except UnsupportedModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        original = unmorph_problem(hidden, key)
    except KeyMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_problem(original, args.output)

    if not args.quiet:
        print(
            f"read {args.input}: {len(hidden.vars)} vars, {len(hidden.rows)} rows"
            f"\nread {key_path}: {len(key.steps)} steps to undo"
            f"\nwrote {args.output}: {len(original.vars)} vars, "
            f"{len(original.rows)} rows"
        )
    return 0


def unhide_sol_main(argv):
    args = build_unhide_sol_parser().parse_args(argv)
    key_path = args.key or args.input + ".key"

    key = _load_key(key_path, _password(args))
    if key is None:
        return 2

    try:
        comments, values = read_solution(args.input)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        original = unmorph_solution(values, key)
    except KeyMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # The objective comment carries through untouched, and is correct to keep:
    # every transformation preserves the objective value pointwise on the
    # feasible set, so the hidden model's value *is* the original's.
    write_solution(args.output, comments, original)

    if not args.quiet:
        print(
            f"read {args.input}: {len(values)} values"
            f"\nwrote {args.output}: {len(original)} values on the original variables"
        )
    return 0


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def _reference_point(args, problem):
    """``(point, label, optimal)`` for the check, or ``None`` after an error."""
    if args.sol:
        try:
            _, values = read_solution(args.sol)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return None
        missing = [var.name for var in problem.vars if var.name not in values]
        if missing:
            print(
                f"error: {args.sol} has no value for {len(missing)} of the model's "
                f"columns, starting with {missing[0]!r}",
                file=sys.stderr,
            )
            return None
        point = [values[var.name] for var in problem.vars]
        return point, f"point from {args.sol}", True

    try:
        status, _, solution = solve_problem(problem, time_limit=args.time_limit)
    except GurobiError as exc:
        print(f"error: could not solve {args.input}: {exc}", file=sys.stderr)
        return None
    if solution is None:
        print(
            f"error: found no solution of {args.input} to check with "
            f"(Gurobi status {status})",
            file=sys.stderr,
        )
        return None
    if status == GRB.OPTIMAL:
        return solution, f"optimum of {args.input}", True
    print(
        f"warning: the solve stopped before optimality (Gurobi status {status}); "
        "checking with the best solution found, and not comparing optima",
        file=sys.stderr,
    )
    return solution, f"best solution of {args.input}", False


def _report_failures(result):
    culprits = result.culprits
    print(
        f"  {len(result.failures)} step(s) out of tolerance, {len(culprits)} of "
        "them made things worse; the first ones:"
    )
    for s in culprits[:MAX_REPORTED_FAILURES]:
        print(
            f"  {s.index:4d}. objective {s.objective:.10g} "
            f"(change {s.error:.2e}), violation {s.violation:.2e}  {s.step}"
        )


def _compare_optima(args, result, optimal):
    """Solve the hidden model: ``(message, failed)``, or ``(None, False)``."""
    if args.no_solve:
        return None, False
    try:
        status, hidden_opt, _ = solve_problem(result.hidden, time_limit=args.time_limit)
    except GurobiError as exc:
        print(f"warning: could not solve the hidden model: {exc}", file=sys.stderr)
        return None, False
    if status != GRB.OPTIMAL:
        print(
            f"warning: the hidden solve stopped before optimality (Gurobi "
            f"status {status}); not comparing optima",
            file=sys.stderr,
        )
        return None, False
    if not optimal:
        return f"hidden optimum {hidden_opt:.10g}", False
    diff = abs(hidden_opt - result.reference) / max(1.0, abs(result.reference))
    message = f"hidden optimum {hidden_opt:.10g}, relative difference {diff:.2e}"
    return message, diff > args.opt_tol


def _check_run(args, problem, point, optimal, seed, label):
    """One checked run; prints its report, returns ``(failed, result, message)``."""
    result = check_morph(
        problem, point, tol=args.tol, **{**_morph_options(args), "seed": seed}
    )
    single = args.runs == 1

    if args.verbose:
        if not single:
            print(f"{label}:")
        for s in result.steps:
            print(
                f"  {s.index:4d}. objective {s.objective:.10g}  "
                f"violation {s.violation:.2e}  {s.step}"
            )

    if single:
        counts = {}
        for step in result.log:
            counts[step.op] = counts.get(step.op, 0) + 1
        print(
            f"applied {len(result.log)} transformations: "
            + ", ".join(f"{op} {n}" for op, n in sorted(counts.items()))
        )
        print(
            f"after every step: worst objective change {result.worst_error:.2e}, "
            f"worst violation {result.worst_violation:.2e} (tol {args.tol:g})"
        )
        if result.failures:
            print("FAIL: the point was broken")
            _report_failures(result)
        hidden = result.hidden
        print(
            f"hidden model: {len(hidden.vars)} vars, {len(hidden.rows)} rows, "
            f"coefficient range {coef_range(hidden):.3g}"
        )

    message, opt_failed = _compare_optima(args, result, optimal)
    failed = opt_failed or not result.ok

    if single:
        if message:
            print(f"{message} (tol {args.opt_tol:g})")
        if opt_failed:
            print("FAIL: the hidden optimum differs from the original's")
    else:
        print(
            f"{label}: {'FAIL' if failed else 'OK'}  "
            f"{len(result.log)} steps, worst objective change "
            f"{result.worst_error:.2e}, worst violation {result.worst_violation:.2e}"
            + (f", {message}" if message else "")
        )
        if result.failures:
            _report_failures(result)
    return failed, result, message


def _replay_command(argv, seed, original_sol):
    """``argv`` for replaying one run: same options, one seed, the saved point."""
    drop_with_value = {"--runs", "--seed", "--sol"}
    drop_flag = {"-x", "--stop-on-failure"}
    kept = []
    it = iter(argv)
    for arg in it:
        name = arg.split("=", 1)[0]
        if name in drop_with_value:
            if "=" not in arg:
                next(it, None)
        elif arg not in drop_flag:
            kept.append(arg)
    return ["mipmorph", "check", *kept, "--seed", str(seed), "--sol", original_sol]


def _save_original(args, problem, point, label):
    """Write the reference point once, so every saved failure can be replayed."""
    os.makedirs(args.failure_dir, exist_ok=True)
    path = os.path.join(args.failure_dir, "original.sol")
    if not os.path.exists(path):
        write_solution(
            path,
            [
                f"# Objective value = {objective_value(problem, point)!r}",
                f"# reference point of {args.input} for mipmorph check: {label}",
            ],
            {var.name: value for var, value in zip(problem.vars, point)},
        )
    return path


def _save_failure(args, argv, seed, result, message, original_sol):
    """Write what reproduces a failed run; returns the paths written."""
    stem = os.path.join(args.failure_dir, f"seed{seed}")
    model_path = stem + ".mps"
    key_path = model_path + ".key"
    sol_path = stem + ".sol"
    report_path = stem + ".txt"

    write_problem(result.hidden, model_path)
    write_key(result.log.key, key_path)
    # The point carried onto the hidden model: what a correct run guarantees is
    # a feasible solution of it with the original's objective value.
    write_solution(
        sol_path,
        [
            f"# Objective value = {result.reference!r}",
            (
                f"# expected solution of {os.path.basename(model_path)}, "
                f"mipmorph check seed {seed}"
            ),
        ],
        result.log.solution,
    )

    failures = result.failures
    bad = {s.index for s in result.culprits}
    lines = [
        f"input: {args.input}",
        f"seed: {seed}",
        f"replay: {' '.join(_replay_command(argv, seed, original_sol))}",
        f"expected objective: {result.reference!r}",
        (
            f"after every step: worst objective change {result.worst_error:.2e}, "
            f"worst violation {result.worst_violation:.2e} (tol {args.tol:g})"
        ),
    ]
    if message:
        lines.append(f"{message} (tol {args.opt_tol:g})")
    if failures:
        lines.append(f"first failing step: {failures[0].index}")
        lines.append(f"steps that made it worse: {', '.join(map(str, sorted(bad)))}")
    else:
        lines.append("no step broke the point")
    lines += ["", "steps (* = made it worse):"]
    lines += [
        f"{'*' if s.index in bad else ' '} {s.index:4d}. "
        f"objective {s.objective!r}  violation {s.violation:.2e}  {s.step}"
        for s in result.steps
    ]
    with open(report_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return model_path, key_path, sol_path, report_path


def check_main(argv):
    argv = list(argv)
    args = build_check_parser().parse_args(argv)
    if args.runs < 1:
        print("error: --runs must be at least 1", file=sys.stderr)
        return 2

    try:
        problem = read_problem(args.input)
    except UnsupportedModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"read {args.input}: {len(problem.vars)} vars, {len(problem.rows)} rows")

    reference = _reference_point(args, problem)
    if reference is None:
        return 2
    point, label, optimal = reference

    start_violation = max_violation(problem, point)
    print(
        f"{label}: objective {objective_value(problem, point):.10g}, "
        f"violation {start_violation:.2e}"
    )
    if start_violation > args.tol:
        print(
            "warning: the point is not feasible for the original model, so the "
            "per-step feasibility check is not meaningful",
            file=sys.stderr,
        )

    # Every run's seed is printed, so a failure can be replayed with --seed.
    if args.seed is not None:
        seeds = [args.seed + i for i in range(args.runs)]
    else:
        seeds = [random.randrange(2**31) for _ in range(args.runs)]
    width = len(str(args.runs))

    failed_seeds = []
    done = 0
    for i, seed in enumerate(seeds, start=1):
        if args.runs == 1:
            print(f"seed {seed}")
        run_label = f"run {i:{width}d}/{args.runs} seed {seed}"
        try:
            failed, result, message = _check_run(
                args, problem, point, optimal, seed, run_label
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        done += 1
        if failed:
            failed_seeds.append(seed)
            original_sol = _save_original(args, problem, point, label)
            paths = _save_failure(args, argv, seed, result, message, original_sol)
            print(
                f"  saved the hidden model, its key, the expected solution and a "
                f"report: {', '.join(paths)}"
            )
            print(f"  replay: {' '.join(_replay_command(argv, seed, original_sol))}")
            if args.stop_on_failure:
                break

    if args.runs > 1:
        stopped = f" (stopped after {done} of {args.runs})" if done < args.runs else ""
        print(
            f"{done} run(s){stopped}, {len(failed_seeds)} failed"
            + (f": seeds {', '.join(map(str, failed_seeds))}" if failed_seeds else "")
        )
    print("FAIL" if failed_seeds else "OK")
    return 1 if failed_seeds else 0


SUBCOMMANDS = {
    "unhide": unhide_main,
    "unhide-sol": unhide_sol_main,
    "check": check_main,
}


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if argv and argv[0] in SUBCOMMANDS:
        return SUBCOMMANDS[argv[0]](argv[1:])
    return hide_main(argv)


if __name__ == "__main__":
    sys.exit(main())
