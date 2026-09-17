"""Command line interface.

Three invocations, dispatched on the first argument::

    python -m mipmorph model.mps -o hidden.mps          # hide, and write a key
    python -m mipmorph unhide hidden.mps -o back.mps    # key + model -> original
    python -m mipmorph unhide-sol hidden.sol -o back.sol

Hiding is the bare form rather than a ``hide`` subcommand so that every existing
command line keeps working unchanged.
"""

import argparse
import os
import sys

from .io import UnsupportedModelError, read_problem, write_problem
from .ir import coef_range
from .key import KeyFormatError, read_key, write_key
from .morph import (
    DEFAULT_DISPATCH_FANOUT,
    DEFAULT_MAX_COEF_RATIO,
    DEFAULT_ROUNDS,
    morph_model,
)
from .transforms import ALL_NAMES
from .unmorph import KeyMismatchError, unmorph_problem, unmorph_solution

WARN_COEF_RATIO = 1e6

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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mipmorph",
        description=(
            "Read a MIP, apply randomized equivalence-preserving "
            "transformations, and write it out under a new name."
        ),
        epilog=(
            "transformations: "
            + ", ".join(ALL_NAMES)
            + ". Subcommands: 'unhide' recovers the original model from a hidden "
            "one and its key, 'unhide-sol' recovers a solution."
        ),
    )
    parser.add_argument("input", help="input model file (.mps, .mps.gz, .lp, ...)")
    parser.add_argument("-o", "--output", required=True, help="output model file")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"number of transformations to apply (default {DEFAULT_ROUNDS})",
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

    try:
        hidden, log = morph_model(
            problem,
            rounds=args.rounds,
            seed=args.seed,
            enable=args.enable,
            disable=args.disable,
            sloppy_scaling=args.sloppy_scaling,
            max_coef_ratio=args.max_coef_ratio,
            mix_max_nnz=args.mix_max_nnz,
            dispatch_fanout=args.dispatch_fanout,
            rename=not args.no_rename,
            shuffle=not args.no_shuffle,
            preserve_objective=args.preserve_objective,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_problem(hidden, args.output)

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


SUBCOMMANDS = {"unhide": unhide_main, "unhide-sol": unhide_sol_main}


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if argv and argv[0] in SUBCOMMANDS:
        return SUBCOMMANDS[argv[0]](argv[1:])
    return hide_main(argv)


if __name__ == "__main__":
    sys.exit(main())
