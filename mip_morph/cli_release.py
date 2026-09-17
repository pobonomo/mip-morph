"""Hardened CLI for the obfuscated release build.

Same read -> morph -> write behaviour as :mod:`mip_morph.cli`, but with every
option and output that would make the transformation *transparent* removed:

* no per-transform log is printed (the release binary is silent on success,
  aside from feasibility / numerical-hardness warnings);
* no ``--enable`` / ``--disable`` options and no listing of the transform
  names, so the set of transformations is never revealed.

The remaining tuning knobs (``--seed``, ``--rounds``, ``--dispatch-fanout``,
...) are kept. This module is compiled to a native extension by
``scripts/build_obfuscated.sh`` and shipped in place of :mod:`mip_morph.cli`.
"""

import argparse
import sys

from .io import UnsupportedModelError, read_problem, write_problem
from .ir import coef_range
from .morph import (
    DEFAULT_DISPATCH_FANOUT,
    DEFAULT_MAX_COEF_RATIO,
    DEFAULT_ROUNDS,
    morph_model,
)

WARN_COEF_RATIO = 1e6


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mipmorph",
        description=(
            "Read a MIP, apply randomized equivalence-preserving "
            "transformations, and write it out under a new name."
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
        help=f"donor scatter fan-out (default {DEFAULT_DISPATCH_FANOUT})",
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
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        problem = read_problem(args.input)
    except UnsupportedModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        # enable/disable are intentionally not exposed here: the release build
        # never reveals or lets the caller select individual transformations.
        hidden, _log = morph_model(
            problem,
            rounds=args.rounds,
            seed=args.seed,
            enable=None,
            disable=None,
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

    # No per-transform log: the applied transformations are never printed.

    obj_nz_before = sum(1 for v in problem.vars if v.obj != 0.0)
    obj_nz_after = sum(1 for v in hidden.vars if v.obj != 0.0)
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


if __name__ == "__main__":
    sys.exit(main())
