"""Undo a hiding run, given the key that recorded it.

Two things make a clean inverse possible, and both are properties the forward
transformations were already careful to have:

**Indices mean the same thing all run.** No transformation ever deletes a column
or a row -- ``split-free`` retires its column instead of removing it -- and every
addition is appended. So the original columns keep indices ``0..n0-1``, the
original rows keep ``0..m0-1``, and whatever a step appended is still sitting at
the tail when that step comes to be undone.

**Steps are undone in reverse.** That is what makes each individual inverse a
local edit. Walking backwards, by the time step *k* is reached the model is
exactly in its post-step-*k* state, so every index the step recorded still points
at what it pointed at then, and the rows a fold read are unmodified since. An
inverse never has to reason about what came after it.

The one thing indices cannot survive is the final shuffle, so recovery starts by
re-indexing the hidden model onto the *pre-shuffle* name order stored in the key
(:func:`align_to_key`). That single step undoes the permutation and absorbs any
reordering the model writer or reader introduced along the way, so nothing after
it has to trust the order rows and columns arrived in.
"""

import warnings

from .ir import FLIP, INF, negate
from .key import structure_fingerprint, values_fingerprint


class KeyMismatchError(Exception):
    """The key does not describe this model."""


def _check(condition, detail):
    if not condition:
        raise KeyMismatchError(detail)


def _fits(step, condition, detail):
    if not condition:
        raise KeyMismatchError(f"cannot undo step '{step.op}': {detail}")


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------


def _index_by_name(items, kind):
    index = {}
    for position, item in enumerate(items):
        if item.name in index:
            raise KeyMismatchError(
                f"the model has two {kind} called {item.name!r}, so it cannot be "
                f"matched against the key by name. This happens when a model with "
                f"duplicate names was hidden with --no-rename; hiding it with "
                f"renaming left on produces a recoverable model."
            )
        index[item.name] = position
    return index


def align_to_key(hidden, key):
    """Return ``hidden`` re-indexed onto the key's pre-shuffle name order.

    Alignment is by name, so it does not matter what order the rows and columns
    actually arrive in -- whether that order came from ``shuffle_all`` or from a
    round trip through a file. What it does require is that the names are unique
    and are exactly the ones the key recorded, which is checked here rather than
    left to fail confusingly further in.
    """
    var_index = _index_by_name(hidden.vars, "columns")
    row_index = _index_by_name(hidden.rows, "rows")

    _check(
        len(hidden.vars) == len(key.hidden_var_names),
        f"the model has {len(hidden.vars)} columns but the key describes a hidden "
        f"model with {len(key.hidden_var_names)}",
    )
    _check(
        len(hidden.rows) == len(key.hidden_row_names),
        f"the model has {len(hidden.rows)} rows but the key describes a hidden "
        f"model with {len(key.hidden_row_names)}",
    )

    try:
        col_order = [var_index[name] for name in key.hidden_var_names]
    except KeyError as exc:
        raise KeyMismatchError(
            f"the model has no column named {exc.args[0]!r}, which the key expects; "
            f"this key belongs to a different model"
        ) from exc
    try:
        row_order = [row_index[name] for name in key.hidden_row_names]
    except KeyError as exc:
        raise KeyMismatchError(
            f"the model has no row named {exc.args[0]!r}, which the key expects; "
            f"this key belongs to a different model"
        ) from exc

    # Exactly ``shuffle_all`` run backwards, on a copy so that inverting does not
    # mutate the caller's model.
    aligned = hidden.copy()
    remap = {old: new for new, old in enumerate(col_order)}
    aligned.vars = [aligned.vars[old] for old in col_order]
    for row in aligned.rows:
        row.coeffs = {remap[j]: c for j, c in row.coeffs.items()}
    aligned.rows = [aligned.rows[old] for old in row_order]
    return aligned


def verify(problem, key):
    """Check an aligned model against the key's fingerprints.

    The two fingerprints fail for genuinely different reasons, so they are worth
    keeping apart. A structural mismatch -- different sparsity, senses or types --
    means the key belongs to another model, and going on from there would produce
    a confidently wrong answer, so it raises. A values-only mismatch means the
    shape is right but some number moved, which is what a lossy round trip
    through a file looks like; the recovery is still the best available and is
    still worth having, so that warns and continues.
    """
    if structure_fingerprint(problem) != key.structure_fingerprint:
        raise KeyMismatchError(
            "the model does not match the key: same names, but a different "
            "structure (sparsity pattern, row senses or variable types). This "
            "key was almost certainly produced from a different model or a "
            "different hiding run."
        )
    if values_fingerprint(problem) != key.values_fingerprint:
        warnings.warn(
            "the model matches the key structurally but some coefficient, bound "
            "or right-hand side differs from what was hidden -- the model file "
            "appears to have lost precision in a round trip. Recovery will "
            "proceed and be correspondingly approximate.",
            stacklevel=2,
        )


# --------------------------------------------------------------------------
# removing what a step appended
# --------------------------------------------------------------------------


def _drop_appended_vars(problem, indices, step):
    """Remove columns that ``step`` appended, which must be the last ones."""
    indices = sorted(indices)
    first = len(problem.vars) - len(indices)
    _fits(
        step,
        indices == list(range(first, len(problem.vars))),
        f"expected it to have added the last {len(indices)} columns "
        f"({list(range(first, len(problem.vars)))}), but it recorded {indices}",
    )
    for row in problem.rows:
        left = [j for j in row.coeffs if j >= first]
        _fits(
            step,
            not left,
            f"row {row.name!r} still refers to column(s) {sorted(left)}, which the "
            f"step added; undoing it would leave a dangling reference",
        )
    del problem.vars[first:]


def _drop_appended_rows(problem, indices, step):
    """Remove rows that ``step`` appended, which must be the last ones."""
    indices = sorted(indices)
    first = len(problem.rows) - len(indices)
    _fits(
        step,
        indices == list(range(first, len(problem.rows))),
        f"expected it to have added the last {len(indices)} rows "
        f"({list(range(first, len(problem.rows)))}), but it recorded {indices}",
    )
    del problem.rows[first:]


def _var(problem, j, step):
    _fits(step, 0 <= j < len(problem.vars), f"column index {j} is out of range")
    return problem.vars[j]


def _row(problem, i, step):
    _fits(step, 0 <= i < len(problem.rows), f"row index {i} is out of range")
    return problem.rows[i]


def _unfold(problem, donor_index, fold, step):
    """Subtract back the multiple of ``donor`` that was added to a target row.

    Restoring the numbers is the subtraction. Restoring the *sparsity* is the
    recorded ``created`` list: those columns were not in the target before the
    fold, so they go back to being absent rather than being stored as a zero (or,
    under inexact arithmetic, as a rounding residue on a column that may be about
    to be removed). Every other column the donor mentions keeps its subtracted
    value -- including one the forward pass cancelled to zero and ``drop_zeros``
    deleted, which the subtraction brings back. Columns the donor does not mention
    are not touched at all.
    """
    donor = _row(problem, donor_index, step)
    target_index = fold["target"]
    target = _row(problem, target_index, step)
    lam = fold["lam"]
    created = set(fold["created"])
    _fits(
        step, donor_index != target_index, "a row cannot have been folded into itself"
    )
    for j, coef in donor.coeffs.items():
        if j in created:
            target.coeffs.pop(j, None)
        else:
            target.coeffs[j] = target.coeffs.get(j, 0.0) - lam * coef
    target.rhs -= lam * donor.rhs


# --------------------------------------------------------------------------
# the per-transformation inverses
# --------------------------------------------------------------------------


def _undo_complement(problem, step):
    """``x -> total - x`` is an involution, so replay it with the same total."""
    j = step.params["var"]
    total = step.params["total"]
    var = _var(problem, j, step)

    for row in problem.rows_with(j):
        coef = row.coeffs[j]
        row.rhs -= coef * total
        row.coeffs[j] = -coef

    problem.obj_offset += var.obj * total
    var.obj = negate(var.obj)


def _undo_shift(problem, step):
    """Replay the substitution with ``-delta``."""
    j = step.params["var"]
    delta = step.params["delta"]
    var = _var(problem, j, step)

    var.lb += delta
    var.ub += delta
    for row in problem.rows_with(j):
        row.rhs += row.coeffs[j] * delta
    problem.obj_offset -= var.obj * delta


def _undo_shift_binary(problem, step):
    """Un-shift, then restore the declared type the forward pass overwrote."""
    _undo_shift(problem, step)
    _var(problem, step.params["var"], step).vtype = step.params["vtype"]


def _undo_scale_var(problem, step):
    """Divide where the forward pass multiplied, and vice versa.

    Dividing by the recorded factor rather than multiplying by a stored
    reciprocal is what keeps this exact: ``1 / 0.1`` is not representable, while
    dividing by a power of two always is.
    """
    j = step.params["var"]
    factor = step.params["factor"]
    _fits(step, factor != 0.0, "the recorded scale factor is zero")
    var = _var(problem, j, step)

    if var.lb > -INF:
        var.lb *= factor
    if var.ub < INF:
        var.ub *= factor
    var.obj /= factor
    for row in problem.rows_with(j):
        row.coeffs[j] /= factor


def _undo_xor_binary(problem, step):
    """Drop the AND/XOR gadget and re-declare the column's original type."""
    j = step.params["var"]
    var = _var(problem, j, step)
    _drop_appended_rows(problem, step.params["rows"], step)
    _drop_appended_vars(
        problem, [step.params["u"], step.params["v"], step.params["w"]], step
    )
    var.vtype = step.params["vtype"]


def _undo_split_free(problem, step):
    """Fold ``x+ - x-`` back into the retired column and revive it."""
    j = step.params["var"]
    pos = step.params["pos"]
    neg = step.params["neg"]
    var = _var(problem, j, step)

    for row in problem.rows:
        if pos not in row.coeffs and neg not in row.coeffs:
            continue
        _fits(
            step,
            pos in row.coeffs and neg in row.coeffs,
            f"row {row.name!r} holds only one of the two split columns",
        )
        # The two halves were written as exact opposites, so either one gives the
        # original coefficient. Averaging them uses both: it returns exactly the
        # same number when they still agree -- ``(c - -c) / 2`` is exact in binary
        # floating point -- and the best available estimate when inexact
        # arithmetic downstream has let them drift apart in the last bit, which
        # is what ``--sloppy-scaling`` does. Only a disagreement far too large to
        # be rounding means the key does not belong to this model.
        coef_pos = row.coeffs[pos]
        coef_neg = row.coeffs[neg]
        scale = max(abs(coef_pos), abs(coef_neg), 1.0)
        _fits(
            step,
            abs(coef_pos + coef_neg) <= 1e-9 * scale,
            f"row {row.name!r} has {coef_pos!r} on the positive part of the split "
            f"column but {coef_neg!r} on the negative part; those are not opposites",
        )
        del row.coeffs[pos]
        del row.coeffs[neg]
        row.coeffs[j] = (coef_pos - coef_neg) / 2.0

    var.lb = step.params["lb"]
    var.ub = step.params["ub"]
    var.obj = step.params["obj"]
    _drop_appended_vars(problem, [pos, neg], step)


def _undo_scale_row(problem, step):
    """Divide the row through, and un-flip the sense if the factor was negative."""
    factor = step.params["factor"]
    _fits(step, factor != 0.0, "the recorded scale factor is zero")
    row = _row(problem, step.params["row"], step)

    if factor < 0:
        row.sense = FLIP[row.sense]
    for j in row.coeffs:
        row.coeffs[j] /= factor
    row.rhs /= factor


def _undo_invert_row(problem, step):
    """Negating a row is an involution."""
    row = _row(problem, step.params["row"], step)
    for j in row.coeffs:
        row.coeffs[j] = -row.coeffs[j]
    row.rhs = -row.rhs
    row.sense = FLIP[row.sense]


def _undo_slack(problem, step):
    """Un-fold, then take the slack column back out and restore the sense.

    The order inside the step matters, because the fold happened *after* the
    slack coefficient went in and therefore carried it along. Un-folding first
    means the donor row still holds that coefficient, exactly as it did when it
    was folded; removing the slack first would subtract the wrong row.
    """
    i = step.params["row"]
    slack = step.params["slack"]
    row = _row(problem, i, step)

    fold = step.params["fold"]
    if fold is not None:
        _unfold(problem, i, fold, step)

    _fits(
        step,
        slack in row.coeffs,
        f"row {row.name!r} has no entry for the slack column it was given",
    )
    del row.coeffs[slack]
    row.sense = step.params["old_sense"]
    _drop_appended_vars(problem, [slack], step)


def _undo_mix_row(problem, step):
    _unfold(problem, step.params["donor"], step.params["fold"], step)


def _undo_dispatch(problem, step):
    """Subtract each multiple back off its target.

    The folds commute: the donor was never one of the targets and was not itself
    modified, and the targets are sampled without replacement, so each pair is
    independent of the others.
    """
    donor = step.params["donor"]
    for fold in step.params["folds"]:
        _unfold(problem, donor, fold, step)


def _undo_mix_obj(problem, step):
    """Take the multiple back out of the cost vector.

    Note the offset sign: the forward pass *subtracts* ``lam * rhs`` from the
    offset, the opposite convention to the row folds, so this adds it back.
    """
    lam = step.params["lam"]
    donor = _row(problem, step.params["donor"], step)
    for j, coef in donor.coeffs.items():
        _var(problem, j, step).obj -= lam * coef
    problem.obj_offset += lam * donor.rhs


INVERSES = {
    "complement": _undo_complement,
    "shift": _undo_shift,
    "shift-binary": _undo_shift_binary,
    "scale-var": _undo_scale_var,
    "xor-binary": _undo_xor_binary,
    "split-free": _undo_split_free,
    "scale-row": _undo_scale_row,
    "invert-row": _undo_invert_row,
    "slack": _undo_slack,
    "mix-row": _undo_mix_row,
    "dispatch": _undo_dispatch,
    "mix-obj": _undo_mix_obj,
}


def unmorph_problem(hidden, key, check=True):
    """Recover the original model from a hidden one and its key.

    Exact for a default (dyadic) hiding run: every scale is a power of two, and
    every fold is undone by subtracting back the same multiple that was added.
    Under ``--sloppy-scaling`` the *forward* pass already rounds, so recovery
    there is exact only to floating-point tolerance.

    What comes back is the model as ``read_problem`` produced it, which is not
    quite the original file: an MPS ``RANGES`` entry is desugared into an
    equality plus an auxiliary column before the IR exists, and explicitly
    stored zeros are dropped at the same point. Both happen upstream of anything
    a key can observe.
    """
    problem = align_to_key(hidden, key)
    if check:
        verify(problem, key)

    for step in reversed(key.steps):
        inverse = INVERSES.get(step.op)
        if inverse is None:
            raise KeyMismatchError(
                f"the key records a transformation this build does not know how "
                f"to undo: {step.op!r}"
            )
        inverse(problem, step)

    _check(
        len(problem.vars) == key.n_orig_vars,
        f"after undoing every step the model has {len(problem.vars)} columns, but "
        f"the key says the original had {key.n_orig_vars}",
    )
    _check(
        len(problem.rows) == key.n_orig_rows,
        f"after undoing every step the model has {len(problem.rows)} rows, but the "
        f"key says the original had {key.n_orig_rows}",
    )

    for var, name in zip(problem.vars, key.orig_var_names):
        var.name = name
    for row, name in zip(problem.rows, key.orig_row_names):
        row.name = name
    problem.name = key.problem_name
    problem.obj_sense = key.obj_sense
    return problem


# --------------------------------------------------------------------------
# solutions
# --------------------------------------------------------------------------


def _map_complement(x, step):
    x[step.params["var"]] = step.params["total"] - x[step.params["var"]]


def _map_shift(x, step):
    x[step.params["var"]] += step.params["delta"]


def _map_scale_var(x, step):
    x[step.params["var"]] *= step.params["factor"]


def _drop_tail(x, indices, step):
    indices = sorted(indices)
    first = len(x) - len(indices)
    _fits(
        step,
        indices == list(range(first, len(x))),
        f"expected the columns it added to be the last {len(indices)}",
    )
    del x[first:]


def _map_xor_binary(x, step):
    # ``x`` itself is untouched by the encoding -- it keeps its bounds, its cost
    # and every row entry, and the linking row makes the fresh binaries agree
    # with it -- so its value in a hidden solution is already the original value.
    _drop_tail(x, [step.params["u"], step.params["v"], step.params["w"]], step)


def _map_split_free(x, step):
    params = step.params
    x[params["var"]] = x[params["pos"]] - x[params["neg"]]
    _drop_tail(x, [params["pos"], params["neg"]], step)


def _map_slack(x, step):
    _drop_tail(x, [step.params["slack"]], step)


def _map_identity(x, step):
    """Row operations and objective mixing do not move the variable space."""


SOLUTION_MAPS = {
    "complement": _map_complement,
    "shift": _map_shift,
    "shift-binary": _map_shift,
    "scale-var": _map_scale_var,
    "xor-binary": _map_xor_binary,
    "split-free": _map_split_free,
    "scale-row": _map_identity,
    "invert-row": _map_identity,
    "slack": _map_slack,
    "mix-row": _map_identity,
    "dispatch": _map_identity,
    "mix-obj": _map_identity,
}


def unmorph_solution(values, key):
    """Translate a solution of the hidden model back to the original variables.

    ``values`` maps hidden column names to values; the result maps original
    column names to values. Only the substitutions move a point -- the row
    operations and objective mixing are identity on the variable space, since
    they rewrite the constraints a point satisfies rather than the point.

    The objective value needs no translation and is not returned: every
    transformation preserves it pointwise on the feasible set.
    """
    missing = [name for name in key.hidden_var_names if name not in values]
    if missing:
        raise KeyMismatchError(
            f"the solution has no value for {len(missing)} of the hidden model's "
            f"columns, starting with {missing[0]!r}; it does not appear to be a "
            f"solution of the model this key describes"
        )

    x = [float(values[name]) for name in key.hidden_var_names]
    for step in reversed(key.steps):
        mapping = SOLUTION_MAPS.get(step.op)
        if mapping is None:
            raise KeyMismatchError(
                f"the key records a transformation this build does not know how "
                f"to undo: {step.op!r}"
            )
        mapping(x, step)

    _check(
        len(x) == key.n_orig_vars,
        f"after undoing every step {len(x)} values are left, but the key says the "
        f"original had {key.n_orig_vars} columns",
    )
    return dict(zip(key.orig_var_names, x))
