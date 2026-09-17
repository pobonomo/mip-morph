"""The recovery key: a structured record of everything hiding did.

Hiding is only reversible if the record of it is *exact*. The human-readable log
that ``morph_model`` returns is not: it formats every float with ``%g``, which
is six significant digits, and it omits parameters outright -- ``dispatch``
names neither its target rows nor any of its per-target multipliers, ``slack``
drops the multiplier it folded with, and ``xor-binary`` never mentions the
variable type it overwrote. So a key records parameters separately from the
description, and :class:`Step` carries both: ``params`` for the inverse engine,
``description`` for people.

Replaying the driver from its seed is not an alternative. The coefficient-range
guard rolls a candidate back *after* the transformation has already drawn from
the RNG, and only accepted steps are logged, so the seed and the list of
accepted steps do not determine one another. A key has to be an explicit edit
list.

On disk a key is deliberately opaque: canonical JSON, then zlib, then base64
behind a magic line. Be clear about what that does and does not buy. Without a
password it is *obfuscation, not encryption* -- anyone who wants the contents
can base64-decode and inflate them in two lines. Pass a password for actual
confidentiality, which matters because a leaked key undoes the hiding
completely.
"""

import base64
import hashlib
import hmac
import json
import os
import zlib
from dataclasses import dataclass, field

KEY_FORMAT = 1

MAGIC = b"MIPMORPH-KEY-1\n"
MODE_PLAIN = b"P"
MODE_ENCRYPTED = b"E"

# scrypt work factors. 128 * r * n is 16 MiB of memory per derivation, which is
# slow enough to make guessing a weak password expensive and fast enough that a
# legitimate unhide does not notice.
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 64 * 1024 * 1024

SALT_BYTES = 16
TAG_BYTES = 32
_WRAP = 76


class KeyFormatError(Exception):
    """The file is not a key, is damaged, or the password is wrong."""


@dataclass
class Step:
    """One applied transformation: what it was, exactly, and how to say it.

    ``op`` is a key of :data:`mip_morph.transforms.TRANSFORMS`. ``params`` holds
    the exact choices the transformation made -- column and row *indices* rather
    than names, because ``rename_all`` rewrites every name and because a
    transformation mints new columns from the current column count. ``__str__``
    hands back ``description``, so a ``Step`` prints exactly as the old log
    string did and existing callers need no change.
    """

    op: str
    params: dict = field(default_factory=dict)
    description: str = ""

    def __str__(self):
        return self.description


@dataclass
class Key:
    """Everything needed to invert one ``morph_model`` run.

    The name lists are what make inversion independent of file ordering.
    ``hidden_var_names``/``hidden_row_names`` are recorded in *pre-shuffle*
    order -- straight after renaming, before the final permutation -- so
    re-indexing a freshly read hidden model onto them undoes the shuffle and
    absorbs any reordering the MPS writer or reader introduced, in one step.
    ``orig_var_names``/``orig_row_names`` then restore the original names.

    Only the original columns and rows need naming for recovery: nothing is ever
    deleted during hiding and every addition is appended, so the original
    columns keep indices ``0..n0-1`` and the original rows keep ``0..m0-1`` for
    the whole run.
    """

    problem_name: str = "model"
    obj_sense: int = 1
    orig_var_names: list = field(default_factory=list)
    orig_row_names: list = field(default_factory=list)
    hidden_var_names: list = field(default_factory=list)
    hidden_row_names: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    structure_fingerprint: str = ""
    values_fingerprint: str = ""

    @property
    def n_orig_vars(self):
        return len(self.orig_var_names)

    @property
    def n_orig_rows(self):
        return len(self.orig_row_names)


# --------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------


def _digest(parts):
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def structure_fingerprint(problem):
    """Digest the model's shape: names, types, senses and sparsity pattern.

    Deliberately free of any coefficient *value*, so it stays intact through a
    writer that rounds. A mismatch means the key belongs to a different model,
    not that the file lost precision -- which is why the two fingerprints are
    reported separately.
    """
    parts = [str(len(problem.vars)), str(len(problem.rows))]
    for var in problem.vars:
        parts += [var.name, var.vtype]
    for row in problem.rows:
        parts += [row.name, row.sense, ",".join(str(j) for j in sorted(row.coeffs))]
    return _digest(parts)


def _number(value):
    """Canonical text for one number: bit-exact, but only about the value.

    ``repr`` of a float round-trips exactly, which is what makes this able to
    notice a genuinely changed coefficient. Two things have to be normalized away
    first, or the digest reports differences that are not differences. A model
    built in Python may hold ``3`` where one read from a file holds ``3.0``, and
    ``repr`` tells those apart while ``==`` does not. And ``-0.0`` is the same
    number as ``0.0`` and behaves identically in every inverse, but again has its
    own ``repr``; adding zero folds it back.
    """
    return repr(float(value) + 0.0)


def values_fingerprint(problem):
    """Digest every number in the model, exactly enough to catch a real change."""
    parts = [_number(problem.obj_offset), str(problem.obj_sense)]
    for var in problem.vars:
        parts += [_number(var.lb), _number(var.ub), _number(var.obj)]
    for row in problem.rows:
        parts.append(_number(row.rhs))
        parts += [f"{j}={_number(c)}" for j, c in sorted(row.coeffs.items())]
    return _digest(parts)


# --------------------------------------------------------------------------
# JSON form
# --------------------------------------------------------------------------


def to_json_obj(key):
    return {
        "format": KEY_FORMAT,
        "problem_name": key.problem_name,
        "obj_sense": key.obj_sense,
        "orig_var_names": list(key.orig_var_names),
        "orig_row_names": list(key.orig_row_names),
        "hidden_var_names": list(key.hidden_var_names),
        "hidden_row_names": list(key.hidden_row_names),
        "structure_fingerprint": key.structure_fingerprint,
        "values_fingerprint": key.values_fingerprint,
        "steps": [
            {"op": s.op, "params": s.params, "description": s.description}
            for s in key.steps
        ],
    }


def from_json_obj(obj):
    if not isinstance(obj, dict) or "format" not in obj:
        raise KeyFormatError("key data is missing its format marker")
    if obj["format"] != KEY_FORMAT:
        raise KeyFormatError(
            f"key format {obj['format']} is not supported "
            f"(this build writes {KEY_FORMAT})"
        )
    try:
        return Key(
            problem_name=obj["problem_name"],
            obj_sense=obj["obj_sense"],
            orig_var_names=obj["orig_var_names"],
            orig_row_names=obj["orig_row_names"],
            hidden_var_names=obj["hidden_var_names"],
            hidden_row_names=obj["hidden_row_names"],
            structure_fingerprint=obj.get("structure_fingerprint", ""),
            values_fingerprint=obj.get("values_fingerprint", ""),
            steps=[
                Step(
                    op=s["op"],
                    params=s.get("params", {}),
                    description=s.get("description", ""),
                )
                for s in obj["steps"]
            ],
        )
    except (TypeError, KeyError) as exc:
        raise KeyFormatError(f"key data is missing field {exc}") from exc


def _dumps(key):
    """Serialize to bytes.

    ``sort_keys`` makes the form canonical, so the same run always produces the
    same key bytes. Floats go through ``repr``, which round-trips exactly in
    Python 3, so a key is bit-exact even for the arbitrary factors that
    ``--sloppy-scaling`` draws -- unlike the ``%g`` of the human-readable log.
    Infinities (``split-free`` records the bounds it overwrote) are written as
    JSON's ``Infinity`` extension, which Python's own parser reads back exactly;
    the container is opaque and internal, so relying on that is cheaper than a
    sentinel encoding.
    """
    return json.dumps(to_json_obj(key), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _loads(data):
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise KeyFormatError(f"key does not contain valid key data: {exc}") from exc
    return from_json_obj(obj)


# --------------------------------------------------------------------------
# opaque container
# --------------------------------------------------------------------------


def _keystream(enc_key, length):
    """HMAC-SHA256 in counter mode."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(enc_key, counter.to_bytes(8, "big"), "sha256").digest()
        counter += 1
    return bytes(out[:length])


def _derive(password, salt):
    """Stretch the password into an encryption key and a separate MAC key."""
    material = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=64,
        maxmem=SCRYPT_MAXMEM,
    )
    return material[:32], material[32:64]


def _xor(data, stream):
    return bytes(a ^ b for a, b in zip(data, stream))


def encode_key(key, password=None):
    """Return the opaque on-disk bytes for ``key``."""
    body = zlib.compress(_dumps(key), 9)
    if password:
        salt = os.urandom(SALT_BYTES)
        enc_key, mac_key = _derive(password, salt)
        cipher = _xor(body, _keystream(enc_key, len(body)))
        tag = hmac.new(mac_key, salt + cipher, "sha256").digest()
        payload = MODE_ENCRYPTED + salt + tag + cipher
    else:
        payload = MODE_PLAIN + body

    armoured = base64.b64encode(payload)
    lines = [armoured[i : i + _WRAP] for i in range(0, len(armoured), _WRAP)]
    return MAGIC + b"\n".join(lines) + b"\n"


def decode_key(data, password=None):
    """Parse the bytes written by :func:`encode_key`.

    Every failure mode -- wrong file, truncated file, flipped byte, wrong
    password -- raises :class:`KeyFormatError`. Nothing is half-parsed: for an
    encrypted key the authentication tag is checked before a single byte is
    decrypted, so a damaged key cannot quietly turn into a plausible wrong one.
    """
    if not data.startswith(MAGIC):
        raise KeyFormatError("not a mip_morph key (bad magic line)")
    try:
        payload = base64.b64decode(data[len(MAGIC) :])
    except (ValueError, TypeError) as exc:
        raise KeyFormatError(f"key is corrupt: {exc}") from exc

    if not payload:
        raise KeyFormatError("key is empty")
    mode, payload = payload[:1], payload[1:]

    if mode == MODE_PLAIN:
        if password:
            raise KeyFormatError("key is not encrypted, but a password was given")
        body = payload
    elif mode == MODE_ENCRYPTED:
        if not password:
            raise KeyFormatError("key is encrypted: a password is required")
        if len(payload) < SALT_BYTES + TAG_BYTES:
            raise KeyFormatError("key is corrupt (truncated header)")
        salt = payload[:SALT_BYTES]
        tag = payload[SALT_BYTES : SALT_BYTES + TAG_BYTES]
        cipher = payload[SALT_BYTES + TAG_BYTES :]
        enc_key, mac_key = _derive(password, salt)
        expected = hmac.new(mac_key, salt + cipher, "sha256").digest()
        if not hmac.compare_digest(tag, expected):
            raise KeyFormatError("key is corrupt or the password is wrong")
        body = _xor(cipher, _keystream(enc_key, len(cipher)))
    else:
        raise KeyFormatError("key is corrupt (unknown container mode)")

    try:
        plain = zlib.decompress(body)
    except zlib.error as exc:
        raise KeyFormatError(f"key is corrupt: {exc}") from exc
    return _loads(plain)


def write_key(key, path, password=None):
    """Write ``key`` to ``path``.

    Without ``password`` the result is obfuscated, not secret; see the module
    docstring.
    """
    with open(path, "wb") as handle:
        handle.write(encode_key(key, password=password))


def read_key(path, password=None):
    with open(path, "rb") as handle:
        return decode_key(handle.read(), password=password)
