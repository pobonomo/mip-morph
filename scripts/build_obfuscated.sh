#!/usr/bin/env bash
#
# Build an obfuscated distribution of mip-morph.
#
# The transformation IP lives in three pure-Python modules — ir.py, morph.py and
# transforms.py — plus a hardened release CLI (cli_release.py) that removes the
# options and output which make the transformation transparent (the per-transform
# log and the --enable/--disable transform-name selection). This script compiles
# those modules to native machine-code extension modules (.so) with Nuitka, then
# assembles a shippable package under dist/ where:
#   * the compiled .so files REPLACE the .py sources, and
#   * the full-featured cli.py is REPLACED by a tiny stub that re-exports the
#     compiled release CLI, so `mipmorph`/`python -m mip_morph` use the hardened
#     interface and the developer-only options never ship.
# The gurobi read/write module (io.py) stays as normal Python, and gurobipy
# itself is never touched — it is imported normally at runtime.
#
# IMPORTANT
#   * A compiled .so is tied to the exact OS/arch and Python minor version it was
#     built on. Run this on each target platform (e.g. your Mac, Python 3.14).
#   * Requires a C compiler (clang via Xcode CLT on macOS) and Nuitka:
#         uv sync --group build          # or: uv pip install nuitka
#   * Run from the repository root:  ./scripts/build_obfuscated.sh
#
set -euo pipefail

# --- locate repo root (parent of this script's dir) --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PKG="mip_morph"
BUILD_DIR="${ROOT}/build_obf"
DIST_PKG="${ROOT}/dist/${PKG}"

# Modules whose source must NOT ship — compiled to .so instead. cli_release is
# the hardened CLI; the full-featured cli.py is replaced by a stub (see below).
SECRET_MODULES=(ir morph transforms cli_release)

PY="${PYTHON:-python}"

echo ">> Using interpreter: $(${PY} -c 'import sys; print(sys.executable, sys.version.split()[0])')"
if ! "${PY}" -c 'import nuitka' 2>/dev/null; then
  echo "ERROR: Nuitka is not installed in this interpreter." >&2
  echo "       Run: uv sync --group build   (or: uv pip install nuitka)" >&2
  exit 1
fi

# --- clean previous outputs --------------------------------------------------
rm -rf "${BUILD_DIR}" "${ROOT}/dist"
mkdir -p "${BUILD_DIR}"

# --- compile each secret module to a native extension ------------------------
# Compiled as package submodules so their relative imports (e.g. `from .ir
# import ...`) resolve against the mip_morph package at runtime.
for mod in "${SECRET_MODULES[@]}"; do
  echo ">> Compiling ${PKG}/${mod}.py -> native extension"
  "${PY}" -m nuitka --module "${PKG}/${mod}.py" \
    --include-package="${PKG}" \
    --output-dir="${BUILD_DIR}"
done

# --- assemble the shippable package under dist/ ------------------------------
echo ">> Assembling dist/${PKG}"
mkdir -p "${DIST_PKG}"

# Copy the package, then drop the secret sources and any bytecode caches.
cp -R "${PKG}/." "${DIST_PKG}/"
rm -rf "${DIST_PKG}/__pycache__"
for mod in "${SECRET_MODULES[@]}"; do
  rm -f "${DIST_PKG}/${mod}.py"
done
# Developer-only: ``mipmorph check`` prints every step it verifies, and only the
# full-featured cli.py, which does not ship either, uses it.
rm -f "${DIST_PKG}/check.py"

# Drop in the freshly built .so extensions (skip Nuitka's .pyi/build folders).
for so in "${BUILD_DIR}"/*.so; do
  cp "${so}" "${DIST_PKG}/"
done

# Replace the full-featured CLI with a stub that re-exports the compiled,
# hardened release CLI. The developer-only options and the per-transform log
# never ship, and cli.py carries no logic worth reading.
rm -f "${DIST_PKG}/cli.py"
cat > "${DIST_PKG}/cli.py" <<'STUB'
"""Release CLI shim: delegates to the compiled, hardened cli_release module."""

from .cli_release import build_parser, main

__all__ = ["build_parser", "main"]
STUB

echo
echo ">> Done. Obfuscated package is at: dist/${PKG}"
echo ">> Sanity checks:"
echo "     ls dist/${PKG}                 # ir/morph/transforms/cli_release are .so; cli.py is a stub"
echo "     PYTHONPATH=dist python -m mip_morph --help   # no --enable/--disable, no transform list"
echo "     PYTHONPATH=dist python -m mip_morph in.mps -o out.mps  # runs silently (no per-transform log)"
echo "     strings dist/${PKG}/transforms*.so | grep -i 'def ' || echo 'no python source'"
echo ">> Verify against the test suite by pointing PYTHONPATH at dist/:"
echo "     PYTHONPATH=dist python -m pytest"
