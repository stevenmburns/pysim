#!/usr/bin/env bash
#
# Build PyNEC (from the python-necpp submodule) and install it into the
# local project venv at .venv/.
#
# Backends (set via PYNEC_BACKEND, default lapacke):
#   lapacke     reference LAPACKE + OpenBLAS pthread
#   atlas       ATLAS clapack
#   mkl         MKL LAPACKE + libgomp threading
#   mkl_intel   MKL LAPACKE + libiomp5 threading
#
# System deps on Ubuntu:
#   common      autoconf automake libtool m4 swig
#   lapacke     libopenblas-pthread-dev liblapacke-dev
#   atlas       libatlas-base-dev
#   mkl,mkl_intel  intel-oneapi-mkl-classic-devel (and sourcing setvars.sh)
#
# SWIG and numpy come from the local venv.
#
# After this script: `from PyNEC import nec_context` should work in .venv.
#
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
VENV="$ROOT/.venv"
BACKEND="${PYNEC_BACKEND:-lapacke}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "error: no .venv at $VENV — create one and pip install -e . first." >&2
    exit 1
fi
if [ ! -d "$ROOT/python-necpp/necpp_src" ] || [ -z "$(ls -A "$ROOT/python-necpp/necpp_src" 2>/dev/null)" ]; then
    echo "error: python-necpp submodule not initialised — run:" >&2
    echo "    git submodule update --init --recursive" >&2
    exit 1
fi
if [ ! -x "$VENV/bin/swig" ]; then
    echo "error: swig not in venv — run: .venv/bin/pip install swig" >&2
    exit 1
fi

echo "PyNEC backend: $BACKEND"

cd "$ROOT/python-necpp/necpp_src"
# Generate configure (idempotent — skip if config.h already exists).
#
# necpp_src/configure.ac's --with-lapack check is hardcoded to look for
# ATLAS's clapack_zgetrf symbol in -llapack. That works for the atlas
# backend but fails for lapacke/mkl: liblapacke-dev and libatlas-base-dev
# cannot coexist on Ubuntu (they Provide the same libblas.so/liblapack.so
# alternatives), so a runner with the lapacke deps installed has no ATLAS
# libraries available and the check fails — even though the runtime code
# uses LAPACKE_zgetrf from libopenblas+liblapacke, never clapack_zgetrf.
#
# For lapacke/mkl we configure --without-lapack (which always passes) and
# rely on PyNEC/setup.py's `_defines` to pass -DLAPACK=1 at compile time,
# so the matrix_algebra.cpp code path gated on LAPACK is live regardless
# of what config.h says. setup.py's PYNEC_BACKEND logic chooses which
# LAPACK implementation we link to at extension-link time (atlas clapack
# vs lapacke vs mkl).
MULTIARCH=$(gcc -print-multiarch 2>/dev/null || echo x86_64-linux-gnu)
if [ ! -f config.h ]; then
    make -f Makefile.git
    if [ "$BACKEND" = "atlas" ]; then
        # Real --with-lapack: configure check runs and finds clapack_zgetrf
        # in libatlas-base-dev's liblapack (under /usr/lib/<MULTIARCH>/atlas/).
        CPPFLAGS="-I/usr/include/${MULTIARCH}" \
            LDFLAGS="-L/usr/lib/${MULTIARCH}/atlas -L/usr/lib/${MULTIARCH}" \
            ./configure --with-lapack
    else
        ./configure --without-lapack
    fi
fi

cd "$ROOT/python-necpp/PyNEC"
# Symlink necpp_src into PyNEC/ so setup.py's relative paths work.
if [ ! -e necpp_src ]; then
    ln -s ../necpp_src .
fi
# SWIG wrapper (idempotent — regenerate if .i is newer than the wrapper).
if [ ! -f PyNEC_wrap.cxx ] || [ PyNEC.i -nt PyNEC_wrap.cxx ]; then
    "$VENV/bin/swig" -Wall -c++ -python PyNEC.i
fi

# Build + install into the venv. --no-build-isolation so setup.py sees
# numpy from .venv. PYNEC_BACKEND propagates to setup.py for backend choice.
PYNEC_BACKEND="$BACKEND" "$VENV/bin/pip" install --no-build-isolation .

echo "OK — PyNEC ($BACKEND) built and installed into $VENV"
"$VENV/bin/python" -c "from PyNEC import nec_context; print('PyNEC import OK')"
