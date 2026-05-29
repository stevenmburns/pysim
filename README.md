# pysim

A pure-Python method-of-moments antenna simulator with optional C++ accelerators (pybind11).

Extracted from [antenna_designer](https://github.com/stevenmburns/antenna_designer).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Test

```bash
pip install pytest numpy scipy matplotlib icecream scikit-rf
pytest tests/
```

## Optional: PyNEC backend

The web UI can run either the in-tree triangular MoM solver or NEC2 via [PyNEC](https://github.com/tmolteno/python-necpp), useful for cross-validation and for the ~5–10× faster single-frequency solves NEC2 delivers. PyNEC is vendored as a git submodule because the PyPI wheel builds are broken on current Python versions.

### Default build (LAPACKE + OpenBLAS)

```bash
git submodule update --init --recursive
pip install swig          # SWIG goes into .venv via pip
sudo apt install autoconf automake libtool m4 \
    libopenblas-pthread-dev liblapacke-dev    # one-time system deps
scripts/build_pynec.sh
```

After the build, `from PyNEC import nec_context` works in `.venv`. The web UI's "solver" tab in the simulation section toggles between pysim and PyNEC at runtime. If the build is skipped the UI silently falls back to pysim.

### Choosing a different LAPACK backend

`scripts/build_pynec.sh` picks the LAPACK implementation from `PYNEC_BACKEND` (default `lapacke`). The matrix solve is the same algorithm in every case; only the library linked at runtime changes. On a 100-director Yagi (2142 segs) at i7-8550U, `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`, the solve+fill wall-times are:

| `PYNEC_BACKEND` | Link target | System deps | NP=4 wall time |
|---|---|---|---|
| `lapacke` (default) | LAPACKE + OpenBLAS pthread | `libopenblas-pthread-dev liblapacke-dev` | 630 ms |
| `atlas` | ATLAS clapack | `libatlas-base-dev` (conflicts with lapacke deps on Ubuntu) | ~900 ms |
| `mkl` | MKL LAPACKE + libgomp | Intel oneAPI MKL, `intel-oneapi-mkl-classic-devel` | 465 ms |
| `mkl_intel` | MKL LAPACKE + libiomp5 | + Intel oneAPI compiler runtime | ~485 ms |

To switch backends, set the env var and force a clean rebuild:

```bash
# Wipe the cached config.h and build dir so the next build re-runs configure
# and re-links with the new backend's libs.
rm -f python-necpp/necpp_src/config.h
rm -rf python-necpp/PyNEC/build

PYNEC_BACKEND=mkl scripts/build_pynec.sh
```

Verify the link target with `ldd $(python -c 'import _PyNEC; print(_PyNEC.__file__)') | grep -iE 'mkl|lapacke|atlas'`.

### Runtime thread pinning

For multi-threaded backends (everything except `atlas`), pick thread counts up front:

```bash
export OMP_NUM_THREADS=$(nproc --all)   # PyNEC matrix fill
export MKL_NUM_THREADS=$(nproc --all)   # MKL backends
export OPENBLAS_NUM_THREADS=1           # muzzle numpy/scipy's idle pool
```

Pinning `OPENBLAS_NUM_THREADS=1` stops numpy/scipy from spinning up their own OpenBLAS thread pool that contends with PyNEC's threads on the same cores. On the 100-director Yagi this is worth ~8% wall time at NP=4.
