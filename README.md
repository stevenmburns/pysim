# momwire

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

momwire can be cross-validated against NEC2 via [PyNEC](https://github.com/tmolteno/python-necpp) (the `tests/test_pynec_backend.py` suite); NEC2 also delivers ~5–10× faster single-frequency solves.

### Install the PyNEC wheel

Install PyNEC from the [python-necpp fork](https://github.com/stevenmburns/python-necpp)'s release. The wheels are self-contained — OpenBLAS is vendored (via scipy-openblas32), so no system BLAS, SWIG, or build toolchain is needed — and cover Linux + Windows on CPython 3.10–3.14:

```bash
pip install PyNEC --no-index \
    --find-links https://github.com/stevenmburns/python-necpp/releases/expanded_assets/v1.7.4-accel.1
```

`--no-index` ensures pip takes the fork's wheel rather than upstream PyNEC on PyPI (same version, but its builds are broken on current Python and it lacks the OpenBLAS/OpenMP work). After install, `from PyNEC import nec_context` works and the cross-validation tests run; without it they're skipped (momwire itself needs no PyNEC).

### Runtime thread pinning

The wheel links OpenBLAS and parallelises the NEC2 matrix fill with OpenMP. Pick thread counts up front:

```bash
export OMP_NUM_THREADS=$(nproc --all)   # PyNEC matrix fill
export OPENBLAS_NUM_THREADS=1           # muzzle numpy/scipy's idle pool
```

Pinning `OPENBLAS_NUM_THREADS=1` stops numpy/scipy from spinning up their own OpenBLAS thread pool that contends with PyNEC's threads on the same cores. On a 100-director Yagi (2142 segs) this is worth ~8% wall time at NP=4.
