import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

# The accelerator is built on both platforms. The vectorization strategy
# differs: Linux/GCC binds the inner sincos to glibc's libmvec (-lmvec) via the
# `omp declare simd` block in _accelerators.cpp; Windows/MSVC has no libmvec, so
# it relies on /arch:AVX2 autovectorization plus OpenMP parallelism. The .cpp
# guards the libmvec-specific declarations to non-MSVC compilers. If the
# extension fails to build/import, triangular.py falls back to pure Python.
if sys.platform == "win32":
    # OpenMP on MSVC is a minefield for this code: /openmp:experimental rejects
    # unsigned loop indices (the kernels use size_t) and silently drops the
    # `reduction` clause from `omp simd` (a correctness hazard), while
    # /openmp:llvm rejects the `omp simd` directive outright. We use
    # /openmp:llvm — it supports the OpenMP 3.0 `collapse` clause and unsigned
    # loop indices, so the parallel-for loops need no changes — and the .cpp
    # neutralizes the `omp simd` directives under _MSC_VER, leaving /arch:AVX2
    # autovectorization to handle the inner loops. /arch:AVX2 matches the Linux
    # AVX2 baseline.
    extra_compile_args = ["/O2", "/arch:AVX2", "/openmp:llvm", "/fp:fast"]
    extra_link_args = []
else:
    extra_compile_args = [
        # Force -O3 -- Debian's Python CFLAGS inject -O2 before our flags
        # and pybind11's default -O3 doesn't override that. Our -O3 here
        # comes after both and wins (gcc takes the last -O).
        "-O3",
        "-fopenmp",
        "-fopenmp-simd",
        # AVX2 + FMA: required for the SIMD inner-loop sincos in
        # _accelerators.cpp to use libmvec (vectorized libm). KBL/HSW
        # and newer Intel; matches what pybind11 release wheels can't
        # assume but a local pip install -e . can.
        "-mavx2",
        "-mfma",
        # `std::cos` / `std::sin` set errno on domain errors by default,
        # which is a global side effect that blocks auto-vectorization.
        # We don't care about errno from a deterministic-domain real input,
        # so disable the side effect to let the vectorizer kick in.
        "-fno-math-errno",
        "-g",
        "-fno-omit-frame-pointer",
        "-std=gnu++11",
    ]
    extra_link_args = ["-fopenmp", "-lpthread", "-lmvec"]

ext_modules = [
    Pybind11Extension(
        "pysim._accelerators",
        ["src/pysim/_accelerators.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    packages=["pysim"],
    package_dir={"": "src/"},
)
