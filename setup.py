from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "pysim._accelerators",
        ["src/pysim/_accelerators.cpp"],
        extra_compile_args=[
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
        ],
        extra_link_args=["-fopenmp", "-lpthread", "-lmvec"],
    ),
]

setup(
    ext_modules=ext_modules,
    packages=["pysim"],
    package_dir={"": "src/"},
)
