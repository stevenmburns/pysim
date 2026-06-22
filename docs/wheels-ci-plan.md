# Plan: cross-platform wheels for the pysim C++ accelerators

Goal: GitHub Actions builds redistributable Python wheels containing the
`pysim._accelerators` C++ extension for **modern Linux and Windows**, attached
to pysim GitHub Releases so `antenna_designer` (which vendors pysim as a
submodule today) can `pip install` a pinned wheel.

Decisions locked (2026-06-22):

| Decision        | Choice                          | Rationale |
|-----------------|---------------------------------|-----------|
| CPU baseline    | **AVX2 + FMA** (Haswell/Zen+)   | "modern" target; keeps the libmvec sincos win on Linux; simplest |
| Windows toolchain | **MSVC `/openmp:llvm`**       | LLVM OpenMP runtime supports `collapse` + `simd` (classic `/openmp` is OpenMP 2.0); no libmvec on Windows |
| Distribution    | **GitHub Release artifacts**    | No public PyPI commitment; antenna_designer pins a release wheel |

This work lives **in the pysim repo** (its own `.github/workflows/`), not in
antenna_designer. antenna_designer's only change is how it consumes pysim.

---

## Why this is two problems

1. **CI mechanics** — multi-Python × Linux+Windows wheel builds: solved by
   [`cibuildwheel`](https://cibuildwheel.pypa.io). Boilerplate.
2. **Windows source portability** — `setup.py` currently *skips the extension
   entirely* on `win32`. Shipping an accelerated Windows wheel is net-new and
   needs source/build changes (below). The good news: `_accelerators.cpp` has
   **no explicit SIMD intrinsics** — all vectorization is OpenMP pragmas +
   compiler autovec, so the math is portable; only the *strategy* is GCC/glibc-
   specific.

---

## Step 1 — Make `setup.py` cross-compile

Replace the `win32 → []` skip with a per-platform flag split (extension built
on both):

```python
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

if sys.platform == "win32":
    # /openmp:llvm => OpenMP 3.1+ (collapse + simd), required by the kernels;
    # classic /openmp is only 2.0. /arch:AVX2 matches the Linux AVX2 baseline.
    # No libmvec on Windows: the declare-simd cos/sin block in the .cpp is
    # guarded out (see Step 2), sincos stays scalar/autovec.
    extra_compile_args = ["/O2", "/arch:AVX2", "/openmp:llvm", "/fp:fast"]
    extra_link_args = []
else:
    extra_compile_args = [
        "-O3", "-fopenmp", "-fopenmp-simd", "-mavx2", "-mfma",
        "-fno-math-errno", "-g", "-fno-omit-frame-pointer", "-std=gnu++11",
    ]
    extra_link_args = ["-fopenmp", "-lpthread", "-lmvec"]

ext_modules = [
    Pybind11Extension(
        "pysim._accelerators",
        ["src/pysim/_accelerators.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(ext_modules=ext_modules, packages=["pysim"], package_dir={"": "src/"})
```

## Step 2 — Guard the libmvec-specific source

In `_accelerators.cpp`, the `#pragma omp declare simd` + `extern "C" double
cos/sin` redeclarations (lines ~23-27) exist only to bind GCC's autovectorizer
to glibc's `_ZGVdN4v_{sin,cos}`. They are meaningless and potentially harmful
under MSVC. Guard them:

```cpp
#if defined(__GNUC__) && !defined(_MSC_VER)
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double cos(double);
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);
#endif
```

Also verify/drop `#include <complex.h>` (line 4) — on MSVC it defines a
`complex` macro that collides with `std::complex`. The file uses `std::complex`
throughout, so `<complex.h>` is likely unnecessary; remove it or guard it.

The `#pragma omp parallel for collapse(2)` and `#pragma omp simd` sites
(many) need no change — `/openmp:llvm` accepts them.

## Step 3 — cibuildwheel config in `pyproject.toml`

```toml
[tool.cibuildwheel]
build = "cp39-* cp310-* cp311-* cp312-* cp313-*"
skip = "*-musllinux_* *_i686 *-win32"   # 64-bit glibc Linux + 64-bit Windows
build-frontend = "build"
test-requires = "pytest numpy scipy"
test-command = "pytest {project}/tests -m 'not slow and not plot'"
build-verbosity = 1

[tool.cibuildwheel.linux]
# glibc >= 2.28 ships libmvec; manylinux2014 (glibc 2.17) does NOT — the
# -lmvec link would fail. _2_28 is the right "modern Linux" floor.
manylinux-x86_64-image = "manylinux_2_28"
environment = { CFLAGS="-mavx2 -mfma" }
```

## Step 4 — The workflow (`.github/workflows/wheels.yml`)

```yaml
name: wheels
on:
  push:
    tags: ["v*"]
  workflow_dispatch:        # manual runs for testing the matrix
  pull_request:
    paths: ["setup.py", "pyproject.toml", "src/pysim/_accelerators.cpp",
            ".github/workflows/wheels.yml"]

jobs:
  build:
    name: wheels-${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }
      - uses: pypa/cibuildwheel@v2.21
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: ./wheelhouse/*.whl

  release:
    needs: build
    if: startsWith(github.ref, 'refs/tags/')
    runs-on: ubuntu-latest
    permissions: { contents: write }
    steps:
      - uses: actions/download-artifact@v4
        with: { path: dist, merge-multiple: true }
      - uses: softprops/action-gh-release@v2
        with: { files: dist/*.whl }
```

## Step 5 — antenna_designer consumes the release wheel

Once pysim publishes wheels on a tag, replace the editable-submodule install of
pysim with a pinned wheel reference (direct URL to the release asset, or a
`pysim @ https://github.com/stevenmburns/pysim/releases/download/vX/...whl`
requirement). Submodule can stay for source/dev; the wheel is for consumers.

---

## Two wrinkles to validate during implementation (not blockers)

1. **`/openmp:llvm` runtime DLL.** It links `libomp140.x86_64.dll` (the LLVM
   OpenMP runtime), which ships with Visual Studio but is **not** part of the
   standard VC redistributable — unlike classic `/openmp`'s `vcomp140.dll`.
   cibuildwheel runs `delvewheel` on Windows and should vendor it into the
   wheel if it's discoverable on PATH; confirm the built wheel actually
   contains an `libomp*.dll` (or that import works on a clean runner).
   **Fallback if bundling is painful:** drop `collapse(2)` → plain
   `parallel for` over the outer index and accept classic `/openmp`
   (`vcomp140.dll`, no bundling); `#pragma omp simd` is then ignored but
   `/arch:AVX2` autovec still applies. Slower Windows sincos, zero DLL hassle.

2. **libmvec under auditwheel.** libmvec is part of glibc, so it's on the
   manylinux allowlist — auditwheel won't try to vendor it and won't reject the
   wheel. Confirm on the first `manylinux_2_28` run; if auditwheel complains,
   the kernel still works without `-lmvec` (scalar sincos), so it's a
   performance, not correctness, fallback.

## Suggested PR sequence (in pysim)

1. PR A: `setup.py` + `.cpp` guards so the extension **builds on Windows
   locally** (verify on a Windows box or a throwaway `windows-latest` run via
   `workflow_dispatch`). Correctness first.
2. PR B: add `[tool.cibuildwheel]` + `wheels.yml`, driven by
   `workflow_dispatch`, until the full matrix is green.
3. PR C: turn on the tag-triggered `release` job; cut a `v0.0.1` tag.
4. antenna_designer: switch its pysim dependency to the release wheel.
