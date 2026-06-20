# Hierarchical (H-matrix / ACA) MoM accelerator — `HMatrixPySim`

`pysim.hmatrix.HMatrixPySim` is a distance-based accelerator for the B-spline
Galerkin MoM. It is a subclass of `BSplinePySim` and reuses that solver's
geometry build, basis-polynomial extraction, kernels, source vectors, and KCL
machinery verbatim. The only thing it replaces is the **dense O(N²)
impedance-matrix assembly + dense LU solve**, with:

- a **block-cluster partition** of Z into *near* (dense) and *far*
  (well-separated, compressible) blocks, and
- **Adaptive Cross Approximation (ACA)** low-rank factors for the far blocks,
- solved iteratively with **GMRES** driven by a fast **H-matvec** and a
  **near-field sparse-LU preconditioner**.

The idea is the standard one from the BEM/MoM literature: the interaction
kernel `G = exp(-jkR)/(4πR)` is smooth once two basis-function supports are
well separated, so the corresponding Z block is numerically low rank and ACA
captures it from `O(r·(m+n))` sampled entries instead of the full `m·n`.

## Status: numpy/scipy prototype (phased plan)

This is a **functional, validated prototype**. The goal of this phase was to
prove the algorithm — accuracy, compression, and operation-count scaling —
in pure numpy/scipy, deferring the constant-factor C++ work. Where it stands:

| Phase | What | State |
|-------|------|-------|
| 0 | On-demand Z block evaluator `zblock(I, J)` | ✅ exact to ~1e-16 vs dense |
| 1 | Cluster tree + admissible/near block partition | ✅ exact cover |
| 2 | ACA far blocks + `HMatrix` container + H-matvec | ✅ |
| 3 | GMRES solve + near-field preconditioner + KCL | ✅ matches dense ~1e-6 |
| 4 | Scaling study + engine/web integration | ✅ |
| 5 | C++ accelerators (ACA fill, H-matvec) | ⏳ future |

## How it works

1. **Cluster tree** (`_aca.build_cluster_tree`): a binary space partition over
   each basis function's support bounding box, bisecting the longest axis at
   the median until clusters hold `aca_leaf_size` (default 32) bases.
2. **Block-cluster partition** (`_aca.build_block_tree`): recursively pair
   clusters; emit a *far* block when admissible
   `min(diam(s), diam(t)) ≤ aca_eta · dist(s, t)` (default `aca_eta=1.0`),
   recurse otherwise, emit a *near* block when two leaves are still
   inadmissible. The far + near leaves tile the n×n index product exactly.
3. **Assembly** (`build_hmatrix`): near blocks are built densely with
   `zblock` (including the analytic same-edge / near-singular path); far
   blocks are compressed with `aca_partial`, sampling the **off-edge kernel
   only** (a far block is well separated by construction, so even two
   segments on the same long wire are far enough apart that the
   `a²`-regularised GL kernel is accurate — no same-edge analytic block is
   ever built). A far block that fails to compress falls back to dense.
4. **Solve** (`_solve_hmatrix`): the dense near blocks form a sparse
   near-field matrix; augmented with the KCL junction rows into the saddle
   system `[Z Aᵀ; A 0]` and factorised once with `splu` — used both as the
   GMRES preconditioner and the initial guess. Z is applied via the
   H-matvec. Free-space, no-enrichment only; ground-plane and
   singular-enrichment cases fall back to the dense `BSplinePySim` path.

## Validation

Against the dense `BSplinePySim` at matched mesh:

- `zblock` (exact `same_edge=True` mode) reproduces arbitrary sub-blocks of
  the dense Z to ~1e-16.
- end-to-end `PysimEngine(...).impedance()` matches the dense bspline engine
  to **~1.1e-4 (rhombic)** and **~2.1e-5 (all 8 bowtiearray2x4 feeds)** at
  `aca_tol=1e-5` — far below the ~1% engine-to-engine variation (1.1e-4 on
  rhombic's 680 Ω is 0.07 Ω). The far-block off-edge GL kernel deviates from
  the dense same-edge analytic moments by ~1e-5 on intra-wire far pairs;
  tightening `aca_tol` and keeping `same_edge=True` recovers ~1e-7 agreement
  at higher fill cost.

## Scaling (`scripts/hmatrix_scaling.py`)

Fixed-length wire, mesh refined, `aca_tol=1e-5`, `aca_eta=2.0`, degree 1:

```
   N   far%  rank  store%    relZ    Dfill(C)  Hbuild(py)  Dsolve  Hsolve  iters
  250   66%  ~5    49.4%   6.6e-08    0.10s     0.27s      0.003s  0.13s     4
  500   82%  ~5    30.9%   1.4e-07    0.40s     0.74s      0.064s  0.36s     5
 1000   91%  ~5    18.9%   7.5e-07    1.57s     2.42s      0.097s  1.10s     6
 2000   95%  ~5    11.1%   1.6e-06    6.12s     8.11s      0.347s  2.06s    12
 4000   98%  ~5     6.4%   3.9e-07   25.05s    29.82s      1.81s   4.94s    21
```

What this shows, and what it doesn't:

- **Storage is the headline win.** It halves on every mesh doubling
  (49% → 6.4% of dense) at **constant far-block rank ~5** — the O(N log N)
  H-matrix behaviour. Same trend on a 0.5λ dipole and a 4λ wire: the
  admissible blocks are sub-wavelength so there is no oscillatory rank
  growth in this size range.
- **Accuracy is flat at ~1e-6** across all N — compression hides no error.
- **Wall-clock is the honest caveat.** The dense *fill* uses the existing
  C++ accelerators; the H build is pure Python, so it only *reaches parity*
  with dense fill near N≈4000 despite touching far fewer entries. The Python
  H-matvec also makes the iterative solve slower than the dense LU at these
  sizes. Both are constant-factor effects the C++ phase (Phase 5) removes;
  the dense fill is ~O(N²) and the dense LU ~O(N³), so the asymptotic
  crossover is already visible in the *fill-work* (entries evaluated) and
  *storage* columns, not yet in pure-Python seconds.

### Known limitations / next work

- **C++ accelerators (Phase 5):** port the ACA fill (per-block row/col
  evaluation) and the H-matvec to C++/OpenMP, the same way the dense path was
  accelerated. This is what turns the proven operation-count win into a
  wall-clock win.
- **Preconditioner strength:** GMRES iteration count grows slowly with N
  (4 → 21 over 250 → 4000) as the near-field preconditioner neglects more of
  the far field. An H-LU or coarse-correction preconditioner would flatten
  this for very large problems.
- **Electrically very large structures:** when admissible blocks exceed ~λ
  the Helmholtz kernel becomes oscillatory and standard ACA rank grows;
  directional/butterfly/HF-FMM techniques would be needed. Not hit by the
  current antenna examples.
- **Ground plane + singular enrichment** in the hierarchical path (currently
  dense fallback).

## Usage

```python
from pysim import HMatrixPySim
from antenna_designer.engines.pysim import PysimEngine

# directly
sim = HMatrixPySim(wires=[...], degree=1, aca_tol=1e-5, aca_eta=2.0)
z, coeffs = sim.compute_impedance()

# via the engine (selectable like any other pysim solver)
eng = PysimEngine(builder, solver=HMatrixPySim,
                  solver_kwargs={"degree": 1, "aca_tol": 1e-5})
z = eng.impedance()
```

Web UI / server: registered under the `hmatrix` model key.

Knobs: `aca_eta` (admissibility looseness), `aca_leaf_size` (cluster leaf
size), `aca_tol` (ACA relative tolerance), `solve_tol` (GMRES tolerance),
plus all the inherited `BSplinePySim` parameters (`degree`, `n_qp_pair`, …).
