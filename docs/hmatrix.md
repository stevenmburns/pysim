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
| 5a | Algorithmic: same-edge fill restricted to near band O(N·leaf) | ✅ |
| 5b | C++ fused off-edge block assembler for the ACA fill | ✅ |
| 5c | C++ H-matvec / stronger preconditioner | ⏳ future (H-matvec not a bottleneck) |

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
   N   far%  rank  store%    relZ    Dfill(C)  Hbuild   Dsolve  Hsolve  iters
  250   66%  ~5    49.4%   6.6e-08    0.10s     0.15s    0.003s  0.09s     4
  500   82%  ~5    30.9%   1.4e-07    0.41s     0.26s    0.070s  0.12s     5
 1000   91%  ~5    18.9%   7.5e-07    1.56s     0.53s    0.103s  0.29s     6
 2000   95%  ~5    11.1%   1.6e-06    6.12s     1.00s    0.302s  0.68s    12
 4000   98%  ~5     6.4%   3.9e-07   24.85s     2.19s    1.761s  1.63s    21
```

(`Hbuild` uses the C++ fused off-edge assembler; `Hsolve` is a full solve
including its own build, which reuses the warm same-edge / context caches.)

What this shows:

- **Storage is the headline win.** It halves on every mesh doubling
  (49% → 6.4% of dense) at **constant far-block rank ~5** — the O(N log N)
  H-matrix behaviour. Same trend on a 0.5λ dipole and a 4λ wire: the
  admissible blocks are sub-wavelength so there is no oscillatory rank
  growth in this size range.
- **Accuracy is flat at ~1e-6** across all N — compression hides no error.
- **The H-build beats the C-accelerated dense fill** for N ≥ ~500 — at N=4000,
  2.2 s vs 25 s, an **11× win that widens with N** (`Hbuild` ~O(N log N),
  `Dfill` ~O(N²)). Two changes got here: (1) an algorithmic fix computing
  same-edge near-field moments only over each near block's contiguous
  sub-range (~2·leaf) instead of the full O(N_edge²) edge block (had been
  ~68% of build time); (2) a C++ fused off-edge block assembler
  (`_accelerators.bspline_assemble_offedge_block`) that quadratures the
  moments and does the Galerkin combine in one pass for the ACA row/column
  sampling, replacing the per-row numpy orchestration + einsum (~2.6×).
- **End-to-end the H-matrix now beats the dense path ~12×** at N=4000
  (~2 s vs ~26 s) and the gap widens. The GMRES iteration count still creeps
  up (4 → 21 over N=250 → 4000) as the near-field preconditioner neglects
  more of the far field — a stronger preconditioner (H-LU / coarse
  correction) is the next lever, not the already-cheap H-matvec.

### Known limitations / next work

- **C++ ACA fill — done.** `_accelerators.bspline_assemble_offedge_block`
  fuses the off-edge moment quadrature and the Galerkin combine for the ACA
  row/column sampling (used by `HMatrixPySim._offedge_block_evaluators`,
  gated by `hmatrix_use_accel`). Falls back to the pure-numpy `zblock` path
  when the extension is absent or `degree > 2`.
- **Remaining C++ targets:** the per-block ACA pivoting loop itself
  (`aca_partial`) and the near-block dense fill still run in Python; the
  H-matvec is already negligible so it is intentionally not a target.
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
