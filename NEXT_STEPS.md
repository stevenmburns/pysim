# Next steps — pysim

Living roadmap of what's done and what's left. Updated as work lands.

## Where we are

The codebase has one active solver and one legacy comparator:

- **`pysim.triangular.TriangularPySim`** (in `src/pysim/triangular.py`) — piecewise-linear (tent) basis with Galerkin testing and analytic singularity extraction. Accepts arbitrary 3D polylines, multiple wires, an optional PEC-image ground plane, and (PR #36) wire-endpoint junctions where K wires meet — KCL is enforced via a Lagrange-multiplier row per junction. Converges fast (~80 segments to NEC accuracy) AND to a finite reactance limit. C++ accelerator (`_accelerators.cpp`) handles the bottleneck quadrature and Z assembly on non-junction geometries; junction geometries take a generalised Python assembly path. Drives every interactive web-UI antenna: inverted V, Yagi (with N directors), moxon, hexbeam, and fan dipole.
- **`pysim.PySim`** (in `src/pysim/__init__.py`) — legacy pulse-basis MoM, kept only as a convergence comparator. Single straight wire, both `engine="python"` and `engine="accelerated"`. Converges slowly (the real part by ~N=1000–5000 segments, the imag part logarithmically *diverges* with N — see [docs/convergence_analysis.md](docs/convergence_analysis.md)).

The legacy `_legacy.py`, the spline experiments (`spline.py`, `bspline.py`, `augmented_spline.py`), the `icecream` dependency, the separate `pysim.yagi.YagiPySim` class, and the separate `BentTriangularPySim` in `triangular_bent.py` are all gone — the latter two were consolidated into `TriangularPySim`. `docs/convergence_analysis.md` documents the NEC validation campaign that motivated the triangular work.

## Done in recent PRs (don't redo)

- **PR #1** — branch infrastructure cleanup
- **PR #2** — `slow`/`plot` pytest markers, CI filter, NEC comparison scripts, [convergence analysis writeup](docs/convergence_analysis.md), port `YagiPySim` to the `(l_endpoints, r_endpoints)` interface
- **PR #3** — `TriangularPySim` v1 with full analytic static-kernel extraction; `scripts/bspline_probe.py`; smoke test asserting it matches NEC to a few Ω
- **PR #4** — Delete spline modules + drop `icecream` dep; wire `engine="accelerated"` into the new `PySim`; delete `_legacy.py`; silence headless-test `FigureCanvasAgg` warnings
- **PR #5** — `TriangularYagiPySim`: multi-wire triangular Galerkin solver (driver + reflector). Same-wire blocks reuse the analytic static-kernel extraction; cross-wire blocks use direct Gauss-Legendre quadrature on the full kernel. `scripts/compare_yagi_nec.py` updated to show both triangular solvers side-by-side. Matches NEC to ~0.1 Ω on R and X at N=160; resolves the "does the Yagi reactance converge to NEC?" question — it does.
- **PR #7** — Interactive inverted-V web UI (FastAPI + Vite/React/TS, under `web/`). WebSocket-driven live solve at sliders for droop angle, halfdriver factor, design freq, measurement freq, N. Smith chart overlay with a debounced ±30% sweep across measurement freq. Canvas shows the wire with current-magnitude color + per-arm `|I|` envelope, scaled by design wavelength with a λ/4 reference bar. No changes to the solver.
- **PR #8** — adds azimuth-plane (xy) far-field polar plot in the top-left of the stage, computed client-side from the segment currents already in the WebSocket response. Shows the figure-8 → fatter-peanut transition as droop closes the V.
- **PR #9** — 2-element Yagi (driver + reflector) added to the interactive UI: geometry tab switcher, per-Yagi controls (driver length factor, reflector length factor, spacing in λ), top-down xy canvas view so the beam axis lives in the far-field cut plane. F/B asymmetry now visible on the azimuth plot.
- **PR #10** — solver perf: dropped `BentTriangularPySim` default `n_qp_off=8 → 4` (free 2× on V, sub-0.02 Ω X error). Added `compute_impedance_swept(k_array)` to both triangular solvers; static kernel + R distances reused across the sweep, only `exp(-jk·R)` and einsum reductions carry a k axis. `/sweep` is ~2× faster, V `/sweep` is ~7× faster overall once combined with the `n_qp_off` change. New `scripts/profile_triangular.py` for future profiling.
- **PR #11** — first C++ accelerator for the triangular solvers: `seg_seg_quad_batch_3d` in `_accelerators.cpp` handles both V's off-edge and Yagi's cross-wire batched quadrature (distinguished by `a²` regularization). OpenMP `parallel for collapse(2)` over `(i, j)` pairs; per-pair `R[q, r]` table is built once and reused across the k axis. The cross-edge/cross-wire block drops from ~190 ms → 4 ms at N=80 (~47×); full V solve goes 250 → 55 ms (~4.6×); V `/sweep` at N=80 goes 3.1 s → 1.7 s (~1.8×). Output verified to ~1e-19 vs numpy. New bottleneck: `_seg_seg_reg_all_batch` (same-edge regularized, still pure numpy).
- **PR #12** — second C++ accelerator: `seg_seg_reg_quad_batch_1d` in `_accelerators.cpp` ports `_seg_seg_reg_all_batch` (same-edge regularized kernel `(exp(-jkR)-1)/(4πR)` on a shared 1D arc). Same `parallel for collapse(2)` + per-pair `R[q,r]` table pattern as PR #11; the kernel is the real-part `cos(kR)-1` plus imag `sin(kR)` directly to match numpy bit-for-bit. Kernel alone is **~14× faster** at typical sizes (numpy 537 → C++ 39 ms at N=80, n_k=41). End-to-end V `/sweep` at N=80, n_k=41 goes **2250 → 982 ms (~2.3×)**; N=40 sweep goes 390 → 211 ms (~1.85×). Output verified to ~1e-14 relative vs numpy. New bottlenecks after PR #12 (on the V `/sweep` workload): (1) numpy einsum / fancy-indexing matrix assembly (item 6e — `compute_impedance_swept` tottime is 488 ms at N=80, 2380 ms at N=160 — *the new dominant cost at N=160*); (2) `numpy.linalg.solve` on the batched `(n_k, M, M)` matrix (item 6d — 609 ms at N=80, 878 ms at N=160). The C++ kernels are now a combined 16% of sweep time.
- **PR #13** — third C++ accelerator: `assemble_Z` in `_accelerators.cpp` ports the scalar/vector-potential matrix assembly that combines J tensors + tangent dot products + `h` into the final `(n_k, n_basis, n_basis)` Z. OpenMP `parallel for collapse(2)` over `(k, m)`; inner `n` loop reads contiguous rows of the J tensors so the inner-loop accesses are sequential (`left_seg`, `right_seg` are simply `m` and `m+1` for V and within-wire for Yagi). Unified kernel — same C++ function serves both `BentTriangularPySim` (per-segment tangents, per-segment h) and `TriangularYagiPySim` (per-wire tangents converted to per-segment). Output matches numpy to ~1e-13 relative. Assembly time alone collapses from **488 → 15 ms at N=80 (~33×)** and **2380 → 36 ms at N=160 (~66×)**. End-to-end speedups: V `/sweep` N=80 goes **1043 → 736 ms (~1.4×)**; V `/sweep` N=160 goes **4207 → 2408 ms (~1.75×)**; Yagi `/sweep` N=80 goes 801 → 560 ms (~1.4×); Yagi `/sweep` N=160 goes 4229 → 3360 ms (~1.26×). After PR #13, **`numpy.linalg.solve` is the new dominant cost** (50% of V `/sweep` at N=80, **60% at N=160**) — item 6d is now the top remaining bottleneck. A smaller next-tier cost has emerged: `np.zeros_like` on the (n_k, N, N) J tensors (~145 ms / 7% at N=160), driven by the ~270 MB of memory it has to touch.
- **PR #14** — web UI freq band tightened from ±30% around 13.625 MHz to 0.8x–1.25x around 14.3 MHz; freq sliders narrowed to match the sweep band (and the measurement freq slider now derives min/max from designFreq so it tracks the sweep endpoints exactly). UI only — no solver changes.
- **PR #15** — PyNEC (NEC2 via [python-necpp](https://github.com/tmolteno/python-necpp)) added as an optional second backend for the web UI. Vendored as a submodule at `python-necpp/` (the PyPI wheel build is broken on Py 3.14; the submodule + `scripts/build_pynec.sh` produces a working install). New `web/pynec_backend.py` mirrors the pysim solve/sweep response shape for both inverted V and Yagi geometries; the server dispatches via a `solver` field on each request (`"pysim"` or `"pynec"`), and the frontend adds a tab in the simulation panel. At N=30 single-frequency solves: PyNEC ~0.8 ms vs pysim ~6 ms (~7.5×); at N=160 PyNEC ~13 ms vs pysim ~50 ms (~4×). Z agreement is ~1 Ω across both antennas. The animation rate isn't actually limited by the solver in either case — the bottleneck is WebSocket round-trip + render. Note: PyNEC's hand-rolled C++ LU solver doesn't link BLAS at all (`./configure --without-lapack`); patching upstream to call `zgetrf`/`zgetrs` is plausible future work that would help at large N.

- **PR #35** — Fan dipole geometry on the PyNEC backend. `_build_fandipole` / `solve_fandipole` produce a K-band cone arrangement (default 2 bands at 20m + 10m) with K arm wires per side meeting at a shared T→S feed gap; PyNEC handles the wire-endpoint junctions natively via shared segment coordinates. UI: new "Fan dipole" tab with sliders for n_bands (1–5), per-band length, cone slope, cone radius; meas-band tab strip parallel to design-band strip so the user can probe each band without retuning; meas-freq slider expanded to the full HF range for fan_dipole and sweep anchored to measFreq so the Smith plot follows the band; in-flight sweep aborts immediately on any slider move so the live /ws solve isn't starved by a stale streaming sweep. Side-view (yz) projection in the antenna canvas. Bug fix to `_segment_centers_to_knot_currents`: added `junction_at_start/end` flags so band-arm inner-knot currents carry the adjacent segment value instead of the open-wire zero — without it the envelope tapered to nothing at the feed-side junction.

- **PR #36** — Junction support in `TriangularPySim` + fan dipole on pysim. New `junctions=[[(wire_idx, "start"|"end"), ...], ...]` constructor arg. Each junction adds K directional tent bases (one per (wire, end) tuple) with the single active wing on the adjacent segment, level 1 at the junction node falling to 0 at the segment's other end. KCL Σ I_k = 0 is enforced symmetrically by a Lagrange-multiplier row per junction (+1 for "start", −1 for "end" matching the outflow sign), so no privileged reference wire is picked. The Z assembly is refactored around per-basis `(seg, L_left, R_right)` support arrays — the new general path handles arbitrary 2-wing layouts including the inactive-wing case for junction directionals. Non-junction geometries take the existing fast path with the C++ accelerator and are bit-exact regressions. `web/server.py` ships `_solve_fandipole` / `_sweep_fandipole` using TriangularPySim with the 2-segment-feed-wire trick (puts an interior knot at the midpoint of T→S so the delta-gap source has a place to live), and the "fan_dipole requires PyNEC" route is gone — the UI's solver-tab toggle works end-to-end. **Known**: 2-band fan dipole on pysim disagrees with PyNEC by ~14 Ω real at the 14.3 MHz design freq. K=2 junctions match the single-polyline equivalent to roundoff (verifying the formulation), but K=3 with sharp angles surfaces a gap — see new item 8 and item 12 in "What's left" for what's known about it and how to chase it.

- **PR #37** — Three improvements to the triangular solver's batched path, primarily targeting multi-band fan-dipole `/sweep` performance:
  1. **6g: `assemble_Z_general` C++ accelerator.** Ports `_assemble_Z_general_batch` (the per-basis `(support_seg, support_L, support_R)` 2-wing assembly used for junction geometries) to C++, mirroring PR #13's `parallel for collapse(2)` over `(k, m)`. Slope = (R − L) / h is precomputed once outside the parallel region; junction directional bases with one inactive wing (L=R=0) pass through the same kernel without a branch. Kernel: **5286 → 45 ms at n_per_wire=21, n_k=41 (~117×)**, exceeding the 33–66× estimate. Bit-exact with the pure-Python reference to 3.3e-16 relative.
  2. **6h: J-build dispatch fusion.** Replaces `_build_J_blocks(_batch)`'s ~150 small per-edge-pair Python dispatches with one all-pairs `seg_seg_quad_batch_3d` call, then per-edge overwrites the same-edge blocks with the analytic + regularized treatment. Same-edge pairs are computed twice (~5% redundant compute) but the dispatch savings dominate. `_build_J_blocks_batch` **593 → 224 ms (~2.65×)** at fan-dipole sizes.
  3. **6i: Schur-complement KCL solve.** Replaces `_solve_with_kcl_batch`'s `(n_k, n_b+n_c, n_b+n_c)` augmented-matrix construction with a Schur complement: solve `Z [w | X] = [v | Aᵀ]` (batched, 1 + n_c RHS), then `S λ = A w` (tiny n_c × n_c), `coeffs = w − X λ`. Avoids materializing the augmented (41, 425, 425) matrix entirely — that construction alone cost ~63 ms / 12% of /sweep. Numerically equivalent to ~6e-15 relative. Saves ~50 ms / ~7% e2e on fan-dipole. Also applied to single-k `_solve_with_kcl` for consistency.

  **Cumulative end-to-end speedup on 5-band fan-dipole `/sweep` (n=21, n_k=41): 6067 → ~750 ms (~8×)** across the three items. Inverted V and Yagi geometries (few edges, no junctions) are unaffected by 6h and 6i; only 6g touches their path, and they didn't use the general assembly anyway, so end-to-end is unchanged.

  After PR #37, the wall-time breakdown on the 5-band fan dipole `/sweep` is roughly: J build ~50%, batched solve (Schur) ~30%, assembly ~10%, glue ~10%. **J build is the dominant cost on all geometries**; see item 6b for the next remaining lever (vectorized libmvec → SLEEF/SVML for AVX-512 sincos).

  New regression tests: `test_assemble_Z_general_cpp_matches_python` (bit-exact C++ vs python on a K=3 junction) and `test_triangular_fandipole_swept_matches_per_freq` (batched-vs-per-freq for K=3, mirroring the existing K=2 test). The Schur-complement change is covered by the existing `test_triangular_k2_junction_*` and `test_triangular_fandipole_two_band_smoke` tests.

## What's left

Ordered by what I'd actually do next, not by what's most ambitious.

### High value, moderate effort

1. ~~**Multi-wire `TriangularYagiPySim`**~~ — done in PR #5. Driver impedance converges to NEC within ~0.1 Ω on R and X at N=160.

2. ~~**Add the triangular solver to `scripts/compare_yagi_nec.py`**~~ — done in PR #5. Both `TriangularPySim` and `TriangularYagiPySim` are now printed side-by-side with NEC for both dipole and Yagi cases.

3. ~~**Investigate observed `TriangularPySim` convergence rate**~~ — investigated in [scripts/triangular_convergence.py](scripts/triangular_convergence.py). Findings: hypothesis (a) [odd-N off-center source] is **wrong** — at any given N, odd-N actually produces a *smaller* absolute error than the adjacent even N (for `point_delta` it's strictly better on the real part, identical on imag). The log-log slope difference between even and odd just reflects different small-N transients, not different asymptotic rates: at N≥80 both parities converge at ~O(1/N^1.2). Hypothesis (b) [delta-gap source projection] is the real cap: switching to a finite-gap source (`finite_gap` in the convergence script) barely changes the rate, only the asymptotic limit, indicating the convergence cap is from the delta-gap *physics* (singular feed-point current), not from the projection method. Conclusion: the right next move for higher accuracy is a magnetic-frill / true finite-gap source — i.e., item 5.

### Medium value, larger effort

4. ~~**Bent wire / arbitrary geometry support for `TriangularPySim`**~~ — done as `BentTriangularPySim` in `src/pysim/triangular_bent.py`. Accepts an arbitrary 3D polyline. Same-edge segment pairs reuse the analytic static-kernel extraction from `TriangularPySim`; cross-edge pairs use wire-radius-regularized 3D Gauss-Legendre quadrature. Per-segment tangents enter the vector-potential assembly via per-sub-rectangle dot products. Validated against NEC for V-dipoles across α∈{0, 15, 30, 45, 60}° — sub-0.3 Ω agreement up to 30° bend, ~5% relative at 60°. *Multi-wire bent geometry (bent Yagi etc.) is a natural follow-up — the same `BentTriangularPySim` scaffolding generalizes by adding wire-boundary tracking like `TriangularYagiPySim`.*

5. ~~**Magnetic-frill or finite-gap feed model**~~ — **investigated and reverted**; null result documented here. Implementation lived briefly on the `magnetic-frill-feed` branch (commits `dbbc300` … reverted) and was a working Tsai/Burke-Poggio coaxial-feed model (`feed_model="magnetic_frill"`, `frill_outer_factor=2.3` for a ~50 Ω feed, matching NEC2's default). Source vector built by analytic DC-kernel projection of `E_z(z) = +V/(2 ln(b/a)) · [exp(−jkR1)/R1 − exp(−jkR2)/R2]` against the linear tent (asinh / sqrt antiderivatives — required because the `1/R` part is sharply peaked inside `|z| ≲ b ≈ 1 mm` while segments at typical N are ~65 mm, far too wide for moderate-nq Gauss-Legendre to resolve) plus a smooth-remainder GL quadrature for the k-dependent `(exp(−jkR)−1)/R` part. Sign was chosen so the b→a limit recovers the delta-gap convention v[m_center]=1; verified by `sum(v) = 1.0` to 1e-8 in the k→0 limit.

   **Why we reverted.** At pysim's typical discretizations the projected source vector spreads only ~1% of its mass onto each immediate neighbor of `m_center` and concentrates the rest there — and *after the matrix solve* this distinction in v washes out almost entirely:
   - Half-wave dipole, N=81 → `|Z_frill − Z_delta| < 0.01 Ω`
   - Short dipole (0.15 λ), N=161 → still ~1 Ω, while both differ from PyNEC by ~12 Ω
   - Convergence rate unchanged: both feed models track at ~O(1/N^0.77) toward PyNEC; the residual is the formulation gap (extended thin-wire kernel etc.), not the source
   - Cost when enabled: +30–120% on `compute_impedance` (Python loop over n_b × 2 wings dominates; ~30 ms / 45 ms `_build_source_vector` overhead at N=160 single / swept-41)

   **The valuable finding — preserved here.** The fan-dipole's ~14 Ω disagreement with PyNEC (item 8) is **not** the feed model. Both pysim feed models give nearly the same Z at the relevant N because h >> b, so the tent basis can't resolve the frill's width and the matrix-solved current responds the same way. This cheaply rules out one whole class of explanation and points to item 12 (cross-wire kernel regularization at the K=3 junction) as the next leverage point — the cone-angle sweep evidence already implicates it.

   **If we ever want it back.** The implementation is in git history at commit `dbbc300`. The frill matters in regimes pysim doesn't currently exercise: N > ~1000 where the delta-gap reactance starts to drift logarithmically (the legacy pulse-basis convergence-failure mode, milder for tent basis but still present), and very-short antennas where the feed-singularity-vs-finite-source distinction is larger relative to |Z|. Re-applying would also benefit from vectorizing the Python loop (the analytic-DC + GL-remainder math is fully vectorizable across bases — drop the per-basis Python dispatch to single-digit-ms overhead).

6. **C++ accelerator: remaining work after PR #37.** Phases 1, 2, 3, and 4 landed: cross-edge/cross-wire quadrature (PR #11), same-edge regularized quadrature (PR #12), Z matrix assembly fast path (PR #13), and Z matrix assembly general path + J-build fusion + Schur KCL solve for junction geometries (PR #37). The wall-time breakdown on the 5-band fan dipole `/sweep` is now roughly: J build ~50% (kernel-bound on AVX2 sincos), batched solve ~25%, assembly ~10%, glue ~15%. Items below are organized by what's actionable; deep micro-optimization of the C++ kernels (item 6b) is *blocked on this development machine* (i7-8550U, Kaby Lake R — AVX/AVX2 + FMA only, no AVX-512). If pysim deploys to a server with AVX-512 the calculus changes; for now the leverage is elsewhere.

   **Done:**

   - ~~**6g. assemble_Z for junction geometries.**~~ — done in PR #37. `_assemble_Z_general_batch` collapses from 5286 ms → 45 ms at n_per_wire=21, n_k=41 (~117×); end-to-end fan-dipole /sweep 6067 → 971 ms (~6.25×). Bit-exact with the pure-Python reference to 3.3e-16 relative.

   - ~~**6h. J-build batching for junction multi-edge geometries.**~~ — done in PR #37. `_build_J_blocks(_batch)` fused into one all-pairs `seg_seg_quad_batch_3d` call plus per-edge same-edge overwrites with the analytic+regularized treatment. Same-edge pairs are computed twice (the off-edge quadrature is wrong for them, then overwritten) but ~5% redundant compute is paid back many times over by eliminating ~150 small Python-side dispatches. `_build_J_blocks_batch` 593 → 224 ms (~2.65×); end-to-end fan-dipole /sweep 971 → 641 ms (~1.5×). Inverted V and Yagi (few edges, ~no dispatch overhead) unaffected.

   - ~~**6i. Schur-complement KCL solve.**~~ — done in PR #37. The KCL-augmented `(n_k, n_b+n_c, n_b+n_c)` matrix construction in `_solve_with_kcl_batch` was costing ~63 ms (12% of fan-dipole /sweep) just to allocate and write the augmented matrix M. Replaced with the Schur complement: solve `Z [w | X] = [v | A.T]` (batched, 1+n_c=3 RHS) then `S λ = A w` (tiny n_c × n_c per k), `coeffs = w − X λ`. Avoids materializing M entirely. Numerically equivalent (~6e-15 relative). Save ~50 ms / ~7% e2e on fan dipole.

   **Remaining work, re-ranked for AVX2-only hardware:**

   - **6d. Batched LU solve via direct LAPACK + OpenMP over k.** *Now the biggest actionable lever.* Post-PR-#37, `np.linalg.solve` is ~25% of fan-dipole /sweep (~180 ms) and ~18% on inverted V N=80 (~31 ms). VTune showed `blas_thread_server` accumulating ~48 thread-seconds — most of it OpenBLAS workers spinning while idle, contending with the C++ kernels' OpenMP threads for cores. The (n_k=41, M=423) shape is awkward for OpenBLAS: each k's `zgetrf`+`zgetrs` is small enough that the inner BLAS parallelism gives little benefit while the thread-pool overhead is large. A direct `zgetrf` + `zgetrs` from C++ with `parallel for` over k (and OPENBLAS_NUM_THREADS=1 inside the LAPACK calls to prevent nested threading) should give near-linear scaling on this shape. Estimated 2-3× on the solve, hence ~1.15-1.25× end-to-end. Implementation: add `solve_kcl_batched` to `_accelerators.cpp` calling LAPACK directly (link `-llapack` or use scipy.linalg.lapack which exposes the same routines without a build dependency).

   - **6j. Runtime BLAS thread cap.** *Easiest win, low effort.* VTune confirms that the default `OPENBLAS_NUM_THREADS=8` (or whatever it picks via openblas-pthreads autodetect) gives 8 OpenBLAS workers spinning to serve a few small solves while 8 OpenMP threads run the C++ kernels — pure contention on this 4-physical-core CPU. An OMP × BLAS thread sweep on 5-band fan dipole showed: defaults give ~825 ms, while `OMP=8 BLAS=4` gives ~711 ms (~14% faster), and `OMP=4 BLAS=1` (current shell default) gives ~812 ms. Best is `OMP=physical_cores BLAS=physical_cores/2` for our (n_k=41, M~400) regime. Could set this at import time in `pysim/__init__.py` (using `threadpoolctl`) or document as a deployment knob.

   - **6c. OpenMP scaling at higher N.** At N=80 the per-`(i, j)` work is small enough that OpenMP scheduling overhead dominates beyond ~4 threads (best at T=2–4 in measurements). At N=160+ this should self-correct since work per thread grows quadratically; revisit once we have an actual workload at that scale.

   - **6k. Optional single-precision (`float32`) pipeline.** Trades accuracy for 2× SIMD lane width — on AVX2 this is the *only* way to widen the math beyond what libmvec already does, since AVX-512 is blocked (item 6b). Tiered options, each strictly extending the previous:
     1. **Sincos only in float32** (cast in/out): ~12% e2e. Per-element cos/sin error ~1e-5 abs (was ~1e-15). J integrals ~2e-6 relative. Low risk.
     2. **Full float32 inner loop, double accumulators**: ~17% e2e. Cast overhead at accumulator boundary eats ~30% of the gain. J integrals still ~2e-6.
     3. **Full float32 J kernel (incl. accumulators)**: ~22-25% e2e. J integrals ~1e-6, final Z error ~0.1 Ω on 50 Ω antennas — same order as existing pysim-vs-NEC discretization error.
     4. **Also float32 in `np.linalg.solve` (`cgesv` instead of `zgesv`)**: another ~10-12% e2e (solve is currently 25% of /sweep, single-complex LU is ~2× faster).
     5. **Also float32 assembly (`assemble_Z_general`)**: another ~5% e2e (assembly is only 10% of /sweep).

     **Cumulative ceiling on AVX2: ~35-40% e2e** for the full-stack float32 mode. Existing tests would fail at every tier: `test_assemble_Z_general_cpp_matches_python` rtol=1e-12 (tier ≥1), `test_triangular_k2_junction_*` atol=1e-9 (tier ≥3), `test_triangular_fandipole_swept_matches_per_freq` atol=1e-6 (tier ≥3). Design implication: add `precision="float"` (or similar) opt-in flag, gate the C++ kernels on it, keep double as default. Existing tests stay; add float32 regression tests with relaxed tolerances (atol ~1e-3 on Z). Suited for the interactive web UI where the user can already see a ~0.1 Ω disagreement between pysim and PyNEC without complaint.

   **Blocked on this hardware:**

   - **6b. Vectorized `cexp(-jkR)` via SLEEF / Intel SVML for AVX-512 sincos.** *Blocked: i7-8550U has no AVX-512.* The C++ quadrature kernels already use libmvec's AVX2 `cos_avx2`/`sin_avx2` (4 doubles per inst), which is the widest sincos this CPU supports. AVX-512 (8 doubles per inst) would notionally give ~1.5-2× on the sincos portion (~22% of J build, ~10% of /sweep), but only on Skylake-X / Ice Lake / Tiger Lake / newer server parts. If pysim deploys to such hardware, this becomes the top lever; on Kaby Lake there is nothing to gain. Smaller AVX2 micro-opts (fused sincos via SLEEF — libmvec lacks vector sincos so cos and sin are two passes; software pipelining the kernel loop) are plausible but add a build dependency for an estimated <10% kernel speedup.

   **Retired:**

   - ~~**6f. `np.zeros_like` on the J tensors.**~~ — obsolete after PR #37. The fused J-build no longer pre-allocates J tensors in Python (the C++ kernel allocates them directly as its output).

   The natural API target remains the *batched* form: each kernel function takes `k_array` and produces `(n_k, N, N)` output. The Python paths should remain as the reference / fallback for platforms without OpenMP. Existing `seg_seg_quad_batch_3d`, `seg_seg_reg_quad_batch_1d`, `assemble_Z`, and `assemble_Z_general` are the pybind11 build templates.

### Validation

7. **Coverage for non-default geometries** — sweep `wavelength`, `halfdriver_factor`, `wire_radius`, verify `TriangularPySim` against NEC. Currently only the default (0.481 λ) dipole has been validated end-to-end.

8. ~~**Third-reference validation**~~ — **resolved with a refinement**. The 2-band fan-dipole disagreement between `TriangularPySim` and PyNEC originally looked like ~14 Ω on R AND ~15 Ω on X, but the X gap was a **geometry artifact**: the old `_FANDIPOLE_RING_5` constant placed K=2 bands at lopsided pentagon positions (36° and 108°, only 72° apart on one side of the cone) rather than at the natural opposite ends of a diameter. After replacing the static prefix with `_fandipole_ring(K)` that distributes K bands evenly at 360°/K (`fandipole-even-ring` branch), the X gap collapses to ~1.5 Ω across N and the only remaining axis is R.

   `scripts/compare_fandipole_solvers.py` 3-way comparison at n_bands=2 (the K=3 junction case), 14.3 MHz design freq, with the corrected even-distribution ring:

   ```
     N |    pysim (R + jX)    |    PyNEC (R + jX)    | pymininec (R + jX)
    21 |   +58.9   -5.3j      |   +51.6   -3.9j      |   +58.8   -5.8j
    41 |   +59.0   -4.9j      |   +49.1   -3.5j      |   +58.5  -11.8j
    81 |   +59.1   -4.6j      |   +46.8   -3.1j      |   +56.9  -34.9j
   ```

   pysim and pymininec — two independently-implemented MoM solvers in different basis families (triangular Galerkin vs pulse) — **agree on R to ~2 Ω across N**. PyNEC's R drifts downward with N (51.6 → 46.8, a further 4.8 Ω drift from N=21→81). The gap to pysim/pymininec is ~7 Ω at N=21 and ~12 Ω at N=81. The X gap is now small in all three solvers. pymininec's X diverges with N as documented (pulse-basis convergence-failure mode), so only its R is useful — R agreement with pysim across N is the load-bearing evidence.

   **Decision**: accept the remaining R disagreement as a known NEC2-vs-others formulation gap and move on. Pysim is not the outlier — NEC2 is. The literature attributes this to NEC2's source-at-K-wire-junction handling and the thin-wire-kernel choice (these caveats apply independent of wire diameter, which is uniform in our model — that ruled out the dissimilar-diameter literature concerns). NEC4 (option a) would be a fourth datapoint if the question resurfaces but is not urgent.

   **Implication for the web UI**: the "solver agreement diagnostic" idea in item 10 should now treat fan-dipole pysim/PyNEC R disagreement as expected (and *growing* with K and N), not a bug indicator. The X agreement is the new baseline.

### Interactive UI follow-ups

10. **UI follow-ups** — Yagi (with N directors), moxon, hexbeam, and fan dipole all ship in the interactive UI. Open work:
    - **Solver overlay on Smith plot**: currently the pysim/pynec tab toggle replaces the displayed solve; an overlay drawing *both* points/sweeps on the Smith chart simultaneously would make solver-disagreement geometries (fan dipole, see PR #36) visually obvious without manual A/B.
    - **Fan dipole 3D rotation**: the side-view (yz plane) collapses x-axis cone variation. The 5-band default has visible x-spread that the projection drops. A simple azimuth-rotation control or isometric view would surface it.
    - **Solver agreement diagnostic**: when pysim and pynec disagree by more than some threshold (say 5% of |Z|), surface a small indicator in the UI noting "solvers differ by X Ω — see geometry-specific known-issue".

11. **Far-field pattern enhancements** (defers from the `far-field-pattern` branch — keep all three; the first cut only shows the xy plane in linear scale, per-frame normalized):
    - **Second cut** — add an elevation slice (yz or xz plane) or a 2D `(θ, φ)` heatmap so the take-off-angle change with droop is also visible.
    - **dB radial axis** with a fixed dynamic range (e.g. 30 dB) — linear hides shallow nulls; dB shows the depth of deep nulls and is the standard radiation-pattern convention.
    - **Absolute directivity (dBi)** — integrate `|E|²` over the sphere for total radiated power, normalize so the radial axis is gain over isotropic. Lets the user compare antennas across geometries (currently per-frame normalization hides "is this 2 dBi or 8 dBi").

### Open research

9. **Higher-order basis functions** — triangular is degree-1 B-spline. Degree-2 (quadratic) or degree-3 (cubic) B-splines should give O(1/N³) or O(1/N⁴) convergence. The scipy `BSpline` machinery handles arbitrary degree; the analytic static-kernel integrals get more terms but the structure is the same (more antiderivatives in `asinh`/`√`).

12. ~~**Cross-wire kernel regularization for close-fanning junctions**~~ — **investigated and ruled out on the `per-pair-kernel-reg` branch**; the regularization is not the cause of the fan-dipole pysim/PyNEC gap. Two probes:

    **Probe 1 (sensitivity)**: hacked the cross-wire block of `_build_J_blocks` to use an `a_xw = factor · a` regularization for cross-wire pairs only, leaving same-wire-different-edge (kink) pairs at `a²`. Swept `factor` over 4 orders of magnitude on the 2-band fan dipole (K=3 junctions at S and T):

    ```
      factor    a_xw (mm)      R         X      |Z-PyNEC|
      1.0000     0.50000     63.40     15.27       22.68
      0.0010     0.00050     63.40     15.27       22.68
    ```

    Per-pair J-matrix entries change by ~0.5% (verified at the junction-adjacent pair), but **the impedance changes by < 0.001 Ω**. The cross-wire regularization is irrelevant to Z for this geometry.

    **Probe 2 (junction multiplicity)**: ran the K=2 case (single 20m band, only K=2 junctions at S and T, no close-fanning K≥3 geometry):

    ```
                       R         X
       K=1 single   pysim: 63.4 + j16.3,  PyNEC: 66.1 + j 1.0  →  ΔX = +15.3
       K=2 double   pysim: 63.4 + j15.3,  PyNEC: 46.3 + j 0.4  →  ΔX = +14.9
       K=3 triple   pysim: 63.5 + j14.5,  PyNEC: 41.5 - j 0.1  →  ΔX = +14.6
    ```

    The ~15 Ω ΔX is **constant across K**. K=1 has no K≥3 junction at all and still shows the same gap. The disagreement is not about junction multiplicity.

    Combined with item 8's pysim-vs-pymininec agreement on R, the conclusion: all three sub-options (per-pair regularization, adaptive junction meshing, sinusoidal-segment basis at junction nodes) were targeting K≥3 junction effects that don't exist as the dominant cause. The dominant effect is NEC2's formulation choices, not anything in pysim's local junction treatment. **Item closed.**

    **Postscript (`fandipole-even-ring` branch, item 8 update)**: after fixing the lopsided pentagon `_FANDIPOLE_RING_5` to evenly distribute K bands at 360°/K, the ~15 Ω X-part of what was being called "the fan-dipole disagreement" turned out to be a *geometry* artifact (lopsided ring) that had been incorrectly attributed to junction/formulation effects. The original PR #36 cone-angle sweep (~7 Ω tracking inter-arm angle) was also partly contaminated by the same ring asymmetry — when n_bands varied while still using the pentagon prefix, the inter-arm angle changes mixed with ring-position bias. The remaining real-part disagreement (~5–17 Ω growing with K and N) is what's left after that contamination is removed; it sits in the same family as item 8's "NEC2 outlier" conclusion. No new actions — item stays closed.

## Key locations

- `src/pysim/triangular.py` — the active solver. `_build_geometry` builds the per-basis support arrays (segments, level-at-left, level-at-right); `_add_junction_bases` (PR #36) appends K directional bases per junction and the KCL constraint matrix `kcl_A`; `_assemble_Z_single` is the fast path used for non-junction geometries (calls the C++ `assemble_Z` accelerator), `_assemble_Z_general_single` is the general path used when junctions exist. The Lagrange-augmented solve lives in `_solve_with_kcl` (single) and `_solve_with_kcl_batch` (swept).
- `web/server.py` — geometry-specific builders (`_solve_inverted_v`, `_solve_yagi`, `_solve_moxon`, `_solve_hexbeam`, `_solve_fandipole`) and their `_sweep_*` counterparts. `_fandipole_geometry` shows the K-band cone with junctions and the feed-wire-as-2-segments trick that puts the delta-gap source on an interior knot.
- `web/pynec_backend.py` — drop-in PyNEC backend mirroring the server's response shape. Useful as a comparator (the UI's solver-tab toggle picks between them).
- `scripts/compare_yagi_nec.py` — the NEC validation harness (single dipole and 2-element Yagi in free space). Requires PyNEC (build via `scripts/build_pynec.sh` after `git submodule update --init --recursive`):
  ```
  .venv/bin/python scripts/compare_yagi_nec.py
  ```
- `scripts/bspline_probe.py` — demonstrates scipy `BSpline` operations (design matrix, derivative, mass matrix vs analytic tent formulas). Reference for any future basis-function work.
- `docs/convergence_analysis.md` — full writeup of the pulse-basis convergence failure that motivated the triangular work. Reference for "why are we doing this."

## Conventions to know

- Always work in a branch; the repo uses rebase-merge so each branch commit lands on `main` verbatim
- CI runs `pytest tests/ -m 'not slow and not plot'` — anything marked `@pytest.mark.slow` or `@pytest.mark.plot` is dev-only
- Default global instructions are in `~/.claude/CLAUDE.md`; project-specific conventions (none currently) would go in `CLAUDE.md` at the repo root
