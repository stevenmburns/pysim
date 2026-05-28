# Next steps — pysim

Living roadmap of what's done and what's left. Updated as work lands.

## Where we are

The codebase currently has three antenna solver classes:

- **`pysim.PySim`** (in `src/pysim/__init__.py`) — pulse-basis MoM, single straight wire, both `engine="python"` and `engine="accelerated"` (C++ via `psi_fusion_trapezoid`). Converges slowly (the real part by ~N=1000–5000 segments, the imag part logarithmically *diverges* with N — see [docs/convergence_analysis.md](docs/convergence_analysis.md)).
- **`pysim.yagi.YagiPySim`** (in `src/pysim/yagi.py`) — same pulse-basis MoM as `PySim`, but two parallel wires (driver + reflector) using the multi-wire stencil that handles wire-boundary non-adjacency. Python only.
- **`pysim.triangular.TriangularPySim`** (in `src/pysim/triangular.py`) — piecewise-linear (tent) basis with Galerkin testing and analytic singularity extraction. Single straight wire only. Converges fast (~80 segments to NEC accuracy) AND to a finite reactance limit. Python only.

The legacy `_legacy.py`, the spline experiments (`spline.py`, `bspline.py`, `augmented_spline.py`), and the `icecream` dependency are all gone. `docs/convergence_analysis.md` documents the NEC validation campaign that motivated the triangular work.

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

## What's left

Ordered by what I'd actually do next, not by what's most ambitious.

### High value, moderate effort

1. ~~**Multi-wire `TriangularYagiPySim`**~~ — done in PR #5. Driver impedance converges to NEC within ~0.1 Ω on R and X at N=160.

2. ~~**Add the triangular solver to `scripts/compare_yagi_nec.py`**~~ — done in PR #5. Both `TriangularPySim` and `TriangularYagiPySim` are now printed side-by-side with NEC for both dipole and Yagi cases.

3. ~~**Investigate observed `TriangularPySim` convergence rate**~~ — investigated in [scripts/triangular_convergence.py](scripts/triangular_convergence.py). Findings: hypothesis (a) [odd-N off-center source] is **wrong** — at any given N, odd-N actually produces a *smaller* absolute error than the adjacent even N (for `point_delta` it's strictly better on the real part, identical on imag). The log-log slope difference between even and odd just reflects different small-N transients, not different asymptotic rates: at N≥80 both parities converge at ~O(1/N^1.2). Hypothesis (b) [delta-gap source projection] is the real cap: switching to a finite-gap source (`finite_gap` in the convergence script) barely changes the rate, only the asymptotic limit, indicating the convergence cap is from the delta-gap *physics* (singular feed-point current), not from the projection method. Conclusion: the right next move for higher accuracy is a magnetic-frill / true finite-gap source — i.e., item 5.

### Medium value, larger effort

4. ~~**Bent wire / arbitrary geometry support for `TriangularPySim`**~~ — done as `BentTriangularPySim` in `src/pysim/triangular_bent.py`. Accepts an arbitrary 3D polyline. Same-edge segment pairs reuse the analytic static-kernel extraction from `TriangularPySim`; cross-edge pairs use wire-radius-regularized 3D Gauss-Legendre quadrature. Per-segment tangents enter the vector-potential assembly via per-sub-rectangle dot products. Validated against NEC for V-dipoles across α∈{0, 15, 30, 45, 60}° — sub-0.3 Ω agreement up to 30° bend, ~5% relative at 60°. *Multi-wire bent geometry (bent Yagi etc.) is a natural follow-up — the same `BentTriangularPySim` scaffolding generalizes by adding wire-boundary tracking like `TriangularYagiPySim`.*

5. **Magnetic-frill or finite-gap feed model** — current code uses a true delta-gap (`v[m_center] = 1.0`). A finite-gap or magnetic-frill source is more physical and will give different (more accurate) reactance for short antennas. Affects the source-vector construction only; the matrix is unchanged. **Promoted from medium- to high-priority by the item 3 investigation**: the convergence cap of `TriangularPySim` at ~O(1/N^1.2) is set by the delta-gap source, not by the basis function — the basis is already paying for higher-order accuracy that the source model is wasting.

6. **C++ accelerator: remaining work after PR #11.** Phase 1 (off-edge + cross-wire) landed; the cross-edge/cross-wire block is now ~4 ms at N=80, basically free. Three remaining wins, ranked by current Amdahl share:

   - **6a. Port `_seg_seg_reg_all_batch` (same-edge regularized) to C++.** This is now the dominant cost: ~33 ms of a 55 ms V solve at N=80 (60%). Kernel shape is similar to `seg_seg_quad_batch_3d` but with a 1D arc-length geometry (all quadrature points on a shared line) and the kernel is `(exp(-jkR) - 1) / (4πR)` instead of `exp(-jkR) / (4πR)`. The R values are k-independent and can be tabulated for all `(q_i, q_j)` pairs once. OpenMP parallelizes over the (i, j) basis pairs the same way. Estimated 1.5–2× further on full solve (or higher for sweep where the R table is amortized across k).

   - **6b. Vectorized `cexp(-jkR)` via SLEEF or Intel SVML.** The C++ kernel currently calls `std::cos`/`std::sin` from libm one element at a time. Replacing with vectorized AVX2/AVX-512 sincos (4 or 8 elements per instruction) is realistically another 2–3× on the kernel itself. Adds a build dependency. Probably defer until 6a is done — both wins compound.

   - **6c. OpenMP scaling at higher N.** At N=80 the per-`(i, j)` work is small enough that OpenMP scheduling overhead dominates beyond ~4 threads (best at T=2–4 in measurements). At N=160+ this should self-correct since work per thread grows quadratically; revisit once we have an actual workload at that scale.

   The natural API target remains the *batched* form: each kernel function takes `k_array` and produces `(n_k, N, N)` output. The Python paths should remain as the reference / fallback for platforms without OpenMP. Existing pulse-basis `psi_fusion_trapezoid` is the pybind11 build template.

   **Beyond the kernels**, two adjacent perf items that may matter once 6a lands:

   - **6d. Batched LU solve.** `compute_impedance_swept` calls `np.linalg.solve(Z, v)` on a `(n_k, M, M)` matrix — already batched, but at N=160 the cubic solve dominates (~45 ms per k at N=160 single, scales O(N³)). Look at whether a per-thread LU factorize-then-solve loop (instead of numpy's stacked LAPACK call) beats it for our (n_k=41, M~160-320) size class.

   - **6e. Matrix assembly.** The einsum / fancy-indexing block that combines J tensors into Z is ~14 ms at N=80 in the new profile. Likely fine as numpy code; only worth touching if it becomes >30% of total after 6a.

### Validation

7. **Coverage for non-default geometries** — sweep `wavelength`, `halfdriver_factor`, `wire_radius`, verify `TriangularPySim` against NEC. Currently only the default (0.481 λ) dipole has been validated end-to-end.

8. **Test against measurements or a third reference** — NEC is a model, not ground truth. Comparison to a published measurement or to `nec4`/MININEC would be more convincing than NEC2 alone. Lower priority; useful as a sanity check.

### Interactive UI follow-ups

10. **Multi-wire UI: Yagi, then bent-wire arrays (hexbeam etc.)** — the current `web/` UI only drives `BentTriangularPySim` (one bent wire). Two natural extensions:
    - **Yagi UI**: add driver + reflector (and optionally director) using `TriangularYagiPySim`, which already exists. Frontend needs a wire-list data model and per-wire controls (length, spacing). Solver is unchanged.
    - **Hexbeam UI**: 6 bent elements arranged in a hex. Requires a new solver class first — combine `BentTriangularPySim`'s polyline support with `TriangularYagiPySim`'s wire-boundary tracking. Per item 4's closing note, this is "natural" but unbuilt.

11. **Far-field pattern enhancements** (defers from the `far-field-pattern` branch — keep all three; the first cut only shows the xy plane in linear scale, per-frame normalized):
    - **Second cut** — add an elevation slice (yz or xz plane) or a 2D `(θ, φ)` heatmap so the take-off-angle change with droop is also visible.
    - **dB radial axis** with a fixed dynamic range (e.g. 30 dB) — linear hides shallow nulls; dB shows the depth of deep nulls and is the standard radiation-pattern convention.
    - **Absolute directivity (dBi)** — integrate `|E|²` over the sphere for total radiated power, normalize so the radial axis is gain over isotropic. Lets the user compare antennas across geometries (currently per-frame normalization hides "is this 2 dBi or 8 dBi").

### Open research

9. **Higher-order basis functions** — triangular is degree-1 B-spline. Degree-2 (quadratic) or degree-3 (cubic) B-splines should give O(1/N³) or O(1/N⁴) convergence. The scipy `BSpline` machinery handles arbitrary degree; the analytic static-kernel integrals get more terms but the structure is the same (more antiderivatives in `asinh`/`√`).

## Key locations

- `src/pysim/triangular.py` — the new solver. Look at `_J_static_all` (the closed-form moment integrals) and `compute_impedance` (the assembly).
- `src/pysim/yagi.py` — the multi-wire pattern. The `node_l_idx`/`node_r_idx` arrays in `compute_impedance` are the trick for handling non-adjacent wire boundaries; this should port to the triangular solver directly.
- `scripts/compare_yagi_nec.py` — the NEC validation harness (single dipole and 2-element Yagi in free space). Requires PyNEC from antenna_designer's venv:
  ```
  PYTHONPATH=/home/smburns/antennas/pysim/src \
      /home/smburns/antennas/antenna_designer/.venv/bin/python \
      scripts/compare_yagi_nec.py
  ```
- `scripts/bspline_probe.py` — demonstrates scipy `BSpline` operations (design matrix, derivative, mass matrix vs analytic tent formulas). Reference for any future basis-function work.
- `docs/convergence_analysis.md` — full writeup of the pulse-basis convergence failure that motivated the triangular work. Reference for "why are we doing this."

## Conventions to know

- Always work in a branch; the repo uses rebase-merge so each branch commit lands on `main` verbatim
- CI runs `pytest tests/ -m 'not slow and not plot'` — anything marked `@pytest.mark.slow` or `@pytest.mark.plot` is dev-only
- Default global instructions are in `~/.claude/CLAUDE.md`; project-specific conventions (none currently) would go in `CLAUDE.md` at the repo root
