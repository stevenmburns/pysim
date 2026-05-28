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

## What's left

Ordered by what I'd actually do next, not by what's most ambitious.

### High value, moderate effort

1. ~~**Multi-wire `TriangularYagiPySim`**~~ — done in PR #5. Driver impedance converges to NEC within ~0.1 Ω on R and X at N=160.

2. ~~**Add the triangular solver to `scripts/compare_yagi_nec.py`**~~ — done in PR #5. Both `TriangularPySim` and `TriangularYagiPySim` are now printed side-by-side with NEC for both dipole and Yagi cases.

3. ~~**Investigate observed `TriangularPySim` convergence rate**~~ — investigated in [scripts/triangular_convergence.py](scripts/triangular_convergence.py). Findings: hypothesis (a) [odd-N off-center source] is **wrong** — at any given N, odd-N actually produces a *smaller* absolute error than the adjacent even N (for `point_delta` it's strictly better on the real part, identical on imag). The log-log slope difference between even and odd just reflects different small-N transients, not different asymptotic rates: at N≥80 both parities converge at ~O(1/N^1.2). Hypothesis (b) [delta-gap source projection] is the real cap: switching to a finite-gap source (`finite_gap` in the convergence script) barely changes the rate, only the asymptotic limit, indicating the convergence cap is from the delta-gap *physics* (singular feed-point current), not from the projection method. Conclusion: the right next move for higher accuracy is a magnetic-frill / true finite-gap source — i.e., item 5.

### Medium value, larger effort

4. ~~**Bent wire / arbitrary geometry support for `TriangularPySim`**~~ — done as `BentTriangularPySim` in `src/pysim/triangular_bent.py`. Accepts an arbitrary 3D polyline. Same-edge segment pairs reuse the analytic static-kernel extraction from `TriangularPySim`; cross-edge pairs use wire-radius-regularized 3D Gauss-Legendre quadrature. Per-segment tangents enter the vector-potential assembly via per-sub-rectangle dot products. Validated against NEC for V-dipoles across α∈{0, 15, 30, 45, 60}° — sub-0.3 Ω agreement up to 30° bend, ~5% relative at 60°. *Multi-wire bent geometry (bent Yagi etc.) is a natural follow-up — the same `BentTriangularPySim` scaffolding generalizes by adding wire-boundary tracking like `TriangularYagiPySim`.*

5. **Magnetic-frill or finite-gap feed model** — current code uses a true delta-gap (`v[m_center] = 1.0`). A finite-gap or magnetic-frill source is more physical and will give different (more accurate) reactance for short antennas. Affects the source-vector construction only; the matrix is unchanged. **Promoted from medium- to high-priority by the item 3 investigation**: the convergence cap of `TriangularPySim` at ~O(1/N^1.2) is set by the delta-gap source, not by the basis function — the basis is already paying for higher-order accuracy that the source model is wasting.

6. **C++ accelerator for the triangular moment integrals** — updated by PR #10's profiling. The dominant cost is now clearly the cross-edge / cross-wire Gauss-Legendre quadrature (`_seg_seg_offedge_quad_batch`, `_seg_seg_cross_quad_batch`, and the same-edge `_seg_seg_reg_all_batch`). At N=80 these are ~80% of total solve time; the einsum reduction and the `exp(-jk·R)` evaluation both run single-threaded inside numpy.

   Three layers of available speedup, none of which numpy currently provides:
   - **OpenMP parallelism** over the outer `(i, j)` segment-pair indices — embarrassingly parallel; ~4–6× on 8 cores.
   - **Vectorized `cexp(-jkR)`** via SLEEF or Intel SVML — libm's complex exp is the inner-loop bottleneck; vectorized versions are 2–4× faster on bulk inputs.
   - **Cache-friendly memory layout** for the 5D `G` tensor — contiguous over the innermost quadrature axis, then over k. NumPy's stride pattern is fine but not tuned.

   Realistic combined gain: 4–10× over current numpy on solver compute. The natural API target is the *batched* form added by PR #10: each kernel function takes `k_array` and produces `(n_k, N, N)` output, so the C++ side gets a tight loop with all the parallelism already exposed at the outer axis. The existing pulse-basis `psi_fusion_trapezoid` accelerator (`src/pysim/_accelerators.cpp`) is the pybind11 build template to mirror.

   Order of attack (cheapest gain first):
   - Start with `_seg_seg_offedge_quad_batch` — it's the biggest line item for V and the cleanest function shape.
   - Then `_seg_seg_cross_quad_batch` (Yagi cross-wire) — same kernel structure, smaller relative impact.
   - Then `_seg_seg_reg_all_batch` — same-edge regularized; has a slightly more complex layout (segment-pair points share quadrature points along the wire) but the same OpenMP + cexp story.

   When this lands, the Python paths should remain as the reference / fallback (CI without compiled extensions); use the same `engine="python" | "accelerated"` switch pattern as the original `PySim`.

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
