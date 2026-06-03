# Next steps — pysim

Living roadmap of what's done and what's left. Updated as work lands.

## Where we are

The codebase has one active solver, two optional comparators (a NEC-faithful and a higher-order-basis arbiter), and one legacy comparator:

- **`pysim.triangular.TriangularPySim`** (in `src/pysim/triangular.py`) — piecewise-linear (tent) basis with Galerkin testing and analytic singularity extraction. Accepts arbitrary 3D polylines, multiple wires, an optional PEC-image ground plane, and (PR #36) wire-endpoint junctions where K wires meet — KCL is enforced via a Lagrange-multiplier row per junction. Converges fast (~80 segments to NEC accuracy) AND to a finite reactance limit. C++ accelerator (`_accelerators.cpp`) handles the bottleneck quadrature and Z assembly on non-junction geometries; junction geometries take a generalised Python assembly path. Drives every interactive web-UI antenna: inverted V, Yagi (with N directors), moxon, hexbeam, and fan dipole.
- **`pysim.sinusoidal.SinusoidalPySim`** (in `src/pysim/sinusoidal.py`, PR #44) — NEC2's three-term (const + sin + cos per segment) basis with closed-form Wu-King coefficients (Eqs 43-64 of the LLNL theory manual). Thin-wire kernel, delta-gap applied-E source, uniform wire radius, free space only. Same constructor surface as `TriangularPySim` (wires / `n_per_edge_per_wire` / junctions / `feed_wire_index` / `feed_arclength`). Used as the in-codebase NEC-faithful comparator — reproduces PyNEC R+jX to ~0.05 Ω on the hentenna across n=15..281 (see item 14).
- **`pysim.bspline.BSplinePySim`** (in `src/pysim/bspline.py`, PR #45) — degree-d B-spline Galerkin MoM (currently d ∈ {1, 2}). Polynomial-per-segment basis with the same multi-wire / polyline / junction / KCL machinery as `TriangularPySim`. Closed-form same-edge static-kernel moment integrals J_pq for p, q ∈ {0..2} sympy-derived once and dumped to `_bspline_static_moments.py` — no runtime sympy dependency. Used as the in-codebase higher-order-basis arbiter: a polynomial basis family completely independent from `TriangularPySim`'s tent basis. d=2 on the hentenna lands at the same 43.05 + j38.84 as `TriangularPySim` (see item 9/13/14 — this settles the arbitration). Pure-numpy (no C++ accelerator yet); free space only.
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

- **PR #44** — `SinusoidalPySim`: from-scratch NEC2 three-term basis solver alongside `TriangularPySim` (default unchanged). Built from the LLNL theory manual (`docs/nec2_theory_manual.pdf`, design memo in `docs/sinusoidal_basis_design.md`), not from the necpp source. Scope: free space, thin-wire kernel, delta-gap, uniform radius, X_i=0 free-end. Straight-dipole result: 69.63 − j18.26 at N=101 vs NEC2 reference 69.64 − j18.21. Hentenna params_50: reproduces PyNEC R+jX to ~0.05 Ω across n=15..281, establishing that the hentenna X drift is intrinsic to the three-term basis (see item 14 for the full sweep and what it does/doesn't settle).

- **PR #45** — `BSplinePySim`: degree-d B-spline Galerkin MoM solver (currently d ∈ {1, 2}) alongside `TriangularPySim` and `SinusoidalPySim`. Polynomial-per-segment basis with the same multi-wire / polyline / K-wire-junction / KCL machinery as `TriangularPySim`. Closed-form same-edge static-kernel moment integrals J_pq for p, q ∈ {0..2} sympy-derived once and dumped to `_bspline_static_moments.py` (no runtime sympy dependency; re-run `scripts/derive_bspline_static_moments.py` with a larger `MAX_D` to extend to higher degrees). Same-edge moments split as analytic static + GL-quadrature smooth-kernel — essential because direct GL converges only logarithmically on the 1/R diagonal. Scope: free space, thin-wire kernel, delta-gap, uniform radius, no C++ accelerator yet (pure-numpy J build and Z assembly). **Hentenna params_50 arbitration result (this is the load-bearing finding)**: at every n probed (15, 21, 41, 81), `BSplinePySim(degree=2)` lands at 43.05 + j38.84 ± 0.01 Ω — *the same Z as `TriangularPySim`* (~43.13 + j38.79 at n=81), with the d=2 basis actually converging *faster* (already at j38.85 by n=15 vs d=1's j37.14). Two independent basis families (d=1 tent, d=2 quadratic — different polynomial degrees, different code paths, different polynomial-moment integrals) converge to the same value. **Settles items 9, 13, 14**: pysim's tent basis is NOT converged-to-the-wrong-place; NEC2's three-term basis is the outlier that drifts super-logarithmically on this geometry. Verified on dipole (69.64 - j18.17 at N=81 vs NEC ref 69.64 - j18.21), 2-element Yagi (agrees with `TriangularPySim` ~0.2 Ω at N=21), and 30° V-dipole polyline (agrees ~0.3 Ω at N=40). New regression test `test_bspline_d2_hentenna_arbitrates_against_triangular` pins the value and the cross-basis agreement.

- **PR #46** — Templated C++ accelerators for `BSplinePySim`: J kernel + static moments + Z assembly. Pure-numpy reference path retained as fall-back; bit-exact regression tests cover both. Brings B-spline single-k solve into the same speed class as Triangular's accelerated path.

- **PR #47** — Singular-basis enrichment at K≥3 junctions for `BSplinePySim`: opt-in `use_singular_enrichment=True` kwarg adds a Φ_sing(u) = (u/h)·log(u/h) basis on each segment adjacent to a K≥enrichment_min_k=3 junction. The basis vanishes at the junction node (matching the finite-current condition) while its derivative has the log singularity that captures the classical K≥3 junction charge-density behaviour. On hentenna-class geometries this flips the R-rate from O(1/N) → ~O(1/N^(d+1)) (basis-limited). C++ `assemble_Z_enrich` accelerator handles the augmented (Z_pp, Z_pe, Z_ep, Z_ee) assembly.

- **PR #48** — Smoothed delta-gap source for `BSplinePySim`: opt-in `feed_smoothing_factor=α` replaces V·δ(s − s_f) with V·g_w(s − s_f) where g_w is a cos² bump of integral 1 and half-width w/2 = α·h_feed/2. Galerkin RHS becomes ⟨Φ_m, g_w(. − s_f)⟩, computed by Gauss-Legendre on the bump's support. Eliminates the log-singular feed-point current that caps delta-gap convergence at O(1/N) regardless of basis degree — smoothed-source convergence is basis-limited (O(1/N³) for d=2). Helps dipoles; orthogonal to PR #47's K≥3 enrichment (both are opt-in independent fixes for different convergence caps).

- **PR #49** — Multi-backend solver UI. Three configurable slots A/B/C on the controls panel, each carrying its own backend choice (Triangular / Sinusoidal / B-spline / PyNEC) + segments/wire + wire radius + model-specific knobs (n_qp_reg/n_qp_off, n_qp_const, degree, n_qp_pair, feed_smoothing_factor, use_singular_enrichment, n_qp_sing, enrichment_min_k, n_qp_source). Per-slot gear menu opens a modal with the backend picker plus all options. Defaults A=Triangular N=40, B=B-spline d=2 N=21 with K≥3 enrichment, C=PyNEC N=41. Server dispatches via `pysim_model` + `model_options` filtered through per-model allowlists; `currents_at_knots(coeffs)` added to all three pysim model classes so the web response carries per-knot complex currents regardless of basis convention. Sinusoidal and B-spline get a `compute_impedance_swept` loop fallback so `/sweep` works for all three.

- **PR #50** — PEC image-method ground for `BSplinePySim` and `SinusoidalPySim`, mirroring the existing TriangularPySim implementation. B-spline: image segments mirrored across z=ground_z, image-tangent dot product applied to flip the horizontal current direction, image J integrals via the standard off-edge full-kernel quadrature, image Z block subtracted from free-space Z; `_assemble_Z` grew an optional td_all override so the same assembly fuses the image-current sign flip. Sinusoidal: `_field_tensor` grew optional `src_centers` / `src_tangents` so the same tangential-field formula computes the image-source field at the original observer points; `_assemble_Z` subtracts image-source Φ tensors before the basis-coefficient sweep. Server `_make_pysim_sim` forwards `ground_z` to all three pysim models; frontend `backendSupportsGround()` enables the ground checkbox on the new slots. Pairing `use_singular_enrichment + ground_z` is rejected at construction (the enrichment-basis image reaction isn't worked out yet); the frontend silently drops enrichment from the request when both are active. Cross-validated on a horizontal dipole at h=7m, N=40: Tri/Sin/BSpl all agree on R within 0.5 Ω and X within 1 Ω.

- **PR #51** — **Sign-bug fix in `assemble_Z_enrich`**: the C++ accelerator stored `dΦ_sing/du_arc = (log(u_norm) + 1)/h` for every enrichment basis, but for orig=1 ("end"-orientation: junction at the segment's right endpoint, u_norm = 1 − t) the chain rule requires du_norm/du_arc = −1/h, so dΦ/du_arc should be negated. The missing sign poisoned the Φ-piece of Z_pe / Z_ep and the mixed-orig off-diagonals of Z_ee. On hentenna-like geometries where mirror K=3 junctions have opposite orig, the bug broke left-right symmetry: at N=21 with enrichment, the upper-end enrichment coefficient at D was 6.6× its upper-start mirror at B; impedance was Z = 42.85 + j44.24 (incorrect). With the fix it's 43.05 + j39.13 — within 0.4 Ω of slot A's Triangular N=40 baseline, and per-knot |I| values across the L-R mirror are now bit-exact equal. Surfaced from a UI report of a visible kink in the left-vertical |I| envelope on hentenna's upper polyline that the right vertical didn't show. New regression test `test_bspline_hentenna_enrichment_left_right_symmetry` pins the exact symmetry at N=21 in <1s; the existing pinned-impedance test for the same geometry moved its values to the corrected ones and switched its convergence-rate fit from R to X (R now hits the few-mΩ noise floor between N=41 and N=81). **Re-opens items 9 / 13 / 14 / item 15 below**: the arbitration value of 43.05 + j38.84 from PR #45 was computed with the buggy sign — the corrected enrichment-on convergence story needs a re-run. *(Closed in the post-#51 re-evaluation — see item 15's table below: the 43.05 + j38.84 asymptote is reconfirmed by the polynomial trio (tri / b2 / b2e). `SinusoidalPySim` continues to track PyNEC's super-log drift as expected (same three-term basis family). d=2-no-enrich was never affected by the bug; d=2-enrich now converges to the same place.)*

- **PR #52** — Closes NEXT_STEPS items 11 / 15(a-d) / 16: post-PR-#51 enrichment re-eval (three new probe scripts + 5-column hentenna sweep + arbitration-test tolerance tighten). Same PR also fixes a σ-handling bug in `SinusoidalPySim.currents_at_knots` that broke L-R symmetry on the hentenna canvas — `currents_at_knots` was multiplying the whole `(A + B·sin + C·cos)` basis evaluation by σ instead of using the `(σA, B, σC)` effective coefficients that `_assemble_Z` uses, which doubled the sin term at σ=−1 junction neighbours. Impedance was unaffected (compute_impedance evaluates at s=0). Regression test `test_sinusoidal_hentenna_left_right_symmetry` pins <1e-12 cross-mirror agreement.

- **Slot-B default flip (this branch)** — Web UI's slot B (`BSplinePySim` d=2 N=21) flips from `use_singular_enrichment=true` to `false` (i.e. inherits the backend default), based on the K=3 junction current-imbalance probe (`scripts/probe_k3_junction_imbalance.py`). The probe measures `tap_ratio = min(|I|)/max(|I|)` across the three wires meeting at each K=3 junction:

  ```
  geometry            tap ratio   normalized magnitudes   character
  ------------------  ----------  ----------------------  -----------------------
  fan dipole at S/T      0.03    (1.00, 0.99, 0.03)      basically K=2 + dead 10m
  hentenna  at B/D       0.16    (1.00, 0.84, 0.16)      dominant pair + small tap
  Y-fixture at S         0.50    (1.00, 0.50, 0.50)      balanced 3-way (2:1:1)
  ```

  The diagnostic: **when current at a K=3 junction collapses to a dominant in/out pair with the third wire as a small tap, the log-singular charge-density cusp is suppressed and enrichment is a no-op** (hentenna, fan dipole). When the split is genuinely 3-way (Y-fixture), the cusp matters and enrichment shifts Z by 0.08 Ω on R — the value the d=1 tent basis converges to independently. Both UI K=3 antennas (hentenna, fan dipole) live in the dominant-pair regime, so flipping the default loses no accuracy on any shipped preset and saves 0.26 Ω X accuracy on hentenna at n=21 (the 15(a) finding). Users hitting a genuinely 3-way K=3 geometry can flip enrichment back on via the slot's gear menu.

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

   **Decision**: accept the remaining R disagreement as a known NEC2-vs-others formulation gap and move on. Pysim is not the outlier — NEC2 is. The literature attributes this to NEC2's source-at-K-wire-junction handling and the thin-wire-kernel choice (these caveats apply independent of wire diameter, which is uniform in our model — that ruled out the dissimilar-diameter literature concerns). NEC4 is *not* a useful arbiter here: it shares NEC2's pulse-ish basis, delta-gap source, and thin-wire kernel for normal-radius wires, and the literature finds NEC4 vs NEC2 differences of <5 Ω for free-space wire antennas with uniform diameter. A genuinely independent arbiter would need a different formulation: higher-order B-spline in pysim (item 9 — cheapest in-codebase path), FDTD, or a surface-MoM with RWG basis.

   **Implication for the web UI**: the "solver agreement diagnostic" idea in item 10 should now treat fan-dipole pysim/PyNEC R disagreement as expected (and *growing* with K and N), not a bug indicator. The X agreement is the new baseline.

13. **Hentenna cross-solver convergence — pysim converges, PyNEC diverges super-logarithmically.** Per-solver convergence sweep on the single-band hentenna at params_50 (28.47 MHz, free-space, r=0.5 mm, uniform N segments per non-feed edge). Feed-wire segment count chosen per backend so the source sits at the geometric centre of the T→S gap: EVEN for pysim (tent-basis interior knot at z=0) and ODD for PyNEC (delta-gap segment centred on z=0). The parity choice biases R by ~1 Ω in PyNEC and X by ~0.04 Ω in pysim if mismatched — both small but worth being right.

    ```
      n   | pys nf |  pysim (R + jX)    | pyn nf |  PyNEC (R + jX)
      15  |    2   |  43.20 + j37.14    |    3   |  45.61  −j5.77
      21  |    4   |  43.17 + j38.07    |    3   |  45.60  −j4.60     ← UI default
      41  |    6   |  43.14 + j38.71    |    5   |  45.44  −j1.84
      81  |   12   |  43.13 + j38.87    |   11   |  45.24  +j1.65
     161  |   24   |  43.13 + j38.90    |   23   |  45.01  +j6.54
     201  |   28   |  43.13 + j38.91    |   29   |  44.91  +j8.67
     281  |   40   |  43.14 + j38.91    |   41   |  44.73 +j12.72
     441  |   64   |  43.14 + j38.91    |   63   |  44.42 +j20.22
     601  |   86   |  43.15 + j38.91    |   85   |  44.16 +j26.75
     661  |   94   |  43.15 + j38.91    |   95   |  44.07 +j29.01     ← last n before thin-wire wall
     701+ |        |                    |        |  error: first-segment midpoint intersects neighbor (seg ≈ 1.7× r on cross-bar)
    ```

    Pysim is stable to 3 sig figs from n=80 onward, locked at ~43.13 + j38.91 (a hint of upward R drift at very large n — 43.13 at n=80, 43.15 at n=661 — likely segment-radius interaction on the pysim side). PyNEC does *not* asymptote: `dX/d(log n)` increases monotonically across the sweep (3.5 at n=15→21, 12.1 at n=201→281, 23.8 at n=601→661 — never constant or decreasing), so the growth is super-logarithmic. The earlier framing "PyNEC is heading toward pysim's converged value" was sloppy — the trend would *cross* +j38.91 only if it kept going indefinitely past it, and it appears to. At the last feasible n (661, just before the cross-bar at 0.676 m / r=0.5 mm hits NEC's segment ≥ 2·radius rule), PyNEC X = +j29.01, still ~10 Ω below pysim and accelerating.

    **Third-solver corroboration on the canonical M-Hentenna (W=λ/6, H=λ/2, feed at λ/10).** Babli/Yannopoulou/Zimourtopoulos ([viXra:1811.0473](https://vixra.org/abs/1811.0473)) report R ≈ 65 Ω, X ≈ 0 Ω at the M-Hentenna's design freq, using Richmond's RICHWIRE (piecewise-sinusoidal MoM at OSU, [Richmond 1974](https://apps.dtic.mil/sti/citations/ADA015377)) — a completely independent code lineage from NEC. Re-running our pysim at the paper's r/λ ≈ 0.0037 scaling: pysim gives 66.75 + j6.0 (R within 2 Ω of RICHWIRE; X near zero), stable across n=12..161. PyNEC can't directly run the paper's geometry — at r/λ ≈ 0.0037 with W=λ/6, the cross-bar fails NEC's segment ≥ 2·radius rule for any n ≥ ~9. So **pysim's basis is corroborated by RICHWIRE on M-Hentenna**, but this only carries over to params_50 by argument — they share the topology, not the dimensions. The paper measured *patterns* in an anechoic chamber, not impedance — no physical Z ground-truth was found in any of the open hentenna literature surveyed (OE9HRV field report, DK7ZB page, portable-antennas.com, the 1982 QST article reference, the 1996 Kinoshita reference; none publish a measured R+jX).

    Likely interpretation: this is the same "imag part diverges with N" pathology already documented for the legacy pulse-basis pysim in [docs/convergence_analysis.md](docs/convergence_analysis.md), surfacing in PyNEC on the hentenna's particular geometry. **Which solver is physically correct remains unsettled** — pysim being converged and PyNEC being non-convergent doesn't automatically make pysim right; pysim's converged value could still reflect a systematic error in the tent basis that doesn't show up as non-convergence (it would just converge to the wrong place). NEC4 is *not* a useful arbiter (same MoM family, same pulse-ish basis, same delta-gap, same thin-wire kernel — literature has NEC2/NEC4 differing by <5 Ω on free-space wire antennas, much smaller than the 10 Ω residual here). What would arbitrate is a method in a fundamentally different formulation: see "What's open" below.

    **Feed-model probe (the same Tsai magnetic-frill from item 5, re-applied on the hentenna).** A subclass of `TriangularPySim` reproducing the dbbc300 frill source vector was run across n=15..161 with b/a=2.3 (NEC2's ~50 Ω default), and a separate b/a sensitivity sweep at fixed n=81:

    ```
        n  |   delta-gap         |   frill (b/a=2.3)
        15 |  43.208 + j37.161   |  43.212 + j37.168
        21 |  43.167 + j38.058   |  43.171 + j38.066
        81 |  43.123 + j38.828   |  43.127 + j38.835
       161 |  43.119 + j38.859   |  43.124 + j38.867

        b/a  |   frill Z at n=81
        1.5  |  43.125 + j38.832
        2.3  |  43.127 + j38.835
        5.0  |  43.140 + j38.849
       10.0  |  43.180 + j38.887
       30.0  |  43.487 + j39.172
    ```

    Frill − delta-gap = (+0.005, +0.008j) Ω at any sensible b/a. Even pushing b/a to 30 only shifts Z by ~0.5 Ω. **The feed-singularity hypothesis is rejected for the hentenna**, same null result as item 5 (V dipoles) and item 12 (fandipole K=3). Pysim's converged Z does *not* depend on whether the feed is modelled as δ-gap or a finite-extent coaxial frill, so calling pysim "the converged answer for a basis-bandlimited feed" was overstated — the source-vector projection isn't the lever. The non-convergence in PyNEC is therefore not a feed-singularity artifact either; the mechanism remains open and lies somewhere in the basis × kernel × junction interaction that the three probes (5, 12, 13) have so far ruled out as feed-related.

    pymininec at n=21 also read −j5.04 (matching PyNEC). Given the convergence picture, this is the same under-convergence at the same coarse n on two pulse-basis solvers — not an independent confirmation of "−j5 is correct." A per-n pymininec sweep would confirm it shows the same super-log climb.

    **nec2c sweep — independent NEC2 implementation, same trajectory.** Re-ran the same hentenna deck through Neoklis Kyriazis's [nec2c](https://manpages.ubuntu.com/manpages/noble/en/man1/nec2c.1.html) (a C translation of the LLNL Fortran, independent code lineage from Tim Molteno's necpp that PyNEC wraps):

    ```
        n   |  nec2c (R + jX)     |  PyNEC (R + jX)     |  Δ (R, X)
        15  |  45.614 − j5.838    |  45.613 − j5.769    |  +0.001, −0.069
        21  |  45.606 − j4.673    |  45.604 − j4.604    |  +0.002, −0.069
        81  |  45.245 + j1.578    |  45.244 + j1.646    |  +0.001, −0.068
       161  |  45.008 + j6.469    |  45.006 + j6.536    |  +0.002, −0.067
       281  |  44.735 +j12.648    |  44.733 +j12.722    |  +0.002, −0.074
       441  |  44.424 +j20.110    |  44.423 +j20.218    |  +0.001, −0.108
    ```

    Two findings: (a) R matches to ~0.002 Ω across the entire range, X to ~0.07 Ω — small enough to be linear-solver / complex-arithmetic precision (different LU paths), so **PyNEC has no implementation bug**; (b) nec2c's X climbs the same super-log curve (dX/d(log n) monotonically increasing from 3.5 to 18.1, same shape as PyNEC) — so **the divergence is the NEC2 algorithm, not any PyNEC artifact**. Two independent NEC2 implementations agree on both the value and the non-convergence.

    **Relation to item 8 (fandipole).** Inverted-looking but structurally consistent: there pysim/pymininec agreed on R against PyNEC's drift; here PyNEC drifts on X with no asymptote in sight. Both cases point to the same conclusion: **on multi-wire/junction geometries with a delta-gap feed, NEC's pulse-ish basis can fail to converge, while pysim's tent basis gives a finite (basis-defined) value**. The `tests/test_pysim.py` comment attributing the fan-dipole pysim/PyNEC gap to "basis-shape at K=3 junctions" should be re-checked against an explicit per-solver convergence sweep — it may be the same non-convergence story.

    **What's open**:
    - Add hentenna to the cross-solver comparison scripts (extend `scripts/compare_fandipole_solvers.py` or a new `compare_hentenna_solvers.py`) and have it plot `X vs log(n)` so the super-log signature is visible at a glance.
    - Per-n pymininec sweep on the hentenna to confirm it shows the same super-log climb (vs converging or hitting a different wall).
    - Re-run the fandipole at multiple n with all three solvers and re-do the `dX/d(log n)` analysis to test whether the K=3 "basis-shape" story is actually a non-convergence story too.
    - Decide what to do about the UI default for hentenna. n=21 is fine for pysim and badly wrong for PyNEC, and unlike a normal under-convergence there's no "high enough N" that fixes PyNEC — it just keeps drifting. Options: keep n=21 + UI warning that PyNEC on hentenna is non-convergent, raise the default and accept the cost, or refuse PyNEC for hentenna at the request layer (least surprising).
    - The mechanism behind PyNEC's super-log divergence remains open. Probes that have ruled out causes: feed model (item 5, item 13 frill probe — frill barely changes pysim's Z), K=3 junction kernel regularization (item 12), junction multiplicity (item 12, K=1/2/3 all show ~constant ΔX). Remaining candidates worth a probe: NEC2's source-on-segment-containing-junction handling; thin-wire kernel behaviour on segments adjacent to the source on a short feed wire; whether pysim's tent basis has its own basis-induced regularization at the junction that PyNEC lacks.
    - **What would arbitrate pysim vs PyNEC** (which was the bigger open question — pysim could still have been converged-to-the-wrong-place). NEC4 won't do it (same MoM family). Independent arbiters, ranked by cost:
      1. ~~**Higher-order B-spline in pysim**~~ (item 9) — **done in PR #45**. `BSplinePySim(degree=2)` at the hentenna params_50 sweep lands at 43.05 + j38.84 across n=15..81 (no drift, no super-log climb, monotonic 0.02 Ω tightening with n). Triangular (d=1) is at ~43.11 + j38.79 at n=81. **The two independent basis families agree to ~0.1 Ω** — and d=2 converges *faster* (already at j38.85 by n=15 where d=1 is still climbing). **This settles the arbitration: pysim's tent basis is correct; NEC's three-term basis is the outlier.** The reasoning is now closed at this question and was never about which solver was "physically correct" in some abstract sense — it's that two basis families that approach the EFIE from completely different polynomial-expansion directions both arrive at the same Z. The remaining hentenna mystery (what *mechanism* in the NEC three-term basis produces the super-log drift) is now item 14's "open at lower priority" — interesting but no longer load-bearing.
      2. **FDTD** (openEMS, Meep, or similar). Completely different formulation — no MoM basis, no delta-gap singularity. Genuinely independent. Now superseded as the *needed* arbiter by item-9 above; would still be a nice belt-and-suspenders check if FDTD lands at 43 + j38.8 too.
      3. **Surface-MoM with RWG basis** treating the wire as a thin cylinder. Different basis, different kernel singularity, same Maxwell. Same status as FDTD: belt-and-suspenders, no longer load-bearing.
      4. **Published measurement.** Hentennas are a 1970s amateur-radio antenna with measured VSWR/Z data possibly in the JA literature; a search and translation pass might find a ground-truth point for params_50. Confirmation rather than arbitration now.

14. **NEC's super-log X drift is intrinsic to the three-term basis — isolated by `SinusoidalPySim`.** Implemented NEC2's basis from the design document (`docs/nec2_theory_manual.pdf`, derivation in `docs/sinusoidal_basis_design.md`) in `src/pysim/sinusoidal.py` (PR #44). From-scratch — shares none of necpp's code; matches the manual's Eqs 43-64 for closed-form Wu-King coefficients, Eqs 76-79 for the per-segment field, Eq 187 for the delta-gap source. Scope: free space, thin-wire kernel, uniform radius, X_i=0 free-end. Hentenna params_50 sweep against PyNEC's numbers from item 13:

    ```
       n   | sin pysim R+jX    | PyNEC R+jX        |  ΔR     ΔX
       15  | 45.611 − j 5.718  | 45.613 − j 5.769  | −0.00  +0.05
       21  | 45.602 − j 4.553  | 45.604 − j 4.604  | −0.00  +0.05
       81  | 45.242 + j 1.693  | 45.244 + j 1.646  | −0.00  +0.05
      161  | 45.004 + j 6.540  | 45.006 + j 6.536  | −0.00  +0.00
      201  | 44.908 + j 8.736  | 44.910 + j 8.670  | −0.00  +0.07
      281  | 44.739 + j13.064  | 44.733 + j12.722  | +0.01  +0.34
      441  | 44.381 + j16.796  | 44.423 + j20.218  | −0.04  −3.42
    ```

    R matches PyNEC to ~0.05 Ω across the entire range. X matches to ~0.05 Ω through n=281, then opens up a 3.4 Ω gap at n=441 (likely a quadrature accuracy issue in my `n_qp_const=8` for the const-current self integral once segments shrink below a critical length — worth a follow-up but doesn't affect the qualitative finding).

    **Settles**: the *NEC side* of "what's open" in item 13. The probes there ruled out feed model, K=3 junction kernel regularization, junction multiplicity, and the second NEC2 implementation (nec2c) as causes. With a from-scratch sinusoidal-basis solver in a completely independent codebase reproducing the drift, the cause is the **three-term basis itself** — neither NEC's kernel implementation, nor its source vector, nor its junction-handling code, nor anything else specific to necpp.

    **Does not settle**: which solver is physically correct. Both the triangular basis (converging to ~43 + j39) and the three-term basis (drifting super-logarithmically) are now characterized; the choice between them is the same arbitration question as before. The remaining arbiters in item 13's list still apply, with the focus narrowed: the question is no longer "what in NEC is causing this" but "which basis converges to the right value."

    **What's next from here**:
    - ~~Higher-order B-spline in pysim (item 9)~~ — **done in PR #45**. `BSplinePySim(degree=2)` at the hentenna params_50 sweep lands at 43.05 + j38.84 across n=15..81 (essentially constant). Triangular at n=81 is 43.11 + j38.79. The two basis families agree to ~0.1 Ω. **Tent is validated; the three-term basis is the outlier**. See item 9 / item 13's "What would arbitrate" list / the PR #45 entry above for the full result.
    - Investigate the n=441 X gap in `SinusoidalPySim` (the only remaining hentenna-related accuracy issue). Likely candidates: (a) quadrature node count for `_field_tensor`'s `int_G0` (currently `n_qp_const=8`; try 16, 32), (b) a sign/normalization edge case in the basis-coefficient closed forms that only matters at very small Δ, (c) PyNEC's own quadrature differing slightly from mine at that resolution — bumping `n_qp_const` and seeing whether my values move toward or away from PyNEC's is the first probe. *Now lower priority since the arbitration is settled — this is just sinusoidal-quadrature accuracy, no longer a load-bearing question.*
    - Add `SinusoidalPySim` and `BSplinePySim` to the cross-solver comparison scripts and the web UI as additional solver tabs. The UI integration would make the basis-induced X drift visible on the Smith chart in real time (sinusoidal drift on hentenna alongside tent and B-spline staying converged).
    - The hentenna `n_feed` parity rule: pysim's triangular basis wants EVEN n_feed (tent-basis interior knot at z=0); pysim's sinusoidal basis wants ODD n_feed (delta-gap segment centred at z=0); pysim's B-spline d=2 wants EVEN n_feed (same as triangular — there's an interior knot at z=0 in both cases). Same as PyNEC's parity for the sinusoidal case — confirmed by item 14's near-zero ΔR. Document this in `web/server.py`'s `_hentenna_geometry` so the multi-way solver toggle picks the right parity per backend.
    - **What's now genuinely open** about the three-term basis super-log drift (lower priority — the arbitration is settled, this is just academic interest): the *mechanism*. The probes have ruled out feed model (items 5, 13, 14), K=3 junction kernel regularization (item 12), junction multiplicity (item 12), and the second NEC2 implementation (nec2c) as causes. The drift is intrinsic to the three-term basis on multi-wire/junction geometries. *Why* — what specifically about the const + sin + cos expansion fails on a K=3 junction-rich, non-resonant geometry — is interesting but not load-bearing for any downstream decision. Theoretical conjecture in conversation logs: as h shrinks, sin(k·h) ≈ k·h and cos(k·h) ≈ 1 - (k·h)²/2 so the basis approaches polynomial form *but* with a k-dependent phase reference s_n that creates near-linear-dependence between adjacent segments at small h.

### B-spline enrichment re-evaluation post-PR-#51 — closed

15. ~~**Re-evaluate the B-spline enrichment basis with PR #51's `orig=1` sign fix.**~~ — **closed across (a)/(b)/(c)/(d) on the `bspline-enrichment-reeval` branch**. Three new probe scripts:
    - `scripts/compare_hentenna_solvers.py` — 4-solver hentenna sweep (Triangular / BSpline d=2 no-enrich / BSpline d=2 enrich / PyNEC) across n ∈ {15, 21, 41, 81, 161}. Item 13's open task ("add hentenna to the cross-solver comparison scripts") shipped here too.
    - `scripts/probe_y_fixture_enrichment.py` — K=3 Y fixture at λ/4 arms, d=1 vs d=2 with/without enrichment, n ∈ {15, 21, 41, 81, 161, 241}.
    - `scripts/probe_bent_dipole_enrichment.py` — 90° bent dipole in both polyline-kink and K=2-junction representations, d=1 and d=2, with/without enrichment_min_k ∈ {2, 3}.

    **(a) Hentenna arbitration value reconfirmed at 43.05 + j38.84.** Post-PR-#51 sweep across all four pysim bases + PyNEC (nfeed=2 for the polynomial bases tri/b2/b2e; nfeed=3 for the three-term bases sin/pynec, matching their source-segment-centering rule):

    ```
       n  |   tri         |   b2 (no enr) |   b2e (enr)   |   sin           |   pynec
       15 | 43.20 + j37.13 | 43.07 + j38.85 | 42.86 + j40.07 | 45.61 − j 5.72 | 45.61 − j 5.77
       21 | 43.16 + j38.03 | 43.07 + j38.85 | 43.03 + j39.09 | 45.60 − j 4.55 | 45.60 − j 4.60
       41 | 43.13 + j38.65 | 43.06 + j38.84 | 43.06 + j38.86 | 45.46 − j 1.78 | 45.44 − j 1.84
       81 | 43.11 + j38.79 | 43.05 + j38.84 | 43.05 + j38.84 | 45.25 + j 1.71 | 45.24 + j 1.65
      161 | 43.11 + j38.82 | 43.05 + j38.84 | 43.05 + j38.84 | 44.98 + j 6.51 | 45.01 + j 6.54
    ```

    The **polynomial trio** (tri / b2 / b2e) converges to the same Z asymptote ≈ 43.05 + j38.84 — the PR #45 / item-9 arbitration value unchanged. The **three-term pair** (sin / pynec) drifts super-log on X; `SinusoidalPySim` tracks `PyNEC` to ~0.05 Ω on R and ~0.07 Ω on X at every n, as predicted by item 14. Two independent basis families, each with multiple independent code paths, with the polynomial side reaching an asymptote that the three-term side cannot — the arbitration is now visible in a single table.

    X-rates over the (41, 81, 161) triple: tri p ≈ 2.19 (basis-limited at degree-1's O(1/N²)), b2 p ≈ 2.53 (tail-noise; already at convergence by n=15), b2e p ≈ 3.23. Three-term pair: sin p ≈ −0.46, pynec p ≈ −0.50 — both anti-convergent, tracking together.

    **The conjecture in the open question was wrong**: the buggy sign wasn't masking a real enrichment speed-up. With the fix, b2e is still *slower at low n* than b2 (b2e at n=15 is 1.2 Ω off the asymptote; b2 is already there to ~0.02 Ω). The enrichment basis adds a higher-order tail but pays its own low-N transient. **Decision**: the slot-B UI default should flip from `use_singular_enrichment=True` to `False` for the hentenna preset — b2 alone at n=21 gives 43.07 + j38.85 (within 0.02 Ω of asymptote) while b2e at n=21 gives 43.03 + j39.09 (0.26 Ω off X). The PR #45 `test_bspline_d2_hentenna_arbitrates_against_triangular` tolerance was tightened from 1.0 Ω to 0.2 Ω on the cross-basis disagreement check now that we know b2 is already at convergence at n=21.

    **(b) d=1 enrichment is a no-op** (Y-fixture, λ/4 arms, K=3 at S):

    ```
       n   |    d=1         |    d=1 enr      |    d=2          |    d=2 enr
       15  | 50.22 + j51.82 | 50.24 + j51.85  | 50.29 + j52.56  | 50.38 + j52.56
       21  | 50.29 + j52.22 | 50.30 + j52.23  | 50.30 + j52.71  | 50.38 + j52.72
       41  | 50.36 + j52.65 | 50.37 + j52.65  | 50.32 + j52.92  | 50.39 + j52.92
       81  | 50.40 + j52.87 | 50.40 + j52.87  | 50.33 + j53.04  | 50.40 + j53.04
      161  | 50.41 + j52.99 | 50.42 + j52.99  | 50.34 + j53.11  | 50.42 + j53.11
      241  | 50.42 + j53.04 | 50.43 + j53.05  | 50.34 + j53.14  | 50.42 + j53.15
    ```

    d=1 ↔ d=1-enrich agreement is ~0.01 Ω at every n with identical fitted rates (p ≈ 1.0 on X for both). Interpretation: the tent basis already has a slope discontinuity at every interior knot — it represents the K=3-junction current cusp "for free", so adding the log-singular shape changes essentially nothing. **No port of enrichment to `TriangularPySim` is warranted.** This is a useful negative result.

    Bonus finding from the Y-fixture (not visible on the hentenna): at d=2, enrichment shifts R by ~0.08 Ω in a way that persists to n=241 (does not shrink). d=2 alone converges to a different asymptote than d=1 alone (~50.34 vs ~50.42); d=2 + enrichment matches d=1 alone. The C¹ d=2 basis can't represent the junction-cusp on its own; enrichment supplies it. Why this doesn't show on the hentenna: there the K=3 junctions sit in a near-resonant standing-wave configuration where current through the junction is small, so the cusp contribution is proportionally tiny. The Y has a feed driving directly into a K=3 junction with no resonance damping; the cusp is loud.

    **(c) enrichment_min_k=2 actively HARMS** (bent-dipole, 90° per arm; full-wave anti-resonance so |Z| is large):

    ```
       k2 d=2  no enrich vs  k2 d=2 enrichment_min_k=2  (ΔR, ΔX in Ω)
       n=15 :  ΔR = −597.18   ΔX = +111.82
       n=21 :  ΔR = −516.82   ΔX = + 93.13
       n=41 :  ΔR = −475.59   ΔX = + 82.96
       n=81 :  ΔR = −469.36   ΔX = + 81.42
    ```

    The two solutions are not converging to the same place. The log-singular shape Φ_sing(u) ~ u·log(u) is *wrong* for K=2 bends: a K=2 bend has KCL `I_in = −I_out` (continuity through the bend, no current splitting), so the local charge density isn't log-singular. Forcing the wrong-shaped basis into the system pulls Z to an incorrect limit. The doc's working hypothesis was "shouldn't help"; the data is stronger — **do not use enrichment_min_k=2**. The default `enrichment_min_k=3` is correct.

    K=1 free-end sub-question: moot in current code. `BSplinePySim`'s constructor rejects single-wire-end "junctions" (`len(jw) < 2: raise ValueError`). PR #48 smoothed source remains the principled fix for the source-singularity rate.

    **(d) polyline-kink ↔ K=2-junction representations are bit-exact equivalent for both d=1 and d=2** under BSpline:

    ```
       pl vs k2 disagreement on the bent dipole (n=15..81):
       d=1:  |ΔR| ≤ 5e-12 Ω, |ΔX| ≤ 5e-12 Ω
       d=2:  |ΔR| ≤ 1e-4 Ω,  |ΔX| ≤ 1e-4 Ω   (still roundoff-class)
    ```

    My hypothesis that the C¹ d=2 polyline basis would "smear" the kink was wrong — BSpline rebuilds the basis with a slope break at every polyline anchor too. **Item 16's open question ("does it actually change any answer?") is settled with a clean NO**, for both bases. The architectural unification reduces to a pure code-clarity question. See item 16 for the updated trade-off.

16. **Architectural question: model bends as K=2 junctions instead of as polyline kinks?** Item 15(d) settled the correctness question — the two representations are bit-exact equivalent for `TriangularPySim` (existing test `test_triangular_k2_junction_equivalent_to_single_polyline`) and for `BSplinePySim` at both d=1 and d=2 (probe_bent_dipole_enrichment.py). So this is purely a code-clarity question now.

    Trade-offs (no correctness component left):

    - **Pro:** the polyline-kink path and the junction path are doing approximately the same thing (enforcing current continuity at a node where the tangent changes). One uniform mechanism is conceptually nicer than two slightly different ones.
    - **Con:** every polyline-style geometry input (every UI antenna currently) would need to either be auto-decomposed at the server level or rewritten to use junctions. Mass refactor.
    - **Con:** the Lagrange-multiplier KCL row adds N_bends rows to the solve matrix. For an inverted V with one bend, that's +1 row. For a moxon with 8 bends, +8 rows. Tiny on N≈400-segment fan dipoles but not free.
    - **Con:** within-wire basis continuity is currently *automatic* (the tent basis spans the kink); junction directional bases are by construction one-wing — losing the natural continuity means the basis at a bend would be K=2 directional rather than a single interior tent.
    - **The earlier "pro" about enrichment scaffolding (item 15(d)) is gone**: 15(c) showed enrichment at K=2 bends is actively harmful, so making bends into K=2 junctions doesn't unlock any enrichment story.

    Recommendation: don't do this refactor. The polyline-kink path is correct, equally accurate as the K=2-junction path, simpler for users, and doesn't pay the Lagrange-row cost. Architectural symmetry is a weak motivator against the implementation cost.

### Visualization fidelity follow-ups

17. **Visualization fidelity gaps from `docs/visualization_audit.md`.** Audit done after PR #51 caught the orig=1 sign bug; doc catalogues remaining mismatches between what the solvers compute and what the canvas renders. Severity grades and recommended fixes are in the doc. Headline:

    - **G1 + A1 + A2 combined fix**: extend `currents_at_knots` to take an optional `s_array` per wire (or fixed segment-quarter-points) so the heatmap stroke and the far-field path can sample at mid-segment points where B-spline d=2 quadratic curvature and the enrichment dip actually live. Single coherent change spanning model + server + frontend. Highest value of the remaining items.
    - **G3**: draw a thin ground-line at `z=0` when `result.ground === true`. ~10 LOC of canvas code, removes a "where is the ground" guessing game from the UI.
    - **G4**: visual sanity check that PyNEC vs Triangular ground at the same height produces meaningfully different far-field lobes (different Fresnel reflection coefficients). No code change expected, just verification.
    - **G2** (Lagrange multipliers exposed via solve API): probably no action, debugging-grade info.

    None of these are urgent. Item 15 is higher-value because it re-tests the arbitration value that several other items hinge on; visualization fidelity is a separable lane.

### Interactive UI follow-ups

10. **UI follow-ups** — Yagi (with N directors), moxon, hexbeam, and fan dipole all ship in the interactive UI. Open work:
    - **Solver overlay on Smith plot**: currently the pysim/pynec tab toggle replaces the displayed solve; an overlay drawing *both* points/sweeps on the Smith chart simultaneously would make solver-disagreement geometries (fan dipole, see PR #36) visually obvious without manual A/B.
    - **Fan dipole 3D rotation**: the side-view (yz plane) collapses x-axis cone variation. The 5-band default has visible x-spread that the projection drops. A simple azimuth-rotation control or isometric view would surface it.
    - **Solver agreement diagnostic**: when pysim and pynec disagree by more than some threshold (say 5% of |Z|), surface a small indicator in the UI noting "solvers differ by X Ω — see geometry-specific known-issue".

11. ~~**Far-field pattern enhancements**~~ — **all three sub-items shipped**. `FarFieldChart` in `web/frontend/src/App.tsx` renders both an azimuth (xy) cone-cut and an elevation (yz) great-circle-cut tab; the radial axis is absolute dBi with a fixed 30-dB display range (DBI_TOP=10, ticks at +6/0/−6/−12/−18); per-direction directivity uses the server-supplied `result.directivity_norm` (sphere-integrated `|E|²` total radiated power) so the gain reading is absolute, not per-frame normalized. View-switcher tabs at the top of the stage select among antenna / azimuth / elevation / smith. PR not separately tagged — the pieces landed across the multi-backend UI work in PR #49 and the ground-plane far-field work in PR #50.

### Open research

9. ~~**Higher-order basis functions**~~ — **done in PR #45** (`BSplinePySim(degree=2)`). Quadratic B-spline Galerkin MoM with the same multi-wire / polyline / K-wire-junction machinery as `TriangularPySim`. The closed-form same-edge static-kernel moment integrals J_pq for p, q ∈ {0..2} are sympy-derived once and dumped to `src/pysim/_bspline_static_moments.py` (no runtime sympy dep; `scripts/derive_bspline_static_moments.py` re-runs the derivation with larger `MAX_D`). Convergence-rate observation matches O(1/N³): on the dipole d=2 reaches the NEC reference at N=15 (j-18.4 vs NEC's j-18.21) where d=1 is still at j-21.3. On the hentenna it converges to 43.05 + j38.84 ± 0.01 Ω at *every* n in {15, 21, 41, 81}. The arbitration use this enabled is the load-bearing result — see item 14 and the PR #45 entry above.

  d > 2 not implemented yet — the static-moment expressions grow combinatorially (8 KB code for d=2; estimated ~30 KB for d=3). Likely unnecessary unless we want to probe a *third* basis family for an even stronger arbitration claim.

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
