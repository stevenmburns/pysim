"""Higher-order B-spline Galerkin MoM solver.

`TriangularSolver` is the degree-1 B-spline (tent) special case; this
module extends to arbitrary degree d on multi-wire polylines with K-wire
junctions, primarily as an in-codebase arbiter for the hentenna question
(NEXT_STEPS.md items 9, 13, 14): does the tent basis converge to the
correct value, or is it converged-to-the-wrong-place?

Scope:
  * arbitrary number of wires; each wire is a polyline (M ≥ 2 anchors)
  * uniform segments per edge, possibly non-uniform across edges
  * free space, thin-wire kernel with a² wire-radius regularization
  * delta-gap "applied-E" source on one feed wire
  * degree d ∈ {1, 2}  (d=1 reproduces the tent basis up to feed convention)
  * K-wire junctions with KCL constraint (Σ outflow currents = 0)
  * NO ground plane (yet)

Same as TriangularSolver, but with a polynomial-of-degree-d on each segment
instead of just a linear ramp. Each interior basis Φ_m spans up to d+1
contiguous segments within a single wire; on each segment in its support
("wing") the basis equals Σ_p C[m, w, p] · u^p with u local arc length.

J_pq[i, j] = ∫∫ u^p u'^q · exp(-jkR)/(4πR) du' du
with R² = |r_i(u) - r_j(u')|² + a²

Galerkin assembly:
    Z_A[m,n]   = jωμ Σ_{a,b} (t_i · t_j) · Σ_{p,q} C[m,a,p] C[n,b,q]
                 · J_{pq}[supp_seg[m,a], supp_seg[n,b]]
    Z_Φ[m,n]   = (1/jωε) Σ_{a,b} Σ_{p≥1,q≥1} p·q · C[m,a,p] C[n,b,q]
                 · J_{p-1,q-1}[supp_seg[m,a], supp_seg[n,b]]

Junction directional bases: at every junction node with K connected wire-
ends we add K boundary bases (B_0 or B_{N+d-1} of each connected wire,
the ones with value 1 at the junction) and enforce KCL via a Lagrange-
multiplier row, mirroring TriangularSolver's treatment.

Feed: v_m = Φ_m(s_f), Z_drive = 1 / (v^T c).
"""

import numpy as np
import scipy.linalg
from scipy.interpolate import BSpline

from ._bspline_kernels import (
    _seg_seg_full_moments_offedge,
    _seg_seg_reg_geometry,
    _seg_seg_reg_moments,
    _seg_seg_reg_moments_from_geometry,
    _seg_seg_reg_moments_from_geometry_swept,
    _seg_seg_static_moments,
)
from ._quadrature import leggauss

try:
    from . import _accelerators as _acc

    _HAVE_BSPLINE_ASSEMBLE_ACCEL = hasattr(_acc, "assemble_Z_bspline")
    _HAVE_ENRICH_ACCEL = hasattr(_acc, "assemble_Z_enrich")
except ImportError:
    _HAVE_BSPLINE_ASSEMBLE_ACCEL = False
    _HAVE_ENRICH_ACCEL = False

_BSPLINE_ASSEMBLE_ACCEL_MAX_D = 2


# Constant Vandermonde inverses for uniform sample points [0, 1/d, ..., 1].
# Used by `_build_basis_polynomials` to convert per-segment basis values at
# d+1 uniform local-u sample points to polynomial coefficients without a
# per-segment scipy.linalg.solve. With u_local = h_seg * [0, 1/d, ..., 1],
# the Vandermonde factors as Vmat = V_unit @ diag(1, h, h², ..., h^d), so
# coeffs_p = (V_unit_inv @ vals)_p / h_seg^p — pure matmul + column scaling.
_V_UNIT_INV: dict[int, np.ndarray] = {
    1: np.array([[1.0, 0.0], [-1.0, 1.0]]),
    2: np.array([[1.0, 0.0, 0.0], [-3.0, 4.0, -1.0], [2.0, -4.0, 2.0]]),
}


# Module-level caches for `_build_geometry` and `_build_basis_polynomials`.
# Both functions are pure functions of immutable geometry inputs (wires +
# n_per_edge_per_wire, plus degree + junctions for the basis case) — they
# don't depend on `k` / wavelength / feed location. The instance-level
# caches in `_cached_geometry` / `_cached_basis_polynomials` only help the
# swept path (where one solver instance handles many k's). The engine
# wrapper instantiates a fresh BSplineSolver per impedance() call, so the
# instance cache is dead for the interactive UI sweep. These module-level
# caches survive across instances and turn a band-sweep of N freqs into
# 1 cold call + (N−1) hot calls for the geometry/basis stages.
#
# FIFO with a small bound — typical interactive use has 1–3 active
# (geometry, degree) combinations at a time.
_GEOMETRY_CACHE: dict = {}
_BASIS_POLY_CACHE: dict = {}
_GEOMETRY_CACHE_MAX = 32
_BASIS_POLY_CACHE_MAX = 32


def _evict_fifo(cache: dict, limit: int) -> None:
    while len(cache) >= limit:
        cache.pop(next(iter(cache)))


def _xfem_projection_coeffs(d):
    """Coefficients c such that P_bubble Φ_sing (t) = Σ_p c_p t^p, where
    P_bubble is the L²-orthogonal projection of Φ_sing(t) = t·log(t) onto
    the subspace of P_d on [0, 1] whose elements vanish at t=0 and t=1.

    Returns an array of length d+1, the monomial coefficients of the
    projection. Pure constants — h-independent and geometry-independent.

    Used to build the "stable" XFEM enrichment basis
        Φ_sing_stable(t) = t·log(t) − Σ_p c_p t^p
    Φ_sing_stable retains both endpoint BCs of Φ_sing — vanishing at
    t=0 (the junction node, required for finite-current KCL at the
    K-wire junction — the KCL constraint only sees polynomial bases,
    so enrichment must self-zero there) and at t=1 (the segment's far
    end, required for current continuity with the adjacent non-enriched
    segment). And on the BC-compatible bubble subspace, the enrichment
    is L²-orthogonal: α_enrich = 0 exactly when the truth's
    BC-compatible-bubble part lives in the polynomial subspace — the
    small-N transient where the original Φ_sing absorbs polynomial
    discretization error is eliminated.

    Bubble basis: b_k(t) = t^(k+1) − t^(k+2) = t·(1−t)·t^k for k = 0..d−2.
    Dimension is d−1 (for d=2 it's 1D = span{t(1−t)}; for d=1 it's empty,
    matching the empirical "d=1 enrichment is a no-op" finding in
    NEXT_STEPS item 15(b)).
    """
    if d < 2:
        return np.zeros(d + 1)
    n_b = d - 1
    # Bubble Gram matrix: ⟨b_i, b_j⟩ = ∫₀¹ (t^(i+1)-t^(i+2))(t^(j+1)-t^(j+2)) dt
    #                  = 1/(i+j+3) − 2/(i+j+4) + 1/(i+j+5)
    G = np.array(
        [
            [
                1.0 / (i + j + 3) - 2.0 / (i + j + 4) + 1.0 / (i + j + 5)
                for j in range(n_b)
            ]
            for i in range(n_b)
        ]
    )
    # Moment vector: ⟨Φ_sing, b_i⟩ = ∫₀¹ t·log(t)·(t^(i+1)-t^(i+2)) dt
    #              = ∫ t^(i+2) log(t) dt − ∫ t^(i+3) log(t) dt
    #              = −1/(i+3)² + 1/(i+4)²    [using ∫₀¹ t^n log(t) dt = −1/(n+1)²]
    m = np.array([-1.0 / (i + 3) ** 2 + 1.0 / (i + 4) ** 2 for i in range(n_b)])
    alpha = np.linalg.solve(G, m)
    # Bubble-coefficient α → monomial coefficients c:
    # P(t) = Σ_k α_k · (t^(k+1) − t^(k+2))  ⇒  c[k+1] += α_k, c[k+2] −= α_k.
    coeffs = np.zeros(d + 1)
    for k in range(n_b):
        coeffs[k + 1] += alpha[k]
        coeffs[k + 2] -= alpha[k]
    return coeffs


class BSplineSolver:
    """Degree-d B-spline Galerkin MoM, multi-wire polylines with junctions.

    Parameters
    ----------
    wires : list of (M, 3) polyline arrays, M ≥ 2 anchors each.
    n_per_edge_per_wire : list of (int | sequence | None). Per-wire segment
        counts per edge. None for a wire ⇒ use `nsegs` on every edge; int ⇒
        same count for every edge; sequence ⇒ explicit per-edge count.
    degree : B-spline degree (1 ≤ degree ≤ 2 currently; static-moment file
        only covers max_d=2). d=1 reproduces the tent basis up to the
        feed-convention difference.
    feed_wire_index : index of the wire carrying the delta-gap source.
    feed_arclength : arc length along the feed wire at which to evaluate
        Φ_m(s_f). Default: feed wire midpoint.
    junctions : list of [(wire_idx, "start"|"end"), ...] tuples, each entry
        one junction node where K wire endpoints meet. Same convention as
        TriangularSolver.
    n_qp_pair : Gauss-Legendre nodes per segment per axis for the smooth-
        kernel piece of same-edge pairs and for all cross-edge / cross-wire
        pairs (full kernel with a² regularization).
    wavelength, halfdriver_factor, wire_radius, nsegs : as in
        TriangularSolver.
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        degree=2,
        feed_wire_index=0,
        feed_arclength=None,
        feeds=None,
        feed_smoothing_factor=None,
        junctions=None,
        n_qp_pair=4,
        n_qp_source=16,
        wavelength=22,
        halfdriver_factor=0.962,
        wire_radius=0.0005,
        nsegs=101,
        ground_z=None,
        use_singular_enrichment=False,
        n_qp_sing=32,
        enrichment_min_k=3,
        enrichment_variant="raw",
        tikhonov_lambda=1e-3,
        auto_tap_ratio_threshold=0.3,
    ):
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if degree > 2:
            raise NotImplementedError(
                "degree > 2 needs scripts/derive_bspline_static_moments.py "
                "to be re-run with a larger MAX_D"
            )
        if not wires:
            raise ValueError("wires must be non-empty")
        if use_singular_enrichment and ground_z is not None:
            raise NotImplementedError(
                "use_singular_enrichment + ground_z together not supported "
                "yet — image reaction for enrichment bases isn't implemented"
            )

        self.degree = int(degree)
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.wire_radius = wire_radius
        self.nsegs = nsegs
        self.ground_z = ground_z
        # Source smoothing: when None (default) → delta-gap at exact midpoint
        # of feed wire. When set to a float α, the delta-gap is replaced by a
        # cos² bump of width w = α · h_feed_segment_at_source centered on s_f,
        # giving basis-limited convergence instead of the O(1/N) delta-gap
        # source-singularity rate. α ≈ 2-4 is a sensible starting point;
        # larger α gives faster basis-limited convergence but the smoothing
        # error from the bump's finite width takes longer to vanish.
        self.feed_smoothing_factor = feed_smoothing_factor
        self.n_qp_source = int(n_qp_source)

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = self.omega / self.c
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")

        n_w = len(self.wires_polylines)
        if n_per_edge_per_wire is None:
            n_per_edge_per_wire = [None] * n_w
        if len(n_per_edge_per_wire) != n_w:
            raise ValueError(
                f"n_per_edge_per_wire length {len(n_per_edge_per_wire)} != n_wires {n_w}"
            )

        self.n_per_edge_per_wire = []
        for i, (pl, npe) in enumerate(zip(self.wires_polylines, n_per_edge_per_wire)):
            n_edges_w = pl.shape[0] - 1
            if npe is None:
                npe = self.nsegs
            if np.isscalar(npe):
                npe = [int(npe)] * n_edges_w
            npe = list(npe)
            if len(npe) != n_edges_w:
                raise ValueError(
                    f"wire {i}: n_per_edge length {len(npe)} != n_edges {n_edges_w}"
                )
            self.n_per_edge_per_wire.append(npe)

        if feeds is None:
            if not (0 <= feed_wire_index < n_w):
                raise ValueError(f"feed_wire_index {feed_wire_index} out of range")
            self.feeds = [(int(feed_wire_index), feed_arclength, 1.0 + 0.0j)]
        else:
            if len(feeds) == 0:
                raise ValueError("feeds must contain at least one entry")
            norm = []
            for i, f in enumerate(feeds):
                if len(f) != 3:
                    raise ValueError(
                        f"feeds[{i}]: expected (wire_index, arclength, voltage), got {f!r}"
                    )
                w_i, arc_i, v_i = f
                if not (0 <= w_i < n_w):
                    raise ValueError(
                        f"feeds[{i}]: wire_index {w_i} out of range [0, {n_w})"
                    )
                arc_i = None if arc_i is None else float(arc_i)
                norm.append((int(w_i), arc_i, complex(v_i)))
            self.feeds = norm

        self.feed_wire_index = self.feeds[0][0]
        self.feed_arclength = self.feeds[0][1]
        self.n_qp_pair = int(n_qp_pair)

        # Singular basis enrichment at K≥`enrichment_min_k` junctions.
        # When enabled, adds ONE extra basis per (wire, end_pos) tuple at each
        # qualifying junction, with shape Φ_sing(u) = (u/h)·log(u/h) on the
        # adjacent segment (u measured from the junction node, so Φ_sing(0)=0
        # matching the finite-current condition while dΦ_sing/du has a log
        # singularity that captures the classical K≥3 junction charge-density
        # singularity). On hentenna-class geometries this flips the R-rate
        # from O(1/N) to ~O(1/N^(d+1)) (basis-limited). Quadrature is GL with
        # `n_qp_sing` nodes per axis (default 32) routed through the C++
        # `assemble_Z_enrich` accelerator.
        self.use_singular_enrichment = bool(use_singular_enrichment)
        self.n_qp_sing = int(n_qp_sing)
        self.enrichment_min_k = int(enrichment_min_k)
        # `enrichment_variant` picks the singular basis shape:
        #   "raw"    → Φ_sing(t) = t·log(t), the unmodified PR #47 shape.
        #             Mixed behavior: captures real cusps where they exist
        #             (balanced K=3 like Y-fixture: ~0.08 Ω R correction)
        #             but also absorbs polynomial discretization error at
        #             small N (hentenna n=21: ~0.26 Ω X transient post-#51
        #             sign fix). PR #45 / #47 default.
        #   "stable" → Φ_sing_stable(t) = t·log(t) − P_bubble(t·log(t)) where
        #             P_bubble is the L²-orthogonal projection onto the
        #             polynomial bubble subspace of P_d that vanishes at
        #             both t=0 and t=1 (preserves Φ_sing's endpoint BCs;
        #             required because the KCL constraint only sees the
        #             polynomial bases). Trade-offs measured on probe
        #             scripts: hentenna large-N converges faster
        #             (X-rate p≈4.10 vs raw's 2.7); fan-dipole gap closes
        #             to noise floor; **hentenna small-N gets a larger
        #             transient (~0.79 Ω X at n=21)**; Y-fixture loses
        #             its 0.08 Ω R cusp benefit. Not "universally safe."
        #             For d=1 the bubble subspace is empty so "stable"
        #             reduces to "raw" identically.
        # "tikhonov" → raw Φ_sing basis, but add λ·s·I to the enrichment
        # block Z_ee at solve time, where s is the average diagonal
        # magnitude of Z_ee (so λ is a dimensionless relative-strength
        # knob). Penalizes ||α_enr||² in the augmented objective; shrinks
        # spurious-large α at small N without re-deriving the basis.
        # λ → 0 ⇒ raw; λ → ∞ ⇒ enrichment effectively off.
        # "auto" → two-pass selectivity. Solve once without enrichment,
        # measure tap_ratio = min|I_wire|/max|I_wire| over wires meeting
        # at each K≥enrichment_min_k junction (the same diagnostic as
        # scripts/probe_k3_junction_imbalance.py), and apply raw
        # enrichment only at junctions where tap_ratio exceeds
        # `auto_tap_ratio_threshold`. Cleanly separates dominant-pair
        # K=3 (hentenna ≈ 0.16, fan-dipole ≈ 0.03) from balanced 3-way
        # (Y-fixture ≈ 0.50). One extra solve per compute_impedance
        # call; second solve skipped if no junction qualifies.
        if enrichment_variant not in ("raw", "stable", "tikhonov", "auto"):
            raise ValueError(
                f"enrichment_variant must be one of 'raw', 'stable', "
                f"'tikhonov', 'auto', got {enrichment_variant!r}"
            )
        self.enrichment_variant = enrichment_variant
        self.tikhonov_lambda = float(tikhonov_lambda)
        self.auto_tap_ratio_threshold = float(auto_tap_ratio_threshold)
        # Populated by compute_impedance when variant="auto" runs the
        # two-pass solve so currents_at_knots can index the enrichment
        # block consistently. None means "ignore auto-selection".
        self._auto_active_junctions = None

        self.junctions = []
        if junctions is not None:
            for j, jw in enumerate(junctions):
                if len(jw) < 2:
                    raise ValueError(f"junction {j}: need >= 2 wire-ends")
                normalized = []
                for w, end in jw:
                    if not (0 <= w < n_w):
                        raise ValueError(
                            f"junction {j}: wire_idx {w} out of range [0, {n_w})"
                        )
                    if end not in ("start", "end"):
                        raise ValueError(
                            f"junction {j}: end must be 'start' or 'end', got {end!r}"
                        )
                    normalized.append((int(w), end))
                self.junctions.append(normalized)

        # `compute_impedance(...)` and `currents_at_knots(coeffs)` both call
        # `_build_geometry()` + `_build_basis_polynomials(geom)` from scratch
        # — on the N=21 hentenna width-sweep harness that's ~90 ms/step of
        # repeated Python work (see scripts/vtune_hentenna_width_sweep.py).
        # Cache both on the instance. Neither depends on `k` (only on the
        # immutable geometry inputs + degree + junctions), so
        # `compute_impedance_swept`'s per-k loop also benefits.
        self._cached_geometry: dict | None = None
        self._cached_basis_polynomials: tuple | None = None

    # ------------------------------------------------------------------
    # Geometry build
    # ------------------------------------------------------------------

    def _build_geometry(self):
        """Discretize all wires, concatenate to global arrays.

        Per-wire metadata is preserved so the basis-polynomial extraction
        (which operates on each wire's clamped knot vector independently)
        can be done wire-by-wire.

        Returns a `geom` dict with:
          per_wire: list of per-wire dicts (seg_l, seg_r, tangents, h_per_seg,
              edge_offsets, edge_arc_edges, arc_at_knot, n_total)
          seg_offsets: list[n_w+1] of global segment index of wire start
          n_segs_total: total segment count across all wires
          h_per_seg: (N_total,) per-segment edge length
          tangents: (N_total, 3) per-segment tangent unit vector
          seg_l, seg_r: (N_total, 3) per-segment 3D endpoint
        """
        if self._cached_geometry is not None:
            return self._cached_geometry
        # Module cache hit → reuse the geom dict identity-stably across
        # solver instances. The basis-polynomial instance cache keys on
        # `cached_geom is geom`, so returning the same object also lets
        # the basis-poly cache resolve through the module path below.
        geom_key = self._geometry_cache_key()
        cached = _GEOMETRY_CACHE.get(geom_key)
        if cached is not None:
            self._cached_geometry = cached
            return cached
        per_wire = []
        seg_offsets = [0]
        h_list = []
        tangents_list = []
        seg_l_list_all = []
        seg_r_list_all = []
        for w_idx, (pl, npe_list) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            seg_l_w = []
            seg_r_w = []
            tan_w = []
            h_w_list = []
            edge_offsets = [0]
            edge_arc_edges = []
            for e_idx in range(pl.shape[0] - 1):
                p0 = pl[e_idx]
                p1 = pl[e_idx + 1]
                edge_vec = p1 - p0
                edge_len = float(np.linalg.norm(edge_vec))
                if edge_len < 1e-15:
                    raise ValueError(f"wire {w_idx} edge {e_idx} has zero length")
                tan = edge_vec / edge_len
                n_e = npe_list[e_idx]
                h_e = edge_len / n_e

                t_node = np.linspace(0.0, 1.0, n_e + 1)
                pts = (1 - t_node[:, None]) * p0[None, :] + t_node[:, None] * p1[
                    None, :
                ]
                seg_l_w.append(pts[:-1])
                seg_r_w.append(pts[1:])
                tan_w.append(np.tile(tan, (n_e, 1)))
                h_w_list.append(np.full(n_e, h_e))
                edge_arc_edges.append(np.linspace(0.0, edge_len, n_e + 1))
                edge_offsets.append(edge_offsets[-1] + n_e)

            seg_l = np.vstack(seg_l_w)
            seg_r = np.vstack(seg_r_w)
            tangents_w = np.vstack(tan_w)
            h_per_seg_w = np.concatenate(h_w_list)
            n_total_w = seg_l.shape[0]
            arc_at_knot = np.concatenate([[0.0], np.cumsum(h_per_seg_w)])

            per_wire.append(
                {
                    "seg_l": seg_l,
                    "seg_r": seg_r,
                    "tangents": tangents_w,
                    "h_per_seg": h_per_seg_w,
                    "edge_offsets": edge_offsets,
                    "edge_arc_edges": edge_arc_edges,
                    "arc_at_knot": arc_at_knot,
                    "n_total": n_total_w,
                }
            )
            seg_offsets.append(seg_offsets[-1] + n_total_w)
            h_list.append(h_per_seg_w)
            tangents_list.append(tangents_w)
            seg_l_list_all.append(seg_l)
            seg_r_list_all.append(seg_r)

        h_per_seg_global = np.concatenate(h_list)
        tangents_global = np.vstack(tangents_list)
        seg_l_global = np.vstack(seg_l_list_all)
        seg_r_global = np.vstack(seg_r_list_all)

        self._cached_geometry = {
            "per_wire": per_wire,
            "seg_offsets": seg_offsets,
            "n_segs_total": seg_offsets[-1],
            "h_per_seg": h_per_seg_global,
            "tangents": tangents_global,
            "seg_l": seg_l_global,
            "seg_r": seg_r_global,
        }
        _evict_fifo(_GEOMETRY_CACHE, _GEOMETRY_CACHE_MAX)
        _GEOMETRY_CACHE[geom_key] = self._cached_geometry
        return self._cached_geometry

    def _geometry_cache_key(self):
        # Bytes view of each wire's float64 polyline + per-wire segmentation.
        # Both are immutable post-__init__, and the geom dict depends on
        # exactly these (see _build_geometry body).
        return (
            tuple(w.tobytes() for w in self.wires_polylines),
            tuple(tuple(npe) for npe in self.n_per_edge_per_wire),
        )

    # ------------------------------------------------------------------
    # Endpoint status (free vs junction)
    # ------------------------------------------------------------------

    def _wire_endpoint_status(self):
        """For each wire, return ("free" | junction_idx, "free" | junction_idx)
        for its (start, end) — the index of the junction connecting it, or
        "free" if the endpoint isn't junctioned.
        """
        n_w = len(self.wires_polylines)
        start_status = ["free"] * n_w
        end_status = ["free"] * n_w
        for j_idx, jw in enumerate(self.junctions):
            for w, end in jw:
                if end == "start":
                    start_status[w] = j_idx
                else:
                    end_status[w] = j_idx
        return start_status, end_status

    # ------------------------------------------------------------------
    # Basis polynomial extraction
    # ------------------------------------------------------------------

    def _build_basis_polynomials(self, geom):
        """Extract polynomial coefficients per (basis, wing).

        For each wire:
          * Build clamped knot vector on the wire's cumulative arc.
          * Determine which of the d+1 boundary bases per end are kept:
              - Free end: drop all d+1 boundary bases (Φ(end) = 0 strictly,
                  AND derivative 0, etc. — for d ≤ 2 this means drop just
                  B_0 because only B_0 has nonzero value, and the higher
                  boundary bases are kept as ordinary interior bases since
                  their value at the end is 0).
              - Junction end: keep the value-1 boundary basis B_0 as a
                  directional basis; keep B_1..B_{d-1} as interior bases.
          * Extract per-segment polynomial coefficients via BSpline +
            Vandermonde (uniform within each segment's local-u range).

        Returns
        -------
        supp_seg, polys : as in the single-wire case, concatenated globally.
        kcl_A : (n_junctions, n_basis_total) Lagrange-multiplier rows
            (+1 / -1 outflow sign per directional basis).
        wire_knots : list of per-wire knot vectors (for the source vector).
        wire_basis_global : list of per-wire (kept_idx, global_basis_idx)
            tuples for the source-vector mapping.
        """
        # Cache key is geometry identity: the result depends only on `geom`
        # (per-wire arc knots), `self.degree`, and `self.junctions` (via
        # _wire_endpoint_status); none change after __init__, so a cached
        # result computed against the same geom dict is still valid.
        cached_geom = self._cached_geometry
        if cached_geom is geom and self._cached_basis_polynomials is not None:
            return self._cached_basis_polynomials
        # Module cache promotes the per-instance memoization across solver
        # instances (the engine wrapper recreates the solver per impedance()
        # call). Key is geometry signature + degree + junctions; the result
        # is k-independent.
        basis_key = (
            self._geometry_cache_key(),
            self.degree,
            tuple(tuple((w, e) for (w, e) in j) for j in self.junctions),
        )
        cached_basis = _BASIS_POLY_CACHE.get(basis_key)
        if cached_basis is not None:
            if cached_geom is geom:
                self._cached_basis_polynomials = cached_basis
            return cached_basis
        d = self.degree
        n_wings = d + 1
        n_poly = d + 1

        start_status, end_status = self._wire_endpoint_status()

        all_supp_seg = []
        all_polys = []
        wire_knots = []
        wire_basis_global = []
        # Track per-junction the list of (directional-basis global idx,
        # outflow sign).
        junction_dirs = {j: [] for j in range(len(self.junctions))}

        m_global = 0
        for w_idx, pw in enumerate(geom["per_wire"]):
            arc = pw["arc_at_knot"]
            wire_arc = arc[-1]
            interior_knots = arc.copy()
            knots = np.concatenate(
                [np.full(d, 0.0), interior_knots, np.full(d, wire_arc)]
            )
            wire_knots.append(knots)
            n_basis_w = len(knots) - d - 1  # = N_w + d

            # Determine kept bases. For d ∈ {1, 2}:
            #   B_0 is the value-1 boundary basis at the start
            #   B_{n_basis_w - 1} is the value-1 boundary basis at the end
            #   B_1, ..., B_{n_basis_w - 2} are interior (value 0 at endpoints)
            kept = []  # list of (basis_j, kind, junction_idx-or-None)
            # Start boundary basis (B_0)
            if start_status[w_idx] == "free":
                pass  # drop
            else:
                kept.append((0, "dir", start_status[w_idx], "start"))
            # Truly interior bases
            for j in range(1, n_basis_w - 1):
                kept.append((j, "int", None, None))
            # End boundary basis (B_{n_basis_w - 1})
            if end_status[w_idx] == "free":
                pass  # drop
            else:
                kept.append((n_basis_w - 1, "dir", end_status[w_idx], "end"))

            seg_off = geom["seg_offsets"][w_idx]
            h_per_seg_w = pw["h_per_seg"]
            arc_at_knot_w = pw["arc_at_knot"]
            n_total_w = pw["n_total"]

            # Vectorize: per-wire single BSpline.design_matrix + constant
            # V_unit_inv lookup replaces per-(basis, wing) BSpline
            # construction + per-segment linspace/vander/solve.
            #
            # Sample points: d+1 uniform u within each segment, in global
            # arc. shape (n_total_w, d+1) → flatten for design_matrix.
            unit = np.linspace(0.0, 1.0, d + 1)  # (d+1,) shared across segs
            u_local_per_seg = h_per_seg_w[:, None] * unit[None, :]  # (N, d+1)
            u_global_per_seg = arc_at_knot_w[:-1, None] + u_local_per_seg
            u_flat = u_global_per_seg.reshape(-1)

            # All basis values at all sample points in one design_matrix call.
            DM = BSpline.design_matrix(u_flat, knots, d).toarray()
            # → (n_total_w, d+1, n_basis_w)
            DM_seg = DM.reshape(n_total_w, d + 1, n_basis_w)

            # V_unit_inv @ vals: convert d+1 basis values per segment to
            # poly coeffs (in u_local). Then divide by h_seg^p column-wise
            # to recover coeffs in u_local = h_seg · u_unit terms.
            V_unit_inv = _V_UNIT_INV[d]
            inv_h_powers = h_per_seg_w[:, None] ** (-np.arange(d + 1))
            # → (N, d+1, n_basis_w): for each segment, polynomial coeff p
            # of each basis function expressed as Σ_p coeffs_p · u_local^p
            poly_per_seg = np.einsum("ij,sjk->sik", V_unit_inv, DM_seg)
            poly_per_seg *= inv_h_powers[:, :, None]

            # Per-basis support range as half-open segment indices [lo, hi).
            # knots = [0]*d + arc + [wire_arc]*d, so knots[j] sits at
            # segment index max(0, j - d), and knots[j+d+1] sits at
            # min(N, j+1). Result: basis j has wings = segments
            # max(0, j-d) .. min(N, j+1) - 1.

            per_basis_local_to_global = {}
            for kept_idx, (j, kind, junc_idx, end_pos) in enumerate(kept):
                seg_lo = max(0, j - d)
                seg_hi = min(n_total_w, j + 1)
                n_actual = seg_hi - seg_lo

                supp_seg_m = np.zeros(n_wings, dtype=np.int64)
                polys_m = np.zeros((n_wings, n_poly), dtype=np.float64)
                supp_seg_m[:n_actual] = seg_off + np.arange(seg_lo, seg_hi)
                polys_m[:n_actual, :] = poly_per_seg[seg_lo:seg_hi, :, j]

                all_supp_seg.append(supp_seg_m)
                all_polys.append(polys_m)
                per_basis_local_to_global[kept_idx] = m_global

                if kind == "dir":
                    sign = +1.0 if end_pos == "start" else -1.0
                    junction_dirs[junc_idx].append((m_global, sign))

                m_global += 1

            wire_basis_global.append((kept, per_basis_local_to_global))

        supp_seg = (
            np.stack(all_supp_seg, axis=0)
            if all_supp_seg
            else (np.zeros((0, n_wings), dtype=np.int64))
        )
        polys = (
            np.stack(all_polys, axis=0)
            if all_polys
            else (np.zeros((0, n_wings, n_poly), dtype=np.float64))
        )
        n_basis_total = supp_seg.shape[0]

        n_junctions = len(self.junctions)
        kcl_A = np.zeros((n_junctions, n_basis_total), dtype=np.float64)
        for j_idx, dirs in junction_dirs.items():
            for m_g, sign in dirs:
                kcl_A[j_idx, m_g] = sign

        result = (supp_seg, polys, kcl_A, wire_knots, wire_basis_global)
        if cached_geom is geom:
            self._cached_basis_polynomials = result
        _evict_fifo(_BASIS_POLY_CACHE, _BASIS_POLY_CACHE_MAX)
        _BASIS_POLY_CACHE[basis_key] = result
        return result

    # ------------------------------------------------------------------
    # J moment integrals
    # ------------------------------------------------------------------

    def _image_positions(self, positions):
        """Mirror an array of 3D positions across z = ground_z."""
        out = positions.copy()
        out[..., 2] = 2 * self.ground_z - out[..., 2]
        return out

    def _image_tangent_dot(self, tangents):
        """t_m · t_image_n with t_image_n = (t_n_x, t_n_y, -t_n_z)."""
        return tangents @ (tangents * np.array([1.0, 1.0, -1.0])).T

    def _build_J_image_blocks(self, geom, k):
        """Build the J moment tensor with j-segments mirrored across the
        PEC ground plane. The image is always far enough from the original
        that the analytic same-edge static + reg split doesn't apply — full
        off-edge quadrature handles every (i, j) pair uniformly.
        """
        a = self.wire_radius
        d = self.degree
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        seg_l_img = self._image_positions(seg_l)
        seg_r_img = self._image_positions(seg_r)
        return _seg_seg_full_moments_offedge(
            seg_l, seg_r, seg_l_img, seg_r_img, a, k, d, self.n_qp_pair
        )

    def _same_edge_prep(self, geom):
        """k-independent per-same-edge precompute hoisted out of the swept-k
        loop: each edge's analytic static-moment block plus the reg-kernel
        quadrature geometry (R table + weighted powers). Returns a list of
        `(global_slice, A_static, reg_geometry)`. Bounded memory — one edge's
        tables at a time, identical footprint to the per-k path; only the
        cheap `exp(-jkR)` + einsum is left per k.
        """
        d = self.degree
        a = self.wire_radius
        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        prep = []
        for w in range(len(per_wire)):
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            base = seg_off[w]
            for i_e in range(len(ed_off) - 1):
                sl = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                A_st = _seg_seg_static_moments(ed_arc[i_e], a, max_d=d)
                reg_geo = _seg_seg_reg_geometry(
                    ed_arc[i_e], a, max_d=d, n_qp=self.n_qp_pair
                )
                prep.append((sl, A_st, reg_geo))
        return prep

    def _build_J_blocks(self, geom, k, same_edge_prep=None):
        """All polynomial moment integrals J_pq[i, j] for p, q ∈ {0..d} and
        every (i, j) global segment pair. Returns shape (d+1, d+1, N, N).

        Fused build mirroring TriangularSolver._build_J_blocks: first compute
        every pair by full GL quadrature on the regularized full kernel
        G = exp(-jkR)/(4πR), R² = |Δr|² + a²; then overwrite same-edge
        blocks with the analytic static + GL-regularized split (essential
        for the log-singular diagonal).

        `same_edge_prep` (from `_same_edge_prep`) lets a swept-k caller share
        the k-independent static + reg-geometry across frequencies; when None
        the same quantities are computed inline (single-k path).
        """
        d = self.degree
        a = self.wire_radius
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]

        # All-pairs full kernel (same a² regularization handles touching
        # segments at kink corners and at junctions to within ~1e-5 at
        # antenna scales; off-segment-pair accuracy is what GL is good at).
        J = _seg_seg_full_moments_offedge(
            seg_l, seg_r, seg_l, seg_r, a, k, d, self.n_qp_pair
        )  # (d+1, d+1, N, N) complex

        # Overwrite each same-edge block with analytic static + reg
        if same_edge_prep is None:
            per_wire = geom["per_wire"]
            seg_off = geom["seg_offsets"]
            for w in range(len(per_wire)):
                pw = per_wire[w]
                ed_off = pw["edge_offsets"]
                ed_arc = pw["edge_arc_edges"]
                base = seg_off[w]
                for i_e in range(len(ed_off) - 1):
                    sl = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                    A_st = _seg_seg_static_moments(ed_arc[i_e], a, max_d=d)
                    A_reg = _seg_seg_reg_moments(
                        ed_arc[i_e], a, k, max_d=d, n_qp=self.n_qp_pair
                    )
                    J[:, :, sl, sl] = A_st + A_reg
        else:
            for sl, A_st, reg in same_edge_prep:
                # `reg` is either a reg-geometry dict (compute this k's block
                # now) or a precomputed (max_d+1, max_d+1, N, N) moment block
                # for this k (swept caller batched it across frequencies).
                A_reg = (
                    _seg_seg_reg_moments_from_geometry(reg, k)
                    if isinstance(reg, dict)
                    else reg
                )
                J[:, :, sl, sl] = A_st + A_reg

        return J

    # ------------------------------------------------------------------
    # Z assembly
    # ------------------------------------------------------------------

    def _assemble_Z(self, J, supp_seg, polys, geom, td_all=None):
        """Assemble the (n_basis, n_basis) complex Z matrix.

        Uses the templated C++ accelerator `assemble_Z_bspline` when
        available and `self.degree` is in its instantiation set; otherwise
        falls back to a numpy-einsum implementation that's a bit-exact
        reference target.

        `td_all` defaults to the free-space tangent dot product matrix
        derived from `geom["tangents"]`. The PEC image build passes its
        own (tx, ty, -tz)-modified table here so the same assembly fuses
        the image-current sign flip.
        """
        d = self.degree
        n_basis, n_wings, n_poly = polys.shape
        assert n_wings == d + 1 and n_poly == d + 1

        if td_all is None:
            tangents = geom["tangents"]
            td_all = tangents @ tangents.T

        if _HAVE_BSPLINE_ASSEMBLE_ACCEL and d <= _BSPLINE_ASSEMBLE_ACCEL_MAX_D:
            return _acc.assemble_Z_bspline(
                np.ascontiguousarray(J, dtype=np.complex128),
                np.ascontiguousarray(supp_seg, dtype=np.int64),
                np.ascontiguousarray(polys, dtype=np.float64),
                np.ascontiguousarray(td_all, dtype=np.float64),
                float(self.omega),
                float(self.eps),
                float(self.mu),
                int(d),
            )

        Z_A = np.zeros((n_basis, n_basis), dtype=np.complex128)
        Z_Phi = np.zeros((n_basis, n_basis), dtype=np.complex128)
        p_vec = np.arange(1, d + 1, dtype=np.float64) if d >= 1 else None

        for a in range(n_wings):
            sm = supp_seg[:, a]
            for b in range(n_wings):
                sn = supp_seg[:, b]
                J_blk = J[:, :, sm[:, None], sn[None, :]]
                td_blk = td_all[sm[:, None], sn[None, :]]

                inner_A = np.einsum(
                    "mp,pPmn,nP->mn", polys[:, a, :], J_blk, polys[:, b, :]
                )
                Z_A += td_blk * inner_A

                if d >= 1:
                    deriv_m = polys[:, a, 1:] * p_vec[None, :]
                    deriv_n = polys[:, b, 1:] * p_vec[None, :]
                    J_blk_lo = J_blk[:d, :d]
                    inner_Phi = np.einsum("mp,pPmn,nP->mn", deriv_m, J_blk_lo, deriv_n)
                    Z_Phi += inner_Phi

        Z_A = 1j * self.omega * self.mu * Z_A
        Z_Phi = Z_Phi / (1j * self.omega * self.eps)
        return Z_A + Z_Phi

    # ------------------------------------------------------------------
    # Source vector
    # ------------------------------------------------------------------

    def _build_source_vector(
        self,
        geom,
        wire_knots,
        wire_basis_global,
        n_basis_total,
        wi=None,
        s_f=None,
    ):
        """Galerkin RHS for either a delta-gap or smoothed source.

        Delta-gap (feed_smoothing_factor=None): v_m = Φ_m(s_f).

        Smoothed source: replace V·δ(s − s_f) with V·g_w(s − s_f) where g_w
        is a cos² bump of integral 1 and half-width w/2 = α·h_feed/2 (with
        α = self.feed_smoothing_factor). Then v_m = ⟨Φ_m, g_w(. − s_f)⟩,
        computed by Gauss-Legendre quadrature on the bump's support. The
        impedance extraction in `compute_impedance` is unchanged:
        I_in = v^T c gives the smoothing-weighted current, and Z = 1/I_in.
        In the α → 0 limit g_w → δ and both v and I_in revert to the
        delta-gap formulas.

        Why this fixes the convergence rate. The delta-gap source produces a
        log singularity in the current at s_f that no polynomial basis can
        represent; the integrated impedance picks up an O(1/N) error term
        regardless of basis degree. The smoothed source has no singularity,
        so the convergence is basis-limited (O(1/N³) for d=2).

        For unit excitation V=1 at a single (wi, s_f); the multi-feed
        caller scales each per-feed vector by V_i and sums them. wi/s_f
        default to self.feeds[0] for back-compat with single-feed callers.
        """
        d = self.degree
        if wi is None:
            wi = self.feeds[0][0]
        arc = geom["per_wire"][wi]["arc_at_knot"]
        wire_arc = arc[-1]
        if s_f is None:
            arc_req = self.feeds[0][1]
            s_f = arc_req if arc_req is not None else wire_arc / 2.0
        knots = wire_knots[wi]
        kept, local_to_global = wire_basis_global[wi]

        if self.feed_smoothing_factor is None:
            # Delta-gap (original)
            DM = BSpline.design_matrix(np.array([s_f]), knots, d).toarray()[0]
            v = np.zeros(n_basis_total, dtype=np.complex128)
            for kept_idx, (j, _kind, _junc_idx, _end_pos) in enumerate(kept):
                m_global = local_to_global[kept_idx]
                v[m_global] = DM[j]
            return v

        # Smoothed source: find the feed segment to set the smoothing width
        # w = α·h_feed. The "feed segment" is the segment containing s_f.
        h_per_seg = geom["per_wire"][wi]["h_per_seg"]
        arc_at_knot = arc
        # Locate segment such that arc_at_knot[seg] <= s_f < arc_at_knot[seg+1]
        seg_idx = int(np.searchsorted(arc_at_knot, s_f, side="right")) - 1
        seg_idx = max(0, min(seg_idx, len(h_per_seg) - 1))
        h_feed = float(h_per_seg[seg_idx])
        alpha = float(self.feed_smoothing_factor)
        smoothing_w = alpha * h_feed
        half_w = smoothing_w / 2.0

        # Clip to wire arc range — if the feed is too close to a wire end,
        # the bump may not fit; in that case the integral is just over the
        # available portion (consistent with the smoothed source convention
        # but breaks symmetry at the wire end).
        s_lo = max(0.0, s_f - half_w)
        s_hi = min(wire_arc, s_f + half_w)
        if s_lo >= s_hi:
            raise ValueError(
                "feed_smoothing_factor too large for wire — bump doesn't fit"
            )

        gl_xi, gl_w = leggauss(self.n_qp_source)
        t = 0.5 * (s_hi + s_lo) + 0.5 * (s_hi - s_lo) * gl_xi
        weights = 0.5 * (s_hi - s_lo) * gl_w

        # Cos² bump on |x| < smoothing_w/2:
        #   g_w(x) = (2/smoothing_w) · cos²(π x / smoothing_w)
        # so ∫ g_w = 1 (since ∫_{-w/2}^{w/2} cos²(πx/w) dx = w/2).
        delta = t - s_f
        in_support = np.abs(delta) < half_w
        g_vals = np.where(
            in_support,
            (2.0 / smoothing_w) * np.cos(np.pi * delta / smoothing_w) ** 2,
            0.0,
        )

        # Evaluate every basis at the quadrature points and integrate.
        DM = BSpline.design_matrix(t, knots, d).toarray()  # (n_qp, n_basis_w_full)
        v_full = np.einsum("qj,q,q->j", DM, g_vals, weights)  # (n_basis_w_full,)

        v = np.zeros(n_basis_total, dtype=np.complex128)
        for kept_idx, (j, _kind, _junc_idx, _end_pos) in enumerate(kept):
            m_global = local_to_global[kept_idx]
            v[m_global] = v_full[j]
        return v

    # ------------------------------------------------------------------
    # KCL solve (Schur complement)
    # ------------------------------------------------------------------

    def _solve_with_kcl(self, Z, v, kcl_A):
        """Constrained solve [Z A^T; A 0] [I; λ] = [v; 0] via Schur.

        Identical structure to TriangularSolver._solve_with_kcl. If kcl_A
        is empty (no junctions), do a plain solve.
        """
        if kcl_A.shape[0] == 0:
            return scipy.linalg.solve(Z, v)
        n_b = Z.shape[0]
        n_c = kcl_A.shape[0]
        rhs = np.empty((n_b, 1 + n_c), dtype=np.complex128)
        rhs[:, 0] = v
        rhs[:, 1:] = kcl_A.T
        sol = scipy.linalg.solve(Z, rhs)
        w = sol[:, 0]
        X = sol[:, 1:]
        lam = scipy.linalg.solve(kcl_A @ X, kcl_A @ w)
        return w - X @ lam

    def _solve_with_kcl_ports(self, Z, V, kcl_A):
        """Multi-port KCL-constrained Schur solve. V: (n_b, n_p), returns
        (n_b, n_p). Matrix-RHS generalisation of `_solve_with_kcl` — all
        n_p source columns share one LU factorisation with the n_c
        constraint columns.
        """
        if kcl_A.shape[0] == 0:
            return scipy.linalg.solve(Z, V)
        n_b, n_p = V.shape
        n_c = kcl_A.shape[0]
        rhs = np.empty((n_b, n_p + n_c), dtype=np.complex128)
        rhs[:, :n_p] = V
        rhs[:, n_p:] = kcl_A.T
        sol = scipy.linalg.solve(Z, rhs)
        W = sol[:, :n_p]
        X = sol[:, n_p:]
        Lam = scipy.linalg.solve(kcl_A @ X, kcl_A @ W)
        return W - X @ Lam

    # ------------------------------------------------------------------
    # Driver impedance
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Singular basis enrichment at K≥3 junctions
    # ------------------------------------------------------------------

    def _enrichment_specs(self, geom, active_junction_indices=None):
        """For each (wire, end_pos) at a K≥enrichment_min_k junction, return
        the global segment index of the adjacent segment and the "orientation"
        flag: u_local from junction = u_seg_left if end_pos == "start",
        h - u_seg_left if end_pos == "end".

        When `active_junction_indices` is provided (a container of indices
        into self.junctions), only those junctions get specs — for the
        "auto" variant's per-junction enrichment selectivity. None means
        all qualifying junctions enrich (raw / stable / tikhonov path).
        """
        specs = []  # list of (junction_idx, wire_w, end_pos, seg_idx, u_origin)
        active_set = (
            None if active_junction_indices is None else set(active_junction_indices)
        )
        for j_idx, jw in enumerate(self.junctions):
            if len(jw) < self.enrichment_min_k:
                continue
            if active_set is not None and j_idx not in active_set:
                continue
            for wire_w, end_pos in jw:
                if end_pos == "start":
                    seg_idx = geom["seg_offsets"][wire_w]
                    u_origin = "left"  # u from junction = u_seg_left
                else:
                    seg_idx = geom["seg_offsets"][wire_w + 1] - 1
                    u_origin = "right"  # u from junction = h - u_seg_left
                specs.append((j_idx, wire_w, end_pos, seg_idx, u_origin))
        return specs

    def _junction_tap_ratios(self, coeffs_poly):
        """Compute tap_ratio = min(|I_wire|) / max(|I_wire|) at each
        junction node, using the polynomial-only coefficient vector from
        a no-enrichment solve. Returns a list of length len(self.junctions);
        entries are None for junctions with K < enrichment_min_k.

        At a junction node only the bspline directional bases are nonzero
        (one per wire end), so the wire-by-wire current magnitudes can be
        read directly out of `currents_at_knots` at the junction-side knot.
        Per-wire |I| values at a K-junction sum to zero by KCL (vector
        sum), but individual magnitudes need not be equal — the ratio
        captures how lopsided the split is.
        """
        # Temporarily disable enrichment so currents_at_knots doesn't try
        # to index past the polynomial block.
        saved = self.use_singular_enrichment
        self.use_singular_enrichment = False
        try:
            I_per_wire = self.currents_at_knots(coeffs_poly)
        finally:
            self.use_singular_enrichment = saved

        ratios = []
        for jw in self.junctions:
            if len(jw) < self.enrichment_min_k:
                ratios.append(None)
                continue
            mags = []
            for wire_w, end_pos in jw:
                knot_idx = 0 if end_pos == "start" else -1
                mags.append(abs(I_per_wire[wire_w][knot_idx]))
            m_max = max(mags)
            ratios.append(0.0 if m_max == 0.0 else min(mags) / m_max)
        return ratios

    @staticmethod
    def _assemble_Z_enrich_numpy(
        spec_seg,
        spec_origin,
        seg_l,
        seg_r,
        h_per_seg,
        td_all,
        supp_seg_poly,
        polys_poly,
        a_squared,
        k,
        omega,
        eps_,
        mu_,
        gl_t01,
        gl_w01,
        proj_coeffs,
    ):
        """Pure-numpy reference for the C++ `assemble_Z_enrich` kernel.

        Mirrors the C++ structure (precompute Φ_sing values/derivatives
        and 3D positions at the enrichment-side quadrature nodes, then
        nested-loop Galerkin sums for Z_ee / Z_pe / Z_ep) so the parity
        test in tests/test_momwire.py catches any drift between the two.

        Also lets BSplineSolver(use_singular_enrichment=True) work without
        the C++ accelerator at all — Windows users hit this path because
        setup.py skips the Pybind11Extension under MSVC (the GCC-only
        `-fopenmp` / `-mavx2` / `-lmvec` flags don't link there).

        Same argument names and semantics as the C++ binding. `proj_coeffs`
        of length d+1 selects raw (all zeros → Φ_sing = t·log(t)) vs
        stable XFEM (subtract Σ_p proj_coeffs[p] t^p from both Φ and its
        derivative).
        """
        n_enrich = spec_seg.shape[0]
        n_poly, n_wings = supp_seg_poly.shape
        d_plus_1 = polys_poly.shape[2]
        n_qp = gl_t01.shape[0]

        Z_pe = np.zeros((n_poly, n_enrich), dtype=np.complex128)
        Z_ep = np.zeros((n_enrich, n_poly), dtype=np.complex128)
        Z_ee = np.zeros((n_enrich, n_enrich), dtype=np.complex128)
        if n_enrich == 0:
            return Z_pe, Z_ep, Z_ee

        inv_4pi = 1.0 / (4.0 * np.pi)
        omega_mu = omega * mu_
        inv_omega_eps = 1.0 / (omega * eps_)
        eps_tiny = 1e-300

        # Derivative coefficients in monomial basis for the proj polynomial:
        # P(t) = Σ_p c_p t^p ⇒ P'(t) = Σ_{p≥1} p·c_p t^(p-1).
        proj_deriv_coeffs = proj_coeffs[1:] * np.arange(1, d_plus_1)

        # Per-enrichment precompute.
        pos_e_all = np.zeros((n_enrich, n_qp, 3))
        sing_val_all = np.zeros((n_enrich, n_qp))
        sing_dval_all = np.zeros((n_enrich, n_qp))
        w_e_all = np.zeros((n_enrich, n_qp))
        polyval = np.polynomial.polynomial.polyval
        for e in range(n_enrich):
            se = int(spec_seg[e])
            orig = int(spec_origin[e])
            he = h_per_seg[se]
            dphi_sign = 1.0 if orig == 0 else -1.0
            t = gl_t01
            u_norm = t if orig == 0 else (1.0 - t)
            u_safe = np.where(u_norm > eps_tiny, u_norm, eps_tiny)
            log_u = np.log(u_safe)
            poly_val = polyval(u_norm, proj_coeffs)
            if proj_deriv_coeffs.size > 0:
                poly_dval = polyval(u_norm, proj_deriv_coeffs)
            else:
                poly_dval = np.zeros_like(u_norm)
            sing_val_all[e] = u_norm * log_u - poly_val
            sing_dval_all[e] = dphi_sign * (log_u + 1.0 - poly_dval) / he
            w_e_all[e] = gl_w01 * he
            pos_e_all[e, :, 0] = (1.0 - t) * seg_l[se, 0] + t * seg_r[se, 0]
            pos_e_all[e, :, 1] = (1.0 - t) * seg_l[se, 1] + t * seg_r[se, 1]
            pos_e_all[e, :, 2] = (1.0 - t) * seg_l[se, 2] + t * seg_r[se, 2]

        seg_e_arr = spec_seg.astype(np.int64, copy=False)

        # Z_ee: symmetric. Fill upper triangle, mirror.
        for e in range(n_enrich):
            for f in range(e, n_enrich):
                td = td_all[seg_e_arr[e], seg_e_arr[f]]
                diff = pos_e_all[e, :, None, :] - pos_e_all[f, None, :, :]
                R = np.sqrt(np.sum(diff * diff, axis=-1) + a_squared)
                iR_4pi = inv_4pi / R
                phase = -k * R
                Gre = np.cos(phase) * iR_4pi
                Gim = np.sin(phase) * iR_4pi
                wprod_A = (w_e_all[e] * sing_val_all[e])[:, None] * (
                    w_e_all[f] * sing_val_all[f]
                )[None, :]
                wprod_P = (w_e_all[e] * sing_dval_all[e])[:, None] * (
                    w_e_all[f] * sing_dval_all[f]
                )[None, :]
                IA_re = np.sum(wprod_A * Gre)
                IA_im = np.sum(wprod_A * Gim)
                IP_re = np.sum(wprod_P * Gre)
                IP_im = np.sum(wprod_P * Gim)
                # Z = jωμ·td·I_A + I_Φ / (jωε)
                Zre = -omega_mu * td * IA_im + IP_im * inv_omega_eps
                Zim = omega_mu * td * IA_re - IP_re * inv_omega_eps
                Z_ee[e, f] = complex(Zre, Zim)
                if e != f:
                    Z_ee[f, e] = Z_ee[e, f]

        # Z_pe / Z_ep. Loop over polynomial bases and their wings;
        # for each wing, precompute poly value/derivative + 3D quad-point
        # positions on the wing's segment, then loop over enrichments.
        # Z_pe and Z_ep are computed independently (no .T shortcut) to
        # mirror the C++ kernel — the kernel is symmetric, so the totals
        # agree to floating-point rounding either way.
        for m in range(n_poly):
            for w_idx in range(n_wings):
                cw = polys_poly[m, w_idx]
                if not np.any(cw != 0.0):
                    continue
                seg_m = int(supp_seg_poly[m, w_idx])
                hm = h_per_seg[seg_m]
                t = gl_t01
                u_arc = t * hm
                pv = polyval(u_arc, cw)
                cw_deriv = cw[1:] * np.arange(1, d_plus_1)
                if cw_deriv.size > 0:
                    dv = polyval(u_arc, cw_deriv)
                else:
                    dv = np.zeros_like(u_arc)
                w_m = gl_w01 * hm
                pos_m = np.empty((n_qp, 3))
                pos_m[:, 0] = (1.0 - t) * seg_l[seg_m, 0] + t * seg_r[seg_m, 0]
                pos_m[:, 1] = (1.0 - t) * seg_l[seg_m, 1] + t * seg_r[seg_m, 1]
                pos_m[:, 2] = (1.0 - t) * seg_l[seg_m, 2] + t * seg_r[seg_m, 2]
                for e in range(n_enrich):
                    seg_e = int(seg_e_arr[e])
                    td_me = td_all[seg_m, seg_e]
                    td_em = td_all[seg_e, seg_m]
                    # Z_pe leg: i = m-axis, j = e-axis
                    diff = pos_m[:, None, :] - pos_e_all[e, None, :, :]
                    R = np.sqrt(np.sum(diff * diff, axis=-1) + a_squared)
                    iR_4pi = inv_4pi / R
                    phase = -k * R
                    Gre = np.cos(phase) * iR_4pi
                    Gim = np.sin(phase) * iR_4pi
                    wprod_A = (w_m * pv)[:, None] * (w_e_all[e] * sing_val_all[e])[
                        None, :
                    ]
                    wprod_P = (w_m * dv)[:, None] * (w_e_all[e] * sing_dval_all[e])[
                        None, :
                    ]
                    pe_IA_re = np.sum(wprod_A * Gre)
                    pe_IA_im = np.sum(wprod_A * Gim)
                    pe_IP_re = np.sum(wprod_P * Gre)
                    pe_IP_im = np.sum(wprod_P * Gim)
                    # Z_ep leg: i = e-axis, j = m-axis
                    diff = pos_e_all[e, :, None, :] - pos_m[None, :, :]
                    R = np.sqrt(np.sum(diff * diff, axis=-1) + a_squared)
                    iR_4pi = inv_4pi / R
                    phase = -k * R
                    Gre = np.cos(phase) * iR_4pi
                    Gim = np.sin(phase) * iR_4pi
                    wprod_A = (w_e_all[e] * sing_val_all[e])[:, None] * (w_m * pv)[
                        None, :
                    ]
                    wprod_P = (w_e_all[e] * sing_dval_all[e])[:, None] * (w_m * dv)[
                        None, :
                    ]
                    ep_IA_re = np.sum(wprod_A * Gre)
                    ep_IA_im = np.sum(wprod_A * Gim)
                    ep_IP_re = np.sum(wprod_P * Gre)
                    ep_IP_im = np.sum(wprod_P * Gim)
                    Z_pe[m, e] += complex(
                        -omega_mu * td_me * pe_IA_im + pe_IP_im * inv_omega_eps,
                        omega_mu * td_me * pe_IA_re - pe_IP_re * inv_omega_eps,
                    )
                    Z_ep[e, m] += complex(
                        -omega_mu * td_em * ep_IA_im + ep_IP_im * inv_omega_eps,
                        omega_mu * td_em * ep_IA_re - ep_IP_re * inv_omega_eps,
                    )

        return Z_pe, Z_ep, Z_ee

    def _enrichment_Z_assemble(
        self, geom, supp_seg_poly, polys_poly, active_junction_indices=None
    ):
        """Assemble the (Z_pe, Z_ep, Z_ee) enrichment blocks via the C++
        accelerator (`momwire._accelerators.assemble_Z_enrich`).

        Z = [[Z_pp, Z_pe],
             [Z_ep, Z_ee]]
        with n_poly + n_enrich basis functions. Z_pp is built by the
        existing polynomial assembly; the three new blocks come from
        Gauss-Legendre quadrature over the (u·log(u/h))-shaped singular
        basis adjacent to each K≥`enrichment_min_k` junction. Z_pe and
        Z_ep are computed independently — the Galerkin .T shortcut is
        mathematically exact for symmetric kernels but the productized
        path computes both halves so a future quadrature change can't
        silently break symmetry.
        """
        specs = self._enrichment_specs(
            geom, active_junction_indices=active_junction_indices
        )
        n_enrich = len(specs)
        if n_enrich == 0:
            return None  # no qualifying junctions → no-op

        spec_seg = np.fromiter((s[3] for s in specs), dtype=np.int64, count=n_enrich)
        spec_origin = np.fromiter(
            (0 if s[4] == "left" else 1 for s in specs),
            dtype=np.int64,
            count=n_enrich,
        )

        tangents = geom["tangents"]
        td_all = np.ascontiguousarray(tangents @ tangents.T, dtype=np.float64)

        gl_xi, gl_w = leggauss(self.n_qp_sing)
        t01 = 0.5 * (gl_xi + 1.0)
        w01 = 0.5 * gl_w

        # "stable" variant subtracts the L²-projection of Φ_sing onto
        # the BC-preserving polynomial bubble subspace; "raw" and
        # "tikhonov" both send zero coefficients (the tikhonov knob is
        # applied at solve time to Z_ee, not at the basis level).
        if self.enrichment_variant == "stable":
            proj_coeffs = _xfem_projection_coeffs(self.degree)
        else:
            proj_coeffs = np.zeros(self.degree + 1)

        kernel_args = (
            spec_seg,
            spec_origin,
            np.ascontiguousarray(geom["seg_l"], dtype=np.float64),
            np.ascontiguousarray(geom["seg_r"], dtype=np.float64),
            np.ascontiguousarray(geom["h_per_seg"], dtype=np.float64),
            td_all,
            np.ascontiguousarray(supp_seg_poly, dtype=np.int64),
            np.ascontiguousarray(polys_poly, dtype=np.float64),
            float(self.wire_radius) ** 2,
            float(self.k),
            float(self.omega),
            float(self.eps),
            float(self.mu),
            np.ascontiguousarray(t01, dtype=np.float64),
            np.ascontiguousarray(w01, dtype=np.float64),
            np.ascontiguousarray(proj_coeffs, dtype=np.float64),
        )
        if _HAVE_ENRICH_ACCEL:
            Z_pe, Z_ep, Z_ee = _acc.assemble_Z_enrich(*kernel_args)
        else:
            Z_pe, Z_ep, Z_ee = self._assemble_Z_enrich_numpy(*kernel_args)

        return {
            "specs": specs,
            "n_enrich": n_enrich,
            "Z_pe": Z_pe,
            "Z_ep": Z_ep,
            "Z_ee": Z_ee,
        }

    # ------------------------------------------------------------------
    # Driver impedance
    # ------------------------------------------------------------------

    def compute_impedance(self, same_edge_prep=None):
        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )
        n_basis_total = supp_seg.shape[0]

        J = self._build_J_blocks(geom, self.k, same_edge_prep=same_edge_prep)
        Z = self._assemble_Z(J, supp_seg, polys, geom)

        if self.ground_z is not None:
            # PEC image method: subtract the same-shape assembly built from
            # J integrals over image segments + (tx, ty, -tz)-modified
            # tangent dot products. The minus sign captures both the
            # image current's horizontal anti-parallel direction and the
            # image charge's sign flip (one minus combined).
            J_img = self._build_J_image_blocks(geom, self.k)
            td_img = self._image_tangent_dot(geom["tangents"])
            Z = Z - self._assemble_Z(J_img, supp_seg, polys, geom, td_all=td_img)

        # Per-feed unit Galerkin source vectors. For multi-feed, the
        # combined RHS is Σ_i V_i · v_i, and each per-feed driving-point
        # current is I_i = v_i^T coeffs by reciprocity of the Galerkin
        # inner product (V=1 source at port i gives v_i; the current
        # sampled at port j by another source is then v_j^T · solve).
        n_feeds = len(self.feeds)
        v_per_feed = []
        for w_i, arc_i, _v in self.feeds:
            arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
            s_f_i = arc_i if arc_i is not None else arc_at_knot[-1] / 2.0
            v_per_feed.append(
                self._build_source_vector(
                    geom,
                    wire_knots,
                    wire_basis_global,
                    n_basis_total,
                    wi=w_i,
                    s_f=s_f_i,
                )
            )
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        v = np.zeros(n_basis_total, dtype=np.complex128)
        for V_i, v_i in zip(voltages, v_per_feed):
            v += V_i * v_i

        def _per_feed_z(coeffs_full):
            """Drive-point impedance per feed. coeffs_full may include the
            enrichment block; v_per_feed entries are zero on that block by
            convention (the feed is not on the enriched segment), so the
            inner product naturally restricts to the polynomial block.
            """
            currents = np.array(
                [v_i @ coeffs_full[: v_i.shape[0]] for v_i in v_per_feed],
                dtype=np.complex128,
            )
            z_per = voltages / currents
            return z_per[0] if n_feeds == 1 else z_per

        # Clear any leftover per-junction selection from a prior solve
        # (variant="auto" repopulates this below; everything else leaves
        # it as None so the standard "all qualifying junctions" path
        # runs in _enrichment_specs).
        self._auto_active_junctions = None

        active_junctions = None
        if self.use_singular_enrichment and self.enrichment_variant == "auto":
            # Pass 1: solve raw (no enrichment) to read tap_ratio at each
            # K≥enrichment_min_k junction. Below the threshold ⇒ dominant-
            # pair geometry, enrichment would absorb spurious polynomial-
            # discretization error — skip. Above ⇒ genuinely balanced
            # K-way split, enrichment captures real cusp physics — keep.
            coeffs_p1 = self._solve_with_kcl(Z, v, kcl_A)
            ratios = self._junction_tap_ratios(coeffs_p1)
            active_junctions = [
                j
                for j, r in enumerate(ratios)
                if r is not None and r > self.auto_tap_ratio_threshold
            ]
            self._auto_active_junctions = active_junctions
            if not active_junctions:
                # No junction qualifies → pass-1 result is the final answer.
                self.z = Z
                return _per_feed_z(coeffs_p1), coeffs_p1

        if self.use_singular_enrichment:
            enrich = self._enrichment_Z_assemble(
                geom, supp_seg, polys, active_junction_indices=active_junctions
            )
            if enrich is not None:
                n_p = n_basis_total
                n_e = enrich["n_enrich"]
                n_total = n_p + n_e
                Z_aug = np.zeros((n_total, n_total), dtype=np.complex128)
                Z_aug[:n_p, :n_p] = Z
                Z_aug[:n_p, n_p:] = enrich["Z_pe"]
                Z_aug[n_p:, :n_p] = enrich["Z_ep"]
                Z_aug[n_p:, n_p:] = enrich["Z_ee"]
                if self.enrichment_variant == "tikhonov":
                    # λ·s·I on the enrichment-block diagonal. s is the
                    # mean diagonal magnitude of Z_ee so λ is a
                    # dimensionless knob independent of problem scale.
                    # If Z_ee is empty (n_e=0) this block is skipped above.
                    s = float(np.mean(np.abs(np.diag(enrich["Z_ee"]))))
                    Z_aug[n_p:, n_p:] += self.tikhonov_lambda * s * np.eye(n_e)
                v_aug = np.zeros(n_total, dtype=np.complex128)
                v_aug[:n_p] = v
                # Enrichment KCL: singular bases vanish at junction → 0 outflow
                kcl_aug = np.zeros((kcl_A.shape[0], n_total), dtype=np.float64)
                kcl_aug[:, :n_p] = kcl_A
                coeffs = self._solve_with_kcl(Z_aug, v_aug, kcl_aug)
                self.z = Z_aug
                return _per_feed_z(coeffs), coeffs

        coeffs = self._solve_with_kcl(Z, v, kcl_A)
        self.z = Z
        return _per_feed_z(coeffs), coeffs

    def compute_y_matrix(self) -> np.ndarray:
        """Short-circuit admittance matrix [Y_sc] at the configured feeds.

        See `TriangularSolver.compute_y_matrix` for the math + intent.
        BSpline's per-feed driving-point current uses the Galerkin
        reciprocity I_i = v_i^T · coeffs (inner product of feed i's
        source vector with the solution). Stacking the N source
        vectors as RHS columns and back-substituting once gives
        Y[i, j] = v_i^T · solve(Z, v_j) in one shot.

        Junctions are handled through the matrix-RHS Schur solve in
        `_solve_with_kcl_ports` — KCL is enforced once across all n_p
        source columns, so the augmentation cost stays O(n_c²) per Y
        rather than scaling with the port count.

        Singular enrichment is still gated; it composes an iterative two-
        pass solve (variant="auto") that needs separate treatment.
        """
        if self.use_singular_enrichment:
            raise NotImplementedError(
                "BSplineSolver.compute_y_matrix doesn't yet support enrichment"
            )

        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )
        n_basis_total = supp_seg.shape[0]

        J = self._build_J_blocks(geom, self.k)
        Z = self._assemble_Z(J, supp_seg, polys, geom)
        if self.ground_z is not None:
            J_img = self._build_J_image_blocks(geom, self.k)
            td_img = self._image_tangent_dot(geom["tangents"])
            Z = Z - self._assemble_Z(J_img, supp_seg, polys, geom, td_all=td_img)

        # One source vector per feed, columns of B.
        n_ports = len(self.feeds)
        B = np.zeros((n_basis_total, n_ports), dtype=np.complex128)
        for j, (w_i, arc_i, _v) in enumerate(self.feeds):
            arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
            s_f_j = arc_i if arc_i is not None else arc_at_knot[-1] / 2.0
            B[:, j] = self._build_source_vector(
                geom,
                wire_knots,
                wire_basis_global,
                n_basis_total,
                wi=w_i,
                s_f=s_f_j,
            )

        X = self._solve_with_kcl_ports(Z, B, kcl_A)  # (n_basis, n_ports)
        return B.T @ X  # Y[i, j] = v_i^T · solve(Z, v_j)

    def compute_y_matrix_swept(self, k_array) -> np.ndarray:
        """Per-frequency Y matrices. Loops over k like
        `compute_impedance_swept`; returns (n_k, n_ports, n_ports).

        Junctions are handled per-k through `_solve_with_kcl_ports` — same
        matrix-RHS Schur path as the single-k Y. Singular enrichment is
        still gated."""
        if self.use_singular_enrichment:
            raise NotImplementedError(
                "BSplineSolver.compute_y_matrix_swept doesn't yet support enrichment"
            )

        k_array = np.asarray(k_array, dtype=float)
        k_save = self.k
        wl_save = self.wavelength
        omega_save = self.omega

        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )
        n_basis_total = supp_seg.shape[0]
        n_ports = len(self.feeds)

        # k-independent source vectors.
        B = np.zeros((n_basis_total, n_ports), dtype=np.complex128)
        for j, (w_i, arc_i, _v) in enumerate(self.feeds):
            arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
            s_f_j = arc_i if arc_i is not None else arc_at_knot[-1] / 2.0
            B[:, j] = self._build_source_vector(
                geom,
                wire_knots,
                wire_basis_global,
                n_basis_total,
                wi=w_i,
                s_f=s_f_j,
            )

        # k-independent static + reg-geometry, shared across the sweep; the
        # reg-kernel moment blocks are batched over all k up front (one einsum
        # per edge instead of one per (edge, k)).
        prep = self._same_edge_prep(geom)
        reg_all = [
            _seg_seg_reg_moments_from_geometry_swept(reg_geo, k_array)
            for _sl, _A_st, reg_geo in prep
        ]

        out = np.zeros((k_array.shape[0], n_ports, n_ports), dtype=np.complex128)
        for ki, kk in enumerate(k_array):
            self.k = float(kk)
            self.omega = self.k * self.c
            self.wavelength = self.c / (self.omega / (2 * np.pi))
            same_edge_k = [
                (sl, A_st, reg_all[e][ki]) for e, (sl, A_st, _g) in enumerate(prep)
            ]
            J = self._build_J_blocks(geom, self.k, same_edge_prep=same_edge_k)
            Z = self._assemble_Z(J, supp_seg, polys, geom)
            if self.ground_z is not None:
                J_img = self._build_J_image_blocks(geom, self.k)
                td_img = self._image_tangent_dot(geom["tangents"])
                Z = Z - self._assemble_Z(J_img, supp_seg, polys, geom, td_all=td_img)
            X = self._solve_with_kcl_ports(Z, B, kcl_A)
            out[ki] = B.T @ X

        self.k = k_save
        self.wavelength = wl_save
        self.omega = omega_save
        return out

    def compute_impedance_swept(self, k_array):
        """Loop over wavenumbers (no batched assembly here yet). Rebinds
        self.k / self.omega / self.wavelength per call and restores them.
        """
        k_array = np.asarray(k_array, dtype=float)
        n_feeds = len(self.feeds)
        if n_feeds == 1:
            z_out = np.zeros(k_array.shape[0], dtype=np.complex128)
        else:
            z_out = np.zeros((k_array.shape[0], n_feeds), dtype=np.complex128)
        k_save = self.k
        wl_save = self.wavelength
        omega_save = self.omega
        # k-independent static + reg-geometry, shared across the sweep; the
        # reg-kernel moment blocks are batched over all k up front (one einsum
        # per edge instead of one per (edge, k)).
        prep = self._same_edge_prep(self._build_geometry())
        reg_all = [
            _seg_seg_reg_moments_from_geometry_swept(reg_geo, k_array)
            for _sl, _A_st, reg_geo in prep
        ]
        for i, kk in enumerate(k_array):
            self.k = float(kk)
            self.omega = self.k * self.c
            self.wavelength = self.c / (self.omega / (2 * np.pi))
            same_edge_k = [
                (sl, A_st, reg_all[e][i]) for e, (sl, A_st, _g) in enumerate(prep)
            ]
            z, _ = self.compute_impedance(same_edge_prep=same_edge_k)
            z_out[i] = z
        self.k = k_save
        self.wavelength = wl_save
        self.omega = omega_save
        return z_out

    def currents_at_knots(self, coeffs, s_array=None):
        """Per-wire complex current at every mesh knot.

        Evaluates Σ_kept c_g · B_{j_local}(s_knot) per wire using scipy's
        B-spline design matrix on the wire's clamped knot vector.

        When `s_array` is provided as a list of 1D arc-length arrays (one per
        wire), the basis sum is evaluated at those arc positions instead of
        the mesh knots. With `use_singular_enrichment=True`, the enrichment
        basis Φ_sing(u) = (u/h)·log(u/h) — non-zero between knots but exactly
        zero AT the bounding knots — is added at sample positions interior to
        the enriched segments. Φ_sing contributes nothing at mesh knots, so
        the s_array=None path is unchanged.

        KCL Lagrange multipliers (trailing entries beyond the polynomial
        and enrichment blocks of `coeffs`) carry no current shape and are
        ignored by this evaluation.
        """
        coeffs = np.asarray(coeffs)
        geom = self._build_geometry()
        supp_seg, _, _, wire_knots, wire_basis_global = self._build_basis_polynomials(
            geom
        )
        n_poly = supp_seg.shape[0]
        d = self.degree

        enrich_specs = None
        if self.use_singular_enrichment:
            # Match the active-junction subset that compute_impedance used
            # so the spec list lines up with the enrichment block of coeffs.
            specs = self._enrichment_specs(
                geom, active_junction_indices=self._auto_active_junctions
            )
            if specs:
                enrich_specs = specs

        out = []
        for w_idx in range(len(self.wires_polylines)):
            arc_at_knot = geom["per_wire"][w_idx]["arc_at_knot"]
            knots_vec = wire_knots[w_idx]
            if s_array is None:
                s_eval = np.clip(arc_at_knot, knots_vec[0], knots_vec[-1])
            else:
                s_eval = np.clip(
                    np.asarray(s_array[w_idx], dtype=np.float64),
                    knots_vec[0],
                    knots_vec[-1],
                )
            I_out = np.zeros(s_eval.shape[0], dtype=np.complex128)
            kept, local_to_global = wire_basis_global[w_idx]
            if s_eval.shape[0] > 0:
                # design_matrix at [0, wire_arc] — clip tiny FP overshoots that
                # would push the endpoint epsilon outside the clamped knot range.
                DM = BSpline.design_matrix(s_eval, knots_vec, d).toarray()
                for kept_idx, (j_local, _, _, _) in enumerate(kept):
                    I_out += coeffs[local_to_global[kept_idx]] * DM[:, j_local]

            if enrich_specs is not None:
                seg_off_w = geom["seg_offsets"][w_idx]
                arc_at_knot_w = geom["per_wire"][w_idx]["arc_at_knot"]
                h_per_seg_w = geom["per_wire"][w_idx]["h_per_seg"]
                for spec_idx, (_, wire_w, _, seg_idx_global, u_origin) in enumerate(
                    enrich_specs
                ):
                    if wire_w != w_idx:
                        continue
                    seg_local = seg_idx_global - seg_off_w
                    seg_l_arc = arc_at_knot_w[seg_local]
                    seg_r_arc = arc_at_knot_w[seg_local + 1]
                    h_seg = h_per_seg_w[seg_local]
                    mask = (s_eval >= seg_l_arc) & (s_eval <= seg_r_arc)
                    if not np.any(mask):
                        continue
                    if u_origin == "left":
                        u_from_junc = s_eval[mask] - seg_l_arc
                    else:
                        u_from_junc = seg_r_arc - s_eval[mask]
                    u_norm = u_from_junc / h_seg
                    # Match the solver's variant: "raw" subtracts nothing,
                    # "stable" subtracts the bubble-subspace projection.
                    # Both variants preserve Φ_sing(0)=Φ_sing(1)=0.
                    phi = np.zeros_like(u_norm)
                    pos = u_norm > 0.0
                    phi[pos] = u_norm[pos] * np.log(u_norm[pos])
                    if self.enrichment_variant == "stable":
                        proj_coeffs = _xfem_projection_coeffs(self.degree)
                        phi = phi - np.polyval(proj_coeffs[::-1], u_norm)
                    I_out[mask] += coeffs[n_poly + spec_idx] * phi
            out.append(I_out)
        return out
