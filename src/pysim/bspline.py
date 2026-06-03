"""Higher-order B-spline Galerkin MoM solver.

`TriangularPySim` is the degree-1 B-spline (tent) special case; this
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

Same as TriangularPySim, but with a polynomial-of-degree-d on each segment
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
multiplier row, mirroring TriangularPySim's treatment.

Feed: v_m = Φ_m(s_f), Z_drive = 1 / (v^T c).
"""

import numpy as np
import scipy.linalg
from scipy.interpolate import BSpline

from ._bspline_kernels import (
    _seg_seg_full_moments_offedge,
    _seg_seg_reg_moments,
    _seg_seg_static_moments,
)

try:
    from . import _accelerators as _acc

    _HAVE_BSPLINE_ASSEMBLE_ACCEL = hasattr(_acc, "assemble_Z_bspline")
except ImportError:
    _HAVE_BSPLINE_ASSEMBLE_ACCEL = False

_BSPLINE_ASSEMBLE_ACCEL_MAX_D = 2


class BSplinePySim:
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
        TriangularPySim.
    n_qp_pair : Gauss-Legendre nodes per segment per axis for the smooth-
        kernel piece of same-edge pairs and for all cross-edge / cross-wire
        pairs (full kernel with a² regularization).
    wavelength, halfdriver_factor, wire_radius, nsegs : as in
        TriangularPySim.
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

        if not (0 <= feed_wire_index < n_w):
            raise ValueError(f"feed_wire_index {feed_wire_index} out of range")
        self.feed_wire_index = feed_wire_index
        self.feed_arclength = feed_arclength
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

        return {
            "per_wire": per_wire,
            "seg_offsets": seg_offsets,
            "n_segs_total": seg_offsets[-1],
            "h_per_seg": h_per_seg_global,
            "tangents": tangents_global,
            "seg_l": seg_l_global,
            "seg_r": seg_r_global,
        }

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
            # build a sample of d+1 uniform points within each segment's
            # local-u range, in GLOBAL arc length (so we can evaluate the
            # BSpline at them)
            # For per-segment extraction, do it inside the loop because h
            # varies across edges.

            per_basis_local_to_global = {}
            for kept_idx, (j, kind, junc_idx, end_pos) in enumerate(kept):
                c_all = np.zeros(n_basis_w, dtype=np.float64)
                c_all[j] = 1.0
                bspl = BSpline(knots, c_all, d, extrapolate=False)

                support_lo = knots[j]
                support_hi = knots[j + d + 1]

                supp_seg_m = np.zeros(n_wings, dtype=np.int64)
                polys_m = np.zeros((n_wings, n_poly), dtype=np.float64)

                # Find segments overlapping the support
                wing = 0
                # iterate over local segments in this wire
                for seg_local in range(pw["n_total"]):
                    seg_l_arc = arc_at_knot_w[seg_local]
                    seg_r_arc = arc_at_knot_w[seg_local + 1]
                    eps_seg = 1e-9 * max(h_per_seg_w[seg_local], 1e-12)
                    if seg_r_arc < support_lo + eps_seg:
                        continue
                    if seg_l_arc > support_hi - eps_seg:
                        break
                    h_seg = h_per_seg_w[seg_local]
                    # uniform sample for Vandermonde
                    u_local = np.linspace(0.0, h_seg, d + 1)
                    u_global = seg_l_arc + u_local
                    vals = bspl(u_global)
                    Vmat = np.vander(u_local, d + 1, increasing=True)
                    coeffs = np.linalg.solve(Vmat, vals)
                    supp_seg_m[wing] = seg_off + seg_local
                    polys_m[wing, :] = coeffs
                    wing += 1
                    if wing >= n_wings:
                        break

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

        return supp_seg, polys, kcl_A, wire_knots, wire_basis_global

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

    def _build_J_blocks(self, geom, k):
        """All polynomial moment integrals J_pq[i, j] for p, q ∈ {0..d} and
        every (i, j) global segment pair. Returns shape (d+1, d+1, N, N).

        Fused build mirroring TriangularPySim._build_J_blocks: first compute
        every pair by full GL quadrature on the regularized full kernel
        G = exp(-jkR)/(4πR), R² = |Δr|² + a²; then overwrite same-edge
        blocks with the analytic static + GL-regularized split (essential
        for the log-singular diagonal).
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

    def _build_source_vector(self, geom, wire_knots, wire_basis_global, n_basis_total):
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
        """
        d = self.degree
        wi = self.feed_wire_index
        arc = geom["per_wire"][wi]["arc_at_knot"]
        wire_arc = arc[-1]
        s_f = self.feed_arclength if self.feed_arclength is not None else wire_arc / 2.0
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

        gl_xi, gl_w = np.polynomial.legendre.leggauss(self.n_qp_source)
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

        Identical structure to TriangularPySim._solve_with_kcl. If kcl_A
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

    # ------------------------------------------------------------------
    # Driver impedance
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Singular basis enrichment at K≥3 junctions
    # ------------------------------------------------------------------

    def _enrichment_specs(self, geom):
        """For each (wire, end_pos) at a K≥enrichment_min_k junction, return
        the global segment index of the adjacent segment and the "orientation"
        flag: u_local from junction = u_seg_left if end_pos == "start",
        h - u_seg_left if end_pos == "end".
        """
        specs = []  # list of (junction_idx, wire_w, end_pos, seg_idx, u_origin)
        for j_idx, jw in enumerate(self.junctions):
            if len(jw) < self.enrichment_min_k:
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

    def _enrichment_Z_assemble(self, geom, supp_seg_poly, polys_poly):
        """Assemble the (Z_pe, Z_ep, Z_ee) enrichment blocks via the C++
        accelerator (`pysim._accelerators.assemble_Z_enrich`).

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
        specs = self._enrichment_specs(geom)
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

        gl_xi, gl_w = np.polynomial.legendre.leggauss(self.n_qp_sing)
        t01 = 0.5 * (gl_xi + 1.0)
        w01 = 0.5 * gl_w

        from pysim._accelerators import assemble_Z_enrich

        Z_pe, Z_ep, Z_ee = assemble_Z_enrich(
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
        )

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

    def compute_impedance(self):
        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )
        n_basis_total = supp_seg.shape[0]

        J = self._build_J_blocks(geom, self.k)
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

        v = self._build_source_vector(
            geom, wire_knots, wire_basis_global, n_basis_total
        )

        if self.use_singular_enrichment:
            enrich = self._enrichment_Z_assemble(geom, supp_seg, polys)
            if enrich is not None:
                n_p = n_basis_total
                n_e = enrich["n_enrich"]
                n_total = n_p + n_e
                Z_aug = np.zeros((n_total, n_total), dtype=np.complex128)
                Z_aug[:n_p, :n_p] = Z
                Z_aug[:n_p, n_p:] = enrich["Z_pe"]
                Z_aug[n_p:, :n_p] = enrich["Z_ep"]
                Z_aug[n_p:, n_p:] = enrich["Z_ee"]
                v_aug = np.zeros(n_total, dtype=np.complex128)
                v_aug[:n_p] = v
                # Enrichment KCL: singular bases vanish at junction → 0 outflow
                kcl_aug = np.zeros((kcl_A.shape[0], n_total), dtype=np.float64)
                kcl_aug[:, :n_p] = kcl_A
                coeffs = self._solve_with_kcl(Z_aug, v_aug, kcl_aug)
                I_at_feed = v_aug @ coeffs
                driver_impedance = 1.0 / I_at_feed
                self.z = Z_aug
                return driver_impedance, coeffs

        coeffs = self._solve_with_kcl(Z, v, kcl_A)
        I_at_feed = v @ coeffs
        driver_impedance = 1.0 / I_at_feed
        self.z = Z
        return driver_impedance, coeffs

    def compute_impedance_swept(self, k_array):
        """Loop over wavenumbers (no batched assembly here yet). Rebinds
        self.k / self.omega / self.wavelength per call and restores them.
        """
        k_array = np.asarray(k_array, dtype=float)
        z_out = np.zeros(k_array.shape[0], dtype=np.complex128)
        k_save = self.k
        wl_save = self.wavelength
        omega_save = self.omega
        for i, kk in enumerate(k_array):
            self.k = float(kk)
            self.omega = self.k * self.c
            self.wavelength = self.c / (self.omega / (2 * np.pi))
            z, _ = self.compute_impedance()
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
            specs = self._enrichment_specs(geom)
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
                    # Φ_sing(u) = (u/h)·log(u/h); the limit at u=0 is 0.
                    phi = np.zeros_like(u_norm)
                    pos = u_norm > 0.0
                    phi[pos] = u_norm[pos] * np.log(u_norm[pos])
                    I_out[mask] += coeffs[n_poly + spec_idx] * phi
            out.append(I_out)
        return out
