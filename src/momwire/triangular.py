"""Triangular-basis Galerkin MoM for multiple bent (polyline) wires.

Each wire is its own polyline of M >= 2 anchor points; edges within a wire
may bend at kinks; basis functions are continuous through kinks within a
wire but vanish at every wire's endpoints (no continuity across wires).
A straight dipole is the degenerate case of a single-edge single-wire
polyline; a parallel-element Yagi is multiple single-edge wires; an
inverted-V is one two-edge wire; moxon/hexbeam are multi-edge multi-wire.

Block structure of the segment-pair J integrals:
  * same-edge same-wire: analytic static kernel + GL-quadrature regular remainder
  * different-edge same-wire: full 3D quadrature with wire-radius regularization
    (keeps the kernel finite at kink corners shared by adjacent edges)
  * cross-wire: full 3D quadrature; same a^2 regularization is applied for
    code reuse — at moxon-scale tip-to-tip distances of order cm with
    a ~ 0.5 mm the regularization shifts cross-wire integrals by ~1e-5,
    well below the discretization error.

Feed: at the interior knot of `feed_wire_index` whose arc-length from that
wire's start is closest to `feed_arclength` (default: midpoint).
"""

import numpy as np
import scipy.linalg

from ._triangular_kernels import (
    _seg_seg_static_all,
    _seg_seg_reg_all,
    _seg_seg_reg_all_batch,
    _seg_seg_offedge_quad,
    _seg_seg_offedge_quad_batch,
)

from ._accel import acc as _acc

_HAVE_ASSEMBLE_Z = _acc is not None and hasattr(_acc, "assemble_Z")
_HAVE_ASSEMBLE_Z_GENERAL = _acc is not None and hasattr(_acc, "assemble_Z_general")


class TriangularSolver:
    """N-wire triangular Galerkin MoM, each wire a polyline with bends.

    wires: list of (M_w, 3) polyline arrays. M_w >= 2 anchor points per wire.
    n_per_edge_per_wire: list of (int or sequence). Per-wire segments per
        polyline edge. None for a wire means use `nsegs` for each of its
        edges; an int means use that count for each edge; a sequence gives a
        per-edge count. If `n_per_edge_per_wire` itself is None, every wire
        uses `nsegs` on every edge.
    feed_wire_index: index of the wire that carries the delta-gap source
        (single-feed back-compat path; ignored when `feeds` is supplied).
    feed_arclength: arc length along the feed wire (from its starting
        anchor) at which to place the source. None picks the midpoint.
        Ignored when `feeds` is supplied.
    feeds: optional list of (wire_index, arclength_or_None, voltage)
        tuples describing multiple delta-gap sources with prescribed
        complex driving voltages. Each entry is treated the same way
        as the single-feed kwargs — `arclength_or_None=None` picks the
        wire's midpoint; `voltage` is a complex scalar so phase shifts
        across feeds are expressed by `V_i = |V| * exp(1j * phi_i)`.
        With N feeds, `compute_impedance()` solves once with the
        weighted RHS Σ_i V_i e_{m_i} and returns the per-feed driving
        point impedance vector Z_i = V_i / coeffs[m_i].
    n_qp_reg: GL points for same-edge regular-kernel integrals.
    n_qp_off: GL points for off-edge / cross-wire integrals.
    wavelength, halfdriver_factor: set the measurement wavenumber k and the
        default-geometry half-driver length. With explicit `wires` the
        geometry is fully determined by the polylines; `halfdriver_factor`
        is then only informational.
    wire_radius: thin-wire radius used in the kernel regularization.
    nsegs: default segment count when `n_per_edge_per_wire` doesn't specify.
    ground_z: if not None, model an infinite PEC ground plane at z = ground_z
        via the image method. The image contribution is computed by mirroring
        every segment's position across z = ground_z and adding the reaction
        of the resulting image current (horizontal components anti-parallel,
        vertical preserved; image charge density is the negative of the
        original). All wires must lie at z >= ground_z + wire_radius for the
        image to be well-separated from the antenna.
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        feed_wire_index=0,
        feed_arclength=None,
        feeds=None,
        n_qp_reg=4,
        n_qp_off=4,
        wavelength=22,
        halfdriver_factor=0.962,
        wire_radius=0.0005,
        nsegs=101,
        ground_z=None,
        junctions=None,
    ):
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.wire_radius = wire_radius
        self.nsegs = nsegs
        self.ground_z = ground_z

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = self.omega / self.c
        self.jomega = 1j * self.omega
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        if not wires:
            raise ValueError("wires must be non-empty")
        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")

        n_w = len(self.wires_polylines)
        if n_per_edge_per_wire is None:
            n_per_edge_per_wire = [None] * n_w
        if len(n_per_edge_per_wire) != n_w:
            raise ValueError(
                f"n_per_edge_per_wire length {len(n_per_edge_per_wire)} "
                f"!= number of wires {n_w}"
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
                    f"wire {i}: n_per_edge length {len(npe)} "
                    f"!= number of edges {n_edges_w}"
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
        self.n_qp_reg = n_qp_reg
        self.n_qp_off = n_qp_off

        # Junctions: list of [(wire_idx, "start"|"end"), ...] tuples — each entry
        # is one junction node where K wire endpoints meet. K directional tent
        # basis functions are added per junction (one per connected wire-end);
        # KCL Σ I_k = 0 is enforced at solve time via a Lagrange-multiplier row
        # so all K directional bases are treated symmetrically.
        self.junctions = []
        if junctions is not None:
            for j, jw in enumerate(junctions):
                if len(jw) < 2:
                    raise ValueError(
                        f"junction {j}: need >= 2 wire-ends, got {len(jw)}"
                    )
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
        # `_build_geometry()`; on the N=21 hentenna width sweep that's ~1.8
        # ms/step of duplicated Python work. Cache on the instance. Geometry
        # is purely a function of the immutable constructor inputs, so this
        # is also safe inside `compute_impedance_swept`'s per-k loop.
        self._cached_geometry: dict | None = None

    def _build_geometry(self):
        """Discretize every wire and concatenate into global arrays.

        Per-wire metadata (`edge_offsets`, `edge_arc_edges`) is preserved so
        the same-wire J build can stay edge-local and use the analytic
        static-kernel formula on each edge.
        """
        if self._cached_geometry is not None:
            return self._cached_geometry
        per_wire = []
        seg_offsets = [0]
        basis_offsets = [0]
        h_list = []
        tangents_list = []
        for w_idx, (pl, npe_list) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            seg_l_list = []
            seg_r_list = []
            tan_list = []
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
                knots = (1 - t_node[:, None]) * p0[None, :] + t_node[:, None] * p1[
                    None, :
                ]
                seg_l_list.append(knots[:-1])
                seg_r_list.append(knots[1:])
                tan_list.append(np.tile(tan, (n_e, 1)))
                h_w_list.append(np.full(n_e, h_e))
                edge_arc_edges.append(np.linspace(0.0, edge_len, n_e + 1))
                edge_offsets.append(edge_offsets[-1] + n_e)

            seg_l = np.vstack(seg_l_list)
            seg_r = np.vstack(seg_r_list)
            tangents = np.vstack(tan_list)
            h_per_seg = np.concatenate(h_w_list)
            n_total = seg_l.shape[0]
            arc_at_knot = np.concatenate([[0.0], np.cumsum(h_per_seg)])

            per_wire.append(
                {
                    "seg_l": seg_l,
                    "seg_r": seg_r,
                    "tangents": tangents,
                    "h_per_seg": h_per_seg,
                    "edge_offsets": edge_offsets,
                    "edge_arc_edges": edge_arc_edges,
                    "arc_at_knot": arc_at_knot,
                    "n_total": n_total,
                }
            )
            seg_offsets.append(seg_offsets[-1] + n_total)
            basis_offsets.append(basis_offsets[-1] + (n_total - 1))
            h_list.append(h_per_seg)
            tangents_list.append(tangents)

        h_per_seg_global = np.concatenate(h_list)
        tangents_global = np.vstack(tangents_list)

        left_segs = []
        right_segs = []
        for w in range(len(per_wire)):
            base = seg_offsets[w]
            N_w = per_wire[w]["n_total"]
            left_segs.append(base + np.arange(N_w - 1, dtype=np.int64))
            right_segs.append(base + np.arange(1, N_w, dtype=np.int64))
        left_seg = np.concatenate(left_segs)
        right_seg = np.concatenate(right_segs)

        geom = {
            "per_wire": per_wire,
            "seg_offsets": seg_offsets,
            "basis_offsets": basis_offsets,
            "n_segs_total": seg_offsets[-1],
            "n_basis_total": basis_offsets[-1],
            "h_per_seg": h_per_seg_global,
            "tangents": tangents_global,
            "left_seg": left_seg,
            "right_seg": right_seg,
        }
        if self.junctions:
            self._add_junction_bases(geom)
        self._cached_geometry = geom
        return geom

    def _add_junction_bases(self, geom):
        """Append directional tent bases at each junction node.

        Each junction with K wire-ends contributes K directional bases —
        per (wire_idx, end_pos) tuple, one basis whose single active wing is
        the tent rising to 1 at the junction node and falling to 0 at the
        adjacent segment's other end. KCL Σ I_k = 0 across the K bases is
        recorded in `kcl_A` for Lagrange-augmented solving.

        Mutates `geom` in place to add:
          support_seg / support_L / support_R   (n_bases_total, 2)
              Per-basis 2-wing support — interior bases keep their (left, right)
              stencil (L_left=0, R_left=1; L_right=1, R_right=0), junction
              directional bases use wing 0 only and zero out wing 1.
          kcl_A                                  (n_junctions, n_bases_total)
              Row j has 1.0 at columns of junction j's K directional bases.
          n_basis_total                          int — updated to include junctions.
        """
        seg_off = geom["seg_offsets"]
        n_interior = geom["n_basis_total"]
        left_seg = geom["left_seg"]
        right_seg = geom["right_seg"]

        # For each (wire_idx, end) tuple at a junction we add one directional
        # basis with tent value +1 at the junction node. KCL "outflow sign":
        # a wire connected at its polyline-start has arc direction pointing
        # outward, so the basis coefficient is the current flowing OUT of the
        # junction (sign +1). A wire connected at its polyline-end has arc
        # direction pointing inward, so the basis coefficient is the current
        # flowing INTO the junction; the outflow contribution is its negative
        # (sign -1).
        junction_dirs = []  # (junction_idx, seg_idx, L, R, kcl_sign)
        for j, jw in enumerate(self.junctions):
            for w, end in jw:
                if end == "start":
                    seg_idx = seg_off[w]
                    L, R = 1.0, 0.0  # junction at seg's left arc end
                    kcl_sign = +1.0
                else:  # "end"
                    seg_idx = seg_off[w + 1] - 1
                    L, R = 0.0, 1.0  # junction at seg's right arc end
                    kcl_sign = -1.0
                junction_dirs.append((j, seg_idx, L, R, kcl_sign))

        n_dir = len(junction_dirs)
        n_total = n_interior + n_dir

        support_seg = np.zeros((n_total, 2), dtype=np.int64)
        support_L = np.zeros((n_total, 2), dtype=np.float64)
        support_R = np.zeros((n_total, 2), dtype=np.float64)

        # Interior bases mirror the existing (left, right) stencil.
        support_seg[:n_interior, 0] = left_seg
        support_seg[:n_interior, 1] = right_seg
        support_L[:n_interior, 0] = 0.0
        support_R[:n_interior, 0] = 1.0
        support_L[:n_interior, 1] = 1.0
        support_R[:n_interior, 1] = 0.0

        # Junction directional bases: wing 0 active, wing 1 inactive (L=R=0
        # contributes nothing to the assembly's slope or level terms; any valid
        # segment index works as a filler).
        for k, (_, seg_idx, L, R, _) in enumerate(junction_dirs):
            m = n_interior + k
            support_seg[m, 0] = seg_idx
            support_L[m, 0] = L
            support_R[m, 0] = R

        kcl_A = np.zeros((len(self.junctions), n_total), dtype=np.float64)
        for k, (j, _, _, _, kcl_sign) in enumerate(junction_dirs):
            kcl_A[j, n_interior + k] = kcl_sign

        geom["support_seg"] = support_seg
        geom["support_L"] = support_L
        geom["support_R"] = support_R
        geom["kcl_A"] = kcl_A
        geom["n_basis_interior"] = n_interior
        geom["n_basis_total"] = n_total

    def _feed_basis_index(self, geom):
        """Global basis index of the (primary) source; back-compat helper
        for callers that still assume a single feed.
        """
        return self._feed_basis_indices(geom)[0]

    def _feed_basis_indices(self, geom):
        """Global basis indices for every entry in `self.feeds`.

        Each feed maps to the interior knot of its wire whose arc-length
        from the wire start is closest to the requested `arclength`
        (None → wire midpoint).
        """
        idx = []
        for w, arc, _v in self.feeds:
            arc_at_knot = geom["per_wire"][w]["arc_at_knot"]
            feed_arc = arc if arc is not None else arc_at_knot[-1] / 2.0
            interior_arc = arc_at_knot[1:-1]
            m_local = int(np.argmin(np.abs(interior_arc - feed_arc)))
            idx.append(int(geom["basis_offsets"][w] + m_local))
        return idx

    def _build_J_blocks(self, geom, k):
        """Per-segment-pair J integrals for a single wavenumber.

        Fused build: one all-pairs off-edge quadrature call (same `a²`
        regularization applies uniformly to different-edge and cross-wire
        pairs), then same-edge blocks are overwritten with the analytic
        static kernel + regularized GL-quadrature remainder. Same-edge
        pairs are computed twice (the off-edge result is wrong for them
        because the kernel is singular), but ~5% redundant compute beats
        ~N_edges² + N_wires² small Python-side dispatches.
        """
        a = self.wire_radius
        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        n_w = len(per_wire)

        seg_l_all = np.vstack([pw["seg_l"] for pw in per_wire])
        seg_r_all = np.vstack([pw["seg_r"] for pw in per_wire])
        J00, J10, J01, J11 = _seg_seg_offedge_quad(
            seg_l_all,
            seg_r_all,
            seg_l_all,
            seg_r_all,
            a,
            k,
            self.n_qp_off,
        )

        for w in range(n_w):
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            n_edges_w = len(ed_off) - 1
            base = seg_off[w]
            for i_e in range(n_edges_w):
                sl = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                A00, A10, A01, A11 = _seg_seg_static_all(ed_arc[i_e], a)
                R00, R10, R01, R11 = _seg_seg_reg_all(ed_arc[i_e], a, k, self.n_qp_reg)
                J00[sl, sl] = A00 + R00
                J10[sl, sl] = A10 + R10
                J01[sl, sl] = A01 + R01
                J11[sl, sl] = A11 + R11

        return J00, J10, J01, J11

    def _build_J_blocks_batch(self, geom, k_array):
        """Batched (n_k, N, N) J integrals over a vector of wavenumbers.
        Fused build — same strategy as `_build_J_blocks`; see that docstring
        for the rationale.
        """
        a = self.wire_radius
        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        n_w = len(per_wire)

        seg_l_all = np.vstack([pw["seg_l"] for pw in per_wire])
        seg_r_all = np.vstack([pw["seg_r"] for pw in per_wire])
        J00, J10, J01, J11 = _seg_seg_offedge_quad_batch(
            seg_l_all,
            seg_r_all,
            seg_l_all,
            seg_r_all,
            a,
            k_array,
            self.n_qp_off,
        )

        for w in range(n_w):
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            n_edges_w = len(ed_off) - 1
            base = seg_off[w]
            for i_e in range(n_edges_w):
                sl = slice(base + ed_off[i_e], base + ed_off[i_e + 1])
                A00, A10, A01, A11 = _seg_seg_static_all(ed_arc[i_e], a)
                R00, R10, R01, R11 = _seg_seg_reg_all_batch(
                    ed_arc[i_e], a, k_array, self.n_qp_reg
                )
                J00[:, sl, sl] = A00[None, :, :] + R00
                J10[:, sl, sl] = A10[None, :, :] + R10
                J01[:, sl, sl] = A01[None, :, :] + R01
                J11[:, sl, sl] = A11[None, :, :] + R11

        return J00, J10, J01, J11

    def _image_positions(self, positions):
        out = positions.copy()
        out[..., 2] = 2 * self.ground_z - out[..., 2]
        return out

    def _image_tangent_dot(self, tangents):
        """t_m · t_image_n with t_image_n = (t_n_x, t_n_y, -t_n_z).

        Equivalent to td_all with the z-z outer product negated.
        """
        return tangents @ (tangents * np.array([1.0, 1.0, -1.0])).T

    def _build_J_image_blocks(self, geom, k):
        """J integrals with each j-segment mirrored across z = ground_z.

        One quadrature call over all (i, j) pairs — the image is far enough
        from the original to never be analytically tractable, even for
        same-edge pairs.
        """
        a = self.wire_radius
        per_wire = geom["per_wire"]
        seg_l_all = np.vstack([pw["seg_l"] for pw in per_wire])
        seg_r_all = np.vstack([pw["seg_r"] for pw in per_wire])
        seg_l_img = self._image_positions(seg_l_all)
        seg_r_img = self._image_positions(seg_r_all)
        return _seg_seg_offedge_quad(
            seg_l_all, seg_r_all, seg_l_img, seg_r_img, a, k, self.n_qp_off
        )

    def _build_J_image_blocks_batch(self, geom, k_array):
        a = self.wire_radius
        per_wire = geom["per_wire"]
        seg_l_all = np.vstack([pw["seg_l"] for pw in per_wire])
        seg_r_all = np.vstack([pw["seg_r"] for pw in per_wire])
        seg_l_img = self._image_positions(seg_l_all)
        seg_r_img = self._image_positions(seg_r_all)
        return _seg_seg_offedge_quad_batch(
            seg_l_all, seg_r_all, seg_l_img, seg_r_img, a, k_array, self.n_qp_off
        )

    def _assemble_Z_single(self, J00, J10, J01, J11, td_all, geom):
        """Assemble the single-k (n_basis, n_basis) Z matrix from the four
        per-segment-pair J tensors and a tangent-dot-product table.

        Identical form for free-space and image contributions — the caller
        is responsible for negating the result before adding it back to
        Z_free when ground is present.
        """
        left_seg = geom["left_seg"]
        right_seg = geom["right_seg"]
        h_per_seg = geom["h_per_seg"]

        hl_m = h_per_seg[left_seg][:, None]
        hl_n = h_per_seg[left_seg][None, :]
        hr_m = h_per_seg[right_seg][:, None]
        hr_n = h_per_seg[right_seg][None, :]

        S = (
            J00[np.ix_(left_seg, left_seg)] / (hl_m * hl_n)
            - J00[np.ix_(left_seg, right_seg)] / (hl_m * hr_n)
            - J00[np.ix_(right_seg, left_seg)] / (hr_m * hl_n)
            + J00[np.ix_(right_seg, right_seg)] / (hr_m * hr_n)
        )
        Z_Phi = S / (1j * self.omega * self.eps)

        td_ll = td_all[np.ix_(left_seg, left_seg)]
        td_lr = td_all[np.ix_(left_seg, right_seg)]
        td_rl = td_all[np.ix_(right_seg, left_seg)]
        td_rr = td_all[np.ix_(right_seg, right_seg)]

        I_A = (
            td_ll * (J11[np.ix_(left_seg, left_seg)] / (hl_m * hl_n))
            + td_lr
            * (
                J10[np.ix_(left_seg, right_seg)] / hl_m
                - J11[np.ix_(left_seg, right_seg)] / (hl_m * hr_n)
            )
            + td_rl
            * (
                J01[np.ix_(right_seg, left_seg)] / hl_n
                - J11[np.ix_(right_seg, left_seg)] / (hr_m * hl_n)
            )
            + td_rr
            * (
                J00[np.ix_(right_seg, right_seg)]
                - J10[np.ix_(right_seg, right_seg)] / hr_m
                - J01[np.ix_(right_seg, right_seg)] / hr_n
                + J11[np.ix_(right_seg, right_seg)] / (hr_m * hr_n)
            )
        )
        Z_A = 1j * self.omega * self.mu * I_A

        return Z_A + Z_Phi

    def _assemble_Z_general_single(self, J00, J10, J01, J11, td_all, geom):
        """Single-k Z assembly using per-basis (segment, L, R) support arrays.

        Handles arbitrary 2-wing basis layouts including junction directional
        bases (where one wing is inactive: L = R = 0 zeroes the slope and
        level so that wing contributes nothing). Bit-exact with
        `_assemble_Z_single` when the support arrays encode the standard
        (left, right) interior-basis stencil.
        """
        support_seg = geom["support_seg"]
        support_L = geom["support_L"]
        support_R = geom["support_R"]
        h_per_seg = geom["h_per_seg"]
        h_supp = h_per_seg[support_seg]
        slope = (support_R - support_L) / h_supp  # (n_basis, 2)

        n_b = support_seg.shape[0]
        S = np.zeros((n_b, n_b), dtype=np.complex128)
        I_A = np.zeros((n_b, n_b), dtype=np.complex128)
        for a in range(2):
            sm = support_seg[:, a]
            Lma, Sma = support_L[:, a], slope[:, a]
            for b in range(2):
                sn = support_seg[:, b]
                Lnb, Snb = support_L[:, b], slope[:, b]
                J00_blk = J00[np.ix_(sm, sn)]
                J01_blk = J01[np.ix_(sm, sn)]
                J10_blk = J10[np.ix_(sm, sn)]
                J11_blk = J11[np.ix_(sm, sn)]
                td_blk = td_all[np.ix_(sm, sn)]
                S += np.outer(Sma, Snb) * J00_blk
                I_A += td_blk * (
                    np.outer(Lma, Lnb) * J00_blk
                    + np.outer(Lma, Snb) * J01_blk
                    + np.outer(Sma, Lnb) * J10_blk
                    + np.outer(Sma, Snb) * J11_blk
                )
        Z_Phi = S / (1j * self.omega * self.eps)
        Z_A = 1j * self.omega * self.mu * I_A
        return Z_A + Z_Phi

    def _solve_with_kcl(self, Z, v, geom):
        """KCL-constrained solve via Schur complement (single-k mirror of
        `_solve_with_kcl_batch` — see that docstring for the derivation).
        """
        A = geom["kcl_A"]
        n_b = Z.shape[0]
        n_c = A.shape[0]
        rhs = np.empty((n_b, 1 + n_c), dtype=np.complex128)
        rhs[:, 0] = v
        rhs[:, 1:] = A.T
        sol = scipy.linalg.solve(Z, rhs)
        w = sol[:, 0]
        X = sol[:, 1:]
        lam = scipy.linalg.solve(A @ X, A @ w)
        return w - X @ lam

    def _solve_with_kcl_ports(self, Z, V, geom):
        """Multi-port single-k KCL-constrained Schur solve.

        V: (n_b, n_p) right-hand side, one column per port. Returns
        (n_b, n_p) coefficients. Same Schur derivation as `_solve_with_kcl`
        but with a matrix RHS — packs all n_p source columns together with
        the n_c constraint columns into a single LU + back-sub call, so
        the factor cost is paid once for all ports.
        """
        A = geom["kcl_A"]
        n_b, n_p = V.shape
        n_c = A.shape[0]
        rhs = np.empty((n_b, n_p + n_c), dtype=np.complex128)
        rhs[:, :n_p] = V
        rhs[:, n_p:] = A.T
        sol = scipy.linalg.solve(Z, rhs)
        W = sol[:, :n_p]  # Z⁻¹ V
        X = sol[:, n_p:]  # Z⁻¹ Aᵀ
        Lam = scipy.linalg.solve(A @ X, A @ W)
        return W - X @ Lam

    def compute_impedance(self, *, ntrap=None):
        geom = self._build_geometry()
        tangents = geom["tangents"]
        has_junctions = bool(self.junctions)
        assemble = (
            self._assemble_Z_general_single
            if has_junctions
            else self._assemble_Z_single
        )

        J_free = self._build_J_blocks(geom, self.k)
        td_free = tangents @ tangents.T
        Z = assemble(*J_free, td_free, geom)

        if self.ground_z is not None:
            # PEC image: subtract sub-assembly built with image J tensors and
            # mirrored tangent dot products. The net image current is
            # anti-parallel for horizontal components / parallel for vertical;
            # the basis-function sign flip and the image-charge sign flip
            # combine to a single minus sign on the entire sub-assembly.
            J_img = self._build_J_image_blocks(geom, self.k)
            td_img = self._image_tangent_dot(tangents)
            Z = Z - assemble(*J_img, td_img, geom)

        self.z = Z

        m_indices = self._feed_basis_indices(geom)
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        v = np.zeros(geom["n_basis_total"], dtype=np.complex128)
        for m_i, v_i in zip(m_indices, voltages):
            v[m_i] += v_i
        if has_junctions:
            coeffs = self._solve_with_kcl(Z, v, geom)
        else:
            coeffs = scipy.linalg.solve(Z, v)
        feed_currents = np.array(
            [coeffs[m_i] for m_i in m_indices], dtype=np.complex128
        )
        z_per_feed = voltages / feed_currents
        driver_impedance = z_per_feed[0] if len(self.feeds) == 1 else z_per_feed
        return driver_impedance, coeffs

    def compute_y_matrix(self) -> np.ndarray:
        """Short-circuit admittance matrix [Y_sc] at the configured feeds.

        Y_sc[i, j] is the current that flows out of port i when port j
        is driven with V_j = 1 and every other port is held at V_k = 0.
        N back-substitutions on the same factored system give all N
        columns in one batched solve — the LU factor cost (which is
        what dominates for our sizes) is paid once.

        The caller can invert Y_sc to recover the open-circuit Z
        matrix used in network analysis (V = Z·I with all other ports
        open / I_k = 0).

        Same Z assembly + ground handling as `compute_impedance`. Only
        the RHS / readout differs. Junctions are handled through the
        port-batched Schur-complement solve `_solve_with_kcl_ports`.
        """
        geom = self._build_geometry()
        tangents = geom["tangents"]
        has_junctions = bool(self.junctions)
        assemble = (
            self._assemble_Z_general_single
            if has_junctions
            else self._assemble_Z_single
        )

        J_free = self._build_J_blocks(geom, self.k)
        td_free = tangents @ tangents.T
        Z = assemble(*J_free, td_free, geom)

        if self.ground_z is not None:
            J_img = self._build_J_image_blocks(geom, self.k)
            td_img = self._image_tangent_dot(tangents)
            Z = Z - assemble(*J_img, td_img, geom)

        self.z = Z

        m_indices = self._feed_basis_indices(geom)
        n_ports = len(m_indices)
        n_basis = geom["n_basis_total"]

        # B is n_basis × n_ports; column j has a unit excitation at
        # port j's basis index. scipy's LAPACK handles the batched
        # RHS in a single LU + N back-subs.
        B = np.zeros((n_basis, n_ports), dtype=np.complex128)
        for j, m_i in enumerate(m_indices):
            B[m_i, j] = 1.0

        if has_junctions:
            X = self._solve_with_kcl_ports(Z, B, geom)
        else:
            X = scipy.linalg.solve(Z, B)
        return X[m_indices, :]

    def currents_at_knots(self, coeffs, s_array=None):
        """Per-wire complex current at every mesh knot, length = M_w knots per wire.

        Tent basis k peaks at its single interior knot with value 1 and is zero
        at every other knot, so the current at interior knot j of wire w is
        just coeffs[basis_offsets[w] + j - 1]. Junction directional bases
        (when present) carry value 1 at the wire-end knot of their (wire, end)
        tuple — those contribute the endpoint currents at junctioned ends.
        Free ends are zero (open-wire BC).

        When `s_array` is provided as a list of 1D arc-length arrays (one per
        wire), the basis sum is evaluated at those arc positions instead of
        the mesh knots. Tent basis is piecewise linear between adjacent knots,
        so this reduces to a linear interpolation of the knot values along
        the wire's cumulative-arc parameterization.
        """
        coeffs = np.asarray(coeffs)
        geom = self._build_geometry()
        per_wire = geom["per_wire"]
        basis_offsets = geom["basis_offsets"]
        n_interior = basis_offsets[-1]

        out = []
        for w_idx, pw in enumerate(per_wire):
            n_knots = pw["arc_at_knot"].shape[0]
            I = np.zeros(n_knots, dtype=np.complex128)
            I[1:-1] = coeffs[basis_offsets[w_idx] : basis_offsets[w_idx + 1]]
            out.append(I)

        if self.junctions:
            k = n_interior
            for jw in self.junctions:
                for w, end in jw:
                    out[w][0 if end == "start" else -1] = coeffs[k]
                    k += 1

        if s_array is None:
            return out

        sampled = []
        for w_idx, sv in enumerate(s_array):
            arc = per_wire[w_idx]["arc_at_knot"]
            knot_I = out[w_idx]
            sv = np.asarray(sv, dtype=np.float64)
            Ire = np.interp(sv, arc, knot_I.real)
            Iim = np.interp(sv, arc, knot_I.imag)
            sampled.append(Ire + 1j * Iim)
        return sampled

    def _assemble_Z_batch(self, J00, J10, J01, J11, td_all, geom, omega_array):
        """Batched (n_k, n_basis, n_basis) Z assembly, mirroring
        `_assemble_Z_single` but with a leading k-axis. Uses the C++
        accelerator when available.
        """
        left_seg = geom["left_seg"]
        right_seg = geom["right_seg"]
        h_per_seg = np.ascontiguousarray(geom["h_per_seg"], dtype=np.float64)
        td_all = np.ascontiguousarray(td_all, dtype=np.float64)

        if _HAVE_ASSEMBLE_Z:
            return _acc.assemble_Z(
                J00,
                J10,
                J01,
                J11,
                h_per_seg,
                td_all,
                left_seg,
                right_seg,
                np.ascontiguousarray(omega_array, dtype=np.float64),
                float(self.eps),
                float(self.mu),
            )

        hl_m = h_per_seg[left_seg][:, None]
        hl_n = h_per_seg[left_seg][None, :]
        hr_m = h_per_seg[right_seg][:, None]
        hr_n = h_per_seg[right_seg][None, :]

        ll = (slice(None), left_seg[:, None], left_seg[None, :])
        lr = (slice(None), left_seg[:, None], right_seg[None, :])
        rl = (slice(None), right_seg[:, None], left_seg[None, :])
        rr = (slice(None), right_seg[:, None], right_seg[None, :])

        S = (
            J00[ll] / (hl_m * hl_n)
            - J00[lr] / (hl_m * hr_n)
            - J00[rl] / (hr_m * hl_n)
            + J00[rr] / (hr_m * hr_n)
        )
        Z_Phi = S / (1j * omega_array[:, None, None] * self.eps)

        td_ll = td_all[np.ix_(left_seg, left_seg)][None, ...]
        td_lr = td_all[np.ix_(left_seg, right_seg)][None, ...]
        td_rl = td_all[np.ix_(right_seg, left_seg)][None, ...]
        td_rr = td_all[np.ix_(right_seg, right_seg)][None, ...]

        I_A = (
            td_ll * (J11[ll] / (hl_m * hl_n))
            + td_lr * (J10[lr] / hl_m - J11[lr] / (hl_m * hr_n))
            + td_rl * (J01[rl] / hl_n - J11[rl] / (hr_m * hl_n))
            + td_rr
            * (J00[rr] - J10[rr] / hr_m - J01[rr] / hr_n + J11[rr] / (hr_m * hr_n))
        )
        Z_A = 1j * omega_array[:, None, None] * self.mu * I_A
        return Z_A + Z_Phi

    def _assemble_Z_general_batch(self, J00, J10, J01, J11, td_all, geom, omega_array):
        """Batched general assembly — same support-array formulation as
        `_assemble_Z_general_single` with a leading k-axis on the J tensors.
        Uses the C++ accelerator when available, otherwise the pure-Python
        path in `_assemble_Z_general_batch_python`.
        """
        support_seg = geom["support_seg"]
        support_L = geom["support_L"]
        support_R = geom["support_R"]
        h_per_seg = np.ascontiguousarray(geom["h_per_seg"], dtype=np.float64)
        td_all = np.ascontiguousarray(td_all, dtype=np.float64)

        if _HAVE_ASSEMBLE_Z_GENERAL:
            return _acc.assemble_Z_general(
                J00,
                J10,
                J01,
                J11,
                h_per_seg,
                td_all,
                np.ascontiguousarray(support_seg, dtype=np.int64),
                np.ascontiguousarray(support_L, dtype=np.float64),
                np.ascontiguousarray(support_R, dtype=np.float64),
                np.ascontiguousarray(omega_array, dtype=np.float64),
                float(self.eps),
                float(self.mu),
            )

        return self._assemble_Z_general_batch_python(
            J00, J10, J01, J11, td_all, geom, omega_array
        )

    def _assemble_Z_general_batch_python(
        self, J00, J10, J01, J11, td_all, geom, omega_array
    ):
        """Pure-numpy reference implementation of the general batch assembly.
        Retained as the C++ fallback and as a bit-exact regression target.
        """
        support_seg = geom["support_seg"]
        support_L = geom["support_L"]
        support_R = geom["support_R"]
        h_per_seg = geom["h_per_seg"]
        h_supp = h_per_seg[support_seg]
        slope = (support_R - support_L) / h_supp  # (n_basis, 2)

        n_b = support_seg.shape[0]
        n_k = J00.shape[0]
        S = np.zeros((n_k, n_b, n_b), dtype=np.complex128)
        I_A = np.zeros((n_k, n_b, n_b), dtype=np.complex128)
        td_all = np.asarray(td_all, dtype=np.float64)
        for a in range(2):
            sm = support_seg[:, a]
            Lma, Sma = support_L[:, a], slope[:, a]
            for b in range(2):
                sn = support_seg[:, b]
                Lnb, Snb = support_L[:, b], slope[:, b]
                idx = (slice(None), sm[:, None], sn[None, :])
                J00_blk = J00[idx]
                J01_blk = J01[idx]
                J10_blk = J10[idx]
                J11_blk = J11[idx]
                td_blk = td_all[np.ix_(sm, sn)][None, ...]
                S += np.outer(Sma, Snb)[None, ...] * J00_blk
                I_A += td_blk * (
                    np.outer(Lma, Lnb)[None, ...] * J00_blk
                    + np.outer(Lma, Snb)[None, ...] * J01_blk
                    + np.outer(Sma, Lnb)[None, ...] * J10_blk
                    + np.outer(Sma, Snb)[None, ...] * J11_blk
                )
        Z_Phi = S / (1j * omega_array[:, None, None] * self.eps)
        Z_A = 1j * omega_array[:, None, None] * self.mu * I_A
        return Z_A + Z_Phi

    def _solve_with_kcl_batch(self, Z, v, geom):
        """Batched KCL-constrained solve via Schur complement.

        Solves the saddle-point system
            [ Z   Aᵀ ] [ I ]   [ v ]
            [ A    0 ] [ λ ] = [ 0 ]
        without materializing the augmented (n_b+n_c, n_b+n_c) matrix M per
        k — which on the 5-band fan dipole alone costs ~63 ms / 12% of
        /sweep just to allocate and fill the (n_k, 425, 425) zero buffer.

        Schur expansion:
            I = Z⁻¹(v − Aᵀ λ),    (A Z⁻¹ Aᵀ) λ = A Z⁻¹ v

        Implementation packs the (1 + n_c) right-hand sides into one batched
        np.linalg.solve on Z, so the solve cost is dominated by the LU
        factorization (the extra n_c RHS add <1% to back-substitution).
        """
        A = geom["kcl_A"]
        n_k = Z.shape[0]
        n_b = Z.shape[1]
        n_c = A.shape[0]
        rhs = np.empty((n_k, n_b, 1 + n_c), dtype=np.complex128)
        rhs[:, :, 0] = v[None, :]
        rhs[:, :, 1:] = A.T[None, :, :]
        sol = np.linalg.solve(Z, rhs)  # (n_k, n_b, 1 + n_c)
        w = sol[:, :, 0]  # Z⁻¹ v   → (n_k, n_b)
        X = sol[:, :, 1:]  # Z⁻¹ Aᵀ → (n_k, n_b, n_c)
        S = np.einsum("cm,kmn->kcn", A, X)  # (n_k, n_c, n_c)
        Aw = np.einsum("cm,km->kc", A, w)  # (n_k, n_c)
        lam = np.linalg.solve(S, Aw[:, :, None])[:, :, 0]  # (n_k, n_c)
        return w - np.einsum("kmc,kc->km", X, lam)

    def _solve_with_kcl_swept_ports(self, Z, V, geom):
        """k-batched and port-batched KCL-constrained Schur solve.

        Z: (n_k, n_b, n_b). V: (n_b, n_p) — shared across k. Returns
        (n_k, n_b, n_p). Same Schur math as `_solve_with_kcl_batch`,
        generalised to a matrix RHS so `compute_y_matrix_swept` can solve
        all ports at all frequencies in one batched call.
        """
        A = geom["kcl_A"]
        n_k = Z.shape[0]
        n_b, n_p = V.shape
        n_c = A.shape[0]
        rhs = np.empty((n_k, n_b, n_p + n_c), dtype=np.complex128)
        rhs[:, :, :n_p] = V[None, :, :]
        rhs[:, :, n_p:] = A.T[None, :, :]
        sol = np.linalg.solve(Z, rhs)
        W = sol[:, :, :n_p]  # (n_k, n_b, n_p)
        X = sol[:, :, n_p:]  # (n_k, n_b, n_c)
        S = np.einsum("cm,kmn->kcn", A, X)  # (n_k, n_c, n_c)
        AW = np.einsum("cm,kmp->kcp", A, W)  # (n_k, n_c, n_p)
        Lam = np.linalg.solve(S, AW)  # (n_k, n_c, n_p)
        return W - np.einsum("kmc,kcp->kmp", X, Lam)

    def compute_impedance_swept(self, k_array):
        """Driver impedance over a batch of wavenumbers, sharing all
        k-independent work (geometry, static kernel, basis stencil).
        """
        k_array = np.asarray(k_array, dtype=float)
        omega_array = k_array * self.c
        geom = self._build_geometry()
        tangents = geom["tangents"]
        has_junctions = bool(self.junctions)
        assemble_batch = (
            self._assemble_Z_general_batch if has_junctions else self._assemble_Z_batch
        )

        J_free = self._build_J_blocks_batch(geom, k_array)
        td_free = tangents @ tangents.T
        Z = assemble_batch(*J_free, td_free, geom, omega_array)

        if self.ground_z is not None:
            J_img = self._build_J_image_blocks_batch(geom, k_array)
            td_img = self._image_tangent_dot(tangents)
            Z = Z - assemble_batch(*J_img, td_img, geom, omega_array)

        m_indices = self._feed_basis_indices(geom)
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        v = np.zeros(geom["n_basis_total"], dtype=np.complex128)
        for m_i, v_i in zip(m_indices, voltages):
            v[m_i] += v_i
        if has_junctions:
            coeffs = self._solve_with_kcl_batch(Z, v, geom)
        else:
            coeffs = np.linalg.solve(Z, v)
        feed_currents = coeffs[:, m_indices]  # (n_k, n_feeds)
        z_per_feed = voltages[None, :] / feed_currents
        return z_per_feed[:, 0] if len(self.feeds) == 1 else z_per_feed

    def compute_y_matrix_swept(self, k_array) -> np.ndarray:
        """Short-circuit Y matrices over a batch of wavenumbers.

        Returns an (n_k, n_ports, n_ports) array. Each Y[k] is the
        admittance matrix at wavenumber k_array[k]; see
        `compute_y_matrix()` for the math. Same single-RHS pattern as
        `compute_impedance_swept`, but with a batched per-port RHS so
        we get all N columns per frequency in one solve.

        Junctions are handled through `_solve_with_kcl_swept_ports`,
        which generalises the per-k single-port Schur path to a matrix
        RHS so all (port × frequency) solves share one LU factorisation
        per frequency.
        """
        k_array = np.asarray(k_array, dtype=float)
        omega_array = k_array * self.c
        geom = self._build_geometry()
        tangents = geom["tangents"]
        has_junctions = bool(self.junctions)
        assemble_batch = (
            self._assemble_Z_general_batch if has_junctions else self._assemble_Z_batch
        )

        J_free = self._build_J_blocks_batch(geom, k_array)
        td_free = tangents @ tangents.T
        Z = assemble_batch(*J_free, td_free, geom, omega_array)

        if self.ground_z is not None:
            J_img = self._build_J_image_blocks_batch(geom, k_array)
            td_img = self._image_tangent_dot(tangents)
            Z = Z - assemble_batch(*J_img, td_img, geom, omega_array)

        m_indices = self._feed_basis_indices(geom)
        n_ports = len(m_indices)
        n_basis = geom["n_basis_total"]

        # Same per-k RHS — broadcast it to (n_k, n_basis, n_ports).
        B = np.zeros((n_basis, n_ports), dtype=np.complex128)
        for j, m_i in enumerate(m_indices):
            B[m_i, j] = 1.0

        if has_junctions:
            coeffs = self._solve_with_kcl_swept_ports(Z, B, geom)
        else:
            B_batch = np.broadcast_to(B, (k_array.shape[0], n_basis, n_ports))
            coeffs = np.linalg.solve(Z, B_batch)  # (n_k, n_basis, n_ports)
        # Y[k, i, j] = current at port i when port j was driven with
        # V = 1, all others V = 0 — i.e. coeffs[k, m_indices[i], j].
        return coeffs[:, m_indices, :]
