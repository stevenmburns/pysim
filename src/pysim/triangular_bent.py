"""Triangular-basis Galerkin MoM for a single bent (polyline) wire in 3D.

Generalizes TriangularPySim to arbitrary 3D wire geometry. The wire is given
as a polyline of M >= 2 anchor points; each polyline edge is straight, and
adjacent edges may bend (different tangent directions). Each edge is
subdivided into n_per_edge equal-length MoM segments.

Same-edge segment pairs use the analytic static-kernel extraction from
TriangularPySim (the wire is straight within an edge). Cross-edge pairs --
including adjacent segments meeting at a kink -- use 3D Gauss-Legendre
quadrature on the full kernel with wire-radius regularization
R = sqrt(|r - r'|^2 + a^2). This keeps the kernel finite at corner points.

The vector-potential assembly uses per-segment tangents and breaks each
basis-pair (m, n) matrix element into four sub-rectangles
(left_m/right_m x left_n/right_n), each weighted by its own tangent
dot product. For a straight wire all tangent dot products collapse to 1
and the formula reduces exactly to TriangularPySim's.
"""
import numpy as np
import scipy.linalg

from .abstract import AbstractPySim
from .triangular import (
    _seg_seg_static_all,
    _seg_seg_reg_all,
    _seg_seg_reg_all_batch,
)


def _seg_seg_offedge_quad(seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k, n_qp):
    """Moment integrals J_pq between segments on different polyline edges.

    Uses 3D Gauss-Legendre quadrature on the regularized kernel
    G(R) = exp(-jkR) / (4 pi R) with R = sqrt(|r_i - r_j|^2 + a^2). The
    regularization keeps G finite even when seg_i and seg_j share a corner
    point.

    seg_l, seg_r: (N, 3) arrays of 3D endpoints.
    Returns (J00, J10, J01, J11), each (N_i, N_j) complex; u_i, u_j are arc
    lengths into each segment from its left endpoint.
    """
    J00, J10, J01, J11 = _seg_seg_offedge_quad_batch(
        seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, np.array([k]), n_qp,
    )
    return J00[0], J10[0], J01[0], J11[0]


def _seg_seg_offedge_quad_batch(seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k_array, n_qp):
    """Batched version of _seg_seg_offedge_quad over a vector of k values.

    Returns (J00, J10, J01, J11) each of shape (n_k, N_i, N_j).
    """
    gl_xi, gl_w = np.polynomial.legendre.leggauss(n_qp)
    t_qp = 0.5 * (gl_xi + 1.0)
    w_qp = 0.5 * gl_w

    len_i = np.linalg.norm(seg_r_i - seg_l_i, axis=1)
    len_j = np.linalg.norm(seg_r_j - seg_l_j, axis=1)

    pos_i = ((1 - t_qp[None, :, None]) * seg_l_i[:, None, :]
             + t_qp[None, :, None] * seg_r_i[:, None, :])
    pos_j = ((1 - t_qp[None, :, None]) * seg_l_j[:, None, :]
             + t_qp[None, :, None] * seg_r_j[:, None, :])

    u_i = t_qp[None, :] * len_i[:, None]
    u_j = t_qp[None, :] * len_j[:, None]
    w_i = w_qp[None, :] * len_i[:, None]
    w_j = w_qp[None, :] * len_j[:, None]

    diff = pos_i[:, :, None, None, :] - pos_j[None, None, :, :, :]
    R = np.sqrt((diff * diff).sum(-1) + a * a)
    G = np.exp(-1j * k_array[:, None, None, None, None] * R[None, ...]) / (
        4 * np.pi * R[None, ...]
    )

    J00 = np.einsum("iq,kiqjr,jr->kij", w_i, G, w_j)
    J10 = np.einsum("iq,iq,kiqjr,jr->kij", w_i, u_i, G, w_j)
    J01 = np.einsum("iq,kiqjr,jr,jr->kij", w_i, G, w_j, u_j)
    J11 = np.einsum("iq,iq,kiqjr,jr,jr->kij", w_i, u_i, G, w_j, u_j)
    return J00, J10, J01, J11


class BentTriangularPySim(AbstractPySim):
    """Triangular Galerkin MoM for a single bent (polyline) wire.

    polyline: (M, 3) array of M >= 2 anchor points in 3D. Edge k goes from
        anchor k to anchor k+1.
    n_per_edge: int or sequence of length M-1. Segments per polyline edge.
        If int, used for every edge.
    feed_arclength: float or None. If None, feed is at the interior knot
        whose total-arc-length position is closest to half the wire's total
        arc length. Otherwise feed at the interior knot closest to this
        arc-length value (measured from the start of the polyline).
    n_qp_reg: GL points per segment for the same-edge regular-kernel integral.
    n_qp_off: GL points per segment for cross-edge integrals.

    Defaults reproduce TriangularPySim: a single straight wire from
    (0, 0, 0) to (0, 2 * halfdriver, 0) with nsegs segments.
    """

    def __init__(self, *, polyline=None, n_per_edge=None, feed_arclength=None,
                 n_qp_reg=4, n_qp_off=4, **kwargs):
        super().__init__(**kwargs)
        if polyline is None:
            L = 2 * self.halfdriver
            polyline = np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]])
        self.polyline = np.asarray(polyline, dtype=float)
        if self.polyline.ndim != 2 or self.polyline.shape[0] < 2 or self.polyline.shape[1] != 3:
            raise ValueError("polyline must be (M, 3) with M >= 2")
        n_edges = self.polyline.shape[0] - 1
        if n_per_edge is None:
            n_per_edge = self.nsegs
        if np.isscalar(n_per_edge):
            n_per_edge = [int(n_per_edge)] * n_edges
        self.n_per_edge = list(n_per_edge)
        if len(self.n_per_edge) != n_edges:
            raise ValueError(
                f"n_per_edge length {len(self.n_per_edge)} != number of polyline edges {n_edges}"
            )
        self.feed_arclength = feed_arclength
        self.n_qp_reg = n_qp_reg
        self.n_qp_off = n_qp_off

    def _build_geometry(self):
        """Discretize the polyline into segments. Returns a dict of per-segment
        arrays plus per-edge metadata.
        """
        seg_l_list = []
        seg_r_list = []
        tangent_list = []
        h_list = []
        edge_offsets = [0]
        edge_arc_edges = []  # per-edge arc-length edge arrays (len n_e + 1)
        edge_lengths = []
        for e_idx in range(self.polyline.shape[0] - 1):
            p0 = self.polyline[e_idx]
            p1 = self.polyline[e_idx + 1]
            edge_vec = p1 - p0
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len < 1e-15:
                raise ValueError(f"polyline edge {e_idx} has zero length")
            tan = edge_vec / edge_len
            n_e = self.n_per_edge[e_idx]
            h_e = edge_len / n_e

            t_node = np.linspace(0.0, 1.0, n_e + 1)
            knots = (1 - t_node[:, None]) * p0[None, :] + t_node[:, None] * p1[None, :]
            seg_l_list.append(knots[:-1])
            seg_r_list.append(knots[1:])
            tangent_list.append(np.tile(tan, (n_e, 1)))
            h_list.append(np.full(n_e, h_e))
            edge_arc_edges.append(np.linspace(0.0, edge_len, n_e + 1))
            edge_offsets.append(edge_offsets[-1] + n_e)
            edge_lengths.append(edge_len)

        seg_l = np.vstack(seg_l_list)
        seg_r = np.vstack(seg_r_list)
        tangents = np.vstack(tangent_list)
        h_per_seg = np.concatenate(h_list)

        # Global arc-length at each segment knot. Length = N_total + 1.
        global_arc_at_knot = np.concatenate([[0.0], np.cumsum(h_per_seg)])
        return {
            "seg_l": seg_l, "seg_r": seg_r,
            "tangents": tangents, "h_per_seg": h_per_seg,
            "edge_offsets": edge_offsets,
            "edge_arc_edges": edge_arc_edges,
            "edge_lengths": edge_lengths,
            "global_arc_at_knot": global_arc_at_knot,
            "n_total": seg_l.shape[0],
        }

    def compute_impedance(self, *, ntrap=None):
        a = self.wire_radius
        k = self.k
        geom = self._build_geometry()
        N = geom["n_total"]
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        tangents = geom["tangents"]
        h_per_seg = geom["h_per_seg"]
        edge_offsets = geom["edge_offsets"]
        edge_arc_edges = geom["edge_arc_edges"]
        global_arc_at_knot = geom["global_arc_at_knot"]
        n_edges = len(edge_offsets) - 1

        # Build J tensors (N, N).
        J00 = np.zeros((N, N), dtype=np.complex128)
        J10 = np.zeros_like(J00)
        J01 = np.zeros_like(J00)
        J11 = np.zeros_like(J00)

        for i_e in range(n_edges):
            for j_e in range(n_edges):
                sli = slice(edge_offsets[i_e], edge_offsets[i_e + 1])
                slj = slice(edge_offsets[j_e], edge_offsets[j_e + 1])
                if i_e == j_e:
                    A00, A10, A01, A11 = _seg_seg_static_all(edge_arc_edges[i_e], a)
                    R00, R10, R01, R11 = _seg_seg_reg_all(
                        edge_arc_edges[i_e], a, k, self.n_qp_reg
                    )
                    J00[sli, sli] = A00 + R00
                    J10[sli, sli] = A10 + R10
                    J01[sli, sli] = A01 + R01
                    J11[sli, sli] = A11 + R11
                else:
                    C00, C10, C01, C11 = _seg_seg_offedge_quad(
                        seg_l[sli], seg_r[sli],
                        seg_l[slj], seg_r[slj],
                        a, k, self.n_qp_off,
                    )
                    J00[sli, slj] = C00
                    J10[sli, slj] = C10
                    J01[sli, slj] = C01
                    J11[sli, slj] = C11

        n_basis = N - 1
        left_seg = np.arange(n_basis)
        right_seg = np.arange(1, N)
        hl_m = h_per_seg[left_seg][:, None]
        hl_n = h_per_seg[left_seg][None, :]
        hr_m = h_per_seg[right_seg][:, None]
        hr_n = h_per_seg[right_seg][None, :]

        # Scalar potential: dPhi_m/ds = +1/hl on left_seg[m], -1/hr on right_seg[m].
        S = (
            J00[np.ix_(left_seg, left_seg)] / (hl_m * hl_n)
            - J00[np.ix_(left_seg, right_seg)] / (hl_m * hr_n)
            - J00[np.ix_(right_seg, left_seg)] / (hr_m * hl_n)
            + J00[np.ix_(right_seg, right_seg)] / (hr_m * hr_n)
        )
        Z_Phi = S / (1j * self.omega * self.eps)

        # Vector potential: per-sub-rectangle tangent dot products.
        td_all = tangents @ tangents.T  # (N, N)
        td_ll = td_all[np.ix_(left_seg, left_seg)]
        td_lr = td_all[np.ix_(left_seg, right_seg)]
        td_rl = td_all[np.ix_(right_seg, left_seg)]
        td_rr = td_all[np.ix_(right_seg, right_seg)]

        I_A = (
            td_ll * (J11[np.ix_(left_seg, left_seg)] / (hl_m * hl_n))
            + td_lr * (
                J10[np.ix_(left_seg, right_seg)] / hl_m
                - J11[np.ix_(left_seg, right_seg)] / (hl_m * hr_n)
            )
            + td_rl * (
                J01[np.ix_(right_seg, left_seg)] / hl_n
                - J11[np.ix_(right_seg, left_seg)] / (hr_m * hl_n)
            )
            + td_rr * (
                J00[np.ix_(right_seg, right_seg)]
                - J10[np.ix_(right_seg, right_seg)] / hr_m
                - J01[np.ix_(right_seg, right_seg)] / hr_n
                + J11[np.ix_(right_seg, right_seg)] / (hr_m * hr_n)
            )
        )
        Z_A = 1j * self.omega * self.mu * I_A

        Z = Z_A + Z_Phi
        self.z = Z

        # Feed location: interior knot closest to feed_arclength (default: midpoint).
        total_arc = global_arc_at_knot[-1]
        feed_arc = self.feed_arclength if self.feed_arclength is not None else total_arc / 2.0
        interior_arc = global_arc_at_knot[1:-1]  # length N - 1 = n_basis
        m_center = int(np.argmin(np.abs(interior_arc - feed_arc)))

        v = np.zeros(n_basis, dtype=np.complex128)
        v[m_center] = 1.0
        coeffs = scipy.linalg.solve(Z, v)
        driver_impedance = 1.0 / coeffs[m_center]
        return driver_impedance, coeffs

    def compute_impedance_swept(self, k_array):
        """Driver impedance over a batch of wavenumbers, sharing all
        k-independent work (geometry, static kernel, h/tangent setup).

        k_array: 1D array-like of wavenumbers (rad/m).
        Returns z_in of shape (n_k,) complex.
        """
        a = self.wire_radius
        k_array = np.asarray(k_array, dtype=float)
        n_k = len(k_array)
        omega_array = k_array * self.c

        geom = self._build_geometry()
        N = geom["n_total"]
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        tangents = geom["tangents"]
        h_per_seg = geom["h_per_seg"]
        edge_offsets = geom["edge_offsets"]
        edge_arc_edges = geom["edge_arc_edges"]
        global_arc_at_knot = geom["global_arc_at_knot"]
        n_edges = len(edge_offsets) - 1

        J00 = np.zeros((n_k, N, N), dtype=np.complex128)
        J10 = np.zeros_like(J00)
        J01 = np.zeros_like(J00)
        J11 = np.zeros_like(J00)

        for i_e in range(n_edges):
            for j_e in range(n_edges):
                sli = slice(edge_offsets[i_e], edge_offsets[i_e + 1])
                slj = slice(edge_offsets[j_e], edge_offsets[j_e + 1])
                if i_e == j_e:
                    A00, A10, A01, A11 = _seg_seg_static_all(edge_arc_edges[i_e], a)
                    R00, R10, R01, R11 = _seg_seg_reg_all_batch(
                        edge_arc_edges[i_e], a, k_array, self.n_qp_reg,
                    )
                    J00[:, sli, sli] = A00[None, :, :] + R00
                    J10[:, sli, sli] = A10[None, :, :] + R10
                    J01[:, sli, sli] = A01[None, :, :] + R01
                    J11[:, sli, sli] = A11[None, :, :] + R11
                else:
                    C00, C10, C01, C11 = _seg_seg_offedge_quad_batch(
                        seg_l[sli], seg_r[sli],
                        seg_l[slj], seg_r[slj],
                        a, k_array, self.n_qp_off,
                    )
                    J00[:, sli, slj] = C00
                    J10[:, sli, slj] = C10
                    J01[:, sli, slj] = C01
                    J11[:, sli, slj] = C11

        n_basis = N - 1
        left_seg = np.arange(n_basis)
        right_seg = np.arange(1, N)
        hl_m = h_per_seg[left_seg][:, None]
        hl_n = h_per_seg[left_seg][None, :]
        hr_m = h_per_seg[right_seg][:, None]
        hr_n = h_per_seg[right_seg][None, :]

        # Batched 2-axis fancy indexing into the (n_k, N, N) J tensors.
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

        td_all = tangents @ tangents.T
        td_ll = td_all[np.ix_(left_seg, left_seg)][None, ...]
        td_lr = td_all[np.ix_(left_seg, right_seg)][None, ...]
        td_rl = td_all[np.ix_(right_seg, left_seg)][None, ...]
        td_rr = td_all[np.ix_(right_seg, right_seg)][None, ...]

        I_A = (
            td_ll * (J11[ll] / (hl_m * hl_n))
            + td_lr * (J10[lr] / hl_m - J11[lr] / (hl_m * hr_n))
            + td_rl * (J01[rl] / hl_n - J11[rl] / (hr_m * hl_n))
            + td_rr * (
                J00[rr]
                - J10[rr] / hr_m
                - J01[rr] / hr_n
                + J11[rr] / (hr_m * hr_n)
            )
        )
        Z_A = 1j * omega_array[:, None, None] * self.mu * I_A

        Z = Z_A + Z_Phi  # (n_k, n_basis, n_basis)

        total_arc = global_arc_at_knot[-1]
        feed_arc = self.feed_arclength if self.feed_arclength is not None else total_arc / 2.0
        interior_arc = global_arc_at_knot[1:-1]
        m_center = int(np.argmin(np.abs(interior_arc - feed_arc)))

        v = np.zeros(n_basis, dtype=np.complex128)
        v[m_center] = 1.0
        coeffs = np.linalg.solve(Z, v)  # (n_k, n_basis)
        return 1.0 / coeffs[:, m_center]
