"""Multi-wire triangular-basis Galerkin MoM for two parallel straight wires.

Extension of TriangularPySim to two parallel y-directed wires (driver +
reflector). Same-wire moment integrals reuse the analytic static-kernel
extraction from TriangularPySim. Cross-wire moments use straight
Gauss-Legendre quadrature on the full kernel: cross-wire R >= spacing
is bounded well above wire_radius, so the kernel is smooth and no
regularization is needed.

Per-wire segmentation: N equal segments per wire. Per-wire basis: N-1
interior tent functions (basis vanishes at wire endpoints, so wire
boundaries are non-adjacent in the global indexing -- the divergence
stencil uses explicit left_seg/right_seg arrays, mirroring the trick
in pysim.yagi.YagiPySim).
"""
import numpy as np
import scipy.linalg

from .abstract import AbstractPySim
from .triangular import (
    _seg_seg_static_all,
    _seg_seg_reg_all,
    _seg_seg_reg_all_batch,
)


def _seg_seg_cross_quad(seg_l_i, seg_r_i, seg_l_j, seg_r_j, k, n_qp):
    """Moment integrals J_pq for segment pairs on different wires.

    J_pq[i, j] = int_{seg i} int_{seg j} u_i^p u_j^q exp(-jkR)/(4 pi R) ds_i ds_j

    R = |r_i(s_i) - r_j(s_j)| is the axis-to-axis distance; no wire-radius
    regularization is applied since the integrand is non-singular.

    seg_l, seg_r: (N, 3) arrays of segment endpoints in 3D.
    Returns (J00, J10, J01, J11), each (N_i, N_j) complex.
    """
    J00, J10, J01, J11 = _seg_seg_cross_quad_batch(
        seg_l_i, seg_r_i, seg_l_j, seg_r_j, np.array([k]), n_qp,
    )
    return J00[0], J10[0], J01[0], J11[0]


def _seg_seg_cross_quad_batch(seg_l_i, seg_r_i, seg_l_j, seg_r_j, k_array, n_qp):
    """Batched version of _seg_seg_cross_quad over a vector of k values."""
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
    R = np.sqrt((diff * diff).sum(-1))
    G = np.exp(-1j * k_array[:, None, None, None, None] * R[None, ...]) / (
        4 * np.pi * R[None, ...]
    )

    J00 = np.einsum("iq,kiqjr,jr->kij", w_i, G, w_j)
    J10 = np.einsum("iq,iq,kiqjr,jr->kij", w_i, u_i, G, w_j)
    J01 = np.einsum("iq,kiqjr,jr,jr->kij", w_i, G, w_j, u_j)
    J11 = np.einsum("iq,iq,kiqjr,jr,jr->kij", w_i, u_i, G, w_j, u_j)
    return J00, J10, J01, J11


class TriangularYagiPySim(AbstractPySim):
    """Two-element Yagi solver with triangular Galerkin MoM.

    Driver at x=0, length 2*halfdriver. Reflector at x=-spacing_factor*halfdriver,
    length 2*reflector_factor*halfdriver. Both y-directed.

    Source: delta-gap at the interior knot of the driver closest to its center.
    """

    def __init__(self, *, n_qp_reg=4, n_qp_cross=4,
                 reflector_factor=1.05, spacing_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_qp_reg = n_qp_reg
        self.n_qp_cross = n_qp_cross
        self.reflector_factor = reflector_factor
        self.spacing_factor = spacing_factor

    def compute_impedance(self, *, ntrap=None):
        N = self.nsegs
        a = self.wire_radius
        k = self.k

        h_driver = self.halfdriver
        L_driver = 2 * h_driver
        L_refl = 2 * self.reflector_factor * h_driver
        spacing = self.spacing_factor * h_driver

        wires = [
            (np.array([0.0, -h_driver, 0.0]),
             np.array([0.0, +h_driver, 0.0])),
            (np.array([-spacing, -self.reflector_factor * h_driver, 0.0]),
             np.array([-spacing, +self.reflector_factor * h_driver, 0.0])),
        ]
        L_per_wire = [L_driver, L_refl]
        h_per_wire = np.array([L_w / N for L_w in L_per_wire])
        n_wires = len(wires)

        seg_l_per_wire = []
        seg_r_per_wire = []
        arc_edges_per_wire = []
        tangents = np.zeros((n_wires, 3))
        for w_idx, ((p0, p1), L_w) in enumerate(zip(wires, L_per_wire)):
            t_node = np.linspace(0.0, 1.0, N + 1)
            knots = (1 - t_node[:, None]) * p0[None, :] + t_node[:, None] * p1[None, :]
            seg_l_per_wire.append(knots[:-1])
            seg_r_per_wire.append(knots[1:])
            arc_edges_per_wire.append(np.linspace(0.0, L_w, N + 1))
            tangents[w_idx] = (p1 - p0) / np.linalg.norm(p1 - p0)

        n_segs_total = n_wires * N
        J00 = np.zeros((n_segs_total, n_segs_total), dtype=np.complex128)
        J10 = np.zeros_like(J00)
        J01 = np.zeros_like(J00)
        J11 = np.zeros_like(J00)

        for w in range(n_wires):
            A00, A10, A01, A11 = _seg_seg_static_all(arc_edges_per_wire[w], a)
            R00, R10, R01, R11 = _seg_seg_reg_all(
                arc_edges_per_wire[w], a, k, self.n_qp_reg
            )
            sl = slice(w * N, (w + 1) * N)
            J00[sl, sl] = A00 + R00
            J10[sl, sl] = A10 + R10
            J01[sl, sl] = A01 + R01
            J11[sl, sl] = A11 + R11

        for i_w in range(n_wires):
            for j_w in range(n_wires):
                if i_w == j_w:
                    continue
                C00, C10, C01, C11 = _seg_seg_cross_quad(
                    seg_l_per_wire[i_w], seg_r_per_wire[i_w],
                    seg_l_per_wire[j_w], seg_r_per_wire[j_w],
                    k, self.n_qp_cross,
                )
                sli = slice(i_w * N, (i_w + 1) * N)
                slj = slice(j_w * N, (j_w + 1) * N)
                J00[sli, slj] = C00
                J10[sli, slj] = C10
                J01[sli, slj] = C01
                J11[sli, slj] = C11

        nb_per_wire = N - 1
        n_basis = n_wires * nb_per_wire
        wire_for_basis = np.repeat(np.arange(n_wires), nb_per_wire)
        m_local = np.tile(np.arange(nb_per_wire), n_wires)
        left_seg = wire_for_basis * N + m_local
        right_seg = wire_for_basis * N + m_local + 1

        h_basis = h_per_wire[wire_for_basis]
        h_m = h_basis[:, None]
        h_n = h_basis[None, :]

        S = (
            J00[np.ix_(left_seg, left_seg)]
            + J00[np.ix_(right_seg, right_seg)]
            - J00[np.ix_(left_seg, right_seg)]
            - J00[np.ix_(right_seg, left_seg)]
        ) / (h_m * h_n)
        Z_Phi = S / (1j * self.omega * self.eps)

        I_A = (
            J11[np.ix_(left_seg, left_seg)] / (h_m * h_n)
            + J10[np.ix_(left_seg, right_seg)] / h_m
            - J11[np.ix_(left_seg, right_seg)] / (h_m * h_n)
            + J01[np.ix_(right_seg, left_seg)] / h_n
            - J11[np.ix_(right_seg, left_seg)] / (h_m * h_n)
            + J00[np.ix_(right_seg, right_seg)]
            - J10[np.ix_(right_seg, right_seg)] / h_m
            - J01[np.ix_(right_seg, right_seg)] / h_n
            + J11[np.ix_(right_seg, right_seg)] / (h_m * h_n)
        )

        t_basis = tangents[wire_for_basis]
        tangent_dot = t_basis @ t_basis.T

        Z_A = 1j * self.omega * self.mu * I_A * tangent_dot
        Z = Z_A + Z_Phi
        self.z = Z

        interior_arc = np.linspace(0.0, L_driver, N + 1)[1:-1]
        m_center = int(np.argmin(np.abs(interior_arc - L_driver / 2)))
        v = np.zeros(n_basis, dtype=np.complex128)
        v[m_center] = 1.0

        coeffs = scipy.linalg.solve(Z, v)
        driver_impedance = 1.0 / coeffs[m_center]
        return driver_impedance, coeffs

    def compute_impedance_swept(self, k_array):
        """Driver impedance over a batch of wavenumbers, sharing all
        k-independent work.

        k_array: 1D array-like of wavenumbers (rad/m).
        Returns z_in of shape (n_k,) complex.
        """
        N = self.nsegs
        a = self.wire_radius
        k_array = np.asarray(k_array, dtype=float)
        n_k = len(k_array)
        omega_array = k_array * self.c

        h_driver = self.halfdriver
        L_driver = 2 * h_driver
        L_refl = 2 * self.reflector_factor * h_driver
        spacing = self.spacing_factor * h_driver

        wires = [
            (np.array([0.0, -h_driver, 0.0]),
             np.array([0.0, +h_driver, 0.0])),
            (np.array([-spacing, -self.reflector_factor * h_driver, 0.0]),
             np.array([-spacing, +self.reflector_factor * h_driver, 0.0])),
        ]
        L_per_wire = [L_driver, L_refl]
        h_per_wire = np.array([L_w / N for L_w in L_per_wire])
        n_wires = len(wires)

        seg_l_per_wire = []
        seg_r_per_wire = []
        arc_edges_per_wire = []
        tangents = np.zeros((n_wires, 3))
        for w_idx, ((p0, p1), L_w) in enumerate(zip(wires, L_per_wire)):
            t_node = np.linspace(0.0, 1.0, N + 1)
            knots = (1 - t_node[:, None]) * p0[None, :] + t_node[:, None] * p1[None, :]
            seg_l_per_wire.append(knots[:-1])
            seg_r_per_wire.append(knots[1:])
            arc_edges_per_wire.append(np.linspace(0.0, L_w, N + 1))
            tangents[w_idx] = (p1 - p0) / np.linalg.norm(p1 - p0)

        n_segs_total = n_wires * N
        J00 = np.zeros((n_k, n_segs_total, n_segs_total), dtype=np.complex128)
        J10 = np.zeros_like(J00)
        J01 = np.zeros_like(J00)
        J11 = np.zeros_like(J00)

        for w in range(n_wires):
            A00, A10, A01, A11 = _seg_seg_static_all(arc_edges_per_wire[w], a)
            R00, R10, R01, R11 = _seg_seg_reg_all_batch(
                arc_edges_per_wire[w], a, k_array, self.n_qp_reg,
            )
            sl = slice(w * N, (w + 1) * N)
            J00[:, sl, sl] = A00[None, :, :] + R00
            J10[:, sl, sl] = A10[None, :, :] + R10
            J01[:, sl, sl] = A01[None, :, :] + R01
            J11[:, sl, sl] = A11[None, :, :] + R11

        for i_w in range(n_wires):
            for j_w in range(n_wires):
                if i_w == j_w:
                    continue
                C00, C10, C01, C11 = _seg_seg_cross_quad_batch(
                    seg_l_per_wire[i_w], seg_r_per_wire[i_w],
                    seg_l_per_wire[j_w], seg_r_per_wire[j_w],
                    k_array, self.n_qp_cross,
                )
                sli = slice(i_w * N, (i_w + 1) * N)
                slj = slice(j_w * N, (j_w + 1) * N)
                J00[:, sli, slj] = C00
                J10[:, sli, slj] = C10
                J01[:, sli, slj] = C01
                J11[:, sli, slj] = C11

        nb_per_wire = N - 1
        n_basis = n_wires * nb_per_wire
        wire_for_basis = np.repeat(np.arange(n_wires), nb_per_wire)
        m_local = np.tile(np.arange(nb_per_wire), n_wires)
        left_seg = wire_for_basis * N + m_local
        right_seg = wire_for_basis * N + m_local + 1

        h_basis = h_per_wire[wire_for_basis]
        h_m = h_basis[:, None]
        h_n = h_basis[None, :]

        ll = (slice(None), left_seg[:, None], left_seg[None, :])
        lr = (slice(None), left_seg[:, None], right_seg[None, :])
        rl = (slice(None), right_seg[:, None], left_seg[None, :])
        rr = (slice(None), right_seg[:, None], right_seg[None, :])

        S = (J00[ll] + J00[rr] - J00[lr] - J00[rl]) / (h_m * h_n)
        Z_Phi = S / (1j * omega_array[:, None, None] * self.eps)

        I_A = (
            J11[ll] / (h_m * h_n)
            + J10[lr] / h_m
            - J11[lr] / (h_m * h_n)
            + J01[rl] / h_n
            - J11[rl] / (h_m * h_n)
            + J00[rr]
            - J10[rr] / h_m
            - J01[rr] / h_n
            + J11[rr] / (h_m * h_n)
        )
        t_basis = tangents[wire_for_basis]
        tangent_dot = t_basis @ t_basis.T

        Z_A = 1j * omega_array[:, None, None] * self.mu * I_A * tangent_dot[None, ...]
        Z = Z_A + Z_Phi  # (n_k, n_basis, n_basis)

        interior_arc = np.linspace(0.0, L_driver, N + 1)[1:-1]
        m_center = int(np.argmin(np.abs(interior_arc - L_driver / 2)))
        v = np.zeros(n_basis, dtype=np.complex128)
        v[m_center] = 1.0
        coeffs = np.linalg.solve(Z, v)
        return 1.0 / coeffs[:, m_center]
