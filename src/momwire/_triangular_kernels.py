"""Segment-pair moment integrals for the triangular-basis Galerkin MoM.

Three flavours of (N_i, N_j) integrals J_pq[i, j] are exposed here:

  * `_seg_seg_static_all`: same-edge (straight wire) pairs, analytic static
    kernel G_static(s, s') = 1/(4 pi sqrt((s-s')^2 + a^2)) with closed-form
    antiderivatives. The smooth remainder is computed by:

  * `_seg_seg_reg_all` / `_seg_seg_reg_all_batch`: same-edge regular kernel
    G_reg = (exp(-jkR) - 1)/(4 pi R) by per-segment Gauss-Legendre quadrature.

  * `_seg_seg_offedge_quad` / `_seg_seg_offedge_quad_batch`: cross-edge /
    cross-wire pairs by full 3D Gauss-Legendre quadrature on the regularized
    kernel G(R) = exp(-jkR)/(4 pi R), R = sqrt(|r_i - r_j|^2 + a^2).

Each J_pq integrates u_i^p u_j^q against the kernel, where u_i, u_j are arc
lengths into each segment from its left endpoint. The four (p, q) combos
(00, 10, 01, 11) are the building blocks TriangularSolver uses to assemble
the per-sub-rectangle contributions of the basis-pair matrix.

Static kernel building blocks (without the 1/(4*pi) factor):
    J_pq(alpha, beta, A, B) = int_alpha^beta int_A^B
        (s - alpha)^p (s' - A)^q / sqrt((s-s')^2 + a^2) ds ds'

Antiderivatives:
    H_0(u) = u asinh(u/a) - sqrt(u^2 + a^2)
    H_1(u) = ((2u^2 + a^2)/4) asinh(u/a) - (u/4) sqrt(u^2 + a^2)
    H_2(u) = (u^3/3) asinh(u/a) - ((u^2 - 2a^2)/9) sqrt(u^2 + a^2)
    S_0(u) = (u/2) sqrt(u^2 + a^2) + (a^2/2) asinh(u/a)
    S_1(u) = (1/3) (u^2 + a^2)^(3/2)
"""

import numpy as np

from ._quadrature import leggauss

try:
    from . import _accelerators as _acc

    _HAVE_REG_ACCEL = hasattr(_acc, "seg_seg_reg_quad_batch_1d")
    _HAVE_OFF_ACCEL = hasattr(_acc, "seg_seg_quad_batch_3d")
except ImportError:
    _HAVE_REG_ACCEL = False
    _HAVE_OFF_ACCEL = False


def _Sigma(u, a):
    return np.sqrt(u * u + a * a)


def _H0(u, a):
    return u * np.arcsinh(u / a) - _Sigma(u, a)


def _H1(u, a):
    return ((2 * u * u + a * a) / 4.0) * np.arcsinh(u / a) - (u / 4.0) * _Sigma(u, a)


def _H2(u, a):
    return (u**3 / 3.0) * np.arcsinh(u / a) - ((u * u - 2 * a * a) / 9.0) * _Sigma(u, a)


def _S0(u, a):
    return (u / 2.0) * _Sigma(u, a) + (a * a / 2.0) * np.arcsinh(u / a)


def _S1(u, a):
    return (_Sigma(u, a) ** 3) / 3.0


def _bracket(F, alpha, beta):
    return F(beta) - F(alpha)


def _J_static_all(alpha, beta, A, B, a):
    """Return (J_00, J_10, J_01, J_11) without the 1/(4*pi) factor.

    Inputs broadcast: scalars or arrays of matching shape.
    """

    def br_H0(c):
        return _bracket(lambda s: _H0(s - c, a), alpha, beta)

    def br_H1(c):
        return _bracket(lambda s: _H1(s - c, a), alpha, beta)

    def br_H2(c):
        return _bracket(lambda s: _H2(s - c, a), alpha, beta)

    def br_S0(c):
        return _bracket(lambda s: _S0(s - c, a), alpha, beta)

    def br_S1(c):
        return _bracket(lambda s: _S1(s - c, a), alpha, beta)

    H0_A = br_H0(A)
    H0_B = br_H0(B)
    H1_A = br_H1(A)
    H1_B = br_H1(B)
    H2_A = br_H2(A)
    H2_B = br_H2(B)
    S0_A = br_S0(A)
    S0_B = br_S0(B)
    S1_A = br_S1(A)
    S1_B = br_S1(B)

    J00 = H0_A - H0_B
    J10 = (H1_A + (A - alpha) * H0_A) - (H1_B + (B - alpha) * H0_B)
    J01 = H1_A - H1_B - (B - A) * H0_B - S0_A + S0_B
    J11 = (
        (H2_A + (A - alpha) * H1_A)
        - (H2_B + (2 * B - alpha - A) * H1_B + (B - alpha) * (B - A) * H0_B)
        - (S1_A + (A - alpha) * S0_A)
        + (S1_B + (B - alpha) * S0_B)
    )
    return J00, J10, J01, J11


def _seg_seg_static_all(seg_endpoints, a):
    """For each segment pair (i, j) on a single straight edge, return all four
    J_pq integrals, each of shape (N, N), divided by 4*pi.
    """
    sl = seg_endpoints[:-1]
    sr = seg_endpoints[1:]
    alpha = sl[:, None]
    beta = sr[:, None]
    A = sl[None, :]
    B = sr[None, :]
    J00, J10, J01, J11 = _J_static_all(alpha, beta, A, B, a)
    inv = 1.0 / (4 * np.pi)
    return J00 * inv, J10 * inv, J01 * inv, J11 * inv


def _seg_seg_reg_all(seg_endpoints, a, k, n_qp):
    """Same-edge regular kernel G_reg = (exp(-jkR) - 1)/(4 pi R) at a single k.

    Returns (J00, J10, J01, J11), each (N, N) complex.
    """
    J00, J10, J01, J11 = _seg_seg_reg_all_batch(
        seg_endpoints,
        a,
        np.array([k]),
        n_qp,
    )
    return J00[0], J10[0], J01[0], J11[0]


def _seg_seg_reg_all_batch(seg_endpoints, a, k_array, n_qp):
    """Batched same-edge regular-kernel integrals over a 1D vector of k values.

    Point-pair distances R are k-independent and computed once; only
    exp(-jk·R) varies over k, which amortizes per-k overhead and keeps R
    in cache across the k-axis reduction.

    Returns (J00, J10, J01, J11), each of shape (n_k, N, N), complex.
    """
    gl_xi, gl_w = leggauss(n_qp)
    if _HAVE_REG_ACCEL:
        gl_t_01 = 0.5 * (gl_xi + 1.0)
        gl_w_01 = 0.5 * gl_w
        return _acc.seg_seg_reg_quad_batch_1d(
            np.ascontiguousarray(seg_endpoints, dtype=np.float64),
            float(a),
            np.ascontiguousarray(k_array, dtype=np.float64),
            np.ascontiguousarray(gl_t_01, dtype=np.float64),
            np.ascontiguousarray(gl_w_01, dtype=np.float64),
        )

    N = len(seg_endpoints) - 1
    n_k = len(k_array)
    sl = seg_endpoints[:-1]
    sr = seg_endpoints[1:]
    half = 0.5 * (sr - sl)
    mid = 0.5 * (sr + sl)
    s_q = (mid[:, None] + half[:, None] * gl_xi[None, :]).ravel()
    w_q = (half[:, None] * gl_w[None, :]).ravel()
    seg_idx = np.repeat(np.arange(N), n_qp)
    u_q = s_q - sl[seg_idx]

    diffs = s_q[:, None] - s_q[None, :]
    R = np.sqrt(diffs * diffs + a * a)
    G_reg = (np.exp(-1j * k_array[:, None, None] * R[None, ...]) - 1.0) / (
        4 * np.pi * R[None, ...]
    )

    w_block = w_q.reshape(N, n_qp)
    u_block = u_q.reshape(N, n_qp)
    G_block = G_reg.reshape(n_k, N, n_qp, N, n_qp)

    J00 = np.einsum("iq,kiqjr,jr->kij", w_block, G_block, w_block)
    J10 = np.einsum("iq,iq,kiqjr,jr->kij", w_block, u_block, G_block, w_block)
    J01 = np.einsum("iq,kiqjr,jr,jr->kij", w_block, G_block, w_block, u_block)
    J11 = np.einsum(
        "iq,iq,kiqjr,jr,jr->kij",
        w_block,
        u_block,
        G_block,
        w_block,
        u_block,
    )
    return J00, J10, J01, J11


def _seg_seg_offedge_quad(seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k, n_qp):
    """Cross-edge / cross-wire integrals at a single k.

    Uses 3D Gauss-Legendre quadrature on the regularized kernel
    G(R) = exp(-jkR) / (4 pi R) with R = sqrt(|r_i - r_j|^2 + a^2). The
    regularization keeps G finite even when seg_i and seg_j share a corner
    point.

    seg_l, seg_r: (N, 3) arrays of 3D endpoints.
    Returns (J00, J10, J01, J11), each (N_i, N_j) complex.
    """
    J00, J10, J01, J11 = _seg_seg_offedge_quad_batch(
        seg_l_i,
        seg_r_i,
        seg_l_j,
        seg_r_j,
        a,
        np.array([k]),
        n_qp,
    )
    return J00[0], J10[0], J01[0], J11[0]


def _seg_seg_offedge_quad_batch(seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k_array, n_qp):
    """Batched cross-edge / cross-wire integrals over a 1D vector of k values.

    Returns (J00, J10, J01, J11), each of shape (n_k, N_i, N_j).
    """
    gl_xi, gl_w = leggauss(n_qp)
    t_qp = 0.5 * (gl_xi + 1.0)
    w_qp = 0.5 * gl_w
    if _HAVE_OFF_ACCEL:
        return _acc.seg_seg_quad_batch_3d(
            np.ascontiguousarray(seg_l_i, dtype=np.float64),
            np.ascontiguousarray(seg_r_i, dtype=np.float64),
            np.ascontiguousarray(seg_l_j, dtype=np.float64),
            np.ascontiguousarray(seg_r_j, dtype=np.float64),
            float(a) * float(a),
            np.ascontiguousarray(k_array, dtype=np.float64),
            t_qp,
            w_qp,
        )

    len_i = np.linalg.norm(seg_r_i - seg_l_i, axis=1)
    len_j = np.linalg.norm(seg_r_j - seg_l_j, axis=1)

    pos_i = (1 - t_qp[None, :, None]) * seg_l_i[:, None, :] + t_qp[
        None, :, None
    ] * seg_r_i[:, None, :]
    pos_j = (1 - t_qp[None, :, None]) * seg_l_j[:, None, :] + t_qp[
        None, :, None
    ] * seg_r_j[:, None, :]

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
