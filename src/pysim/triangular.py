"""Triangular-basis Galerkin MoM for thin straight wires (v3).

Mixed-potential formulation with full analytic static-kernel extraction for
both vector and scalar potential terms. The smooth remainder (e^{-jkR}-1)/R
is handled by per-segment Gauss-Legendre quadrature.

Static kernel: G_static(s, s') = 1/(4 pi sqrt((s-s')^2 + a^2))

Building blocks (without the 1/(4*pi) factor):
    J_pq(alpha, beta, A, B) = int_alpha^beta int_A^B
        (s - alpha)^p (s' - A)^q / sqrt((s-s')^2 + a^2) ds ds'   for p,q in {0,1}

Antiderivatives:
    H_0(u) = u asinh(u/a) - sqrt(u^2 + a^2)
    H_1(u) = ((2u^2 + a^2)/4) asinh(u/a) - (u/4) sqrt(u^2 + a^2)
    H_2(u) = (u^3/3) asinh(u/a) - ((u^2 - 2a^2)/9) sqrt(u^2 + a^2)
    S_0(u) = (u/2) sqrt(u^2 + a^2) + (a^2/2) asinh(u/a)
    S_1(u) = (1/3) (u^2 + a^2)^(3/2)

After inner s' integration (the antiderivative of 1/sqrt((s-s')^2+a^2) is
-asinh((s-s')/a)) the outer integrals are linear combinations of H_p and S_p.
"""
import numpy as np
import scipy.linalg

from .abstract import AbstractPySim


def _Sigma(u, a):
    return np.sqrt(u * u + a * a)


def _H0(u, a):
    return u * np.arcsinh(u / a) - _Sigma(u, a)


def _H1(u, a):
    return ((2 * u * u + a * a) / 4.0) * np.arcsinh(u / a) - (u / 4.0) * _Sigma(u, a)


def _H2(u, a):
    return (u ** 3 / 3.0) * np.arcsinh(u / a) - ((u * u - 2 * a * a) / 9.0) * _Sigma(u, a)


def _S0(u, a):
    return (u / 2.0) * _Sigma(u, a) + (a * a / 2.0) * np.arcsinh(u / a)


def _S1(u, a):
    return (_Sigma(u, a) ** 3) / 3.0


def _bracket(F, alpha, beta):
    """Evaluate F at beta and subtract F at alpha (broadcasts)."""
    return F(beta) - F(alpha)


def _J_static_all(alpha, beta, A, B, a):
    """Return (J_00, J_10, J_01, J_11) without the 1/(4*pi) factor.

    Each output broadcasts: inputs can be scalars or arrays of matching shape.
    """
    def br_H0(c): return _bracket(lambda s: _H0(s - c, a), alpha, beta)
    def br_H1(c): return _bracket(lambda s: _H1(s - c, a), alpha, beta)
    def br_H2(c): return _bracket(lambda s: _H2(s - c, a), alpha, beta)
    def br_S0(c): return _bracket(lambda s: _S0(s - c, a), alpha, beta)
    def br_S1(c): return _bracket(lambda s: _S1(s - c, a), alpha, beta)

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
    """For each segment pair (i, j), return all four J_pq integrals.

    Returns (J00, J10, J01, J11) each of shape (N, N), divided by 4*pi.
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
    """For each segment pair (i, j), return all four J_pq integrals for the
    regular kernel G_reg = (exp(-jkR) - 1)/(4 pi R).

    Returns (J00, J10, J01, J11) each of shape (N, N), complex.
    """
    J00, J10, J01, J11 = _seg_seg_reg_all_batch(
        seg_endpoints, a, np.array([k]), n_qp,
    )
    return J00[0], J10[0], J01[0], J11[0]


def _seg_seg_reg_all_batch(seg_endpoints, a, k_array, n_qp):
    """Batched version of _seg_seg_reg_all.

    k_array: 1D array of measurement wavenumbers.
    Returns (J00, J10, J01, J11) each of shape (n_k, N, N), complex.

    The point-pair distances R are k-independent and computed once; only
    exp(-jk·R) varies over k, which amortizes per-k overhead and keeps R
    in cache across the k-axis reduction.
    """
    N = len(seg_endpoints) - 1
    n_k = len(k_array)
    gl_xi, gl_w = np.polynomial.legendre.leggauss(n_qp)
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
        "iq,iq,kiqjr,jr,jr->kij", w_block, u_block, G_block, w_block, u_block,
    )
    return J00, J10, J01, J11


class TriangularPySim(AbstractPySim):
    def __init__(self, *, n_qp_reg=4, **kwargs):
        super().__init__(**kwargs)
        self.n_qp_reg = n_qp_reg

    def compute_impedance(self, *, ntrap=None):
        # ntrap accepted for API compat; ignored.
        N = self.nsegs
        L = 2 * self.halfdriver
        a = self.wire_radius
        k = self.k
        h = L / N

        seg_edges = np.linspace(0.0, L, N + 1)
        n_basis = N - 1   # interior tents

        # Per-segment-pair moment integrals: static (analytic) + regular (GL).
        A00, A10, A01, A11 = _seg_seg_static_all(seg_edges, a)
        R00, R10, R01, R11 = _seg_seg_reg_all(seg_edges, a, k, self.n_qp_reg)
        J00 = A00 + R00
        J10 = A10 + R10
        J01 = A01 + R01
        J11 = A11 + R11

        # --- Scalar potential ---
        # dPhi_m = +1/h on segment m, -1/h on segment m+1
        # Z_Phi[m,n] = (1/(jwe)) * (1/h^2) * (J00[m,n] - J00[m,n+1] - J00[m+1,n] + J00[m+1,n+1])
        S = (J00[:N - 1, :N - 1] + J00[1:, 1:]
             - J00[:N - 1, 1:] - J00[1:, :N - 1]) / (h * h)
        Z_Phi = S / (1j * self.omega * self.eps)

        # --- Vector potential ---
        # Phi_m on seg i:
        #   i = m   (left):  Phi_m = u/h        (c=0, d=1/h)
        #   i = m+1 (right): Phi_m = 1 - u/h    (c=1, d=-1/h)
        # Same for Phi_n on seg j.
        #
        # Per sub-rectangle contributions to int int Phi_m G Phi_n:
        #   (m, n)     left-left:   (1/h^2) J11[m, n]
        #   (m, n+1)   left-right:  (1/h)   J10[m, n+1] - (1/h^2) J11[m, n+1]
        #   (m+1, n)   right-left:  (1/h)   J01[m+1, n] - (1/h^2) J11[m+1, n]
        #   (m+1, n+1) right-right: J00[m+1, n+1]
        #                          - (1/h)(J01[m+1, n+1] + J10[m+1, n+1])
        #                          + (1/h^2) J11[m+1, n+1]
        I_A = (
            (J11[:N - 1, :N - 1]) / (h * h)
            + (J10[:N - 1, 1:] / h) - (J11[:N - 1, 1:] / (h * h))
            + (J01[1:, :N - 1] / h) - (J11[1:, :N - 1] / (h * h))
            + (J00[1:, 1:])
            - (J01[1:, 1:] + J10[1:, 1:]) / h
            + (J11[1:, 1:] / (h * h))
        )
        Z_A = 1j * self.omega * self.mu * I_A

        Z = Z_A + Z_Phi
        self.z = Z

        # Source at basis closest to L/2.
        knots_full = np.concatenate([[0.0], seg_edges, [L]])
        interior = slice(1, len(knots_full) - 3)  # = slice(1, N)
        apex = knots_full[interior.start + 1 : interior.stop + 1]
        m_center = int(np.argmin(np.abs(apex - L / 2)))
        v = np.zeros(n_basis, dtype=np.complex128)
        v[m_center] = 1.0

        coeffs = scipy.linalg.solve(Z, v)
        driver_impedance = 1.0 / coeffs[m_center]
        return driver_impedance, coeffs
