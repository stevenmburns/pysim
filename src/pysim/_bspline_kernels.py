"""Polynomial-moment integrals for the B-spline Galerkin MoM.

Two flavours of same-edge moment integrals are exposed:

  * `_seg_seg_static_moments`: closed-form static-kernel piece
        S_pq[i, j] = ∫∫ (s-α_i)^p (s'-A_j)^q / (4π √((s-s')²+a²)) ds' ds
    for p, q ∈ {0, ..., max_d}, on every (i, j) pair of an edge's N segments.
    The closed form (sympy-derived once and dumped to
    `_bspline_static_moments.py`) handles the log singularity on the (i, i)
    diagonal that Gauss-Legendre quadrature converges on only
    logarithmically.

  * `_seg_seg_reg_moments`: smooth-kernel piece
        R_pq[i, j] = ∫∫ (s-α_i)^p (s'-A_j)^q · (exp(-jkR)-1)/(4π R) ds' ds
    by per-segment Gauss-Legendre quadrature on the bounded difference
    (exp(-jkR) - 1)/R (limit -jk at R = 0).

The full moment is S_pq + R_pq on same-edge pairs; for cross-edge / cross-
wire pairs (not in this first-cut single-wire scope) the unregularized
GL quadrature on G = exp(-jkR)/(4π R) is fine because R ≥ a there.

For the triangular basis (degree 1) this is bit-for-bit equivalent to the
existing `_seg_seg_static_all` + `_seg_seg_reg_all` kernels; the new
module is just the generalization to higher polynomial moments.
"""

import numpy as np

from ._bspline_static_moments import J_static_moment

MAX_D_SUPPORTED = 2


def _seg_seg_static_moments(seg_endpoints, a, max_d):
    """Closed-form same-edge static-kernel moment integrals.

    seg_endpoints: (N+1,) array of arc lengths along a single straight edge.
    Returns J_static of shape (max_d+1, max_d+1, N, N), with the 1/(4π)
    prefactor folded in.
    """
    if max_d > MAX_D_SUPPORTED:
        raise NotImplementedError(
            f"max_d={max_d}: only {MAX_D_SUPPORTED} pre-derived. Run "
            "scripts/derive_bspline_static_moments.py with a larger MAX_D."
        )
    sl = np.ascontiguousarray(seg_endpoints[:-1], dtype=np.float64)
    sr = np.ascontiguousarray(seg_endpoints[1:], dtype=np.float64)
    alpha = sl[:, None]
    beta = sr[:, None]
    A = sl[None, :]
    B = sr[None, :]
    n_d = max_d + 1
    N = len(sl)
    out = np.empty((n_d, n_d, N, N), dtype=np.float64)
    inv4pi = 1.0 / (4 * np.pi)
    for p in range(n_d):
        for q in range(n_d):
            out[p, q] = J_static_moment(p, q, alpha, beta, A, B, a) * inv4pi
    return out


def _seg_seg_reg_moments(seg_endpoints, a, k, max_d, n_qp):
    """Smooth-kernel piece (exp(-jkR) - 1)/(4π R) over polynomial moments
    on every same-edge segment pair, via Gauss-Legendre quadrature.

    seg_endpoints: (N+1,) array of arc lengths along a single straight edge.
    a, k: regularization radius and wavenumber.
    max_d: maximum moment degree (inclusive).
    n_qp: Gauss-Legendre nodes per segment per axis.

    Returns J_reg of shape (max_d+1, max_d+1, N, N) complex.
    """
    gl_xi, gl_w = np.polynomial.legendre.leggauss(n_qp)
    t01 = 0.5 * (gl_xi + 1.0)
    w01 = 0.5 * gl_w

    sl = seg_endpoints[:-1]
    h_seg = seg_endpoints[1:] - sl
    N = len(sl)

    s_q = sl[:, None] + t01[None, :] * h_seg[:, None]  # (N, n_qp) global arc
    u_q = (t01[None, :] * h_seg[:, None]) * np.ones((N, 1))  # (N, n_qp) local
    w_q = (w01[None, :] * h_seg[:, None]) * np.ones((N, 1))

    s_flat = s_q.ravel()
    diff = s_flat[:, None] - s_flat[None, :]
    R = np.sqrt(diff * diff + a * a)
    # (exp(-jkR) - 1) / (4π R). At R = a small, this is bounded → -jk/(4π) in
    # the a → 0, kR → 0 limit; no quadrature pathology.
    G_reg = (np.exp(-1j * k * R) - 1.0) / (4 * np.pi * R)

    # u^p evaluated at every quadrature node, weight-folded
    u_pow = np.stack([u_q**p for p in range(max_d + 1)], axis=0)  # (max_d+1, N, n_qp)
    wu_pow = w_q[None, :, :] * u_pow

    G_block = G_reg.reshape(N, n_qp, N, n_qp)
    # J_reg[p, P, i, j] = sum_{q, r} wu_pow[p, i, q] G[i, q, j, r] wu_pow[P, j, r]
    J_reg = np.einsum("piq,iqjr,Pjr->pPij", wu_pow, G_block, wu_pow)
    return J_reg


def _seg_seg_full_moments_offedge(
    seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k, max_d, n_qp
):
    """Full-kernel moment integrals on (cross-edge / cross-wire) pairs.

    For pairs where the two segments don't share any common arc, R ≥ h > a
    away from the singular diagonal, so direct GL on the regularized
    G = exp(-jkR)/(4π R) is accurate.

    Returns (max_d+1, max_d+1, N_i, N_j) complex. (Provided here for
    future polyline / multi-wire extension; the single-straight-wire first
    cut doesn't call it.)
    """
    gl_xi, gl_w = np.polynomial.legendre.leggauss(n_qp)
    t01 = 0.5 * (gl_xi + 1.0)
    w01 = 0.5 * gl_w

    len_i = np.linalg.norm(seg_r_i - seg_l_i, axis=1)
    len_j = np.linalg.norm(seg_r_j - seg_l_j, axis=1)

    pos_i = (1 - t01[None, :, None]) * seg_l_i[:, None, :] + t01[
        None, :, None
    ] * seg_r_i[:, None, :]
    pos_j = (1 - t01[None, :, None]) * seg_l_j[:, None, :] + t01[
        None, :, None
    ] * seg_r_j[:, None, :]

    u_i = t01[None, :] * len_i[:, None]
    u_j = t01[None, :] * len_j[:, None]
    w_i = w01[None, :] * len_i[:, None]
    w_j = w01[None, :] * len_j[:, None]

    diff = pos_i[:, :, None, None, :] - pos_j[None, None, :, :, :]
    R = np.sqrt((diff * diff).sum(-1) + a * a)
    G = np.exp(-1j * k * R) / (4 * np.pi * R)

    u_pow_i = np.stack([u_i**p for p in range(max_d + 1)], axis=0)
    u_pow_j = np.stack([u_j**p for p in range(max_d + 1)], axis=0)
    wu_i = w_i[None, :, :] * u_pow_i
    wu_j = w_j[None, :, :] * u_pow_j

    return np.einsum("piq,iqjr,Pjr->pPij", wu_i, G, wu_j)
