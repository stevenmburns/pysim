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
from ._quadrature import leggauss

from ._accel import acc as _acc

_HAVE_BSPLINE_ACCEL = _acc is not None and hasattr(_acc, "seg_seg_full_moments_bspline")
_HAVE_BSPLINE_STATIC_ACCEL = _acc is not None and hasattr(
    _acc, "seg_seg_static_moments_bspline_uniform"
)
_HAVE_BSPLINE_REG_SWEPT_ACCEL = _acc is not None and hasattr(
    _acc, "seg_seg_reg_moments_bspline_swept"
)

# Currently the C++ accelerator has explicit instantiations for D in {1, 2}.
# Extend by adding `seg_seg_full_moments_bspline_kernel<3>(...)` and a switch
# case in src/momwire/_accelerators.cpp.
_BSPLINE_ACCEL_MAX_D = 2

MAX_D_SUPPORTED = 2


def _seg_seg_static_moments(seg_endpoints, a, max_d):
    """Closed-form same-edge static-kernel moment integrals.

    seg_endpoints: (N+1,) array of arc lengths along a single straight edge.
    Returns J_static of shape (max_d+1, max_d+1, N, N), with the 1/(4π)
    prefactor folded in.

    Fast path: when the edge segments are uniform-h, J_pq[i, j] depends only
    on (j-i)·h (the integrand is translation-invariant in the arc-length
    direction along the straight edge), so the (N, N) matrix is Toeplitz —
    2N-1 unique values per moment instead of N². At N=21 this is ~10× faster
    than the dense evaluation; at N=81 it's ~40×.
    """
    if max_d > MAX_D_SUPPORTED:
        raise NotImplementedError(
            f"max_d={max_d}: only {MAX_D_SUPPORTED} pre-derived. Run "
            "scripts/derive_bspline_static_moments.py with a larger MAX_D."
        )
    sl = np.ascontiguousarray(seg_endpoints[:-1], dtype=np.float64)
    sr = np.ascontiguousarray(seg_endpoints[1:], dtype=np.float64)
    N = len(sl)
    h_seg = sr - sl
    n_d = max_d + 1
    inv4pi = 1.0 / (4 * np.pi)

    uniform = N >= 1 and np.allclose(h_seg, h_seg[0], rtol=1e-12, atol=1e-15)
    if not uniform:
        alpha = sl[:, None]
        beta = sr[:, None]
        A = sl[None, :]
        B = sr[None, :]
        out = np.empty((n_d, n_d, N, N), dtype=np.float64)
        for p in range(n_d):
            for q in range(n_d):
                out[p, q] = J_static_moment(p, q, alpha, beta, A, B, a) * inv4pi
        return out

    # Uniform-h fast paths
    h = float(h_seg[0])
    if _HAVE_BSPLINE_STATIC_ACCEL and max_d <= _BSPLINE_ACCEL_MAX_D:
        # C++ inlined sympy-derived closed forms — ~50× faster than numpy
        # because each call escapes per-op dispatch overhead.
        return _acc.seg_seg_static_moments_bspline_uniform(
            float(h), float(a), int(N), int(max_d)
        )

    # numpy Toeplitz fallback: J_pq[i, j] = vals_pq[j - i + (N - 1)]
    delta = np.arange(-(N - 1), N, dtype=np.float64)
    alpha = np.zeros_like(delta)
    beta = np.full_like(delta, h)
    A = delta * h
    B = (delta + 1.0) * h
    j_minus_i = np.arange(N)[None, :] - np.arange(N)[:, None]
    gather_idx = j_minus_i + (N - 1)
    out = np.empty((n_d, n_d, N, N), dtype=np.float64)
    for p in range(n_d):
        for q in range(n_d):
            vals = J_static_moment(p, q, alpha, beta, A, B, a) * inv4pi
            out[p, q] = vals[gather_idx]
    return out


def _seg_seg_reg_geometry(seg_endpoints, a, max_d, n_qp):
    """k-independent precompute for `_seg_seg_reg_moments`.

    Everything in the smooth-kernel moment integral except the `exp(-jkR)`
    phase: the pair-distance table R and the weight-folded local-coordinate
    powers. Hoisting this out of a swept-k loop turns the per-k same-edge
    work into a single `exp(-jkR)` + einsum (see
    `_seg_seg_reg_moments_from_geometry`). Bounded memory — one edge's
    (N·n_qp, N·n_qp) R table at a time, same as the per-k path.

    Returns a dict consumed by `_seg_seg_reg_moments_from_geometry`.
    """
    gl_xi, gl_w = leggauss(n_qp)
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

    # u^p evaluated at every quadrature node, weight-folded
    u_pow = np.stack([u_q**p for p in range(max_d + 1)], axis=0)  # (max_d+1, N, n_qp)
    wu_pow = w_q[None, :, :] * u_pow

    return {"R": R, "wu_pow": wu_pow, "N": N, "n_qp": n_qp}


def _seg_seg_reg_moments_from_geometry(geo, k):
    """Per-k smooth-kernel moment block from a `_seg_seg_reg_geometry` dict."""
    R = geo["R"]
    wu_pow = geo["wu_pow"]
    N = geo["N"]
    n_qp = geo["n_qp"]
    # Single-k case (the non-swept compute_impedance): the same streaming C++
    # kernel serves it with a length-1 k axis, which we squeeze back off. This
    # is the ~65% of a single d=2 solve that the numpy einsum below otherwise
    # dominates. Bit-close (different reduction order); numpy stays the fallback.
    if _HAVE_BSPLINE_REG_SWEPT_ACCEL:
        return _acc.seg_seg_reg_moments_bspline_swept(
            np.ascontiguousarray(R, dtype=np.float64),
            np.ascontiguousarray(wu_pow, dtype=np.float64),
            np.ascontiguousarray(np.asarray([k], dtype=np.float64)),
        )[0]
    # (exp(-jkR) - 1) / (4π R). At R = a small, this is bounded → -jk/(4π) in
    # the a → 0, kR → 0 limit; no quadrature pathology.
    G_reg = (np.exp(-1j * k * R) - 1.0) / (4 * np.pi * R)
    G_block = G_reg.reshape(N, n_qp, N, n_qp)
    # J_reg[p, P, i, j] = sum_{q, r} wu_pow[p, i, q] G[i, q, j, r] wu_pow[P, j, r]
    return np.einsum("piq,iqjr,Pjr->pPij", wu_pow, G_block, wu_pow)


def _seg_seg_reg_moments_from_geometry_swept(geo, k_array, max_chunk_bytes=256 << 20):
    """Batched `_seg_seg_reg_moments_from_geometry` over a vector of k.

    Returns (n_k, max_d+1, max_d+1, N, N). The only k-dependent factor is the
    phase `exp(-jkR)`, so R and the weighted powers are reused across the
    whole sweep and the (q, r) quadrature reduction is done once per edge as a
    single batched einsum instead of n_k small ones.

    The (chunk, N·n_qp, N·n_qp) phase intermediate is the memory hot-spot, so
    k is processed in chunks sized to keep it under `max_chunk_bytes`. The
    returned moment block is n_qp² smaller than that intermediate, so storing
    it for the whole sweep is cheap relative to the per-k full-J the caller
    already materializes.
    """
    R = geo["R"]
    wu_pow = geo["wu_pow"]
    N = geo["N"]
    n_qp = geo["n_qp"]
    n_d = wu_pow.shape[0]
    k_array = np.asarray(k_array, dtype=float)
    n_k = k_array.shape[0]

    # Streaming C++ kernel: evaluates exp(-jkR) once per (iq, jr, k) and
    # accumulates straight into the (n_d, n_d) moment block, so it never
    # materializes the (chunk, N*n_qp, N*n_qp) phase intermediate this numpy
    # path has to chunk under max_chunk_bytes. Bit-close (different reduction
    # order) to the einsum below, which stays as the fallback.
    if _HAVE_BSPLINE_REG_SWEPT_ACCEL:
        return _acc.seg_seg_reg_moments_bspline_swept(
            np.ascontiguousarray(R, dtype=np.float64),
            np.ascontiguousarray(wu_pow, dtype=np.float64),
            np.ascontiguousarray(k_array, dtype=np.float64),
        )

    out = np.empty((n_k, n_d, n_d, N, N), dtype=np.complex128)
    bytes_per_k = R.size * 16  # complex128 phase table for one k
    chunk = max(1, int(max_chunk_bytes // max(bytes_per_k, 1)))
    inv4pi_R = 1.0 / (4 * np.pi * R)
    for c0 in range(0, n_k, chunk):
        kk = k_array[c0 : c0 + chunk]
        G = (np.exp(-1j * kk[:, None, None] * R[None, :, :]) - 1.0) * inv4pi_R[
            None, :, :
        ]
        G_block = G.reshape(kk.shape[0], N, n_qp, N, n_qp)
        out[c0 : c0 + chunk] = np.einsum(
            "piq,kiqjr,Pjr->kpPij", wu_pow, G_block, wu_pow, optimize=True
        )
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
    geo = _seg_seg_reg_geometry(seg_endpoints, a, max_d, n_qp)
    return _seg_seg_reg_moments_from_geometry(geo, k)


def _seg_seg_full_moments_offedge(
    seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k, max_d, n_qp
):
    """Full-kernel moment integrals on all segment pairs.

    Returns (max_d+1, max_d+1, N_i, N_j) complex. Uses the C++ accelerator
    when available and `max_d` is in the instantiation set; otherwise falls
    back to the pure-numpy reference.

    The same a² wire-radius regularization handles diagonals (i = j) where
    R could otherwise vanish, and touching segments at kink corners; the
    bspline solver overwrites the same-edge blocks with analytic_static +
    GL_reg afterwards, so the accuracy of this call only needs to be good
    on far / nearly-far pairs.
    """
    gl_xi, gl_w = leggauss(n_qp)
    t01 = 0.5 * (gl_xi + 1.0)
    w01 = 0.5 * gl_w

    if _HAVE_BSPLINE_ACCEL and max_d <= _BSPLINE_ACCEL_MAX_D:
        return _acc.seg_seg_full_moments_bspline(
            np.ascontiguousarray(seg_l_i, dtype=np.float64),
            np.ascontiguousarray(seg_r_i, dtype=np.float64),
            np.ascontiguousarray(seg_l_j, dtype=np.float64),
            np.ascontiguousarray(seg_r_j, dtype=np.float64),
            float(a) * float(a),
            float(k),
            int(max_d),
            np.ascontiguousarray(t01, dtype=np.float64),
            np.ascontiguousarray(w01, dtype=np.float64),
        )

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
