"""Probe the convergence rate of TriangularPySim and its source-model variants.

The NEXT_STEPS proposal flagged that TriangularPySim converges empirically at
~O(1/N^1.3), well short of the theoretical O(1/N^2) for triangular Galerkin.
Two hypotheses were on the table:

  (a) Off-center source for odd N: m_center is exact-center only for even N.
  (b) Point-evaluation source projection (v[m_center] = 1.0) is not the right
      Galerkin projection of a true delta-gap, and the delta-gap itself is
      unphysical (infinite source energy).

This script measures both. It sweeps N over even-only and odd-only sequences,
tries three source models, and reports the log-log slope of |Z(N) - Z_ref|
versus N. The reference Z_ref is the highest-N even sample of the cleanest-
converging method, so the absolute slope numbers should be read as "how fast
do the lower-N samples approach this method's own asymptote."

Source models tested:
  - "point_delta":    v[m_center] = 1, others = 0 (current TriangularPySim
                      behavior). For odd N this puts the source at the off-
                      center apex closest to L/2.
  - "galerkin_delta": v[m] = Phi_m(L/2), giving v[m_center] = 1 for even N,
                      v[m_left] = v[m_right] = 0.5 for odd N (proper Galerkin
                      projection of delta(s - L/2)). Z = 1 / I(L/2) where
                      I(L/2) = sum(coeffs[m] * Phi_m(L/2)).
  - "finite_gap":     V(s) = 1/h_gap over a one-segment gap centered at L/2,
                      0 elsewhere. v[m] = integral Phi_m(s) V(s) ds. Z = V_total
                      / I(L/2) where V_total = 1. For even N the gap is one
                      segment; for odd N the gap straddles the L/2 knot
                      (one segment of width h on each side, treating L/2 as
                      the gap midpoint).

Run from the project venv:
    PYTHONPATH=/home/smburns/antennas/pysim/src .venv/bin/python \\
        scripts/triangular_convergence.py
"""
import numpy as np
import scipy.linalg

from pysim.abstract import AbstractPySim
from pysim.triangular import _seg_seg_static_all, _seg_seg_reg_all


def solve_with_source(N, halfdriver, wire_radius, k, omega, eps, mu, n_qp_reg,
                      source_model):
    """Build the triangular MoM system and solve with one of three source
    models. Returns the driver impedance Z.
    """
    L = 2 * halfdriver
    a = wire_radius
    h = L / N

    seg_edges = np.linspace(0.0, L, N + 1)
    n_basis = N - 1

    A00, A10, A01, A11 = _seg_seg_static_all(seg_edges, a)
    R00, R10, R01, R11 = _seg_seg_reg_all(seg_edges, a, k, n_qp_reg)
    J00 = A00 + R00
    J10 = A10 + R10
    J01 = A01 + R01
    J11 = A11 + R11

    S = (J00[: N - 1, : N - 1] + J00[1:, 1:]
         - J00[: N - 1, 1:] - J00[1:, : N - 1]) / (h * h)
    Z_Phi = S / (1j * omega * eps)

    I_A = (
        (J11[: N - 1, : N - 1]) / (h * h)
        + (J10[: N - 1, 1:] / h) - (J11[: N - 1, 1:] / (h * h))
        + (J01[1:, : N - 1] / h) - (J11[1:, : N - 1] / (h * h))
        + (J00[1:, 1:])
        - (J01[1:, 1:] + J10[1:, 1:]) / h
        + (J11[1:, 1:] / (h * h))
    )
    Z_A = 1j * omega * mu * I_A
    Z = Z_A + Z_Phi

    apex_arc = np.linspace(h, (N - 1) * h, N - 1)  # interior knot positions

    if source_model == "point_delta":
        # Current TriangularPySim behavior.
        m_center = int(np.argmin(np.abs(apex_arc - L / 2)))
        v = np.zeros(n_basis, dtype=np.complex128)
        v[m_center] = 1.0
        coeffs = scipy.linalg.solve(Z, v)
        return 1.0 / coeffs[m_center]

    if source_model == "galerkin_delta":
        # Proper Galerkin projection of delta(s - L/2): v[m] = Phi_m(L/2).
        # Phi_m(L/2) = max(0, 1 - |apex[m] - L/2|/h).
        diff = np.abs(apex_arc - L / 2)
        phi_at_center = np.where(diff < h, 1.0 - diff / h, 0.0)
        v = phi_at_center.astype(np.complex128)
        coeffs = scipy.linalg.solve(Z, v)
        I_at_center = phi_at_center @ coeffs
        return 1.0 / I_at_center

    if source_model == "finite_gap":
        # V(s) = 1/h over a one-segment-wide gap centered at L/2.
        # v[m] = integral Phi_m(s) V(s) ds (total source integral is 1).
        # Gap endpoints: [L/2 - h/2, L/2 + h/2].
        gap_lo, gap_hi = L / 2 - h / 2, L / 2 + h / 2
        v = np.zeros(n_basis, dtype=np.complex128)
        # For each tent m, integrate Phi_m over [gap_lo, gap_hi] then divide by h.
        # Use simple closed-form: Phi_m is piecewise linear on three pieces
        # over [apex-h, apex+h]; compute the intersection with [gap_lo, gap_hi].
        for m in range(n_basis):
            apex = apex_arc[m]
            # Phi_m(s) = max(0, 1 - |s - apex|/h)
            lo = max(gap_lo, apex - h)
            hi = min(gap_hi, apex + h)
            if hi <= lo:
                continue
            # Split into the two linear pieces (left of apex, right of apex).
            integral = 0.0
            l1, h1 = lo, min(hi, apex)
            if h1 > l1:
                # Phi(s) = 1 - (apex - s)/h = 1 - apex/h + s/h
                # integral = (h1 - l1) - (apex/h)(h1 - l1) + (h1^2 - l1^2)/(2h)
                integral += (h1 - l1) * (1 - apex / h) + (h1 ** 2 - l1 ** 2) / (2 * h)
            l2, h2 = max(lo, apex), hi
            if h2 > l2:
                # Phi(s) = 1 - (s - apex)/h = 1 + apex/h - s/h
                integral += (h2 - l2) * (1 + apex / h) - (h2 ** 2 - l2 ** 2) / (2 * h)
            v[m] = integral / h
        coeffs = scipy.linalg.solve(Z, v)
        # Current at L/2: I(L/2) = sum(coeffs[m] * Phi_m(L/2)).
        diff = np.abs(apex_arc - L / 2)
        phi_at_center = np.where(diff < h, 1.0 - diff / h, 0.0)
        I_at_center = phi_at_center @ coeffs
        return 1.0 / I_at_center

    raise ValueError(f"unknown source_model: {source_model!r}")


def loglog_slope(ns, errs):
    """Linear regression of log(err) vs log(N). Returns slope (negative if
    err decreases with N)."""
    ns = np.asarray(ns, dtype=float)
    errs = np.asarray(errs, dtype=float)
    mask = errs > 0
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(ns[mask]), np.log(errs[mask]), 1)[0])


def main():
    sim = AbstractPySim()
    cfg = dict(
        halfdriver=sim.halfdriver, wire_radius=sim.wire_radius, k=sim.k,
        omega=sim.omega, eps=sim.eps, mu=sim.mu, n_qp_reg=4,
    )

    even_ns = [10, 20, 40, 80, 160, 320, 640]
    odd_ns = [11, 21, 41, 81, 161, 321, 641]

    models = ["point_delta", "galerkin_delta", "finite_gap"]

    # Compute Z(N) for every (model, N).
    results = {}
    for m in models:
        even_zs = []
        odd_zs = []
        for N in even_ns:
            z = solve_with_source(N, source_model=m, **cfg)
            even_zs.append(z)
        for N in odd_ns:
            z = solve_with_source(N, source_model=m, **cfg)
            odd_zs.append(z)
        results[m] = (np.array(even_zs), np.array(odd_zs))

    # Use the highest-N even sample of finite_gap as the cross-method reference:
    # it's the most physically reasonable model (the others have known issues at
    # the source point). This is a sanity check that all methods agree in the
    # limit; the per-method convergence rates use each method's own asymptote.
    z_cross_ref = results["finite_gap"][0][-1]
    print(f"Cross-method reference (finite_gap, N={even_ns[-1]}):")
    print(f"  Z = {z_cross_ref.real:.5f} + j{z_cross_ref.imag:.5f}")
    print()

    for m in models:
        even_zs, odd_zs = results[m]
        z_self_ref = even_zs[-1]
        print(f"=== {m} ===")
        print(f"  asymptote (N={even_ns[-1]}, even): "
              f"{z_self_ref.real:.5f} + j{z_self_ref.imag:.5f}")
        diff_cross = z_self_ref - z_cross_ref
        print(f"  vs finite_gap asymptote: "
              f"{diff_cross.real:+.4f} + j{diff_cross.imag:+.4f}")

        print(f"  {'N':>5}  {'Z(N)':^24}  {'|err re|':>10}  {'|err im|':>10}")
        for N, z in zip(even_ns + odd_ns,
                        list(even_zs) + list(odd_zs)):
            err_r = abs(z.real - z_self_ref.real)
            err_i = abs(z.imag - z_self_ref.imag)
            tag = "even" if N % 2 == 0 else "odd "
            print(f"  {N:5d} ({tag}) {z.real:8.4f}+j{z.imag:8.4f}  "
                  f"{err_r:10.5f}  {err_i:10.5f}")

        # Drop the last point (it IS the reference -> err = 0).
        for label, ns, zs in (("even", even_ns[:-1], even_zs[:-1]),
                              ("odd", odd_ns[:-1], odd_zs[:-1])):
            err_r = np.abs(np.array([z.real for z in zs]) - z_self_ref.real)
            err_i = np.abs(np.array([z.imag for z in zs]) - z_self_ref.imag)
            sr = loglog_slope(ns, err_r)
            si = loglog_slope(ns, err_i)
            print(f"  log-log slope  {label}: real {sr:+.3f}, imag {si:+.3f}")
        print()


if __name__ == "__main__":
    main()
