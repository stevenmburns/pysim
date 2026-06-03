"""Probe whether the singular-enrichment basis helps degree-1 BSpline on a
single-K=3-junction fixture (NEXT_STEPS item 15(b)).

The enrichment basis Φ_sing(u) = (u/h)·log(u/h) has no dependence on the
polynomial degree d — its shape is independent of d, and Z_pe / Z_ep just
project the enrichment against whatever polynomial bases exist on the
adjacent segment. So the question is degree-agnostic: does the log-singular
shape accelerate convergence at K≥3 junctions regardless of d?

Geometry — a "Y" with a 2-segment feed wire holding the delta-gap source:

      arm3 (S → +60°)
            \\
             \\
    arm1 ----T-----S---- arm2 (S → −60°, on +x side)
             \\
              \\
              (single K=3 junction at S)
              K=2 junction at T (wires 0,1)

Five wires:
  0: feed gap T → S         (2 segs, source on midpoint knot)
  1: arm at T → −x          (free end at far tip)
  2: arm at S → cos+60°,sin+60°  (free end at far tip)
  3: arm at S → cos−60°,sin−60°  (free end at far tip)

Junctions:
  at T: K=2 (wires 0, 1)
  at S: K=3 (wires 0, 2, 3)   ← the probe junction

Each non-feed arm has length L = λ/4 at 28.47 MHz (≈ 2.63 m); the three
free ends sit on a circle. The K=3 junction at S is the only place where
the classical charge-density log singularity can show up; the K=2 at T is
a benign continuity node.

Probes:
  d=1 no enrich, d=1 enrich        — the load-bearing 15(b) question
  d=2 no enrich, d=2 enrich        — reference

If d=1+enrichment converges meaningfully faster than d=1 no-enrich, the
enrichment basis applies to the TriangularPySim tent basis in principle
too (would need a port). If not, the enrichment value is d-specific (the
log singularity is only useful when paired with a polynomial basis of
degree ≥ 2, perhaps because it interacts with the segment-quadratic
shape in a way the tent can't represent).

Run from the project root:

    PYTHONPATH=. .venv/bin/python scripts/probe_y_fixture_enrichment.py
    PYTHONPATH=. .venv/bin/python scripts/probe_y_fixture_enrichment.py \\
        --n-list 15,21,41,81,161,241

Result (2026-06-03), λ/4 arms at 28.47 MHz, nfeed=2, wire_radius=0.5 mm:

       n  |    d=1         |    d=1 enr      |    d=2          |    d=2 enr
      15  | 50.22 + j51.82 | 50.24 + j51.85  | 50.29 + j52.56  | 50.38 + j52.56
      21  | 50.29 + j52.22 | 50.30 + j52.23  | 50.30 + j52.71  | 50.38 + j52.72
      41  | 50.36 + j52.65 | 50.37 + j52.65  | 50.32 + j52.92  | 50.39 + j52.92
      81  | 50.40 + j52.87 | 50.40 + j52.87  | 50.33 + j53.04  | 50.40 + j53.04
     161  | 50.41 + j52.99 | 50.42 + j52.99  | 50.34 + j53.11  | 50.42 + j53.11
     241  | 50.42 + j53.04 | 50.43 + j53.05  | 50.34 + j53.14  | 50.42 + j53.15

Findings:
  - **d=1 enrichment is a no-op** (within ~0.01 Ω of d=1 no-enrich at every
    n; identical fitted rate p≈1.0 on X). The tent basis already has a
    slope discontinuity at every knot — it captures the K=3-junction
    current cusp "for free", so the log-singular enrichment basis adds
    nothing the polynomial system isn't already representing. No port to
    TriangularPySim is warranted by this fixture.

  - **d=2 enrichment shifts R by ~0.08 Ω** in a way that persists to
    n=241 (does not shrink). d=2 alone converges to a different R
    asymptote (~50.34) than d=1 alone (~50.42). d=2 + enrichment
    matches d=1 alone. The interpretation: the C¹ d=2 basis can't
    represent a slope discontinuity at the K=3 junction (its derivative
    is continuous across knots by construction); enrichment supplies
    the missing cusp shape.

  - Why this doesn't show up on the hentenna: there, b2 (no enrich) and
    b2e converge to the SAME Z within 0.001 Ω at n=161. The hentenna's
    K=3 junctions sit in a near-resonant standing-wave configuration
    where the current flowing through the junction is small, so the
    cusp contribution is proportionally tiny. The Y fixture has a feed
    driving directly into the K=3 junction with no resonance damping —
    the cusp is loud and the C¹ d=2 basis's failure to represent it is
    measurable.

Implication for the UI default (slot B): the post-PR-#51 hentenna data
already argued for flipping enrichment OFF on hentenna (b2 alone is
converged at n=21). The Y-fixture data refines that: enrichment is a
no-op at d=1 and a 0.08-Ω-class correction at d=2 on geometries where
the K=3 junction carries non-trivial current. For the hentenna
specifically it's safe to flip off; for a generic K=3-junction antenna
b2e remains the more accurate default at the cost of one extra basis
per junction-wire endpoint.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pysim.bspline import BSplinePySim


C_LIGHT = 299_792_458.0


def _y_fixture(freq_mhz: float, eps_feed: float = 0.05):
    """Build the K=3 Y fixture. Returns (wires, junctions, wavelength)."""
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    L = wavelength / 4.0  # each arm: λ/4 ≈ 2.63 m at 28.47 MHz

    T = (-eps_feed, 0.0, 0.0)
    S = (+eps_feed, 0.0, 0.0)
    arm1_end = (T[0] - L, 0.0, 0.0)  # arm 1 along −x
    c60 = float(np.cos(np.pi / 3.0))
    s60 = float(np.sin(np.pi / 3.0))
    arm2_end = (S[0] + L * c60, +L * s60, 0.0)  # arm 2 at +60°
    arm3_end = (S[0] + L * c60, -L * s60, 0.0)  # arm 3 at −60°

    wires = [
        np.array([T, S], dtype=float),
        np.array([T, arm1_end], dtype=float),
        np.array([S, arm2_end], dtype=float),
        np.array([S, arm3_end], dtype=float),
    ]
    junctions = [
        [(0, "start"), (1, "start")],  # K=2 at T
        [(0, "end"), (2, "start"), (3, "start")],  # K=3 at S
    ]
    return wires, junctions, wavelength


def solve(
    n: int,
    *,
    degree: int,
    use_enrichment: bool,
    freq_mhz: float = 28.47,
    eps_feed: float = 0.05,
) -> tuple[complex, float]:
    wires, junctions, wavelength = _y_fixture(freq_mhz, eps_feed=eps_feed)
    nfeed = 2  # EVEN — interior knot at the geometric centre of the feed gap
    npe = [[nfeed], [n], [n], [n]]
    sim = BSplinePySim(
        degree=degree,
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
        use_singular_enrichment=use_enrichment,
        enrichment_min_k=3,  # only S qualifies
    )
    t0 = time.perf_counter()
    z, _ = sim.compute_impedance()
    return complex(z), (time.perf_counter() - t0) * 1e3


def _fit_x_rate(ns: list[int], xs: list[float]):
    out = []
    for i in range(len(ns) - 2):
        n1, n2, n3 = ns[i], ns[i + 1], ns[i + 2]
        d12 = xs[i] - xs[i + 1]
        d23 = xs[i + 1] - xs[i + 2]
        if d12 * d23 <= 0 or abs(d23) < 1e-9:
            out.append((n1, n2, n3, float("nan"), float("nan")))
            continue
        p = float(np.log(abs(d12 / d23)) / np.log(n2 / n1))
        r = n3 / n2
        x_inf = xs[i + 2] - d23 / (r**p - 1.0)
        out.append((n1, n2, n3, p, float(x_inf)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-list", type=str, default="15,21,41,81,161")
    ap.add_argument("--freq", type=float, default=28.47)
    args = ap.parse_args()
    ns = [int(x) for x in args.n_list.split(",") if x.strip()]

    wavelength = C_LIGHT / (args.freq * 1e6)
    L = wavelength / 4.0
    print(
        f"Y-fixture, K=3 junction at S, free ends at distance L=λ/4≈{L:.3f} m "
        f"({args.freq:.2f} MHz, λ={wavelength:.4f} m)\n"
    )

    cols = [
        ("d=1", 1, False),
        ("d=1 enr", 1, True),
        ("d=2", 2, False),
        ("d=2 enr", 2, True),
    ]
    results: dict[str, dict[int, tuple[complex, float]]] = {c[0]: {} for c in cols}

    w = 22
    print("  n  | " + " | ".join(f"{name:^{w}}" for name, _, _ in cols))
    print(" " * 5 + "+-" + "-+-".join("-" * w for _ in cols))
    for n in ns:
        cells = []
        for name, d, enr in cols:
            z, ms = solve(n, degree=d, use_enrichment=enr, freq_mhz=args.freq)
            results[name][n] = (z, ms)
            cells.append(f"{z.real:+8.3f} {z.imag:+8.3f}j ({ms:5.0f}ms)")
        print(f" {n:>3} | " + " | ".join(f"{c:^{w}}" for c in cells))

    print()
    print("X-convergence rate fit p in X(N) = X_inf + C/N^p:")
    for name, _, _ in cols:
        xs = [results[name][n][0].imag for n in ns]
        rs = [results[name][n][0].real for n in ns]
        print(f"  {name}:  last n={ns[-1]}: {rs[-1]:+8.3f} + j{xs[-1]:+8.3f}")
        for n1, n2, n3, p, x_inf in _fit_x_rate(ns, xs):
            tag = f"({n1},{n2},{n3})"
            if np.isnan(p):
                print(f"    {tag:>13s}  p=nan")
            else:
                print(f"    {tag:>13s}  p={p:5.2f}   X_inf≈{x_inf:+8.4f}")


if __name__ == "__main__":
    main()
