"""Fan-dipole BSpline d=2 enrichment probe — settles whether the UI slot-B
default should flip enrichment OFF globally or keep ON for fan-dipole-like
geometries (NEXT_STEPS item 15(a) follow-up).

Hentenna data (PR #52 / compare_hentenna_solvers.py): enrichment-on costs
0.26 Ω X accuracy at n=21 because the hentenna's K=3 junctions sit in a
near-resonant standing-wave config where the cusp current is tiny; extra
enrichment DOFs absorb discretization-error mass at small n and only
converge to ~zero at large n.

Y-fixture data (probe_y_fixture_enrichment.py): enrichment-on shifts d=2 R
by 0.08 Ω toward the d=1 tent asymptote (the physically correct value) and
that shift persists to n=241. The C¹ d=2 basis can't represent the K=3
junction cusp without enrichment; the tent basis can.

Open question for the global default: where on this spectrum does the fan
dipole live? At n_bands=2 (the UI default), there are K=3 junctions at S
and T (one feed wire + 2 band wires per side). Whether the current
flowing through those junctions is "hentenna-like" (small) or "Y-fixture-
like" (large) determines whether enrichment is the right call.

Probe at the UI's default 2-band fan dipole at 14.3 MHz, sweeping n ∈
{15, 21, 41, 81} for BSpline d=2 with and without enrichment. Triangular
included as a reference (it's the d=1 tent basis equivalent — same role
as in the Y-fixture probe).

Run from project root:

    PYTHONPATH=. .venv/bin/python scripts/probe_fandipole_enrichment.py
    PYTHONPATH=. .venv/bin/python scripts/probe_fandipole_enrichment.py \\
        --n-bands 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pysim.triangular import TriangularPySim
from pysim.bspline import BSplinePySim
from web import server as web_server


def _make_req(n_per_wire: int, n_bands: int, design_freq_mhz: float = 14.3) -> dict:
    """UI-default fan dipole request (matches scripts/compare_fandipole_solvers.py)."""
    band_lengths_full = [10.2551, 5.2691, 4.0, 3.0, 2.5]
    return {
        "n_per_wire": n_per_wire,
        "n_bands": n_bands,
        "band_lengths_m": band_lengths_full[:n_bands],
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": design_freq_mhz,
        "wire_radius": 0.0005,
        "slope": 0.5,
        "cone_radius_m": 0.12,
        "t0_factor": float(np.sqrt(2.0)),
        "ground": False,
        "height_m": 0.0,
    }


def _solve(n: int, n_bands: int, freq: float, model: str) -> tuple[complex, float]:
    """model ∈ {'tri', 'b2', 'b2e'}."""
    req = _make_req(n, n_bands, freq)
    g = web_server._fandipole_geometry(req)
    wavelength = 299_792_458.0 / (freq * 1e6)
    common = dict(
        wires=g["wires"],
        n_per_edge_per_wire=g["n_per_edge"],
        feed_wire_index=0,
        feed_arclength=g["feed_arclength"],
        wavelength=wavelength,
        wire_radius=req["wire_radius"],
        nsegs=n,
        junctions=g["junctions"],
    )
    if model == "tri":
        sim = TriangularPySim(**common)
    elif model == "b2":
        sim = BSplinePySim(degree=2, use_singular_enrichment=False, **common)
    elif model == "b2e":
        sim = BSplinePySim(degree=2, use_singular_enrichment=True, **common)
    else:
        raise ValueError(model)
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
    ap.add_argument("--n-list", type=str, default="15,21,41,81")
    ap.add_argument("--n-bands", type=int, default=2)
    ap.add_argument("--freq", type=float, default=14.3)
    args = ap.parse_args()
    ns = [int(x) for x in args.n_list.split(",") if x.strip()]

    print(
        f"Fan dipole, n_bands={args.n_bands}, freq={args.freq:.2f} MHz, "
        f"K={1 + args.n_bands} junctions at S and T\n"
    )

    cols = [("tri", "tri"), ("b2", "b2"), ("b2e", "b2e")]
    results: dict[str, dict[int, tuple[complex, float]]] = {c[0]: {} for c in cols}

    w = 24
    print("  n  | " + " | ".join(f"{name:^{w}}" for name, _ in cols))
    print(" " * 5 + "+-" + "-+-".join("-" * w for _ in cols))
    for n in ns:
        cells = []
        for name, model in cols:
            z, ms = _solve(n, args.n_bands, args.freq, model)
            results[name][n] = (z, ms)
            cells.append(f"{z.real:+8.3f} {z.imag:+8.3f}j ({ms:5.0f}ms)")
        print(f" {n:>3} | " + " | ".join(f"{c:^{w}}" for c in cells))

    print()
    print("Per-n agreement (Ω):")
    print("   n   |Δ tri↔b2 |  |Δ tri↔b2e|  |Δ b2↔b2e|")
    for n in ns:
        tri = results["tri"][n][0]
        b2 = results["b2"][n][0]
        b2e = results["b2e"][n][0]
        print(
            f"  {n:>3}   "
            f"R:{abs(tri.real - b2.real):6.3f} X:{abs(tri.imag - b2.imag):6.3f}    "
            f"R:{abs(tri.real - b2e.real):6.3f} X:{abs(tri.imag - b2e.imag):6.3f}    "
            f"R:{abs(b2.real - b2e.real):6.3f} X:{abs(b2.imag - b2e.imag):6.3f}"
        )

    print()
    print("X-convergence rate fit p in X(N) = X_inf + C/N^p:")
    if len(ns) >= 3:
        for name, _ in cols:
            xs = [results[name][n][0].imag for n in ns]
            rs = [results[name][n][0].real for n in ns]
            print(f"  {name}:  last n={ns[-1]}: {rs[-1]:+7.3f} + j{xs[-1]:+7.3f}")
            for n1, n2, n3, p, x_inf in _fit_x_rate(ns, xs):
                tag = f"({n1},{n2},{n3})"
                if np.isnan(p):
                    print(f"    {tag:>13s}  p=nan")
                else:
                    print(f"    {tag:>13s}  p={p:5.2f}   X_inf≈{x_inf:+8.3f}")


if __name__ == "__main__":
    main()
