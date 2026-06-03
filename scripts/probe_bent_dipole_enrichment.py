"""Probe whether singular-enrichment helps at K=2 bend junctions
(NEXT_STEPS item 15(c)) and whether the polyline-kink representation
agrees with the K=2-junction representation under BSpline d=2
(NEXT_STEPS item 15(d)).

The fixture is a symmetric 90° bent dipole at 28.47 MHz, total per-arm
length L per branch (L/2 horizontal, L/2 vertical), feed gap T→S of
width 2·eps_feed at the centre.

Two representations of the same physical antenna:

  "polyline" mode — three wires, bends are within-wire kinks:
     0: feed gap T → S                (2 segs)
     1: S → bend_pos → tip_pos        (3-anchor polyline, 2 edges)
     2: T → bend_neg → tip_neg        (3-anchor polyline, 2 edges)
     Junctions: K=2 at S (wires 0,1), K=2 at T (wires 0,2).
     Bends are *not* junctions — they are polyline kinks; basis
     continuity across the bend is automatic.

  "k2" mode — five wires, bends are explicit K=2 junctions:
     0: feed gap T → S                (2 segs)
     1: S → bend_pos                  (straight, 1 edge)
     2: bend_pos → tip_pos            (straight, 1 edge)
     3: T → bend_neg                  (straight, 1 edge)
     4: bend_neg → tip_neg            (straight, 1 edge)
     Junctions: K=2 at S, T, bend_pos, bend_neg.
     Continuity at the bend is enforced by KCL Lagrange row.

15(d) prediction: under TriangularPySim the two modes are bit-exact
equivalent (test_triangular_k2_junction_equivalent_to_single_polyline
pins this). The same equivalence should hold for BSplinePySim d=1
(degree-1 B-spline = tent). For d=2 it does NOT necessarily hold: the
C¹ d=2 polyline-kink basis is *continuous in slope* across the bend
(its quadratic shape spans the kink), while the K=2-junction
representation only enforces C⁰ continuity (current value matches,
slope can jump). The current-derivative physics at a sharp wire bend
has a slope discontinuity (the charge density picks up a corner
contribution), so the K=2 representation should be MORE physical and
the polyline-kink representation MORE BIASED for d=2.

15(c) prediction: forcing enrichment_min_k=2 in the "k2" mode adds the
log-singular shape at each K=2 bend. The Y-fixture (probe 15(b))
showed d=2-enrich adds 0.08 Ω of "cusp correction" to d=2-no-enrich at
a K=3 junction; here we test whether the same correction shows up at
K=2 bends.

Probes per n:
  pl  d=1, d=2          — polyline mode, no enrichment
  k2  d=1, d=2          — K=2-junction mode, no enrichment
  k2  d=2 enr_min_k=2   — K=2-junction mode, enrichment at every K=2
  k2  d=2 enr_min_k=3   — K=2-junction mode, enrichment OFF (no K=3 here)

The k2 d=2 enr_min_k=3 column is a control: it's the same fixture as
k2 d=2 no-enrich but explicitly via the enrichment-enabled code path,
catching any "enrichment infrastructure changes Z even when zero
enrichment bases get added" bugs.

Run from the project root:

    PYTHONPATH=. .venv/bin/python scripts/probe_bent_dipole_enrichment.py

Results (2026-06-03), L = λ/2 per arm (full-wave dipole, anti-resonant
so |Z| is large; the *relative* per-mode disagreement is what we care
about). nfeed=2, wire_radius=0.5 mm, 28.47 MHz free space:

       n   |   pl d=1            |   pl d=2            |   k2 d=1            |   k2 d=2            |   k2 d=2 enr=2      |   k2 d=2 enr=3
       15  | 3428.41 − j4391.85  | 3967.55 − j4494.82  | 3428.41 − j4391.85  | 3967.55 − j4494.82  | 3370.37 − j4383.00  | 3967.55 − j4494.82
       21  | 3413.00 − j4388.78  | 3978.00 − j4496.29  | 3413.00 − j4388.78  | 3978.00 − j4496.29  | 3461.18 − j4403.16  | 3978.00 − j4496.29
       41  | 3400.82 − j4386.53  | 3989.08 − j4497.96  | 3400.82 − j4386.53  | 3989.08 − j4497.96  | 3513.49 − j4415.00  | 3989.08 − j4497.96
       81  | 3395.22 − j4385.57  | 3990.76 − j4498.44  | 3395.22 − j4385.57  | 3990.76 − j4498.44  | 3521.40 − j4417.02  | 3990.76 − j4498.44

Pairwise findings:

  15(d) — pl vs k2 at d=1 and d=2: **bit-exact equivalent** at every n
    (|ΔR| < 0.0001 Ω, |ΔX| < 0.0001 Ω). The polyline-kink and K=2-
    junction representations produce mathematically identical Z under
    BSpline d=1 AND d=2 — the hypothesis that the C¹ d=2 polyline basis
    would "smear" the kink was wrong; BSpline rebuilds the basis with
    a slope break at the polyline anchor too. The architectural
    unification question (model bends as K=2 junctions) is therefore
    PURE CODE CLARITY — not a correctness question — for both bases.

  control (k2 d=2 vs k2 d=2 enr_min_k=3 with no K=3 junctions): bit-exact
    at every n. The enrichment infrastructure correctly no-ops when no
    junctions meet the K threshold.

  15(c) — enrichment_min_k=2 vs no-enrich at d=2: SHIFTS Z BY ~500 Ω ON R
    AND ~80 Ω ON X at every n. The two answers do not converge to the
    same place; the gap is roughly constant in N. The working hypothesis
    in NEXT_STEPS was that enrichment "shouldn't help" at K=2 bends;
    the data is stronger — it ACTIVELY HARMS. The log-singular shape
    Φ_sing(u) ~ u·log(u) is wrong for K=2 bends: a K=2 bend has KCL
    I_in = −I_out (continuity through the bend, no current "splitting")
    so the charge density at the bend is NOT log-singular. Adding a
    basis with the wrong singularity structure over-fits the solution
    space and pulls Z to a different (incorrect) limit.

  15(c) K=1 sub-question (free-end enrichment): the BSpline constructor
    rejects K=1 junction entries (`len(jw) < 2: raise ValueError`),
    so enrichment_min_k=1 has no effect on free wire ends without a
    code change. The doc's working hypothesis was confused — free ends
    aren't junctions in the constructor's vocabulary. PR #48 smoothed
    source remains the principled fix for the delta-gap source-
    singularity rate at K=1-ish ends.

Implication for the BSpline default: **keep enrichment_min_k=3**. The
default is correct; lowering it to K=2 is actively harmful. This
strengthens the case for the slot-B UI default tracked in 15(a):
even when enrichment is on, only K=3+ junctions get the basis, and
those are the only ones where it represents physics.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pysim.bspline import BSplinePySim


C_LIGHT = 299_792_458.0


def _geometry(freq_mhz: float, eps_feed: float = 0.05):
    """Return (anchors_dict, wavelength) for the bent dipole.

    Per-arm length L = λ/2 so each half-arm is L/2 = λ/4 (resonant-ish).
    """
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    L = wavelength / 2.0
    half = L / 2.0
    T = (0.0, -eps_feed, 0.0)
    S = (0.0, +eps_feed, 0.0)
    bend_pos = (0.0, eps_feed + half, 0.0)
    tip_pos = (0.0, eps_feed + half, +half)
    bend_neg = (0.0, -eps_feed - half, 0.0)
    tip_neg = (0.0, -eps_feed - half, -half)
    return {
        "T": T,
        "S": S,
        "bend_pos": bend_pos,
        "tip_pos": tip_pos,
        "bend_neg": bend_neg,
        "tip_neg": tip_neg,
    }, wavelength


def _wires_polyline(g: dict):
    wires = [
        np.array([g["T"], g["S"]], dtype=float),
        np.array([g["S"], g["bend_pos"], g["tip_pos"]], dtype=float),
        np.array([g["T"], g["bend_neg"], g["tip_neg"]], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],  # K=2 at S
        [(0, "start"), (2, "start")],  # K=2 at T
    ]
    return wires, junctions


def _wires_k2(g: dict):
    wires = [
        np.array([g["T"], g["S"]], dtype=float),
        np.array([g["S"], g["bend_pos"]], dtype=float),
        np.array([g["bend_pos"], g["tip_pos"]], dtype=float),
        np.array([g["T"], g["bend_neg"]], dtype=float),
        np.array([g["bend_neg"], g["tip_neg"]], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],  # K=2 at S
        [(0, "start"), (3, "start")],  # K=2 at T
        [(1, "end"), (2, "start")],  # K=2 at bend_pos
        [(3, "end"), (4, "start")],  # K=2 at bend_neg
    ]
    return wires, junctions


def solve(
    n: int,
    *,
    mode: str,
    degree: int,
    enrichment_min_k: int | None,
    freq_mhz: float = 28.47,
    eps_feed: float = 0.05,
) -> tuple[complex, float]:
    g, wavelength = _geometry(freq_mhz, eps_feed=eps_feed)
    if mode == "pl":
        wires, junctions = _wires_polyline(g)
        npe = [[2], [n, n], [n, n]]
    elif mode == "k2":
        wires, junctions = _wires_k2(g)
        npe = [[2], [n], [n], [n], [n]]
    else:
        raise ValueError(mode)
    kwargs = dict(
        degree=degree,
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    )
    if enrichment_min_k is not None:
        kwargs["use_singular_enrichment"] = True
        kwargs["enrichment_min_k"] = enrichment_min_k
    else:
        kwargs["use_singular_enrichment"] = False
    sim = BSplinePySim(**kwargs)
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

    g, wavelength = _geometry(args.freq)
    half = wavelength / 4.0
    print(
        f"90° bent dipole, half-arm = λ/4 ≈ {half:.3f} m   "
        f"({args.freq:.2f} MHz, λ={wavelength:.4f} m)\n"
    )

    cols = [
        ("pl d=1", "pl", 1, None),
        ("pl d=2", "pl", 2, None),
        ("k2 d=1", "k2", 1, None),
        ("k2 d=2", "k2", 2, None),
        ("k2 d=2 enr=2", "k2", 2, 2),
        ("k2 d=2 enr=3", "k2", 2, 3),
    ]
    results: dict[str, dict[int, tuple[complex, float]]] = {c[0]: {} for c in cols}

    w = 19
    print("  n  | " + " | ".join(f"{name:^{w}}" for name, *_ in cols))
    print(" " * 5 + "+-" + "-+-".join("-" * w for _ in cols))
    for n in ns:
        cells = []
        for name, mode, d, enr in cols:
            z, ms = solve(
                n, mode=mode, degree=d, enrichment_min_k=enr, freq_mhz=args.freq
            )
            results[name][n] = (z, ms)
            cells.append(f"{z.real:+7.2f}{z.imag:+7.2f}j ({ms:4.0f}ms)")
        print(f" {n:>3} | " + " | ".join(f"{c:^{w}}" for c in cells))

    print()
    print("Pairwise checks (smaller is better agreement; values in Ω):")
    print()
    # pl d=1 vs k2 d=1 — should be bit-exact (tent test_triangular_k2_junction_*)
    # pl d=2 vs k2 d=2 — open question for 15(d)
    # k2 d=2 vs k2 d=2 enr=3 — should be bit-exact (enrichment infra adds nothing when no junctions qualify)
    # k2 d=2 vs k2 d=2 enr=2 — 15(c): does enr at K=2 bends shift Z?
    pairs = [
        ("pl d=1", "k2 d=1", "15(d) at d=1 (expected: bit-exact)"),
        ("pl d=2", "k2 d=2", "15(d) at d=2 (open: polyline-kink vs K=2-junction)"),
        (
            "k2 d=2",
            "k2 d=2 enr=3",
            "control: enr-infra no-op when no junctions qualify",
        ),
        ("k2 d=2", "k2 d=2 enr=2", "15(c): enrichment at K=2 bends shift"),
    ]
    for a, b, label in pairs:
        print(f"  {label}")
        print("    n     ΔR (Ω)     ΔX (Ω)")
        for n in ns:
            za = results[a][n][0]
            zb = results[b][n][0]
            print(f"    {n:>3}   {zb.real - za.real:+8.4f}   {zb.imag - za.imag:+8.4f}")
        print()

    print("X-convergence rate fit p in X(N) = X_inf + C/N^p:")
    for name, *_ in cols:
        xs = [results[name][n][0].imag for n in ns]
        rs = [results[name][n][0].real for n in ns]
        print(f"  {name}:  last n={ns[-1]}: {rs[-1]:+7.2f} + j{xs[-1]:+7.2f}")
        for n1, n2, n3, p, x_inf in _fit_x_rate(ns, xs):
            tag = f"({n1},{n2},{n3})"
            if np.isnan(p):
                print(f"    {tag:>15s}  p=nan")
            else:
                print(f"    {tag:>15s}  p={p:5.2f}   X_inf≈{x_inf:+8.3f}")


if __name__ == "__main__":
    main()
