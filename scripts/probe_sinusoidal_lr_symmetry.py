"""Quantify L-R asymmetry of SinusoidalPySim's per-knot currents on the
hentenna. The hentenna is mirror-symmetric about y=0, so |I| at each
knot on the upper-right vertical should equal |I| at the mirror knot
on the upper-left vertical, and similarly for the cross-bar halves.

Visual symptom: a kink/spike on the left half of the cross-bar that
the right half doesn't show (bug2.png).

Hypothesis: the σ (arc-flip) handling in `currents_at_knots` doesn't
match the σ handling in `_assemble_Z`. _assemble_Z applies σ to the
A and C coefficients only (the sin term picks up its σ inside
sin(σ·k·s) already), while currents_at_knots multiplies the whole
(A + B·sin + C·cos) expression by σ — adding a spurious 2·B·sin(k·s)
error when σ=−1.

Run:
    PYTHONPATH=. .venv/bin/python scripts/probe_sinusoidal_lr_symmetry.py
"""

from __future__ import annotations

import numpy as np

from pysim.sinusoidal import SinusoidalPySim


C_LIGHT = 299_792_458.0
FREQ_MHZ = 28.47
WIDTH_FACTOR = 0.1378
TOP_HEIGHT_FACTOR = 0.5081
MID_HEIGHT_FACTOR = 0.1094
EPS_FEED = 0.05
WIRE_RADIUS = 0.0005


def build_sim(n: int) -> SinusoidalPySim:
    wavelength = C_LIGHT / (FREQ_MHZ * 1e6)
    half_w = wavelength * WIDTH_FACTOR / 2.0
    z_mid = wavelength * (MID_HEIGHT_FACTOR - TOP_HEIGHT_FACTOR)
    z_bot = -wavelength * TOP_HEIGHT_FACTOR
    A = (0.0, half_w, 0.0)
    B = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, EPS_FEED, z_mid)
    C = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E = (0.0, -half_w, z_bot)
    T = (0.0, -EPS_FEED, z_mid)
    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B], dtype=float),
        np.array([B, A, C, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E, F, B], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    nfeed = 3
    return SinusoidalPySim(
        wires=wires,
        n_per_edge_per_wire=[[nfeed], [n], [n, n, n], [n], [n, n, n]],
        feed_wire_index=0,
        feed_arclength=EPS_FEED,
        wavelength=wavelength,
        wire_radius=WIRE_RADIUS,
        nsegs=n,
        junctions=junctions,
    )


def main():
    n = 21
    sim = build_sim(n)
    z, alpha = sim.compute_impedance()
    print(f"hentenna n={n}: Z = {z.real:+.4f} {z.imag:+.4f}j Ω\n")
    knots = sim.currents_at_knots(alpha)

    # Wire 1 (S→B): right half of cross-bar, n+1 knots.
    # Wire 3 (T→D): left half of cross-bar, n+1 knots.
    # Mirror symmetry: |I(s along wire 1)| = |I(s along wire 3)| at the
    # same arc distance from the feed-side end of the cross-bar.
    w1 = knots[1]
    w3 = knots[3]
    print(f"Cross-bar halves ({len(w1)} knots each, arc from feed-side junction):")
    print("  i  |I_w1 (right)|   |I_w3 (left)|   ratio    Δ|I|")
    for i in range(len(w1)):
        m1 = abs(w1[i])
        m3 = abs(w3[i])
        ratio = m1 / m3 if m3 > 1e-15 else float("inf")
        print(f"  {i:2d}    {m1:.6e}   {m3:.6e}   {ratio:7.4f}   {m1 - m3:+.3e}")

    # Wire 2 (B→A→C→D): upper rectangle. Knot 0 is B, knot n_w2 is D.
    # Mirror about y=0 reverses the path → mirror(knot k from B) = knot
    # (n_w2 - k) from D, i.e. the path is its own mirror traversed
    # backwards. Compare |I[k]| with |I[n_w2 - k]|.
    w2 = knots[2]
    print(f"\nUpper rectangle (wire 2, {len(w2)} knots, B→A→C→D vs reversed):")
    print("  i   k_mirror   |I[i]|       |I[k_mirror]|   Δ|I|")
    halfn = len(w2) // 2
    max_dev = 0.0
    for i in range(halfn + 1):
        km = len(w2) - 1 - i
        m1 = abs(w2[i])
        m2 = abs(w2[km])
        dev = abs(m1 - m2)
        max_dev = max(max_dev, dev)
        if i < 5 or i >= halfn - 2:
            print(f"  {i:2d}   {km:2d}        {m1:.6e}   {m2:.6e}   {dev:+.3e}")
    print(f"  max |Δ|I|| on wire 2: {max_dev:.4e}")

    # Wire 4 (D→E→F→B): lower rectangle. Same self-mirror story.
    w4 = knots[4]
    print(f"\nLower rectangle (wire 4, {len(w4)} knots, D→E→F→B vs reversed):")
    max_dev = 0.0
    for i in range(len(w4) // 2 + 1):
        km = len(w4) - 1 - i
        m1 = abs(w4[i])
        m2 = abs(w4[km])
        max_dev = max(max_dev, abs(m1 - m2))
    print(f"  max |Δ|I|| on wire 4: {max_dev:.4e}")


if __name__ == "__main__":
    main()
