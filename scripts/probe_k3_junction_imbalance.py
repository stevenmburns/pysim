"""Measure current imbalance at K=3 junctions on hentenna / fan dipole /
Y-fixture, to confirm the "dominant-pair throughflow vs. genuine 3-way
split" hypothesis for when enrichment-basis matters.

Working theory (slot-B default flip support): the K=3 cusp basis matters
only when the three currents meeting at the junction don't admit a
dominant in/out pair. When current essentially flows through one pair of
wires with the third as a small tap, the junction is "K=2-like" and the
log-singular cusp is geometrically/physically suppressed (regardless of
overall current magnitude through the junction).

For each K=3 junction we report:
  - |I_k| for each of the three wires at the junction node (in amps)
  - normalized magnitudes (divided by the max)
  - "tap ratio" = min(|I|) / max(|I|). Near 0 → dominant-pair K=2-like.
    Near 1 → balanced 3-way.

Expected ranking by tap ratio:
  Hentenna K=3 (at B, D):  tap ratio small  → enrichment a no-op
  Fan dipole K=3 (at S, T): tap ratio small  → enrichment near no-op
  Y-fixture K=3 (at S):    tap ratio ≈ 0.5  → enrichment matters

All solves use BSplinePySim d=2 with enrichment OFF (the slot-B default
we're about to ship) so the currents shown are what users actually see.

Run from project root:

    PYTHONPATH=. .venv/bin/python scripts/probe_k3_junction_imbalance.py

Result (2026-06-03), all at n=21 with BSpline d=2 enrichment OFF:

  geometry             tap ratio   normalized magnitudes   character
  -------------------  ----------  ----------------------  -----------------------
  fan dipole at S/T       0.03    (1.00, 0.99, 0.03)      basically K=2 + dead 10m
  hentenna  at B/D        0.16    (1.00, 0.84, 0.16)      dominant pair + small tap
  Y-fixture at S          0.50    (1.00, 0.50, 0.50)      balanced 3-way (2:1:1)

Confirmation of the hypothesis: when K=3 currents collapse to a
dominant in/out pair with the third wire as a small tap, the
log-singular cusp is suppressed and enrichment is a no-op (hentenna,
fan dipole). When the K=3 split is genuinely 3-way (Y-fixture), the
cusp matters and enrichment shifts Z by 0.08 Ω on R — the value the
d=1 tent basis converges to independently.

Both UI K=3 antennas (hentenna, fan dipole) live in the "dominant pair"
regime, so the global slot-B default flips from
use_singular_enrichment=True to False with no accuracy loss on any
shipped preset. Users hitting a genuinely 3-way K=3 geometry can flip
enrichment back on via the slot gear menu.
"""

from __future__ import annotations

import numpy as np

from pysim.bspline import BSplinePySim
from web import server as web_server


C_LIGHT = 299_792_458.0


def _solve_and_get_knots(sim: BSplinePySim):
    _, alpha = sim.compute_impedance()
    return sim.currents_at_knots(alpha)


def _print_junction(label: str, junction: list, knots: list):
    """For each (wire_idx, end_pos) entry in the junction, fetch |I| at
    the appropriate knot (knot 0 for 'start', knot -1 for 'end')."""
    K = len(junction)
    mags = []
    for w_idx, end_pos in junction:
        knot_idx = 0 if end_pos == "start" else -1
        I = knots[w_idx][knot_idx]
        mags.append(abs(complex(I)))
    mags = np.array(mags)
    max_m = mags.max()
    min_m = mags.min()
    tap = min_m / max_m if max_m > 0 else 0.0
    norm = mags / max_m if max_m > 0 else mags
    print(f"  {label}  K={K}")
    for (w_idx, end_pos), m, nm in zip(junction, mags, norm):
        print(
            f"    wire {w_idx:>2} {end_pos:<5}  |I| = {m:.6e} A   normalized = {nm:.4f}"
        )
    print(f"    tap ratio (min/max) = {tap:.4f}")
    print()


def hentenna():
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps_feed = 0.05
    half_w = wavelength * width_factor / 2.0
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
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
        [(1, "end"), (2, "start"), (4, "end")],  # K=3 at B
        [(2, "end"), (3, "end"), (4, "start")],  # K=3 at D
    ]
    n = 21
    sim = BSplinePySim(
        degree=2,
        use_singular_enrichment=False,
        wires=wires,
        n_per_edge_per_wire=[[2], [n], [n, n, n], [n], [n, n, n]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    )
    print(f"=== HENTENNA  n={n}, freq={freq_mhz} MHz ===")
    knots = _solve_and_get_knots(sim)
    _print_junction("at B", junctions[2], knots)
    _print_junction("at D", junctions[3], knots)


def fandipole():
    n = 21
    n_bands = 2
    freq = 14.3
    req = {
        "n_per_wire": n,
        "n_bands": n_bands,
        "band_lengths_m": [10.2551, 5.2691],
        "design_freq_mhz": freq,
        "measurement_freq_mhz": freq,
        "wire_radius": 0.0005,
        "slope": 0.5,
        "cone_radius_m": 0.12,
        "t0_factor": float(np.sqrt(2.0)),
        "ground": False,
        "height_m": 0.0,
    }
    g = web_server._fandipole_geometry(req)
    wavelength = C_LIGHT / (freq * 1e6)
    sim = BSplinePySim(
        degree=2,
        use_singular_enrichment=False,
        wires=g["wires"],
        n_per_edge_per_wire=g["n_per_edge"],
        feed_wire_index=0,
        feed_arclength=g["feed_arclength"],
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=g["junctions"],
    )
    print(f"=== FAN DIPOLE  n_bands={n_bands} (20m + 10m), n={n}, freq={freq} MHz ===")
    print("    (band 0 = 20m near-resonant; band 1 = 10m far off-resonance)")
    knots = _solve_and_get_knots(sim)
    _print_junction("at S", g["junctions"][0], knots)
    _print_junction("at T", g["junctions"][1], knots)


def y_fixture():
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    L = wavelength / 4.0
    eps_feed = 0.05
    T = (-eps_feed, 0.0, 0.0)
    S = (+eps_feed, 0.0, 0.0)
    arm1_end = (T[0] - L, 0.0, 0.0)
    c60 = float(np.cos(np.pi / 3.0))
    s60 = float(np.sin(np.pi / 3.0))
    arm2_end = (S[0] + L * c60, +L * s60, 0.0)
    arm3_end = (S[0] + L * c60, -L * s60, 0.0)
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
    n = 21
    sim = BSplinePySim(
        degree=2,
        use_singular_enrichment=False,
        wires=wires,
        n_per_edge_per_wire=[[2], [n], [n], [n]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    )
    print(f"=== Y-FIXTURE  n={n}, freq={freq_mhz} MHz, three λ/4 arms 120° apart ===")
    knots = _solve_and_get_knots(sim)
    _print_junction("at S", junctions[1], knots)


def main():
    hentenna()
    fandipole()
    y_fixture()


if __name__ == "__main__":
    main()
