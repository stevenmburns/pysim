"""Profile the triangular MoM solvers to see where time is spent.

Runs BentTriangularPySim (inverted V) and TriangularYagiPySim (driver+reflector)
at a few segment counts and prints:

  1. Wall-clock timing (median of repeats)
  2. cProfile output trimmed to the top consumers

Usage:
    python scripts/profile_triangular.py            # default settings
    python scripts/profile_triangular.py --n 80     # specific N
    python scripts/profile_triangular.py --kind v   # only V (or 'yagi')
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from pysim.triangular_bent import BentTriangularPySim
from pysim.triangular_yagi import TriangularYagiPySim


C_LIGHT = 299_792_458.0


def make_v(n: int, design_freq_mhz: float = 13.625, droop_deg: float = 30.0):
    wavelength = C_LIGHT / (design_freq_mhz * 1e6)
    arm_len = 0.962 * wavelength / 4.0
    alpha = np.deg2rad(droop_deg)
    cos_a, sin_a = float(np.cos(alpha)), float(np.sin(alpha))
    polyline = np.array([
        [-arm_len * cos_a, 0.0, -arm_len * sin_a],
        [0.0, 0.0, 0.0],
        [arm_len * cos_a, 0.0, -arm_len * sin_a],
    ])
    sim = BentTriangularPySim(wavelength=wavelength, halfdriver_factor=0.962, nsegs=n)
    sim.polyline = polyline
    sim.n_per_edge = [n, n]
    return sim


def make_yagi(n: int, design_freq_mhz: float = 13.625):
    wavelength = C_LIGHT / (design_freq_mhz * 1e6)
    sim = TriangularYagiPySim(
        wavelength=wavelength,
        halfdriver_factor=0.962,
        nsegs=n,
        reflector_factor=1.05,
        spacing_factor=0.6,
    )
    return sim


def time_runs(sim_factory, n: int, repeats: int = 7) -> list[float]:
    # Warm up once (touches numpy LU caches, etc.).
    sim = sim_factory(n)
    sim.compute_impedance()
    times = []
    for _ in range(repeats):
        sim = sim_factory(n)
        t0 = time.perf_counter()
        sim.compute_impedance()
        times.append((time.perf_counter() - t0) * 1e3)
    return sorted(times)


def profile_one(sim_factory, n: int, label: str, top: int = 18) -> None:
    sim = sim_factory(n)
    sim.compute_impedance()  # warm
    sim = sim_factory(n)
    pr = cProfile.Profile()
    pr.enable()
    sim.compute_impedance()
    pr.disable()

    times = time_runs(sim_factory, n)
    median = times[len(times) // 2]
    print(f"\n=== {label} N={n}  median={median:.1f} ms  range=[{times[0]:.1f}, {times[-1]:.1f}] ===")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(top)
    # Trim the header noise; keep only the table.
    out = s.getvalue()
    # Skip the first few lines (header + blank); keep from "ncalls" onward.
    keep = out.split("ncalls", 1)
    if len(keep) == 2:
        print("ncalls" + keep[1])
    else:
        print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="single N to profile")
    ap.add_argument("--kind", choices=["v", "yagi", "both"], default="both")
    args = ap.parse_args()

    ns = [args.n] if args.n is not None else [30, 60, 80, 160]

    if args.kind in ("v", "both"):
        for n in ns:
            profile_one(make_v, n, "BentTriangularPySim (V)")
    if args.kind in ("yagi", "both"):
        for n in ns:
            profile_one(make_yagi, n, "TriangularYagiPySim (Yagi)")


if __name__ == "__main__":
    main()
