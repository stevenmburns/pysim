"""Animation factor-cache demo for the array-block solver.

Shows the two reuse levers `ArrayBlockPySim` exposes for animating an array:

  * Phase / excitation sweep — geometry (and therefore Z and its factored
    preconditioner) is fixed; only the feed voltages change. The cached
    operator is reused wholesale, so frames after the first are just multi-RHS
    back-substitutions: near-instant.

  * Spacing sweep — element positions change but the elements themselves are
    identical, so the dense self-block assembly is reused across frames and
    only the cheap low-rank coupling blocks recompute.

Reads the wall-clock and the cache hit/build counters per frame so the reuse
is visible. The pure-Python ACA fill carries large constant factors — read the
*ratio* of cached to first-frame time, not the absolutes.

Run:  PYTHONPATH=. .venv/bin/python scripts/array_block_animation.py
      PYTHONPATH=. .venv/bin/python scripts/array_block_animation.py --design invveearray
"""

import argparse
import time

import numpy as np

from antenna_designer.engines.pysim import PysimEngine
from importlib import import_module

from pysim.array_block import ArrayBlockPySim, cache_stats, reset_array_caches


def _sim(builder_cls, params, solver=ArrayBlockPySim):
    b = builder_cls(params)
    eng = PysimEngine(b, solver=solver, solver_kwargs={"degree": 2})
    return eng._make_solver(wavelength=eng._wavelength_for(b.freq))


def phase_sweep(builder_cls, base_params, phases):
    print("\n--- phase sweep (geometry fixed; operator + factor reused) ---")
    print(
        f"{'frame':>5} {'phase_lr':>9} {'Z0':>20} {'op_build':>9} {'op_hit':>7} {'t(s)':>7}"
    )
    reset_array_caches()
    for i, ph in enumerate(phases):
        p = dict(base_params)
        p["phase_lr"] = ph
        t0 = time.perf_counter()
        z, _ = _sim(builder_cls, p).compute_impedance()
        dt = time.perf_counter() - t0
        st = cache_stats()
        z0 = np.atleast_1d(z)[0]
        print(
            f"{i:>5} {ph:>9.1f} {f'{z0.real:.1f}{z0.imag:+.1f}j':>20} "
            f"{st['operator_build']:>9} {st['operator_hit']:>7} {dt:>7.3f}"
        )


def spacing_sweep(builder_cls, base_params, spacings):
    print("\n--- spacing sweep (elements identical; self-blocks reused) ---")
    print(
        f"{'frame':>5} {'del_y':>7} {'Z0':>20} "
        f"{'sb_build':>9} {'sb_hit':>7} {'uACA':>5} {'t(s)':>7}"
    )
    reset_array_caches()
    for i, dy in enumerate(spacings):
        p = dict(base_params)
        p["del_y"] = dy
        t0 = time.perf_counter()
        sim = _sim(builder_cls, p)
        z, _ = sim.compute_impedance()
        dt = time.perf_counter() - t0
        st = cache_stats()
        z0 = np.atleast_1d(z)[0]
        print(
            f"{i:>5} {dy:>7.2f} {f'{z0.real:.1f}{z0.imag:+.1f}j':>20} "
            f"{st['self_block_build']:>9} {st['self_block_hit']:>7} "
            f"{sim._last_n_coupling_aca:>5} {dt:>7.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="bowtiearray2x4")
    args = ap.parse_args()
    mod = import_module(f"antenna_designer.designs.{args.design}")
    builder_cls = mod.Builder
    base = dict(builder_cls.default_params)
    print(f"=== {args.design} animation factor-cache demo ===")
    phase_sweep(builder_cls, base, [0.0, 30.0, 60.0, 90.0, 120.0])
    spacing_sweep(builder_cls, base, [3.5, 4.0, 4.5, 5.0])


if __name__ == "__main__":
    main()
