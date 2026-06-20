"""Scaling study: ArrayBlockPySim vs dense BSplinePySim vs HMatrixPySim.

Sweeps mesh density on the array designs and reports, at each total-unknowns N:

  * storage / compression — block scalars stored vs dense N^2
  * unique ACA            — coupling blocks actually compressed (Toeplitz reuse)
  * GMRES iters           — block-Jacobi iteration count (per RHS)
  * accuracy              — |Y_method - Y_dense| / |Y_dense| (full Y matrix)
  * wall-clock            — dense assemble+solve vs accelerator build+solve

The pure-Python ACA fill carries large constant factors; read the *slopes*
and the storage/iteration columns, not just the absolute wall-clock.

Run:  PYTHONPATH=. .venv/bin/python scripts/array_block_scaling.py
      PYTHONPATH=. .venv/bin/python scripts/array_block_scaling.py --design invveearray
      PYTHONPATH=. .venv/bin/python scripts/array_block_scaling.py --nsegs 11,15,21,27
"""

import argparse
import time

import numpy as np

from antenna_designer.engines.pysim import PysimEngine
from importlib import import_module

from pysim.bspline import BSplinePySim
from pysim.hmatrix import HMatrixPySim
from pysim.array_block import ArrayBlockPySim


def _make(builder, solver, nsegs, degree=2):
    params = dict(builder.default_params)
    params["nominal_nsegs"] = nsegs
    b = type(builder)(params)
    eng = PysimEngine(b, solver=solver, solver_kwargs={"degree": degree})
    return eng._make_solver(wavelength=eng._wavelength_for(b.freq))


def _time_y(sim):
    t0 = time.perf_counter()
    Y = np.asarray(sim.compute_y_matrix(), dtype=np.complex128)
    return Y, time.perf_counter() - t0


def run(design, nsegs_list, degree=2):
    mod = import_module(f"antenna_designer.designs.{design}")
    builder0 = mod.Builder()
    print(f"\n=== {design}  (degree={degree}) ===")
    hdr = (
        f"{'nsegs':>5} {'N':>6} {'P':>3} | "
        f"{'dense s':>8} | "
        f"{'H comp':>7} {'H acc':>8} {'H s':>7} | "
        f"{'AB comp':>7} {'uACA':>5} {'iters':>5} {'AB acc':>8} {'AB s':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for nsegs in nsegs_list:
        dsim = _make(builder0, BSplinePySim, nsegs, degree)
        Yd, td = _time_y(dsim)
        N = dsim._build_basis_polynomials(dsim._build_geometry())[0].shape[0]

        hsim = _make(builder0, HMatrixPySim, nsegs, degree)
        Yh, th = _time_y(hsim)
        Hc = hsim._hmatrix.stats()["compression"]
        h_acc = np.abs(Yh - Yd).max() / np.abs(Yd).max()

        asim = _make(builder0, ArrayBlockPySim, nsegs, degree)
        Ya, ta = _time_y(asim)
        AB = asim._hmatrix  # compute_y_matrix stores the operator here
        Ac = AB.stats()["compression"]
        a_acc = np.abs(Ya - Yd).max() / np.abs(Yd).max()
        uaca = asim._last_n_coupling_aca
        iters = max(asim._last_solve_iters)
        P = len(AB.groups)

        print(
            f"{nsegs:>5} {N:>6} {P:>3} | "
            f"{td:>8.3f} | "
            f"{Hc:>7.1%} {h_acc:>8.1e} {th:>7.3f} | "
            f"{Ac:>7.1%} {uaca:>5} {iters:>5} {a_acc:>8.1e} {ta:>7.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="bowtiearray2x4")
    ap.add_argument("--nsegs", default="11,15,21,27")
    ap.add_argument("--degree", type=int, default=2)
    args = ap.parse_args()
    nsegs_list = [int(x) for x in args.nsegs.split(",")]
    run(args.design, nsegs_list, degree=args.degree)


if __name__ == "__main__":
    main()
