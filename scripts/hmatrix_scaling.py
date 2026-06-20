"""Scaling study: HMatrixPySim (hierarchical / ACA) vs dense BSplinePySim.

Sweeps a fixed-length straight wire at increasing segment count N and reports,
for each N:

  * storage      — H-matrix complex scalars stored vs dense N^2
  * rank         — mean / max far-block ACA rank
  * far%         — fraction of the matrix area in admissible (far) blocks
  * fill-work    — matrix *entries evaluated* by each method (the algorithmic
                   fill cost, independent of Python constant factors):
                   dense = N^2; H = near_area + sum_far rank*(rows+cols)
  * accuracy     — |Z_hmat - Z_dense| / |Z_dense|
  * wall-clock   — dense assemble+solve vs H build+solve (honest; the pure-
                   Python prototype carries large constant factors the C++
                   phase removes — read the *slopes*, not the absolutes)

Run:  python scripts/hmatrix_scaling.py
"""

import time

import numpy as np

from pysim.bspline import BSplinePySim
from pysim.hmatrix import HMatrixPySim

WAVELENGTH = 22.0
LEN_WL = 4.0  # wire length in wavelengths (fixed as N grows)
ACA_TOL = 1e-5
ACA_ETA = 2.0
DEGREE = 1


def _wire():
    half = 0.5 * LEN_WL * WAVELENGTH
    return [np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])]


def _fill_work(H):
    near = sum(D.size for _, _, D in H.near)
    far = sum(U.shape[1] * (U.shape[0] + V.shape[1]) for _, _, U, V in H.far)
    return near + far


def _dense_solve(sim):
    geom = sim._build_geometry()
    ss, po, kcl, wk, wbg = sim._build_basis_polynomials(geom)
    n = ss.shape[0]
    t0 = time.perf_counter()
    Z = sim._assemble_Z(sim._build_J_blocks(geom, sim.k), ss, po, geom)
    t_fill = time.perf_counter() - t0
    v_list = []
    for wi, arc, _ in sim.feeds:
        ak = geom["per_wire"][wi]["arc_at_knot"]
        sf = arc if arc is not None else ak[-1] / 2
        v_list.append(sim._build_source_vector(geom, wk, wbg, n, wi=wi, s_f=sf))
    v = v_list[0]
    t0 = time.perf_counter()
    c = sim._solve_with_kcl(Z, v, kcl)
    t_solve = time.perf_counter() - t0
    z = sim.feeds[0][2] / (v_list[0] @ c)
    return z, t_fill, t_solve, n


def main():
    ns = [250, 500, 1000, 2000, 4000]
    print(
        f"{'N':>6} {'n':>6} {'far%':>6} {'rank':>9} "
        f"{'store%':>7} {'fill%':>7} {'relZ':>9} "
        f"{'Dfill':>7} {'Dslv':>7} {'Hbld':>7} {'Hslv':>7} {'it':>4}"
    )
    for nsegs in ns:
        wire = _wire()
        dense = BSplinePySim(
            wires=wire,
            degree=DEGREE,
            n_per_edge_per_wire=[[nsegs]],
            wavelength=WAVELENGTH,
        )
        zd, t_dfill, t_dsolve, n = _dense_solve(dense)

        hmat = HMatrixPySim(
            wires=wire,
            degree=DEGREE,
            n_per_edge_per_wire=[[nsegs]],
            wavelength=WAVELENGTH,
            aca_tol=ACA_TOL,
            aca_eta=ACA_ETA,
        )
        t0 = time.perf_counter()
        H = hmat.build_hmatrix()
        t_hbuild = time.perf_counter() - t0
        st = H.stats()
        part = hmat.build_partition()

        t0 = time.perf_counter()
        zh, _ = hmat.compute_impedance()
        t_hsolve = time.perf_counter() - t0

        relz = abs(zh - zd) / abs(zd)
        fill_pct = _fill_work(H) / (n * n)
        print(
            f"{nsegs:>6} {n:>6} {part['stats']['far_frac'] * 100:>5.0f}% "
            f"{st['mean_rank']:>4.1f}/{st['max_rank']:>3d} "
            f"{st['compression'] * 100:>6.1f}% {fill_pct * 100:>6.1f}% "
            f"{relz:>9.1e} "
            f"{t_dfill:>6.3f}s {t_dsolve:>6.3f}s "
            f"{t_hbuild:>6.3f}s {t_hsolve:>6.3f}s "
            f"{max(hmat._last_solve_iters):>4d}"
        )


if __name__ == "__main__":
    main()
