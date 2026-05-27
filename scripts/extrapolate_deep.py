"""Push the NewPySim dipole sweep out to nsegs=5001 to disambiguate:
   (a) power-law extrapolation is reliable -> p stays near 0.25
   (b) fit is wrong -> p drifts as nsegs grows
"""
import time

import numpy as np
from scipy.optimize import curve_fit

from pysim import PySim as NewPySim


def power_law(N, z_inf, c, p):
    return z_inf + c / N**p


def fit(ns, values, p0=None):
    ns = np.asarray(ns, dtype=float)
    values = np.asarray(values, dtype=float)
    if p0 is None:
        p0 = [values[-1], values[0] - values[-1], 1.0]
    popt, _ = curve_fit(power_law, ns, values, p0=p0, maxfev=20000)
    z_inf, c, p = popt
    residual = np.max(np.abs(power_law(ns, *popt) - values))
    return z_inf, c, p, residual


def main():
    ns = [101, 201, 401, 801, 1601, 3001, 4001, 5001]
    zs = []
    for n in ns:
        t = time.time()
        z, _ = NewPySim(nsegs=n).compute_impedance(ntrap=8)
        zs.append(z)
        print(
            f"  nsegs={n:5d}  Z = {z.real:9.4f} + j{z.imag:9.4f}   "
            f"({time.time()-t:6.1f}s)"
        )
    zs = np.array(zs)

    nec = complex(69.64, -18.21)
    print(f"\nNEC reference: {nec.real:.3f} + j{nec.imag:.3f}\n")

    print(f"{'window':<22}  {'p (re)':>8}  {'p (im)':>8}  "
          f"{'Z_inf (re)':>12}  {'Z_inf (im)':>12}")
    print("-" * 70)
    # Fit progressively larger trailing windows so we can see if p stabilizes.
    for k in range(3, len(ns) + 1):
        sub_ns = ns[-k:]
        sub_re = zs[-k:].real
        sub_im = zs[-k:].imag
        try:
            zr, _, pr, _ = fit(sub_ns, sub_re)
            zi, _, pi, _ = fit(sub_ns, sub_im)
            label = f"last {k} (N>={sub_ns[0]})"
            print(f"{label:<22}  {pr:8.3f}  {pi:8.3f}  {zr:12.3f}  {zi:12.3f}")
        except RuntimeError as e:
            print(f"  last {k}: fit failed: {e}")


if __name__ == "__main__":
    main()
