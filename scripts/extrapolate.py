"""Extrapolate NewPySim / YagiPySim impedance to infinite nsegs.

Both classes use pulse-basis / point-matching MoM, which converges slowly with
nsegs. Fit Z(N) = Z_inf + c/N^p (separately to real and imag parts) and report
Z_inf, the power-law exponent p, and the fit residuals. Where available,
compare against the NEC2 free-space reference.

Run with antenna_designer's venv (which has PyNEC):

    PYTHONPATH=/home/smburns/antennas/pysim/src \
        /home/smburns/antennas/antenna_designer/.venv/bin/python \
        scripts/extrapolate.py
"""
import time

import numpy as np
from scipy.optimize import curve_fit

from pysim import PySim as NewPySim
from pysim.yagi import YagiPySim


def power_law(N, z_inf, c, p):
    return z_inf + c / N**p


def fit_extrapolation(ns, zs, label):
    """Fit Z(N) = Z_inf + c/N^p to (ns, zs). zs is complex.

    Returns (z_inf_complex, residual_real, residual_imag, p_real, p_imag).
    """
    ns = np.asarray(ns, dtype=float)

    def fit_one(values, tag):
        # Initial guesses: Z_inf ~ last point, c ~ (first - last), p ~ 1
        p0 = [values[-1], values[0] - values[-1], 1.0]
        try:
            popt, _ = curve_fit(power_law, ns, values, p0=p0, maxfev=10000)
        except RuntimeError as e:
            print(f"  {label} ({tag}): fit failed: {e}")
            return values[-1], np.nan, np.nan
        z_inf, c, p = popt
        residual = np.max(np.abs(power_law(ns, *popt) - values))
        return z_inf, residual, p

    z_inf_r, res_r, p_r = fit_one(zs.real, "real")
    z_inf_i, res_i, p_i = fit_one(zs.imag, "imag")
    return complex(z_inf_r, z_inf_i), res_r, res_i, p_r, p_i


def sweep(label, fn, ns):
    print(f"\n{label} sweep:")
    print(f"  {'nsegs':>6}  {'Z':^28}  {'wall':>6}")
    zs = []
    for n in ns:
        t = time.time()
        z, _ = fn(n)
        zs.append(z)
        print(f"  {n:6d}  {z.real:9.4f} + j{z.imag:9.4f}     {time.time()-t:5.2f}s")
    return np.array(zs)


def report(label, ns, zs, nec_ref=None):
    z_inf, res_r, res_i, p_r, p_i = fit_extrapolation(ns, zs, label)
    print(f"\n{label} extrapolation:")
    print("  fit: Z(N) = Z_inf + c/N^p")
    print(f"  p (real) = {p_r:.3f},  p (imag) = {p_i:.3f}")
    print(f"  max residual: real={res_r:.4f} Ω, imag={res_i:.4f} Ω")
    print(f"  Z_inf = {z_inf.real:7.3f} + j{z_inf.imag:7.3f}")
    if nec_ref is not None:
        err = z_inf - nec_ref
        print(f"  NEC reference: {nec_ref.real:7.3f} + j{nec_ref.imag:7.3f}")
        print(f"  Z_inf - NEC:   {err.real:+7.3f} + j{err.imag:+7.3f}")


def main():
    # Sweep nsegs (geometric so power-law fit is well-conditioned).
    dipole_ns = [101, 201, 401, 801, 1601]
    yagi_ns = [101, 201, 401, 801]   # Yagi is 2N×2N so 1601 would be slow.

    nec_dipole = complex(69.64, -18.21)
    nec_yagi = complex(77.28, 6.74)

    dipole_zs = sweep(
        "NewPySim dipole",
        lambda n: NewPySim(nsegs=n).compute_impedance(ntrap=8),
        dipole_ns,
    )
    report("NewPySim dipole", dipole_ns, dipole_zs, nec_ref=nec_dipole)

    yagi_zs = sweep(
        "YagiPySim 2-element",
        lambda n: YagiPySim(nsegs=n).compute_impedance(ntrap=8),
        yagi_ns,
    )
    report("YagiPySim 2-element", yagi_ns, yagi_zs, nec_ref=nec_yagi)


if __name__ == "__main__":
    main()
