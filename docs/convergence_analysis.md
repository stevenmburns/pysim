# Convergence analysis: pysim vs NEC2

This document records a validation campaign against NEC2 (PyNEC) for the
single-wire dipole and the two-element Yagi, and characterizes how pysim's
impedance results converge as `nsegs` is increased.

## Test geometry

Both models use pysim's default parameters:

- wavelength = 22 m (freq = 13.627 MHz)
- halfdriver_factor = 0.962, so halfdriver = 5.291 m
- wire_radius = 0.0005 m
- For the Yagi, the reflector is `1.05 × halfdriver` long and spaced
  `halfdriver` (≈ 0.241 λ) behind the driver

NEC2 is run on the matching geometry in free space (no ground, no copper
loss): a single wire for the dipole comparison, two parallel wires for the
Yagi comparison. The driver is fed at the center segment (single contiguous
wire, no split feed gap) so the comparison is apples-to-apples with pysim's
feed model.

## NEC2 reference values

NEC2 converges in segment count almost immediately:

| n_seg | Dipole Z (Ω) | Yagi Z (Ω) |
|---:|---|---|
|  21 | 69.69 − j18.67 | 77.09 + j6.45 |
|  41 | 69.65 − j18.41 | 77.19 + j6.61 |
| 101 | 69.64 − j18.21 | 77.28 + j6.74 |

Reference values used below: **dipole = 69.64 − j18.21 Ω**, **Yagi = 77.28 +
j6.74 Ω**.

## pysim sweep data

`NewPySim` (single-wire dipole) at `ntrap=8`:

| nsegs | Re(Z) | Im(Z) | wall |
|---:|---:|---:|---:|
|  101 | 74.6411 | +12.1566 |   0.0s |
|  201 | 72.6522 |  +3.3437 |   0.5s |
|  401 | 71.3992 |  −3.6575 |   0.7s |
|  801 | 70.5559 |  −9.6050 |   1.8s |
| 1601 | 69.9514 | −14.9412 |   6.5s |
| 3001 | 69.5317 | −19.4680 |  23.2s |
| 4001 | 69.3684 | −21.4745 |  47.3s |
| 5001 | 69.2517 | −23.0100 |  53.9s |

`YagiPySim` (2-element driver + reflector) at `ntrap=8`:

| nsegs | Re(Z) | Im(Z) | wall |
|---:|---:|---:|---:|
|  101 | 84.6914 | +37.4430 |   0.1s |
|  201 | 81.7401 | +28.7205 |   0.4s |
|  401 | 79.8827 | +21.6482 |   1.9s |
|  801 | 78.6714 | +15.5312 |   8.8s |

## Extrapolation attempt

Fitting `Z(N) = Z_inf + c / N^p` (3-parameter nonlinear least squares,
real and imaginary parts independently) to progressively narrowing trailing
windows of the dipole sweep:

| window | p (Re) | p (Im) | Z_inf (Re) | Z_inf (Im) |
|:---|---:|---:|---:|---:|
| last 8 (N≥101)  | 0.551 | 0.198 | 68.59 | −52.45 |
| last 7 (N≥201)  | 0.486 | 0.153 | 68.38 | −64.21 |
| last 6 (N≥401)  | 0.431 | 0.116 | 68.17 | −79.47 |
| last 5 (N≥801)  | 0.385 | 0.088 | 67.98 | −99.46 |
| last 4 (N≥1601) | 0.345 | 0.066 | 67.80 | −126.23 |
| last 3 (N≥3001) | 0.320 | 0.053 | 67.67 | −150.97 |

If the power law were valid, both `p` and `Z_inf` would stabilize as we
restricted to the asymptotic region. They do not. The imag exponent
collapses toward 0 (degenerating the model to `Z_inf + c`, i.e., no
convergence), and the fitted `Z_inf` for the imag part marches off to −∞.

The real-part fit is more stable: `p ≈ 0.3–0.4` and `Z_inf ≈ 68 Ω`. So the
real-part *does* converge to a finite limit, just slowly, and slightly below
NEC's 69.64.

## Findings

### 1. Real part: slow convergence to a value ~1% below NEC

`NewPySim`'s real part converges to roughly **68 Ω**, vs NEC's 69.64.
The ~1% gap is a formulation difference: pulse-basis / point-matching MoM
vs NEC's sinusoidal-basis / Galerkin MoM converge to slightly different
radiation resistances. Neither is "wrong" — they're different
discretizations of the same continuous problem.

### 2. Imaginary part: divergent, not convergent

The reactance does **not** have a finite limit in this formulation. The
per-doubling change in Im(Z) has stabilized at roughly −5 Ω across the
high-N region:

| N₁ → N₂ | ΔIm | per `log₂(N₂/N₁)` |
|---|---|---|
|  801 → 1601 | −5.34 | −5.34 |
| 1601 → 3001 | −4.53 | −4.97 |
| 3001 → 5001 | −3.54 | −4.78 |

That's the signature of **logarithmic divergence**: Im(Z) ≈ Im₀ − α·log(N)
with α ≈ 5 Ω per doubling. As `nsegs → ∞`, the reactance → −∞.

The cause is well-known in the MoM literature: a strict delta-gap voltage
source combined with pulse basis functions produces a non-integrable local
current singularity at the feed. With each segment refinement we resolve
more of that singularity and accumulate more reactive contribution that
never settles.

### 3. The Yagi inherits the same behavior

`YagiPySim`'s imag part drops from +37 (N=101) to +16 (N=801) — already
trending past NEC's +6.7 and on the same logarithmic-divergence trajectory
as the dipole. Real part drops from 85 to 79, heading toward NEC's 77.
Same physics, same conclusions.

### 4. `AugmentedSplinePySim` is broken

For the dipole at nsegs=101: it returns roughly `78 + j956 Ω`, which
*diverges* in the wrong direction (gets worse with nsegs). The return
signature `(driver_impedance, (i, orig_i, matched_i))` and the way the
spline is layered on top of the pulse-basis solve suggest it's fitting a
spline to post-hoc currents rather than solving the MoM in a spline basis.
Whatever this code is doing, it's not a working spline MoM and doesn't help
the convergence problem.

## Implications

- **`NewPySim` and `YagiPySim` are not buggy** — they correctly implement
  pulse-basis / point-matching MoM. The slow real-part convergence and
  divergent imag part are well-documented characteristics of that
  formulation.
- **You cannot match NEC just by increasing nsegs.** The reactance gap
  *grows* with refinement.
- **For practical antenna design** (where reactance matters as much as
  radiation resistance), the current pysim is not a substitute for NEC.

## Next steps

The right fix is **richer basis functions**:

1. **Triangular ("rooftop") basis** — piecewise-linear, support over two
   adjacent segments, current continuous everywhere. Convergence jumps
   from `O(1/N^0.5)` or worse to `O(1/N²)`. Charge density becomes
   piecewise constant instead of a Dirac comb at every segment boundary.
   The natural pairing is **Galerkin testing** with the same triangular
   functions, giving a symmetric matrix.
2. **Sinusoidal basis** — what NEC uses. The natural solution of
   Pocklington's equation in the thin-wire limit. Higher implementation
   cost but the gold standard.

The `spline.py` and `bspline.py` modules in this repo appear to be reaching
for option 1 (or a higher-order generalization). Whether they ever land at
a working MoM solver — not just a post-hoc spline fit — is an open
question that this campaign didn't answer.

A second mitigation, independent of basis choice, is a **better feed model**
(e.g., magnetic-frill or finite-gap feed instead of strict delta-gap).
That alone can move the imag-part divergence from `log(N)` down to a
finite limit even with pulse basis.

## Scripts used

- `scripts/compare_yagi_nec.py` — runs both pysim models and PyNEC against
  the same geometry, prints side-by-side impedance results.
- `scripts/extrapolate.py` — fits `Z(N) = Z_inf + c/N^p` to the first
  five-point dipole and Yagi sweeps.
- `scripts/extrapolate_deep.py` — pushes the dipole sweep out to N=5001,
  fits progressively narrowing trailing windows to test whether the power
  law stabilizes (it doesn't).

All three require PyNEC, which is in the antenna_designer venv but not in
pysim's venv. Invocation:

```
PYTHONPATH=/home/smburns/antennas/pysim/src \
    /home/smburns/antennas/antenna_designer/.venv/bin/python \
    scripts/<script>.py
```
