# gprMax analysis

Notes from evaluating gprMax as (1) a candidate third-reference solver for the
pysim/PyNEC fan-dipole junction discrepancy (NEXT_STEPS item 8) and (2) a tool
for antenna-over-terrain scenarios that exceed the assumptions of NEC's
Sommerfeld-ground MoM formulation.

## What gprMax is

Open-source FDTD electromagnetic simulator originally built in 1996 at the
University of Edinburgh for Ground Penetrating Radar modelling, generalised
since then for arbitrary EM wave propagation. Latest is 3.1.7 ("Big Smoke"),
January 2024. Python 3 with Cython hot loops, OpenMP CPU solver, CUDA GPU
solver, OpenCL available.

Solves Maxwell's equations on a 3D Yee grid in the time domain — a broadband
result from one run via a Gaussian/Ricker pulse excitation, vs the
frequency-domain MoM family (NEC, pysim) that needs one matrix solve per
frequency point.

### Method comparison

|                    | NEC / pysim (MoM)                       | gprMax (FDTD)                                     |
| ------------------ | --------------------------------------- | ------------------------------------------------- |
| Domain             | Frequency                               | Time (broadband from one run)                     |
| Geometry           | Wire segments / patches in free space   | Voxelised volumes — air, soil, dielectrics, PECs  |
| Best at            | Thin-wire antennas, open radiation      | Wave propagation through inhomogeneous media      |
| Scaling            | ~N^2 to N^3 in segment count            | ~N^4 in grid cells (3 spatial + time)             |
| Open boundary      | Free-space Green's function             | PML absorbing boundary                            |
| GPU                | No                                      | Yes (CUDA, ~20× single-GPU speedup vs 4-core CPU) |

### Notable features

- Multi-pole **Debye/Drude/Lorenz** dispersive materials (real soils, lossy
  media, water)
- **Peplinski** mixing model for realistic moist-soil ε and σ from
  moisture/clay/sand fractions
- **Sub-gridding** — finer cells locally without paying everywhere
- **STL → voxel** toolbox for complex imported geometry
- Rough-surface generation for realistic soil topography
- Antenna excitation via `#transmission_line` attached to a thin-wire dipole

### Workflow

Mostly deck-driven, not fluent Python: write a `.in` input file, run
`python -m gprMax model.in`, read the resulting `.out` (HDF5) with
`tools.plot_Ascan` / `plot_Bscan` / `plot_antenna_params`.

## Question 1: gprMax for the pysim/PyNEC fan-dipole junction discrepancy

**Verdict: not a useful third reference for this specific dispute.**

### How gprMax handles 3-way wire junctions

Wires are **Yee cell edges** (the `#edge` command sets PEC or other properties
on the edge between two grid nodes). A junction is just a Yee node where three
or more edges meet, and KCL is satisfied automatically because the Ampère
loops around those edges share faces and are stepped together. There is no
special "junction object" — KCL is a consequence of the grid update equations
holding everywhere.

So in principle, junction handling is "for free" — no Lagrange multiplier, no
basis-function bookkeeping, no inter-arm-angle problem analogous to the pysim
cross-wire-kernel regularisation issue (NEXT_STEPS item 12).

### Why it would not resolve the dispute cleanly

1. **Geometry constraints.** Wires on the base Yee grid must be axis-aligned.
   A fan dipole has K arms emerging at oblique angles (13°–71° in the cone
   sweep). Those arms must either be **stair-stepped** along grid axes — which
   introduces a geometry error that grows with cone angle, exactly the
   variable the dispute hinges on — or modelled with a **thin-wire
   correction** (Holland / Edelvik / Umashankar–Taflove family). gprMax's
   documented dipole examples are mostly axis-aligned.

2. **Scale problem.** At 14.3 MHz, λ ≈ 21 m → standard FDTD wants Δ ≤ λ/10 ≈
   2 m cells. Wire radius is ~1 mm. The Holland thin-wire correction is
   calibrated for **straight** wires; how well it handles a 3-wire node at
   oblique angles is itself an open question — and a regime where gprMax has
   no documented validation.

3. **Different method family.** gprMax's disagreement with pysim/PyNEC would
   be non-diagnostic. If gprMax matched PyNEC, it could just mean both share
   an approximation pysim gets right. If it matched neither (most likely,
   given the method differences), a third disagreeing number wouldn't resolve
   the question.

### Better third references for the dispute

NEXT_STEPS item 8 options (a) **NEC4** and (b) **MININEC** are stronger,
because they live in the same MoM family with thin-wire kernels — apples to
apples on the formulation. NEC4 specifically has the **extended thin-wire
kernel** and improved junction treatment that NEC2 (PyNEC) lacks — direct
successor designed to fix the class of issue we're seeing.

## Question 2: dipole 7 m above a 15 m levee above a 1 km river

**Verdict: yes, this is exactly the kind of problem gprMax was built for.**

This case is where NEC's flat-infinite-half-space Sommerfeld ground breaks
down: three distinct media (air / earth / water) with non-flat boundaries and
a bounded river that reflects off the far bank.

### Feasibility scales with frequency

Domain: roughly 1200 m × ~500 m × ~80 m (river width + bank buffers + air
column + earth column). PML eats ~10 cells per face.

| Band         | λ      | Cell size (λ/10) | Total cells | Single-GPU runtime estimate |
| ------------ | ------ | ---------------- | ----------- | --------------------------- |
| HF, 14 MHz   | 21 m   | 2 m              | ~6M         | minutes                     |
| HF, 28 MHz   | 11 m   | 1 m              | ~50M        | ~hour                       |
| VHF, 50 MHz  | 6 m    | 0.5 m            | ~380M       | multi-GPU, several hours    |
| VHF, 144 MHz | 2 m    | 0.2 m            | ~7.5B       | HPC cluster only            |
| UHF, 400+ MHz| <1 m   | <0.1 m           | >100B       | not practical               |

HF is comfortable. VHF gets hard. UHF is out without exotic resources or
aggressive sub-gridding.

### Schematic input

```text
# Materials (Peplinski for soil, constant ε,σ for first cut)
#material: 80 0.05 1 0 water       # fresh river, σ depends on minerals
#material: 9  0.005 1 0 levee_soil # moist compacted earth
#material: 5  0.001 1 0 dry_bank   # drier far-bank substrate

# Geometry (slabs and boxes; levee via #triangle decomposition or STL)
#box: 0 0 0 1200 500 30 dry_bank          # base earth layer
#box: 100 0 30 1100 500 30 water          # 1 km wide, 30 m deep river
#triangle: ...                            # levee wedge — crown 15 m above water

# Dipole, 7 m above levee crown (crown at z=45 → dipole at z=52)
#edge: 595 200 52 605 200 52 pec
#transmission_line: y 600 200 52 50 my_gaussian_pulse

# Excitation
#waveform: gaussiandot 1 14e6 my_gaussian_pulse

# Outputs
#rx: 600 200 52                           # feed-point for Z / S11
#snapshot: ...                            # field movies in a vertical slice
#rx_array: ...                            # for near-to-far-field transform
```

`#edge` here is schematic; an actual run would use the `user_libs` antenna
modules or a `#python:` block to place the dipole, especially if a thin-wire
correction is wanted instead of straight stair-stepped PEC edges.

### What you can extract

1. **Driving-point Z and S11** at the dipole feed — how the levee + river
   shifts impedance vs free-space or flat-PEC-ground baselines.
2. **Field snapshots** in a vertical slice — surface wave, air/water
   refraction, far-bank reflection.
3. **Radiation pattern with terrain** via gprMax's near-to-far-field
   transform — correctly accounts for levee shadow and water reflection.
4. **Cross-river field strength** at receive points on the far bank — link
   budget.

### Caveats worth knowing up front

- Dipole modelling: gprMax has a working dipole example. For wire radius much
  smaller than a cell, use the Holland/Edelvik thin-wire correction. Stair-
  stepped PEC edges are fine for an axis-aligned wire; otherwise sub-grid or
  use the thin-wire model.
- Water electrical properties matter a lot. Fresh river water
  (σ ≈ 0.01–0.05 S/m) reflects very differently from brackish/salt
  (σ ≈ 1–5 S/m).
- Soil moisture matters a lot. Levee at 15% vs 30% moisture content shifts ε
  by 2–3× and σ by ~10×. Use Peplinski if you have soil composition data;
  otherwise pick reasonable constants and sweep them.
- Drive with a broadband Gaussian pulse to get input impedance across a band
  in one run, or sinusoidal CW for a steady-state pattern at a single
  frequency.
- Run length: enough for waves to traverse the longest path several times.
  For 1.2 km at ~c, ~4 μs of physical time → at Δt ~3 ns (CFL at Δ=1m),
  ~1500 steps. Cheap.

## Bottom line

- gprMax is **not the right tool** for resolving the fan-dipole junction
  discrepancy with pysim/PyNEC. Reach for NEC4 or MININEC instead — same MoM
  family, direct apples-to-apples comparison.
- gprMax **is the right tool** for antenna-over-real-terrain scenarios
  (levee, river, layered ground, near-field structures). At HF on a single
  GPU these are minutes-to-hours simulations. The 3-medium / non-flat-
  interface part — the part that breaks NEC's Sommerfeld ground and pysim's
  free-space-plus-image formulation — is what FDTD handles natively.

## References

- [gprMax.com](https://www.gprmax.com/)
- [gprMax on GitHub](https://github.com/gprMax/gprMax)
- [gprMax paper, Computer Physics Communications](https://www.sciencedirect.com/science/article/pii/S0010465516302533)
- [gprMax input commands](https://docs.gprmax.com/en/latest/input.html)
- [gprMax antenna examples](https://docs.gprmax.com/en/latest/examples_antennas.html)
- [gprMax GPR modelling guidance (Peplinski, dispersive materials)](https://docs.gprmax.com/en/latest/gprmodelling.html)
- [Edelvik, "An improved thin-wire model for FDTD"](https://www.researchgate.net/publication/3122006_An_improved_thin-wire_model_for_FDTD)
