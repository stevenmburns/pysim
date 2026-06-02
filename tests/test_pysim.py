import pytest
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

from pysim.bspline import BSplinePySim
from pysim.sinusoidal import SinusoidalPySim
from pysim.triangular import TriangularPySim
from pysim._accelerators import dist_outer_product

import numpy as np


def test_extension():
    nsegs = 20
    pts = np.array([[0, 0, z] for z in range(nsegs + 1)]) / (2 * nsegs)

    result = dist_outer_product(pts, pts)
    expected = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("nsegs", [20, 40, 80])
def test_triangular_dipole_smoke(nsegs):
    L = 2 * 0.962 * 22 / 4
    sim = TriangularPySim(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]])],
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
    )
    z, c = sim.compute_impedance()
    assert c.shape == (nsegs - 1,)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # NEC reference for the default dipole geometry: 69.64 - j18.21.
    # Triangular basis converges quickly; even N=20 is within ~2 Ohm on the
    # real part and ~2 Ohm on the imag.
    assert abs(z.real - 69.64) < 3.0
    assert abs(z.imag - (-18.21)) < 6.0


@pytest.mark.parametrize("nsegs", [20, 40, 80])
def test_triangular_two_wire_yagi_smoke(nsegs):
    # Driver + 1.05x reflector at 1 halfdriver spacing — the classic 2-element
    # Yagi case. Mutual coupling pushes the driver Z away from bare-dipole
    # 69.6 - j18.2 toward roughly 77 + j6.
    hd = 0.962 * 22 / 4  # matches TriangularPySim defaults
    sp = hd
    driver = np.array([[0.0, -hd, 0.0], [0.0, hd, 0.0]])
    refl = np.array([[-sp, -1.05 * hd, 0.0], [-sp, 1.05 * hd, 0.0]])
    z, c = TriangularPySim(
        wires=[driver, refl],
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
    ).compute_impedance()
    assert c.shape == (2 * (nsegs - 1),)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    assert 65.0 < z.real < 85.0
    assert -10.0 < z.imag < 25.0


@pytest.mark.parametrize("nsegs", [20, 40, 80])
def test_triangular_collinear_polyline(nsegs):
    # A "bent" wire whose polyline anchors happen to be collinear should give
    # nearly the same answer as a single-edge straight wire: the only path
    # difference is that cross-edge pairs go through quadrature instead of
    # the analytic formula.
    L = 2 * 0.962 * 22 / 4
    straight = np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]])
    polyline = np.array([[0.0, 0.0, 0.0], [0.0, L / 2, 0.0], [0.0, L, 0.0]])
    z_straight, _ = TriangularPySim(
        wires=[straight], n_per_edge_per_wire=[[nsegs]], nsegs=nsegs
    ).compute_impedance()
    # Use n_qp_off=8 so the artificial cross-edge quadrature at the fake
    # corner has the same precision as the analytic same-edge path.
    z_bent, _ = TriangularPySim(
        wires=[polyline],
        n_per_edge_per_wire=[[nsegs // 2, nsegs // 2]],
        nsegs=nsegs,
        n_qp_off=8,
    ).compute_impedance()
    assert abs(z_bent - z_straight) < 0.2


def test_triangular_v_dipole_smoke():
    # 30-deg V-dipole: arms bent away from the y-axis in the y-z plane.
    L = 2 * 0.962 * 22 / 4
    half = L / 2
    alpha = np.radians(30)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    polyline = np.array(
        [
            [0.0, -half * cos_a, -half * sin_a],
            [0.0, 0.0, 0.0],
            [0.0, +half * cos_a, -half * sin_a],
        ]
    )
    z, c = TriangularPySim(
        wires=[polyline], n_per_edge_per_wire=[[40, 40]], nsegs=80
    ).compute_impedance()
    assert c.shape == (79,)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # Bending lowers R and pushes X more negative compared to straight (69.6 - j18.5).
    assert 30.0 < z.real < 65.0
    assert z.imag < -25.0


@pytest.mark.parametrize("nsegs", [20, 40])
def test_triangular_swept_matches_per_freq(nsegs):
    # Build a small two-wire moxon-like geometry; the batched solver must
    # agree with single-freq calls to machine precision.
    L = 2 * 0.962 * 22 / 4
    halfL = L / 2
    driver = np.array(
        [
            [-0.1, -halfL, 0.0],
            [0.3, -halfL, 0.0],
            [0.3, -0.05, 0.0],
            [0.3, 0.05, 0.0],
            [0.3, halfL, 0.0],
            [-0.1, halfL, 0.0],
        ]
    )
    refl = np.array(
        [
            [-0.2, halfL, 0.0],
            [-0.6, halfL, 0.0],
            [-0.6, -halfL, 0.0],
            [-0.2, -halfL, 0.0],
        ]
    )
    sim = TriangularPySim(
        wires=[driver, refl],
        n_per_edge_per_wire=[[4, nsegs, 1, nsegs, 4], [4, nsegs, 4]],
        nsegs=nsegs,
        feed_wire_index=0,
    )
    z_single, _ = sim.compute_impedance()
    k_arr = np.array([sim.k])
    z_swept = sim.compute_impedance_swept(k_arr)
    assert abs(z_single - z_swept[0]) < 1e-9
    assert np.isfinite(z_single.real) and np.isfinite(z_single.imag)


def test_triangular_moxon_smoke():
    # Approximate moxon at 28.57 MHz with the antenna_designer default
    # parameters. Sanity-check R/X land in plausible bands and currents
    # come out finite.
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.57
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    halfdriver = 0.962 * wavelength / 4
    aspect_ratio = 0.3646
    tipspacer_factor = 0.0773
    t0_factor = 0.4078
    long_ = 2 * halfdriver / (1 + 2 * aspect_ratio * t0_factor)
    short_ = aspect_ratio * long_
    tipspacer = short_ * tipspacer_factor
    t0 = short_ * t0_factor
    eps = 0.05

    def rx(p):
        return (-p[0], p[1], p[2])

    def ry(p):
        return (p[0], -p[1], p[2])

    S = (short_ / 2, eps, 0.0)
    A = (S[0], long_ / 2, 0.0)
    B = (A[0] - t0, A[1], 0.0)
    Cc = (B[0] - tipspacer, B[1], 0.0)
    D = rx(A)
    E = ry(D)
    F = ry(Cc)
    G = ry(B)
    H = ry(A)
    T = ry(S)

    driver = np.array([G, H, T, S, A, B], dtype=float)
    reflector = np.array([Cc, D, E, F], dtype=float)

    sim = TriangularPySim(
        wires=[driver, reflector],
        n_per_edge_per_wire=[[8, 21, 1, 21, 8], [8, 21, 8]],
        feed_wire_index=0,
        nsegs=40,
        wavelength=wavelength,
        halfdriver_factor=0.962,
    )
    z, c = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # Moxons are nominally tuned for ~50 Ω at resonance; with the canonical
    # antenna_designer factors and 28.57 MHz design freq we see ~70 + j10
    # which is a reasonable working point (the canonical design is tuned
    # for a slightly different free-space target than ours).
    assert 40.0 < z.real < 110.0
    assert -30.0 < z.imag < 40.0


def test_triangular_hexbeam_smoke():
    # Single-band hexbeam at 28.47 MHz with the antenna_designer default
    # factors (halfdriver=2.82m, tipspacer=0.1312, t0=0.1243). Hexbeams
    # are tuned for ~50 Ω.
    import math

    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    halfdriver = 2.82
    tipspacer_factor = 0.1312
    t0_factor = 0.1243
    radius = halfdriver / (2 - t0_factor - tipspacer_factor)
    tipspacer = radius * tipspacer_factor
    t0 = radius * t0_factor
    t1 = radius - tipspacer - t0
    eps = 0.05
    cos30 = math.sqrt(3) / 2
    sin30 = 0.5

    def rx(p):
        return (-p[0], p[1], p[2])

    def ry(p):
        return (p[0], -p[1], p[2])

    A = (radius * cos30, radius * sin30, 0.0)
    B = (A[0] - t1 * cos30, A[1] + t1 * sin30, 0.0)
    D = (0.0, radius, 0.0)
    Cc = (D[0] + t0 * cos30, D[1] - t0 * sin30, 0.0)
    E = rx(A)
    F = ry(E)
    G = ry(D)
    H = ry(Cc)
    I_ = ry(B)
    J = ry(A)
    S = (eps * cos30, eps * sin30, 0.0)
    T = ry(S)

    driver = np.array([I_, J, T, S, A, B], dtype=float)
    reflector = np.array([Cc, D, E, F, G, H], dtype=float)

    sim = TriangularPySim(
        wires=[driver, reflector],
        n_per_edge_per_wire=[[15, 21, 1, 21, 15], [3, 21, 21, 21, 3]],
        feed_wire_index=0,
        nsegs=40,
        wavelength=wavelength,
        halfdriver_factor=1.071,
    )
    z, c = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # Hexbeam at the canonical free-space design point lands near 50+j20.
    assert 30.0 < z.real < 75.0
    assert -10.0 < z.imag < 45.0


# ---- Junctions (K wires meeting at a node) ----


def test_triangular_k2_junction_equivalent_to_single_polyline():
    """A K=2 junction at a kink is mathematically equivalent to a single
    polyline with that kink as an interior knot — the Lagrange-augmented
    KCL constraint reduces the two directional bases to one effective DOF
    matching the interior tent basis. Should agree to roundoff.
    """
    # Bent dipole, kink at (0, 0, -2), feed mid-arm (NOT at the kink).
    pl_single = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    sim_single = TriangularPySim(
        wires=[pl_single],
        n_per_edge_per_wire=[[15, 15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        nsegs=15,
        wire_radius=0.0005,
    )
    z_single, _ = sim_single.compute_impedance()

    # Same geometry split into 2 wires joined by a K=2 junction at the kink.
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    sim_junction = TriangularPySim(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        nsegs=15,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )
    z_junction, _ = sim_junction.compute_impedance()
    assert abs(z_junction - z_single) < 1e-9, (
        f"K=2 junction Z={z_junction}, single-polyline Z={z_single}"
    )


def test_triangular_k2_junction_swept_matches_per_freq():
    """Batched swept solver with junctions should match per-freq solves."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        nsegs=15,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )
    sim_sweep = TriangularPySim(wavelength=22, **common)
    C_LIGHT = 299_792_458.0
    freqs_mhz = np.array([10.0, 14.0, 20.0])
    k_array = 2 * np.pi * freqs_mhz * 1e6 / C_LIGHT
    z_swept = sim_sweep.compute_impedance_swept(k_array)
    for f, zs in zip(freqs_mhz, z_swept):
        sim_f = TriangularPySim(wavelength=C_LIGHT / (f * 1e6), **common)
        z_f, _ = sim_f.compute_impedance()
        assert abs(zs - z_f) < 1e-9, f"f={f}: swept={zs}, single={z_f}"


def test_triangular_hentenna_smoke():
    """Single-band hentenna at 28.47 MHz with the antenna_designer params_50
    factors. Geometry is a tall narrow rectangular loop with a horizontal
    cross-bar near the bottom; the feed sits in a small gap (T,S) at the
    middle of the cross-bar. K=3 junctions at B (right of cross-bar) and D
    (left of cross-bar) where the cross-bar half meets the upper and lower
    loop perimeters; K=2 junctions at S and T where the cross-bar halves
    meet the feed wire.

        C----------------------------A
        |                            |
        |                            |
        D------------T--S------------B
        |                            |
        |                            |
        E----------------------------F
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    # antenna_designer hentenna params_50 (tuned for ~50 Ω feed).
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps = 0.05

    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor

    A = (0.0, half_w, 0.0)
    B = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps, z_mid)
    C = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E = (0.0, -half_w, z_bot)
    T = (0.0, -eps, z_mid)

    N = 21
    Nfeed = 3
    wires = [
        np.array([T, S], dtype=float),  # 0: feed gap
        np.array([S, B], dtype=float),  # 1: right half of cross-bar
        np.array(
            [B, A, C, D], dtype=float
        ),  # 2: upper rectangle (right-up-top-down to D)
        np.array([T, D], dtype=float),  # 3: left half of cross-bar
        np.array([D, E, F, B], dtype=float),  # 4: lower rectangle (down-bottom-up to B)
    ]
    n_per_edge_per_wire = [[Nfeed], [N], [N, N, N], [N], [N, N, N]]
    junctions = [
        [(0, "end"), (1, "start")],  # at S
        [(0, "start"), (3, "start")],  # at T
        [(1, "end"), (2, "start"), (4, "end")],  # at B (K=3)
        [(2, "end"), (3, "end"), (4, "start")],  # at D (K=3)
    ]

    sim = TriangularPySim(
        wires=wires,
        n_per_edge_per_wire=n_per_edge_per_wire,
        feed_wire_index=0,
        feed_arclength=eps,
        wavelength=wavelength,
        nsegs=N,
        wire_radius=0.0005,
        junctions=junctions,
    )
    z, c = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # Hentenna params_50 is tuned for ~50 Ω at 28.47 MHz in NEC2; the
    # triangular Galerkin basis lands within a similar window. Use the same
    # generous bands as the moxon/hexbeam smoke tests.
    assert 25.0 < z.real < 110.0, f"R={z.real} out of plausible 50Ω-tuned range"
    assert -40.0 < z.imag < 60.0, f"X={z.imag} out of plausible 50Ω-tuned range"


@pytest.mark.parametrize("degree,nsegs", [(2, 21), (2, 81)])
def test_bspline_dipole_converges_to_nec(degree, nsegs):
    """BSplinePySim degree-2 (quadratic) on the default half-wave dipole.
    With higher-order bases and analytic singularity subtraction we expect
    rapid convergence to the NEC reference 69.64 - j18.21; even N=21 should
    be within ~1 Ω.
    """
    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    sim = BSplinePySim(
        wires=wires,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        degree=degree,
    )
    z, coeffs = sim.compute_impedance()
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(coeffs).all()
    # Half-wave dipole NEC reference: 69.64 - j18.21
    assert abs(z.real - 69.64) < 1.0, f"R={z.real}"
    assert abs(z.imag - (-18.21)) < 1.0, f"X={z.imag}"


@pytest.mark.parametrize("nsegs", [21, 41, 101])
def test_sinusoidal_dipole_matches_nec2(nsegs):
    """SinusoidalPySim implements NEC2's three-term basis (Eqs 43-64 of the
    LLNL theory manual). On a straight dipole it should match PyNEC/nec2c
    to <0.1 Ohm — the only differences are floating-point and quadrature.
    """
    wires = [np.array([[0.0, 0.0, -5.291], [0.0, 0.0, 5.291]], dtype=float)]
    sim = SinusoidalPySim(
        wires=wires,
        n_per_edge_per_wire=[[nsegs]],
        wavelength=22.0,
        wire_radius=0.0005,
        nsegs=nsegs,
    )
    z, _ = sim.compute_impedance()
    # NEC2 reference at this geometry (docs/convergence_analysis.md):
    # 69.69 - j18.67 at N=21, 69.64 - j18.21 at N=101.
    assert 69.5 < z.real < 69.8, f"R={z.real}"
    assert -19.0 < z.imag < -17.5, f"X={z.imag}"


def test_sinusoidal_hentenna_reproduces_pynec():
    """Sinusoidal-basis pysim reproduces PyNEC's hentenna numbers to
    ~0.05 Ohm. Validates the K=2/K=3 junction-basis path against the
    NEXT_STEPS.md item 13 PyNEC reference.
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps, z_mid)
    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    junctions = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    n = 21
    # Sinusoidal basis: ODD n_feed parity → delta-gap segment centred at z=0.
    nfeed = 3
    sim = SinusoidalPySim(
        wires=wires,
        n_per_edge_per_wire=[[nfeed], [n], [n, n, n], [n], [n, n, n]],
        feed_wire_index=0,
        feed_arclength=eps,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    )
    z, _ = sim.compute_impedance()
    # PyNEC reference at n=21 (NEXT_STEPS.md item 13): 45.604 - j4.604.
    assert abs(z.real - 45.604) < 0.1, f"R={z.real}"
    assert abs(z.imag - (-4.604)) < 0.1, f"X={z.imag}"


def test_triangular_fandipole_two_band_smoke():
    """Two-band fan dipole (cone arrangement from antenna_designer) modelled
    with pysim junctions at S and T. Verifies the K=3 path runs, converges,
    and produces plausible Z near the design freq (resonant 20m band).
    """
    import math

    C_LIGHT = 299_792_458.0
    band_lengths = [10.2551, 5.2691]
    slope = 0.5
    cone_radius = 0.12
    t0 = cone_radius * math.sqrt(2.0)
    eps = 0.01
    Zc = 1.0 / math.sqrt(1.0 + slope**2)
    Zs = slope * Zc
    S = (0.0, eps, 0.0)
    T = (0.0, -eps, 0.0)
    C = (S[0], S[1] + t0 * Zc, S[2] - t0 * Zs)
    lst = [
        (math.cos(math.pi * i / 180), math.sin(math.pi * i / 180))
        for i in range(36, 360, 72)
    ][:2]
    A_pos = [
        (
            C[0] + cone_radius * x,
            C[1] + cone_radius * y * Zs,
            C[2] + cone_radius * y * Zc,
        )
        for (x, y) in lst
    ]
    ls = [
        band_lengths[i] / 2 - math.sqrt(sum((s - a) ** 2 for s, a in zip(S, A_pos[i])))
        for i in range(2)
    ]
    B_pos = [(a[0], a[1] + l * Zc, a[2] - l * Zs) for l, a in zip(ls, A_pos)]
    A_neg = [(a[0], -a[1], a[2]) for a in A_pos]
    B_neg = [(b[0], -b[1], b[2]) for b in B_pos]

    N = 21
    wires = [np.array([T, S], dtype=float)]
    n_per_edge = [[2]]
    for i in range(2):
        wires.append(np.array([S, A_pos[i], B_pos[i]], dtype=float))
        n_per_edge.append([N, N])
    for i in range(2):
        wires.append(np.array([T, A_neg[i], B_neg[i]], dtype=float))
        n_per_edge.append([N, N])
    junctions = [
        [(0, "end"), (1, "start"), (2, "start")],  # at S
        [(0, "start"), (3, "start"), (4, "start")],  # at T
    ]
    for fmhz in [14.3, 28.47]:
        wavelength = C_LIGHT / (fmhz * 1e6)
        sim = TriangularPySim(
            wires=wires,
            n_per_edge_per_wire=n_per_edge,
            feed_wire_index=0,
            feed_arclength=eps,
            wavelength=wavelength,
            nsegs=N,
            wire_radius=0.0005,
            junctions=junctions,
        )
        z, coeffs = sim.compute_impedance()
        assert np.isfinite(z.real) and np.isfinite(z.imag)
        assert np.isfinite(coeffs).all()
        # Triangular Galerkin on this multi-wire cone topology lands ~60+j0
        # at the design freqs; PyNEC pulse basis gives ~46+j0 — the gap is
        # the basis-shape difference at K=3 junctions. The smoke window
        # below tolerates both solvers' typical answers.
        assert 30.0 < z.real < 90.0, f"f={fmhz}: R={z.real} out of plausible range"
        assert -50.0 < z.imag < 60.0, f"f={fmhz}: X={z.imag} out of plausible range"


def _fandipole_two_band_sim(N, wavelength):
    """Helper: build the same K=3 two-band fan dipole used in the smoke test.
    Returned simulator has junctions=[S, T] each connecting 3 wire ends.
    """
    import math

    band_lengths = [10.2551, 5.2691]
    slope = 0.5
    cone_radius = 0.12
    t0 = cone_radius * math.sqrt(2.0)
    eps = 0.01
    Zc = 1.0 / math.sqrt(1.0 + slope**2)
    Zs = slope * Zc
    S = (0.0, eps, 0.0)
    T = (0.0, -eps, 0.0)
    C = (S[0], S[1] + t0 * Zc, S[2] - t0 * Zs)
    lst = [
        (math.cos(math.pi * i / 180), math.sin(math.pi * i / 180))
        for i in range(36, 360, 72)
    ][:2]
    A_pos = [
        (
            C[0] + cone_radius * x,
            C[1] + cone_radius * y * Zs,
            C[2] + cone_radius * y * Zc,
        )
        for (x, y) in lst
    ]
    ls = [
        band_lengths[i] / 2 - math.sqrt(sum((s - a) ** 2 for s, a in zip(S, A_pos[i])))
        for i in range(2)
    ]
    B_pos = [(a[0], a[1] + l * Zc, a[2] - l * Zs) for l, a in zip(ls, A_pos)]
    A_neg = [(a[0], -a[1], a[2]) for a in A_pos]
    B_neg = [(b[0], -b[1], b[2]) for b in B_pos]

    wires = [np.array([T, S], dtype=float)]
    n_per_edge = [[2]]
    for i in range(2):
        wires.append(np.array([S, A_pos[i], B_pos[i]], dtype=float))
        n_per_edge.append([N, N])
    for i in range(2):
        wires.append(np.array([T, A_neg[i], B_neg[i]], dtype=float))
        n_per_edge.append([N, N])
    junctions = [
        [(0, "end"), (1, "start"), (2, "start")],
        [(0, "start"), (3, "start"), (4, "start")],
    ]
    return TriangularPySim(
        wires=wires,
        n_per_edge_per_wire=n_per_edge,
        feed_wire_index=0,
        feed_arclength=eps,
        wavelength=wavelength,
        nsegs=N,
        wire_radius=0.0005,
        junctions=junctions,
    )


def test_assemble_Z_general_cpp_matches_python():
    """C++ assemble_Z_general must agree bit-for-bit with the pure-Python
    reference path on a K=3-junction fan dipole. Drives the same J tensors
    through both paths so any kernel divergence shows up here as ULP-level
    error.
    """
    from pysim import triangular as _trimod
    from pysim._accelerators import assemble_Z_general as _cpp_general

    C_LIGHT = 299_792_458.0
    sim = _fandipole_two_band_sim(N=11, wavelength=C_LIGHT / 14.3e6)
    k_array = 2 * np.pi * np.array([12.0e6, 14.3e6, 18.0e6]) / C_LIGHT
    omega_array = k_array * sim.c

    geom = sim._build_geometry()
    tangents = geom["tangents"]
    td_all = tangents @ tangents.T
    J00, J10, J01, J11 = sim._build_J_blocks_batch(geom, k_array)

    Z_py = sim._assemble_Z_general_batch_python(
        J00, J10, J01, J11, td_all, geom, omega_array
    )
    Z_cpp = _cpp_general(
        J00,
        J10,
        J01,
        J11,
        np.ascontiguousarray(geom["h_per_seg"], dtype=np.float64),
        np.ascontiguousarray(td_all, dtype=np.float64),
        np.ascontiguousarray(geom["support_seg"], dtype=np.int64),
        np.ascontiguousarray(geom["support_L"], dtype=np.float64),
        np.ascontiguousarray(geom["support_R"], dtype=np.float64),
        np.ascontiguousarray(omega_array, dtype=np.float64),
        float(sim.eps),
        float(sim.mu),
    )
    np.testing.assert_allclose(Z_cpp, Z_py, rtol=1e-12, atol=1e-12)
    # Also exercise the dispatched accelerator-vs-python branch explicitly.
    saved = _trimod._HAVE_ASSEMBLE_Z_GENERAL
    try:
        _trimod._HAVE_ASSEMBLE_Z_GENERAL = False
        Z_dispatch_py = sim._assemble_Z_general_batch(
            J00, J10, J01, J11, td_all, geom, omega_array
        )
    finally:
        _trimod._HAVE_ASSEMBLE_Z_GENERAL = saved
    Z_dispatch_cpp = sim._assemble_Z_general_batch(
        J00, J10, J01, J11, td_all, geom, omega_array
    )
    np.testing.assert_allclose(Z_dispatch_cpp, Z_dispatch_py, rtol=1e-12, atol=1e-12)


def test_triangular_fandipole_swept_matches_per_freq():
    """Batched K=3-junction solve must agree with per-frequency solves to
    roundoff. Catches any regression in the C++ general-assembly path.
    """
    C_LIGHT = 299_792_458.0
    freqs_mhz = np.array([12.0, 14.3, 21.0, 28.47])
    k_array = 2 * np.pi * freqs_mhz * 1e6 / C_LIGHT
    sim_sweep = _fandipole_two_band_sim(N=11, wavelength=22.0)
    z_swept = sim_sweep.compute_impedance_swept(k_array)
    for f, zs in zip(freqs_mhz, z_swept):
        sim_f = _fandipole_two_band_sim(N=11, wavelength=C_LIGHT / (f * 1e6))
        z_f, _ = sim_f.compute_impedance()
        assert abs(zs - z_f) < 1e-6, f"f={f} MHz: swept={zs}, single={z_f}"


# ---- PEC ground (image method) ----


def _h_dipole(L, h):
    return np.array([[0.0, -L / 2, h], [0.0, L / 2, h]])


def test_ground_none_matches_free_space_bit_exact():
    # ground_z=None must take the same code path as the no-argument case.
    L = 2 * 0.962 * 22 / 4
    poly = _h_dipole(L, 0.0)
    z_no, _ = TriangularPySim(
        wires=[poly], n_per_edge_per_wire=[[40]], nsegs=40
    ).compute_impedance()
    z_none, _ = TriangularPySim(
        wires=[poly], n_per_edge_per_wire=[[40]], nsegs=40, ground_z=None
    ).compute_impedance()
    assert z_no == z_none


def test_ground_horizontal_dipole_at_height_recovers_free_space():
    # As h -> infinity above PEC, the image vanishes and Z -> Z_free.
    L = 2 * 0.962 * 22 / 4
    N = 30
    z_free, _ = TriangularPySim(
        wires=[_h_dipole(L, 0.0)], n_per_edge_per_wire=[[N]], nsegs=N
    ).compute_impedance()
    z_high, _ = TriangularPySim(
        wires=[_h_dipole(L, 100.0)],  # ~5 wavelengths up
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    ).compute_impedance()
    # At ~5λ height the image is weak but not negligible — a couple of Ohms
    # of shift on R and X is expected.
    assert abs(z_high.real - z_free.real) < 2.0
    assert abs(z_high.imag - z_free.imag) < 3.0


def test_ground_horizontal_dipole_at_zero_height_shorts_out():
    # As h -> 0 above PEC, the anti-parallel image cancels the antenna and
    # the radiated power (and hence the input resistance) goes to zero.
    L = 2 * 0.962 * 22 / 4
    z_lo, _ = TriangularPySim(
        wires=[_h_dipole(L, 0.01)],
        n_per_edge_per_wire=[[40]],
        nsegs=40,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_lo.real) < 0.5  # essentially zero radiation resistance


def test_ground_swept_matches_single_freq_with_ground():
    L = 2 * 0.962 * 22 / 4
    N = 30
    h = 5.0
    sim = TriangularPySim(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    )
    z_single, _ = sim.compute_impedance()
    z_swept = sim.compute_impedance_swept(np.array([sim.k]))[0]
    assert abs(z_single - z_swept) < 1e-9
