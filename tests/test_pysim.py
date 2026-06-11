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

import numpy as np


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


def test_triangular_y_matrix_with_junctions_single_feed():
    """compute_y_matrix on a K=2-junction antenna with a single feed should
    return [[1/Z]] where Z is what compute_impedance reports. Tests that the
    new junction path through _solve_with_kcl_ports doesn't regress against
    the existing single-port KCL solve."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        nsegs=15,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )
    sim = TriangularPySim(**common)
    z, _ = sim.compute_impedance()
    Y = TriangularPySim(**common).compute_y_matrix()
    assert Y.shape == (1, 1)
    assert abs(Y[0, 0] - 1.0 / z) < 1e-10, f"Y[0,0]={Y[0, 0]}, 1/Z={1.0 / z}"


def test_triangular_y_matrix_with_junctions_multi_feed():
    """compute_y_matrix on a K=2-junction antenna with two feeds (one per
    wire) should equal the N-independent-solves reference (drive each port
    with V=1 and read coeffs at every feed's basis index)."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wavelength=22,
        nsegs=15,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )

    Y = TriangularPySim(**common).compute_y_matrix()
    assert Y.shape == (2, 2)
    assert abs(Y[0, 1] - Y[1, 0]) < 1e-10, "Y not symmetric (reciprocity)"

    Y_ref = np.zeros((2, 2), dtype=np.complex128)
    for j in range(2):
        feeds_j = [
            (w, arc, 1.0 + 0j if k == j else 0.0 + 0j)
            for k, (w, arc, _) in enumerate(common["feeds"])
        ]
        ref_kwargs = {**common, "feeds": feeds_j}
        sim_j = TriangularPySim(**ref_kwargs)
        _z, coeffs = sim_j.compute_impedance()
        m_indices = sim_j._feed_basis_indices(sim_j._build_geometry())
        Y_ref[:, j] = [coeffs[m] for m in m_indices]
    assert np.allclose(Y, Y_ref, atol=1e-10), f"Y - Y_ref:\n{Y - Y_ref}"


def test_triangular_y_matrix_swept_with_junctions_matches_per_freq():
    """Batched swept Y matrix with junctions should match per-frequency Y
    matrices computed via compute_y_matrix."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        nsegs=15,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )
    C_LIGHT = 299_792_458.0
    freqs_mhz = np.array([10.0, 14.0, 20.0])
    k_array = 2 * np.pi * freqs_mhz * 1e6 / C_LIGHT

    sim_swept = TriangularPySim(wavelength=22, **common)
    Y_swept = sim_swept.compute_y_matrix_swept(k_array)
    assert Y_swept.shape == (3, 2, 2)

    for i, f in enumerate(freqs_mhz):
        sim_f = TriangularPySim(wavelength=C_LIGHT / (f * 1e6), **common)
        Y_f = sim_f.compute_y_matrix()
        assert np.allclose(Y_swept[i], Y_f, atol=1e-10), (
            f"f={f}: swept Y differs from per-k Y"
        )


def test_sinusoidal_y_matrix_with_junctions_single_feed():
    """SinusoidalPySim handles junctions structurally (via the N±(i) neighbour
    topology with σ=±1 signs) rather than via a Lagrange-augmented KCL, so
    compute_y_matrix doesn't need a separate junction path. Lock that in:
    Y[0,0] = 1/Z on a K=2-junction bent dipole."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )
    z, _ = SinusoidalPySim(**common).compute_impedance()
    Y = SinusoidalPySim(**common).compute_y_matrix()
    assert Y.shape == (1, 1)
    assert abs(Y[0, 0] - 1.0 / z) < 1e-9, f"Y[0,0]={Y[0, 0]}, 1/Z={1.0 / z}"


def test_sinusoidal_y_matrix_with_junctions_multi_feed():
    """Two feeds (one per wire) sharing a K=2 junction. Y should be near-
    symmetric (sinusoidal's MoM isn't quite as tight as triangular's at
    fixed segmentation, so use a looser tolerance) and match the N-solve
    reference."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wavelength=22,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )

    Y = SinusoidalPySim(**common).compute_y_matrix()
    assert Y.shape == (2, 2)
    assert abs(Y[0, 1] - Y[1, 0]) < 1e-6, "Y not symmetric within MoM tolerance"

    Y_ref = np.zeros((2, 2), dtype=np.complex128)
    for j in range(2):
        feeds_j = [
            (w, arc, 1.0 + 0j if k == j else 0.0 + 0j)
            for k, (w, arc, _) in enumerate(common["feeds"])
        ]
        ref_kwargs = {**common, "feeds": feeds_j}
        sim_j = SinusoidalPySim(**ref_kwargs)
        _z, alpha = sim_j.compute_impedance()
        geom = sim_j._build_geometry()
        _G, seg_view = sim_j._assemble_Z(geom, sim_j.k)
        Y_ref[:, j] = [
            sim_j._feed_segment_current(alpha, seg_view, fi) for fi in geom["feed_segs"]
        ]
    assert np.allclose(Y, Y_ref, atol=1e-10), f"Y - Y_ref:\n{Y - Y_ref}"


def test_sinusoidal_y_matrix_swept_with_junctions_matches_per_freq():
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
    )
    C_LIGHT = 299_792_458.0
    freqs_mhz = np.array([10.0, 14.0, 20.0])
    k_array = 2 * np.pi * freqs_mhz * 1e6 / C_LIGHT

    sim_swept = SinusoidalPySim(wavelength=22, **common)
    Y_swept = sim_swept.compute_y_matrix_swept(k_array)
    assert Y_swept.shape == (3, 2, 2)

    for i, f in enumerate(freqs_mhz):
        sim_f = SinusoidalPySim(wavelength=C_LIGHT / (f * 1e6), **common)
        Y_f = sim_f.compute_y_matrix()
        assert np.allclose(Y_swept[i], Y_f, atol=1e-10), (
            f"f={f}: swept Y differs from per-k Y"
        )


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_y_matrix_with_junctions_single_feed(degree):
    """compute_y_matrix on a K=2-junction antenna with a single feed should
    return [[1/Z]] where Z is what compute_impedance reports. Bspline uses
    the Lagrange-augmented KCL identically to triangular but with the
    Galerkin reciprocity Y = B^T X readout. Covers both d=1 (tent-equivalent)
    and d=2 (the default quadratic) bases."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feed_wire_index=0,
        feed_arclength=2.5,
        wavelength=22,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
        degree=degree,
    )
    z, _ = BSplinePySim(**common).compute_impedance()
    Y = BSplinePySim(**common).compute_y_matrix()
    assert Y.shape == (1, 1)
    assert abs(Y[0, 0] - 1.0 / z) < 1e-10, (
        f"d={degree}: Y[0,0]={Y[0, 0]}, 1/Z={1.0 / z}"
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_y_matrix_with_junctions_multi_feed(degree):
    """Two feeds on a K=2-junction antenna. Y should be symmetric and match
    the N-independent-solves reference at both d=1 and d=2."""
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wavelength=22,
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
        degree=degree,
    )

    Y = BSplinePySim(**common).compute_y_matrix()
    assert Y.shape == (2, 2)
    assert abs(Y[0, 1] - Y[1, 0]) < 1e-10, f"d={degree}: Y not symmetric (reciprocity)"

    # Reference: N independent solves, Y[:, j] = B^T @ coeffs_j.
    Y_ref = np.zeros((2, 2), dtype=np.complex128)
    for j in range(2):
        feeds_j = [
            (w, arc, 1.0 + 0j if k == j else 0.0 + 0j)
            for k, (w, arc, _) in enumerate(common["feeds"])
        ]
        sim_j = BSplinePySim(**{**common, "feeds": feeds_j})
        _z, coeffs = sim_j.compute_impedance()
        geom = sim_j._build_geometry()
        supp_seg, polys, _kcl, wk, wbg = sim_j._build_basis_polynomials(geom)
        n_bt = supp_seg.shape[0]
        for i, (w_i, arc_i, _v) in enumerate(common["feeds"]):
            arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
            s_f = arc_i if arc_i is not None else arc_at_knot[-1] / 2.0
            v_i = sim_j._build_source_vector(geom, wk, wbg, n_bt, wi=w_i, s_f=s_f)
            Y_ref[i, j] = v_i @ coeffs
    assert np.allclose(Y, Y_ref, atol=1e-10), f"d={degree}: Y - Y_ref:\n{Y - Y_ref}"


@pytest.mark.parametrize("degree", [1, 2])
def test_bspline_y_matrix_swept_with_junctions_matches_per_freq(degree):
    pl0 = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, -2.0]])
    pl1 = np.array([[0.0, 0.0, -2.0], [0.0, 5.0, 0.0]])
    common = dict(
        wires=[pl0, pl1],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wire_radius=0.0005,
        junctions=[[(0, "end"), (1, "start")]],
        degree=degree,
    )
    C_LIGHT = 299_792_458.0
    freqs_mhz = np.array([10.0, 14.0, 20.0])
    k_array = 2 * np.pi * freqs_mhz * 1e6 / C_LIGHT

    sim_swept = BSplinePySim(wavelength=22, **common)
    Y_swept = sim_swept.compute_y_matrix_swept(k_array)
    assert Y_swept.shape == (3, 2, 2)

    for i, f in enumerate(freqs_mhz):
        sim_f = BSplinePySim(wavelength=C_LIGHT / (f * 1e6), **common)
        Y_f = sim_f.compute_y_matrix()
        assert np.allclose(Y_swept[i], Y_f, atol=1e-10), (
            f"d={degree}, f={f}: swept Y differs from per-k Y"
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


def test_bspline_cpp_assemble_z_matches_numpy():
    """The C++ assemble_Z_bspline accelerator must agree with the numpy
    reference path bit-exactly (modulo floating-point reduction order ~1e-12
    relative). Run by toggling the dispatch flag in pysim.bspline.
    """
    import pysim.bspline as bmod
    from pysim.bspline import BSplinePySim

    if not bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL:
        pytest.skip("Z assembly accelerator not built")

    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    sim = BSplinePySim(wires=wires, n_per_edge_per_wire=[[21]], nsegs=21, degree=2)

    # Driver impedance via C++ path
    z_cpp, _ = sim.compute_impedance()

    # Force numpy fallback by flipping the module flag
    saved = bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL
    try:
        bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL = False
        # Re-build sim to invalidate any cached Z (compute_impedance is stateless re Z)
        z_np, _ = BSplinePySim(
            wires=wires, n_per_edge_per_wire=[[21]], nsegs=21, degree=2
        ).compute_impedance()
    finally:
        bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL = saved

    rel = abs(z_cpp - z_np) / abs(z_np)
    assert rel < 1e-12, f"C++ vs numpy Z assembly disagreement: rel diff {rel}"


def test_bspline_cpp_kernel_matches_numpy():
    """The C++ B-spline moment-integral and static-moments accelerators must
    agree with the pure-numpy reference to ~1e-9 relative (the floating-point
    reduction-order error is below GL quadrature precision and the closed-form
    arithmetic precision).
    """
    from pysim._bspline_kernels import (
        _seg_seg_full_moments_offedge,
        _seg_seg_static_moments,
        _HAVE_BSPLINE_ACCEL,
        _HAVE_BSPLINE_STATIC_ACCEL,
    )

    if not _HAVE_BSPLINE_ACCEL or not _HAVE_BSPLINE_STATIC_ACCEL:
        pytest.skip("C++ accelerators not built")

    # Force the numpy reference paths via monkeypatching the module-level flags
    import pysim._bspline_kernels as kmod

    N = 12
    seg = np.linspace(0, 6.0, N + 1)
    seg_l = np.column_stack([np.zeros(N), seg[:-1], np.zeros(N)])
    seg_r = np.column_stack([np.zeros(N), seg[1:], np.zeros(N)])

    J_cpp = _seg_seg_full_moments_offedge(seg_l, seg_r, seg_l, seg_r, 0.0005, 0.3, 2, 4)
    S_cpp = _seg_seg_static_moments(seg, 0.0005, max_d=2)

    # Force numpy paths
    saved_full = kmod._HAVE_BSPLINE_ACCEL
    saved_static = kmod._HAVE_BSPLINE_STATIC_ACCEL
    try:
        kmod._HAVE_BSPLINE_ACCEL = False
        kmod._HAVE_BSPLINE_STATIC_ACCEL = False
        J_np = _seg_seg_full_moments_offedge(
            seg_l, seg_r, seg_l, seg_r, 0.0005, 0.3, 2, 4
        )
        S_np = _seg_seg_static_moments(seg, 0.0005, max_d=2)
    finally:
        kmod._HAVE_BSPLINE_ACCEL = saved_full
        kmod._HAVE_BSPLINE_STATIC_ACCEL = saved_static

    rel_J = np.max(np.abs(J_cpp - J_np) / (np.abs(J_np) + 1e-30))
    rel_S = np.max(np.abs(S_cpp - S_np) / (np.abs(S_np) + 1e-30))
    # Both paths evaluate sympy-derived closed forms but through different
    # math libraries (numpy / libm), so float-precision differences in
    # arcsinh / sqrt cause ~1e-9 relative deviation. Well below the
    # GL-quadrature precision and the antenna-Z noise floor.
    assert rel_J < 1e-8, f"J kernel rel diff {rel_J}"
    assert rel_S < 1e-7, f"Static kernel rel diff {rel_S}"


def test_bspline_cpp_degree1_matches_numpy():
    """Exercise the C++ degree=1 (max_d=1) template instantiations of the
    B-spline accelerators -- the moment kernel<1>, the static moments at
    max_d=1, and assemble_Z_bspline_kernel<1> -- which the degree=2 tests
    never reach. Each must agree with the pure-numpy reference path.
    """
    import pysim.bspline as bmod
    from pysim.bspline import BSplinePySim
    import pysim._bspline_kernels as kmod
    from pysim._bspline_kernels import (
        _seg_seg_full_moments_offedge,
        _seg_seg_static_moments,
    )

    if not (
        kmod._HAVE_BSPLINE_ACCEL
        and kmod._HAVE_BSPLINE_STATIC_ACCEL
        and bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL
    ):
        pytest.skip("B-spline C++ accelerators not built")

    # --- moment kernel<1> + static moments at max_d=1 (off-edge + same-edge) --
    N = 12
    seg = np.linspace(0.0, 6.0, N + 1)
    seg_l = np.column_stack([np.zeros(N), seg[:-1], np.zeros(N)])
    seg_r = np.column_stack([np.zeros(N), seg[1:], np.zeros(N)])

    J_cpp = _seg_seg_full_moments_offedge(seg_l, seg_r, seg_l, seg_r, 0.0005, 0.3, 1, 4)
    S_cpp = _seg_seg_static_moments(seg, 0.0005, max_d=1)
    # max_d=1 -> (d+1, d+1) = (2, 2) leading dims
    assert J_cpp.shape[:2] == (2, 2)
    assert S_cpp.shape[:2] == (2, 2)

    saved_full = kmod._HAVE_BSPLINE_ACCEL
    saved_static = kmod._HAVE_BSPLINE_STATIC_ACCEL
    try:
        kmod._HAVE_BSPLINE_ACCEL = False
        kmod._HAVE_BSPLINE_STATIC_ACCEL = False
        J_np = _seg_seg_full_moments_offedge(
            seg_l, seg_r, seg_l, seg_r, 0.0005, 0.3, 1, 4
        )
        S_np = _seg_seg_static_moments(seg, 0.0005, max_d=1)
    finally:
        kmod._HAVE_BSPLINE_ACCEL = saved_full
        kmod._HAVE_BSPLINE_STATIC_ACCEL = saved_static

    rel_J = np.max(np.abs(J_cpp - J_np) / (np.abs(J_np) + 1e-30))
    rel_S = np.max(np.abs(S_cpp - S_np) / (np.abs(S_np) + 1e-30))
    assert rel_J < 1e-8, f"degree=1 J kernel rel diff {rel_J}"
    assert rel_S < 1e-7, f"degree=1 static kernel rel diff {rel_S}"

    # --- assemble_Z_bspline_kernel<1> via a full degree=1 dipole solve --------
    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]
    z_cpp, _ = BSplinePySim(
        wires=wires, n_per_edge_per_wire=[[21]], nsegs=21, degree=1
    ).compute_impedance()

    saved_asm = bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL
    try:
        bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL = False
        z_np, _ = BSplinePySim(
            wires=wires, n_per_edge_per_wire=[[21]], nsegs=21, degree=1
        ).compute_impedance()
    finally:
        bmod._HAVE_BSPLINE_ASSEMBLE_ACCEL = saved_asm

    rel_z = abs(z_cpp - z_np) / abs(z_np)
    assert rel_z < 1e-12, f"degree=1 Z assembly C++ vs numpy rel diff {rel_z}"


def test_triangular_reg_and_offedge_numpy_matches_cpp():
    """The C++ same-edge-regularized (seg_seg_reg_quad_batch_1d) and cross-edge
    (seg_seg_quad_batch_3d) accelerators must agree with the pure-numpy
    reference paths in _triangular_kernels to ~1e-9 relative. Toggling the
    _HAVE_*_ACCEL flags also exercises the numpy fallbacks themselves (taken
    on platforms where the extension isn't built).
    """
    import pysim._triangular_kernels as tk
    from pysim._triangular_kernels import (
        _seg_seg_reg_all_batch,
        _seg_seg_offedge_quad_batch,
    )

    if not (tk._HAVE_REG_ACCEL and tk._HAVE_OFF_ACCEL):
        pytest.skip("triangular C++ accelerators not built")

    a = 0.0005
    k_array = np.array([0.3, 0.7])
    n_qp = 4
    N = 8

    # same-edge regularized: arc-length endpoints along one straight edge
    seg = np.linspace(0.0, 4.0, N + 1)

    # cross-edge: two parallel edges offset 0.5 m in x, same y-extent
    yl = np.linspace(0.0, 4.0, N + 1)
    edge_i_l = np.column_stack([np.zeros(N), yl[:-1], np.zeros(N)])
    edge_i_r = np.column_stack([np.zeros(N), yl[1:], np.zeros(N)])
    edge_j_l = np.column_stack([np.full(N, 0.5), yl[:-1], np.zeros(N)])
    edge_j_r = np.column_stack([np.full(N, 0.5), yl[1:], np.zeros(N)])

    reg_cpp = _seg_seg_reg_all_batch(seg, a, k_array, n_qp)
    off_cpp = _seg_seg_offedge_quad_batch(
        edge_i_l, edge_i_r, edge_j_l, edge_j_r, a, k_array, n_qp
    )

    saved_reg, saved_off = tk._HAVE_REG_ACCEL, tk._HAVE_OFF_ACCEL
    try:
        tk._HAVE_REG_ACCEL = False
        tk._HAVE_OFF_ACCEL = False
        reg_np = _seg_seg_reg_all_batch(seg, a, k_array, n_qp)
        off_np = _seg_seg_offedge_quad_batch(
            edge_i_l, edge_i_r, edge_j_l, edge_j_r, a, k_array, n_qp
        )
    finally:
        tk._HAVE_REG_ACCEL = saved_reg
        tk._HAVE_OFF_ACCEL = saved_off

    # --- kink-corner pair: two edges sharing an endpoint at a 90deg angle.   --
    # Edge i lies on -y, edge j on +x; the right end of i[-1] coincides with
    # the left end of j[0] at the origin. This exercises the a^2 regularization
    # in seg_seg_quad_batch_3d on the touching pair where unregularized R
    # would vanish at the shared corner, in isolation from the assembly-level
    # tests that normally cover this case.
    Nk = 4
    yk = np.linspace(-2.0, 0.0, Nk + 1)
    xk = np.linspace(0.0, 2.0, Nk + 1)
    kink_i_l = np.column_stack([np.zeros(Nk), yk[:-1], np.zeros(Nk)])
    kink_i_r = np.column_stack([np.zeros(Nk), yk[1:], np.zeros(Nk)])
    kink_j_l = np.column_stack([xk[:-1], np.zeros(Nk), np.zeros(Nk)])
    kink_j_r = np.column_stack([xk[1:], np.zeros(Nk), np.zeros(Nk)])
    kink_cpp = _seg_seg_offedge_quad_batch(
        kink_i_l, kink_i_r, kink_j_l, kink_j_r, a, k_array, n_qp
    )
    try:
        tk._HAVE_OFF_ACCEL = False
        kink_np = _seg_seg_offedge_quad_batch(
            kink_i_l, kink_i_r, kink_j_l, kink_j_r, a, k_array, n_qp
        )
    finally:
        tk._HAVE_OFF_ACCEL = saved_off

    for label, cpp, npref in [
        ("reg", reg_cpp, reg_np),
        ("offedge-parallel", off_cpp, off_np),
        ("offedge-kink-corner", kink_cpp, kink_np),
    ]:
        for idx, (jc, jn) in enumerate(zip(cpp, npref)):
            # Corner pair must be finite (the a^2 regularization keeps it so).
            assert np.isfinite(jc).all(), f"{label} J[{idx}] C++ not finite"
            assert np.isfinite(jn).all(), f"{label} J[{idx}] numpy not finite"
            rel = np.max(np.abs(jc - jn) / (np.abs(jn) + 1e-30))
            assert rel < 1e-9, f"{label} J[{idx}] C++ vs numpy rel diff {rel}"


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


def test_bspline_d2_dipole_smoothed_source():
    """Source smoothing via `feed_smoothing_factor` (α) replaces the delta
    gap with a cos² bump of width w = α·h_feed_segment, integrated against
    each basis. On the dipole this removes the source-localized current
    singularity that otherwise caps the integrated-impedance convergence
    at O(1/N) regardless of basis degree.

    Two checks:
      1. Pin α=4 at n=81 to the recorded productized value.
      2. Fit R(N) = R_inf + C/N^p over n ∈ {21, 41, 81} with α=2 and assert
         the rate clearly lifts above the delta-gap baseline (empirically
         the baseline runs ~1.20; α=2 lifts to ~1.55).
    """
    L = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])]

    def sweep(alpha, ns):
        Zs = []
        for n in ns:
            z, _ = BSplinePySim(
                wires=wires,
                n_per_edge_per_wire=[[n]],
                nsegs=n,
                degree=2,
                feed_smoothing_factor=alpha,
            ).compute_impedance()
            Zs.append(z)
        return Zs

    # 1. Pin α=4 at n=81 (smoothing-on converged value; ±0.1 Ω R, ±0.5 Ω X
    #    leaves headroom for compiler/platform jitter while staying tighter
    #    than the gap to the delta-gap baseline at the same n).
    z = sweep(4.0, [81])[0]
    assert abs(z.real - 69.78) < 0.1, f"α=4 n=81 R={z.real}, expected ≈69.78"
    assert abs(z.imag - (-18.02)) < 0.5, f"α=4 n=81 X={z.imag}, expected ≈-18.02"

    # 2. R-rate lift at α=2 vs delta-gap baseline.
    ns = [21, 41, 81]
    Rs_delta = [zz.real for zz in sweep(None, ns)]
    Rs_a2 = [zz.real for zz in sweep(2.0, ns)]

    def rate(vals):
        d12 = vals[0] - vals[1]
        d23 = vals[1] - vals[2]
        assert d12 * d23 > 0, f"R differences sign-flipped (noise floor): {vals}"
        return np.log(abs(d12 / d23)) / np.log(ns[1] / ns[0])

    p_delta = rate(Rs_delta)
    p_a2 = rate(Rs_a2)
    # Floor of 1.4 has margin vs the empirical α=2 rate of ~1.55; the
    # +0.2 lift assertion catches a silent regression where α=2 happens
    # to land near 69.78 without actually improving the rate.
    assert p_a2 > 1.4, (
        f"α=2 R rate p={p_a2:.2f} below the 1.4 floor (delta p={p_delta:.2f})"
    )
    assert p_a2 > p_delta + 0.2, (
        f"α=2 did not clearly lift R rate vs delta-gap: {p_delta:.2f} → {p_a2:.2f}"
    )


def test_bspline_d2_hentenna_arbitrates_against_triangular():
    """Degree-2 B-spline on the hentenna converges to the SAME value as the
    triangular basis (within ~0.1 Ω), independently arbitrating the
    NEXT_STEPS items 13/14 question: triangular is NOT converged-to-the-
    wrong-place; NEC's three-term basis is the outlier that drifts super-log.

    Two independent basis families (degree-1 tent, degree-2 quadratic) land
    on the same impedance at the canonical n=21 hentenna sweep point.
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps_feed = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
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
    # tent and bspline both want EVEN nfeed (interior knot at z=0).
    nfeed = 2
    npe = [[nfeed], [n], [n, n, n], [n], [n, n, n]]
    z_tri, _ = TriangularPySim(
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    ).compute_impedance()
    z_b2, _ = BSplinePySim(
        degree=2,
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    ).compute_impedance()
    # Triangular at n=21: 43.158 + j38.027
    # B-spline d=2 at n=21: 43.066 + j38.849
    # The two converge to different small-N transients but agree at the
    # asymptote (~43.05 R, ~38.85 X at n=161) — they're independent
    # basis families that BOTH reject the NEC super-log drift.
    assert abs(z_tri.real - 43.16) < 0.1, f"tri R={z_tri.real}"
    assert abs(z_tri.imag - 38.03) < 0.1, f"tri X={z_tri.imag}"
    assert abs(z_b2.real - 43.07) < 0.1, f"bsp d=2 R={z_b2.real}"
    assert abs(z_b2.imag - 38.85) < 0.1, f"bsp d=2 X={z_b2.imag}"
    # Cross-basis disagreement bound. Post-PR-#51 the n=15..161 sweep
    # (scripts/compare_hentenna_solvers.py) pins the asymptote at 43.05 +
    # j38.84 for BOTH bases. At n=21 specifically tri and b2 differ by
    # ~0.09 Ω on R (b2 is essentially converged; tri still tightening
    # from its degree-1 O(1/N²) slope) and by ~0.82 Ω on X (where tri
    # has a larger small-N transient). Tighten R to 0.15 Ω; keep X at
    # 0.9 Ω — both leave ~15-30% headroom over the actual gap and would
    # fail loudly if either basis silently drifts off the arbitration
    # asymptote.
    assert abs(z_tri.real - z_b2.real) < 0.15, (
        f"basis disagreement on R: tri={z_tri.real}, bsp={z_b2.real}"
    )
    assert abs(z_tri.imag - z_b2.imag) < 0.9, (
        f"basis disagreement on X: tri={z_tri.imag}, bsp={z_b2.imag}"
    )


def test_bspline_d2_hentenna_singular_enrichment():
    """Singular basis enrichment at K≥3 junctions flips the hentenna R-rate
    from O(1/N) to ~O(1/N^(d+1)). Pin the converged R/X at n=81 and assert
    that the fitted convergence rate p in Z(N) = Z_inf + C/N^p satisfies
    p > 2.5 on R — both checks catch silent regressions.

    Reference values (productized C++ path, 2026-06):
      n=21  → 42.8574 + j44.2296
      n=41  → 43.0766 + j39.2185
      n=81  → 43.0858 + j38.9038
      n=161 → 43.0845 + j38.8749
    Rate fit on R over the four points gives p ≈ 2.74.
    """
    pytest.importorskip("pysim._accelerators")
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps_feed = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
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

    nfeed = 3
    ns = [21, 41, 81]
    Rs = []
    Xs = []
    for n in ns:
        npe = [[nfeed], [n], [n, n, n], [n], [n, n, n]]
        z, _ = BSplinePySim(
            degree=2,
            wires=wires,
            n_per_edge_per_wire=npe,
            feed_wire_index=0,
            feed_arclength=eps_feed,
            wavelength=wavelength,
            wire_radius=0.0005,
            nsegs=n,
            junctions=junctions,
            use_singular_enrichment=True,
        ).compute_impedance()
        Rs.append(z.real)
        Xs.append(z.imag)

    # Pin n=81 to the recorded productized value (5e-3 Ω tolerance leaves
    # headroom for compiler/platform jitter but catches any constant-offset
    # drift much smaller than the prototype's value-spread to non-enriched).
    assert abs(Rs[2] - 43.0866) < 5e-3, f"R(n=81)={Rs[2]}, expected ≈43.0866"
    assert abs(Xs[2] - 38.8740) < 5e-3, f"X(n=81)={Xs[2]}, expected ≈38.8740"

    # Fit Z = Z_inf + C/N^p on the X component over n in {21, 41, 81}.
    # (X used to be R, but with the enrichment-orig sign fix the R
    # convergence is fast enough that R hits the few-mΩ noise floor
    # between N=41 and N=81, making the sign-based rate estimator
    # unreliable. X still has a few tens of mΩ of headroom at these N
    # and converges monotonically.) Three-point Richardson-style:
    #   p ≈ log( (X(N1) - X(N2)) / (X(N2) - X(N3)) ) / log(N2/N1)
    # with N1 < N2 < N3. The same leading constant cancels.
    dX_12 = Xs[0] - Xs[1]
    dX_23 = Xs[1] - Xs[2]
    assert dX_12 * dX_23 > 0, (
        f"X differences sign-flipped — noise floor reached too early; Xs={Xs}"
    )
    p = np.log(abs(dX_12 / dX_23)) / np.log(ns[1] / ns[0])
    assert p > 2.5, f"X convergence rate p={p:.2f} below the 2.5 floor (Xs={Xs})"


def test_bspline_d2_hentenna_enrichment_stable_variant():
    """Pin the stable-XFEM hentenna asymptote and verify that:
      (a) at n=81 it lands within ~0.005 Ω of the raw variant — the two
          variants must converge to the same Z, only with different
          small-N transients;
      (b) at d=1 the stable variant is bit-exact to raw — the BC-preserving
          polynomial bubble subspace is empty for d=1 (P_1 ∩ {p(0)=p(1)=0}
          = {0}), so projection coeffs are all zero and Φ_sing_stable
          identically equals Φ_sing. This pins the "d=1 enrichment is a
          no-op" symmetry between variants.

    Reference values (productized C++ path, this branch, n=81):
      stable: 43.0864 + j38.8757
      raw   : 43.0866 + j38.8740
      diff  : ~0.0002 Ω R, ~0.0017 Ω X
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps_feed = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
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
    n = 81
    npe = [[3], [n], [n, n, n], [n], [n, n, n]]
    common = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
        use_singular_enrichment=True,
    )
    z_raw, _ = BSplinePySim(
        degree=2, enrichment_variant="raw", **common
    ).compute_impedance()
    z_stb, _ = BSplinePySim(
        degree=2, enrichment_variant="stable", **common
    ).compute_impedance()
    # (a) Both variants converge to the same Z; pin stable's value and the
    # raw-stable agreement so any future projection-coefficient drift fails
    # loudly. The tighter check is the raw-stable diff: 0.005 Ω covers
    # rounding noise but catches any sign / shape regression.
    assert abs(z_stb.real - 43.0864) < 5e-3
    assert abs(z_stb.imag - 38.8757) < 5e-3
    assert abs(z_raw - z_stb) < 5e-3

    # (b) At d=1 the bubble subspace is empty → stable = raw bit-exact.
    z1_raw, _ = BSplinePySim(
        degree=1, enrichment_variant="raw", **common
    ).compute_impedance()
    z1_stb, _ = BSplinePySim(
        degree=1, enrichment_variant="stable", **common
    ).compute_impedance()
    assert z1_raw == z1_stb


def test_bspline_d2_hentenna_enrichment_tikhonov_variant():
    """Pin the two limit cases of the tikhonov variant:
      (a) λ=0 reduces to raw bit-exact (the penalty disappears);
      (b) λ→∞ reduces to use_singular_enrichment=False (penalty
          dominates and α_enr → 0).

    These limits are what makes tikhonov a knob rather than a separate
    variant; if either limit drifts, the dimensionless scaling (λ·s with
    s = mean |diag(Z_ee)|) is wrong and the knob loses its meaning.
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    eps_feed = 0.05
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
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
    n = 21  # small enough to see the tikhonov effect; UI default
    npe = [[3], [n], [n, n, n], [n], [n, n, n]]
    common = dict(
        degree=2,
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    )
    z_raw, _ = BSplinePySim(
        **common, use_singular_enrichment=True, enrichment_variant="raw"
    ).compute_impedance()
    z_tik_zero, _ = BSplinePySim(
        **common,
        use_singular_enrichment=True,
        enrichment_variant="tikhonov",
        tikhonov_lambda=0.0,
    ).compute_impedance()
    z_off, _ = BSplinePySim(**common, use_singular_enrichment=False).compute_impedance()
    z_tik_big, _ = BSplinePySim(
        **common,
        use_singular_enrichment=True,
        enrichment_variant="tikhonov",
        tikhonov_lambda=1e6,
    ).compute_impedance()
    # (a) λ=0 → raw bit-exact
    assert z_raw == z_tik_zero
    # (b) λ→∞ → no-enrichment to ~1e-6 relative
    assert abs(z_tik_big - z_off) / abs(z_off) < 1e-6


def test_bspline_enrichment_auto_two_pass_selects_correctly():
    """The 'auto' variant runs a two-pass solve: pass 1 without enrichment
    measures tap_ratio = min(|I_wire|)/max(|I_wire|) at each K≥enrichment_min_k
    junction; pass 2 applies raw enrichment only at junctions where
    tap_ratio > auto_tap_ratio_threshold.

    Two canonical geometries pin the per-junction decision:
      (a) **Hentenna** — dominant-pair K=3 (tap_ratio ≈ 0.16): auto must
          select no junctions and the result must equal no-enrichment
          bit-exact (the +0.25 Ω X small-N transient that raw introduces
          is the regression this is meant to prevent).
      (b) **Y-fixture** — balanced 3-way K=3 (tap_ratio ≈ 0.50): auto
          must select the K=3 junction (index 1; the K=2 at index 0 is
          correctly excluded by enrichment_min_k=3) and the result must
          equal raw bit-exact (preserves the legitimate cusp).
    """
    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    eps_feed = 0.05

    # --- (a) Hentenna ---
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
    h_wires = [
        np.array([T, S], dtype=float),
        np.array([S, B_], dtype=float),
        np.array([B_, A, C_, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E_, F, B_], dtype=float),
    ]
    h_juncs = [
        [(0, "end"), (1, "start")],
        [(0, "start"), (3, "start")],
        [(1, "end"), (2, "start"), (4, "end")],
        [(2, "end"), (3, "end"), (4, "start")],
    ]
    n = 21
    h_kw = dict(
        degree=2,
        wires=h_wires,
        n_per_edge_per_wire=[[3], [n], [n, n, n], [n], [n, n, n]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=h_juncs,
    )
    z_h_off, _ = BSplinePySim(**h_kw, use_singular_enrichment=False).compute_impedance()
    sim_h_auto = BSplinePySim(
        **h_kw, use_singular_enrichment=True, enrichment_variant="auto"
    )
    z_h_auto, _ = sim_h_auto.compute_impedance()
    assert sim_h_auto._auto_active_junctions == []
    assert z_h_auto == z_h_off

    # --- (b) Y-fixture ---
    L = wavelength / 4.0
    T_ = (-eps_feed, 0.0, 0.0)
    S_ = (+eps_feed, 0.0, 0.0)
    a1 = (T_[0] - L, 0.0, 0.0)
    c60 = float(np.cos(np.pi / 3.0))
    s60 = float(np.sin(np.pi / 3.0))
    a2 = (S_[0] + L * c60, +L * s60, 0.0)
    a3 = (S_[0] + L * c60, -L * s60, 0.0)
    y_wires = [
        np.array([T_, S_], dtype=float),
        np.array([T_, a1], dtype=float),
        np.array([S_, a2], dtype=float),
        np.array([S_, a3], dtype=float),
    ]
    y_juncs = [
        [(0, "start"), (1, "start")],  # K=2 at T (skipped by enrichment_min_k=3)
        [(0, "end"), (2, "start"), (3, "start")],  # K=3 at S (the probe junction)
    ]
    n_y = 41
    y_kw = dict(
        degree=2,
        wires=y_wires,
        n_per_edge_per_wire=[[2], [n_y], [n_y], [n_y]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n_y,
        junctions=y_juncs,
        enrichment_min_k=3,
    )
    z_y_raw, _ = BSplinePySim(
        **y_kw, use_singular_enrichment=True, enrichment_variant="raw"
    ).compute_impedance()
    sim_y_auto = BSplinePySim(
        **y_kw, use_singular_enrichment=True, enrichment_variant="auto"
    )
    z_y_auto, _ = sim_y_auto.compute_impedance()
    assert sim_y_auto._auto_active_junctions == [1]
    assert z_y_auto == z_y_raw


def test_bspline_assemble_z_enrich_cpp_matches_numpy():
    """The C++ `assemble_Z_enrich` accelerator must agree with the
    pure-numpy reference across all four enrichment variants. The
    reference path is what runs on platforms where the Pybind11 extension
    isn't built (Windows — setup.py skips it because the GCC-only
    `-fopenmp` / `-mavx2` / `-lmvec` flags don't link under MSVC).

    Sweep:
      * Two geometries — hentenna (small-tap K=3, 2 enrichable junctions)
        and Y-fixture (balanced 3-way K=3, 1 enrichable junction). Together
        they exercise the multi-junction Z_ee block, the L-R symmetric
        enrichment pair (hentenna), and the auto-variant's per-junction
        filter (Y-fixture's K=2 junction must be excluded).
      * All four variants — raw / stable / tikhonov / auto. raw and stable
        differ in the kernel's `proj_coeffs` argument; tikhonov uses the
        raw kernel call but adds λ·I in Python; auto routes through raw
        with a junction filter and may take the pass-1-only path (which
        skips the kernel entirely — that path agrees trivially but is
        still worth exercising as a "no kernel call" case).

    1e-12 relative is well below GL quadrature precision and lets the
    two implementations differ only in floating-point reduction order
    (numpy's pairwise vs C++'s sequential).
    """
    import pysim.bspline as bmod

    # If the C++ extension wasn't built (Windows wheel, or local build
    # without pybind11), there's no C++ kernel to compare against. The
    # numpy reference is the only path; the parity sweep below would
    # NameError trying to call into the missing _acc.
    if not bmod._HAVE_ENRICH_ACCEL:
        pytest.skip("C++ assemble_Z_enrich not available — numpy-only build")

    C_LIGHT = 299_792_458.0
    freq_mhz = 28.47
    wavelength = C_LIGHT / (freq_mhz * 1e6)
    eps_feed = 0.05

    # Hentenna
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    half_w = wavelength * width_factor / 2
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
    h_kw = dict(
        degree=2,
        wires=[
            np.array([T, S], dtype=float),
            np.array([S, B_], dtype=float),
            np.array([B_, A, C_, D], dtype=float),
            np.array([T, D], dtype=float),
            np.array([D, E_, F, B_], dtype=float),
        ],
        n_per_edge_per_wire=[[3], [21], [21, 21, 21], [21], [21, 21, 21]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=21,
        junctions=[
            [(0, "end"), (1, "start")],
            [(0, "start"), (3, "start")],
            [(1, "end"), (2, "start"), (4, "end")],
            [(2, "end"), (3, "end"), (4, "start")],
        ],
    )

    # Y-fixture
    L = wavelength / 4.0
    T_ = (-eps_feed, 0.0, 0.0)
    S_ = (+eps_feed, 0.0, 0.0)
    a1 = (T_[0] - L, 0.0, 0.0)
    c60 = float(np.cos(np.pi / 3.0))
    s60 = float(np.sin(np.pi / 3.0))
    a2 = (S_[0] + L * c60, +L * s60, 0.0)
    a3 = (S_[0] + L * c60, -L * s60, 0.0)
    y_kw = dict(
        degree=2,
        wires=[
            np.array([T_, S_], dtype=float),
            np.array([T_, a1], dtype=float),
            np.array([S_, a2], dtype=float),
            np.array([S_, a3], dtype=float),
        ],
        n_per_edge_per_wire=[[2], [41], [41], [41]],
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=41,
        junctions=[
            [(0, "start"), (1, "start")],
            [(0, "end"), (2, "start"), (3, "start")],
        ],
        enrichment_min_k=3,
    )

    def run(kw, variant):
        z, _ = BSplinePySim(
            **kw,
            use_singular_enrichment=True,
            enrichment_variant=variant,
            tikhonov_lambda=0.1,
        ).compute_impedance()
        return z

    for label, kw in [("hentenna", h_kw), ("y-fixture", y_kw)]:
        for variant in ("raw", "stable", "tikhonov", "auto"):
            saved = bmod._HAVE_ENRICH_ACCEL
            try:
                bmod._HAVE_ENRICH_ACCEL = True
                z_cpp = run(kw, variant)
                bmod._HAVE_ENRICH_ACCEL = False
                z_np = run(kw, variant)
            finally:
                bmod._HAVE_ENRICH_ACCEL = saved
            rel = abs(z_cpp - z_np) / abs(z_cpp)
            assert rel < 1e-12, (
                f"{label} variant={variant}: C++ vs numpy enrich kernel "
                f"disagreement rel={rel:.2e} (cpp={z_cpp}, np={z_np})"
            )


def test_bspline_hentenna_enrichment_left_right_symmetry():
    """Hentenna is mirror-symmetric about y=0, so the BSpline+enrichment solve
    must produce mirror-symmetric per-knot currents on the upper and lower
    polylines. A sign bug in the enrichment-basis derivative — where
    `dΦ_sing/du_arc` was computed without the chain-rule sign flip for
    "end"-orientation bases (junction at the segment's right endpoint) —
    used to break this exact symmetry: junction-D current magnitudes were
    several percent off from their junction-B mirrors, with the residual
    only converging to zero as N→∞. Caught here at modest N so any
    re-introduction of the bug fails loudly.
    """
    pytest.importorskip("pysim._accelerators")
    design_freq_mhz = 28.47
    C_LIGHT = 299_792_458.0
    wavelength = C_LIGHT / (design_freq_mhz * 1e6)
    width_factor = 0.1378
    top_height_factor = 0.5081
    mid_height_factor = 0.1094
    half_w = wavelength * width_factor / 2.0
    eps_feed = 0.05
    z_mid = wavelength * (mid_height_factor - top_height_factor)
    z_bot = -wavelength * top_height_factor
    A = (0.0, half_w, 0.0)
    B_ = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C_ = (0.0, -half_w, 0.0)
    D = (0.0, -half_w, z_mid)
    E_ = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)
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
    npe = [[4], [n], [n, n, n], [n], [n, n, n]]
    sim = BSplinePySim(
        wires=wires,
        n_per_edge_per_wire=npe,
        feed_wire_index=0,
        feed_arclength=eps_feed,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        degree=2,
        junctions=junctions,
        use_singular_enrichment=True,
    )
    _, coeffs = sim.compute_impedance()
    currents = sim.currents_at_knots(coeffs)

    # The K=3 junction directional polynomial bases sit on the polyline
    # endpoint knots of `upper` and `lower`. By antenna L-R mirror, the
    # current magnitudes at D (left) must equal those at B (right) to
    # machine precision: ratio == 1 within ~1e-6.
    upper = currents[2]
    lower = currents[4]
    assert abs(abs(upper[0]) - abs(upper[-1])) / abs(upper[0]) < 1e-6, (
        f"upper |I| at B={abs(upper[0]):.6f} vs D={abs(upper[-1]):.6f} "
        "(L/R mirror should be exact)"
    )
    assert abs(abs(lower[0]) - abs(lower[-1])) / abs(lower[0]) < 1e-6, (
        f"lower |I| at D={abs(lower[0]):.6f} vs B={abs(lower[-1]):.6f} "
        "(L/R mirror should be exact)"
    )

    # Interior polynomial bases on `upper` near the two K=3 junctions
    # should also mirror. The pre-fix bug surfaced as ~5% asymmetry on the
    # interior basis adjacent to the directional one; cap at 0.01% here
    # so any re-introduction of the orig=1 sign flip fails immediately.
    for k in range(1, 6):
        ratio = abs(upper[k]) / abs(upper[-1 - k])
        assert abs(ratio - 1.0) < 1e-4, (
            f"upper knot pair (k={k}, n-1-{k}) ratio={ratio:.6f} — L/R asymmetry"
        )


def test_sinusoidal_field_tensor_cpp_matches_numpy():
    """The C++ `sinusoidal_field_tensor` accelerator and the pure-numpy
    reference path must produce bit-equivalent (Phi_const, Phi_sin, Phi_cos)
    on representative geometries. Anything looser than ~1e-13 relative
    indicates the C++ kernel diverged from the formula in sinusoidal.py.
    """
    import pysim.sinusoidal as sin_mod

    if not sin_mod._HAVE_FIELD_TENSOR:
        pytest.skip("C++ accelerator not built")

    # Bent two-edge polyline so the (m, n) loop sees varying tangents.
    wires = [np.array([[0.0, 0.0, -5.0], [0.5, 0.0, 0.0], [0.0, 0.0, 5.0]])]
    for n in (15, 41):
        sim = SinusoidalPySim(
            wires=wires,
            n_per_edge_per_wire=[[n, n]],
            wavelength=22.0,
            wire_radius=5e-4,
            nsegs=n,
        )
        geom = sim._build_geometry()
        # C++ path
        Pc_cpp, Ps_cpp, Pco_cpp = sim._field_tensor(geom, sim.k)
        # Numpy reference path
        sin_mod._HAVE_FIELD_TENSOR = False
        try:
            Pc_np, Ps_np, Pco_np = sim._field_tensor(geom, sim.k)
        finally:
            sin_mod._HAVE_FIELD_TENSOR = True
        for name, a_cpp, a_np in [
            ("Phi_const", Pc_cpp, Pc_np),
            ("Phi_sin", Ps_cpp, Ps_np),
            ("Phi_cos", Pco_cpp, Pco_np),
        ]:
            denom = max(np.max(np.abs(a_np)), 1e-30)
            rel = np.max(np.abs(a_cpp - a_np)) / denom
            assert rel < 1e-13, f"n={n} {name} max rel diff = {rel:.3e}"


def test_sinusoidal_hentenna_left_right_symmetry():
    """Hentenna is mirror-symmetric about y=0, so SinusoidalPySim's per-knot
    currents on the cross-bar halves (wires 1 and 3) and on the upper /
    lower rectangles (wires 2 and 4) must be mirror-symmetric to machine
    precision.

    Pre-fix bug: `currents_at_knots` multiplied the whole (A + B·sin +
    C·cos) basis evaluation by σ instead of using the (σA, B, σC) effective
    coefficients that `_assemble_Z` already uses. At σ=−1 junction
    neighbours (K=2 junctions where both wires "start" or both "end" at
    the node, plus the σ=−1 entries at K=3 junctions) this added a
    spurious 2·B·sin(k·s) term, surfacing as asymmetric kinks at the
    junction-adjacent knots. Caught here at modest N so any re-introduction
    fails loudly.
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
    sim = SinusoidalPySim(
        wires=wires,
        n_per_edge_per_wire=[[3], [n], [n, n, n], [n], [n, n, n]],
        feed_wire_index=0,
        feed_arclength=eps,
        wavelength=wavelength,
        wire_radius=0.0005,
        nsegs=n,
        junctions=junctions,
    )
    _, alpha = sim.compute_impedance()
    knots = sim.currents_at_knots(alpha)

    # Cross-bar halves wire 1 (S→B) vs wire 3 (T→D): same arc-direction
    # mirror, so |I_w1[i]| should equal |I_w3[i]| for all i.
    w1 = np.abs(knots[1])
    w3 = np.abs(knots[3])
    max_dev = float(np.max(np.abs(w1 - w3)))
    assert max_dev < 1e-12, (
        f"cross-bar halves max |Δ|I|| = {max_dev:.2e} — L/R asymmetry"
    )

    # Upper rectangle wire 2 (B→A→C→D) is its own mirror traversed
    # backwards: |I_w2[i]| should equal |I_w2[-1-i]| for all i.
    w2 = np.abs(knots[2])
    max_dev = float(np.max(np.abs(w2 - w2[::-1])))
    assert max_dev < 1e-12, (
        f"upper rectangle max self-reversed |Δ|I|| = {max_dev:.2e} — L/R asymmetry"
    )

    # Lower rectangle wire 4 (D→E→F→B): same self-mirror story.
    w4 = np.abs(knots[4])
    max_dev = float(np.max(np.abs(w4 - w4[::-1])))
    assert max_dev < 1e-12, (
        f"lower rectangle max self-reversed |Δ|I|| = {max_dev:.2e} — L/R asymmetry"
    )


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (TriangularPySim, {}),
        (SinusoidalPySim, {}),
        (BSplinePySim, {"degree": 2}),
    ],
)
def test_currents_at_knots_s_array_matches_default_at_knots(model_cls, kwargs):
    """`currents_at_knots(coeffs, s_array=[knot_arcs_per_wire])` must agree
    bit-for-bit with the default `currents_at_knots(coeffs)` at the mesh
    knots, for every basis family. Guards against future basis-evaluation
    drift between the two paths.
    """
    wires = [np.array([[0.0, 0.0, -0.24], [0.0, 0.0, 0.24]])]
    nsegs = 21
    sim = model_cls(
        wires=wires,
        n_per_edge_per_wire=[[nsegs]],
        wavelength=1.0,
        wire_radius=1e-3,
        nsegs=nsegs,
        **kwargs,
    )
    _, coeffs = sim.compute_impedance()
    knot_default = sim.currents_at_knots(coeffs)

    geom = sim._build_geometry()
    if isinstance(sim, SinusoidalPySim):
        first = geom["wire_first"][0]
        last = geom["wire_last"][0]
        wire_h = geom["seg_h"][first : last + 1]
        arc_at_knot = np.concatenate([[0.0], np.cumsum(wire_h)])
    else:
        arc_at_knot = geom["per_wire"][0]["arc_at_knot"]

    knot_via_s = sim.currents_at_knots(coeffs, s_array=[arc_at_knot])
    np.testing.assert_allclose(knot_via_s[0], knot_default[0], rtol=0, atol=1e-12)

    # Sampling at quarter-points should pass through the knot values exactly
    # at every even-indexed sample (where samples = knots interleaved with
    # midpoints).
    mid_arc = 0.5 * (arc_at_knot[:-1] + arc_at_knot[1:])
    sample_arc = np.empty(2 * mid_arc.shape[0] + 1)
    sample_arc[0::2] = arc_at_knot
    sample_arc[1::2] = mid_arc
    sample_I = sim.currents_at_knots(coeffs, s_array=[sample_arc])[0]
    np.testing.assert_allclose(sample_I[0::2], knot_default[0], rtol=0, atol=1e-12)
    # Mid-segment samples should be finite and on roughly the right scale.
    assert np.isfinite(sample_I).all()


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
    pytest.importorskip("pysim._accelerators")
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


# ---- PEC ground on BSplinePySim (image method, mirrors Triangular) ----


def test_bspline_ground_none_matches_free_space_bit_exact():
    L = 2 * 0.962 * 22 / 4
    poly = _h_dipole(L, 0.0)
    z_no, _ = BSplinePySim(
        wires=[poly], n_per_edge_per_wire=[[40]], nsegs=40, degree=2
    ).compute_impedance()
    z_none, _ = BSplinePySim(
        wires=[poly],
        n_per_edge_per_wire=[[40]],
        nsegs=40,
        degree=2,
        ground_z=None,
    ).compute_impedance()
    assert z_no == z_none


def test_bspline_ground_horizontal_dipole_at_height_recovers_free_space():
    L = 2 * 0.962 * 22 / 4
    N = 30
    z_free, _ = BSplinePySim(
        wires=[_h_dipole(L, 0.0)], n_per_edge_per_wire=[[N]], nsegs=N, degree=2
    ).compute_impedance()
    z_high, _ = BSplinePySim(
        wires=[_h_dipole(L, 100.0)],  # ~5 wavelengths up
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        degree=2,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_high.real - z_free.real) < 2.0
    assert abs(z_high.imag - z_free.imag) < 3.0


def test_bspline_ground_horizontal_dipole_at_zero_height_shorts_out():
    L = 2 * 0.962 * 22 / 4
    z_lo, _ = BSplinePySim(
        wires=[_h_dipole(L, 0.01)],
        n_per_edge_per_wire=[[40]],
        nsegs=40,
        degree=2,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_lo.real) < 0.5


def test_bspline_ground_agrees_with_triangular_at_moderate_height():
    # Tent-basis (O(1/N) R-rate) and B-spline d=2 (basis-limited) converge
    # to the same Z as N→∞ but disagree by their respective truncation
    # errors at finite N. R is the physically meaningful number for ground
    # effects (image symmetry doubles it near h=λ/4); X has higher
    # basis-sensitivity at this N. Tolerances reflect that.
    L = 2 * 0.962 * 22 / 4
    N = 40
    h = 7.0
    z_tri, _ = TriangularPySim(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    ).compute_impedance()
    z_bsp, _ = BSplinePySim(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        degree=2,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_bsp.real - z_tri.real) < 0.03 * abs(z_tri.real)
    assert abs(z_bsp.imag - z_tri.imag) < 0.10 * max(abs(z_tri.imag), 5.0)


def test_bspline_ground_swept_matches_single_freq():
    L = 2 * 0.962 * 22 / 4
    N = 30
    h = 5.0
    sim = BSplinePySim(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        degree=2,
        ground_z=0.0,
    )
    z_single, _ = sim.compute_impedance()
    z_swept = sim.compute_impedance_swept(np.array([sim.k]))[0]
    assert abs(z_single - z_swept) < 1e-9


# ---- PEC ground on SinusoidalPySim (NEC three-term basis, image method) ----


def test_sinusoidal_ground_none_matches_free_space_bit_exact():
    L = 2 * 0.962 * 22 / 4
    poly = _h_dipole(L, 0.0)
    z_no, _ = SinusoidalPySim(
        wires=[poly], n_per_edge_per_wire=[[40]], nsegs=40
    ).compute_impedance()
    z_none, _ = SinusoidalPySim(
        wires=[poly], n_per_edge_per_wire=[[40]], nsegs=40, ground_z=None
    ).compute_impedance()
    assert z_no == z_none


def test_sinusoidal_ground_horizontal_dipole_at_height_recovers_free_space():
    L = 2 * 0.962 * 22 / 4
    N = 30
    z_free, _ = SinusoidalPySim(
        wires=[_h_dipole(L, 0.0)], n_per_edge_per_wire=[[N]], nsegs=N
    ).compute_impedance()
    z_high, _ = SinusoidalPySim(
        wires=[_h_dipole(L, 100.0)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_high.real - z_free.real) < 2.0
    assert abs(z_high.imag - z_free.imag) < 3.0


def test_sinusoidal_ground_horizontal_dipole_at_zero_height_shorts_out():
    L = 2 * 0.962 * 22 / 4
    z_lo, _ = SinusoidalPySim(
        wires=[_h_dipole(L, 0.01)],
        n_per_edge_per_wire=[[40]],
        nsegs=40,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_lo.real) < 0.5


def test_sinusoidal_ground_agrees_with_triangular_at_moderate_height():
    L = 2 * 0.962 * 22 / 4
    N = 40
    h = 7.0
    z_tri, _ = TriangularPySim(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    ).compute_impedance()
    z_sin, _ = SinusoidalPySim(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    ).compute_impedance()
    assert abs(z_sin.real - z_tri.real) < 0.03 * abs(z_tri.real)
    assert abs(z_sin.imag - z_tri.imag) < 0.10 * max(abs(z_tri.imag), 5.0)


def test_sinusoidal_ground_swept_matches_single_freq():
    L = 2 * 0.962 * 22 / 4
    N = 30
    h = 5.0
    sim = SinusoidalPySim(
        wires=[_h_dipole(L, h)],
        n_per_edge_per_wire=[[N]],
        nsegs=N,
        ground_z=0.0,
    )
    z_single, _ = sim.compute_impedance()
    z_swept = sim.compute_impedance_swept(np.array([sim.k]))[0]
    assert abs(z_single - z_swept) < 1e-9


def test_bspline_ground_with_enrichment_raises():
    L = 2 * 0.962 * 22 / 4
    import pytest

    with pytest.raises(NotImplementedError):
        BSplinePySim(
            wires=[_h_dipole(L, 5.0)],
            n_per_edge_per_wire=[[20]],
            nsegs=20,
            degree=2,
            ground_z=0.0,
            use_singular_enrichment=True,
        )


# ---------------------------------------------------------------------------
# Multi-feed support (delta-gap excitations with prescribed complex voltages)
# ---------------------------------------------------------------------------


def _two_dipoles(halfdriver, spacing):
    a = np.array([[0.0, -halfdriver, 0.0], [0.0, halfdriver, 0.0]])
    b = np.array([[spacing, -halfdriver, 0.0], [spacing, halfdriver, 0.0]])
    return [a, b]


def test_triangular_single_feed_via_feeds_kwarg_matches_legacy():
    L = 2 * 0.962 * 22 / 4
    nsegs = 40
    wires = [np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]])]
    z_legacy, _ = TriangularPySim(
        wires=wires, n_per_edge_per_wire=[[nsegs]], nsegs=nsegs
    ).compute_impedance()
    z_new, _ = TriangularPySim(
        wires=wires,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j)],
    ).compute_impedance()
    assert abs(z_new - z_legacy) < 1e-12


def test_triangular_multifeed_two_dipoles_in_phase():
    # Two parallel dipoles, both fed with V=1 (no phase shift). The per-feed
    # impedances should be equal by symmetry.
    hd = 0.962 * 22 / 4
    nsegs = 40
    wires = _two_dipoles(hd, spacing=2.0)
    sim = TriangularPySim(
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, 1.0 + 0.0j)],
    )
    z_per_feed, c = sim.compute_impedance()
    assert z_per_feed.shape == (2,)
    assert np.isfinite(c).all()
    assert abs(z_per_feed[0] - z_per_feed[1]) / abs(z_per_feed[0]) < 1e-6
    # Mutual coupling at this spacing shifts Z away from the isolated
    # dipole value but it should remain bounded and positive (in-phase
    # coupling raises R well above the isolated 70 ohm).
    assert 30.0 < z_per_feed[0].real < 200.0


def test_triangular_multifeed_phase_shift_changes_driving_point():
    # Same geometry as the in-phase test; flipping one feed by 180 degrees
    # must change the driving-point impedance (V1+V2 mode -> V1-V2 mode).
    hd = 0.962 * 22 / 4
    nsegs = 40
    wires = _two_dipoles(hd, spacing=2.0)

    z_inphase, _ = TriangularPySim(
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, 1.0 + 0.0j)],
    ).compute_impedance()

    z_anti, _ = TriangularPySim(
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, -1.0 + 0.0j)],
    ).compute_impedance()

    # Anti-phase pair excites the (Z_self - Z_mut) mode; in-phase excites
    # (Z_self + Z_mut). With nonzero mutual coupling the two driving-point
    # impedances must differ noticeably.
    assert abs(z_inphase[0] - z_anti[0]) > 5.0


def test_triangular_multifeed_consistency_via_port_z_matrix():
    # Cross-check: compute the 2x2 port impedance matrix by two unit
    # excitations and confirm V = Z_port @ I matches an arbitrary
    # combined-voltage solve.
    hd = 0.962 * 22 / 4
    nsegs = 40
    wires = _two_dipoles(hd, spacing=2.0)
    kw = dict(wires=wires, n_per_edge_per_wire=[[nsegs], [nsegs]], nsegs=nsegs)

    # Column 0: V=(1,0).
    sim_a = TriangularPySim(**kw, feeds=[(0, None, 1.0 + 0.0j), (1, None, 0.0 + 0.0j)])
    _, coeffs_a = sim_a.compute_impedance()
    m = sim_a._feed_basis_indices(sim_a._build_geometry())
    I_a = coeffs_a[m]
    # Column 1: V=(0,1).
    sim_b = TriangularPySim(**kw, feeds=[(0, None, 0.0 + 0.0j), (1, None, 1.0 + 0.0j)])
    _, coeffs_b = sim_b.compute_impedance()
    I_b = coeffs_b[m]

    # Y matrix (port-admittance) columns are the currents at the two ports.
    Y = np.column_stack([I_a, I_b])
    Z_port = np.linalg.inv(Y)

    # Arbitrary phased excitation: V = (1, exp(j*60deg)).
    V = np.array([1.0 + 0j, np.exp(1j * np.pi / 3)])
    sim_c = TriangularPySim(
        **kw,
        feeds=[(0, None, V[0]), (1, None, V[1])],
    )
    z_per_feed, coeffs_c = sim_c.compute_impedance()
    I_c = coeffs_c[m]

    # Linearity: I_c must equal Y @ V.
    assert np.allclose(I_c, Y @ V, rtol=1e-8, atol=1e-12)
    # And the reported per-feed driving-point Z must match V / (Y @ V).
    assert np.allclose(z_per_feed, V / (Y @ V), rtol=1e-8, atol=1e-12)
    # Z_port reciprocity sanity (free-space, no junctions): Z_port should be
    # symmetric to within the discretization noise.
    assert abs(Z_port[0, 1] - Z_port[1, 0]) / abs(Z_port[0, 0]) < 1e-6


def test_triangular_multifeed_swept_matches_single_k():
    hd = 0.962 * 22 / 4
    nsegs = 30
    wires = _two_dipoles(hd, spacing=2.5)
    feeds = [(0, None, 1.0 + 0.0j), (1, None, np.exp(1j * np.pi / 4))]
    sim = TriangularPySim(
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=feeds,
    )
    z_single, _ = sim.compute_impedance()
    z_swept = sim.compute_impedance_swept(np.array([sim.k]))
    assert z_swept.shape == (1, 2)
    assert np.allclose(z_swept[0], z_single, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Same multi-feed checks against BSplinePySim and SinusoidalPySim.
# ---------------------------------------------------------------------------


def _mk_sim(cls, *, wires, n_per_edge_per_wire, nsegs, feeds=None, **extra):
    kw = dict(
        wires=wires,
        n_per_edge_per_wire=n_per_edge_per_wire,
        nsegs=nsegs,
    )
    if feeds is not None:
        kw["feeds"] = feeds
    if cls is BSplinePySim:
        kw["degree"] = 2
    kw.update(extra)
    return cls(**kw)


@pytest.mark.parametrize("cls", [BSplinePySim, SinusoidalPySim])
def test_multifeed_single_feed_via_feeds_kwarg_matches_legacy(cls):
    L = 2 * 0.962 * 22 / 4
    nsegs = 40
    wires = [np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]])]
    z_legacy, _ = _mk_sim(
        cls, wires=wires, n_per_edge_per_wire=[[nsegs]], nsegs=nsegs
    ).compute_impedance()
    z_new, _ = _mk_sim(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j)],
    ).compute_impedance()
    assert abs(z_new - z_legacy) < 1e-9


@pytest.mark.parametrize("cls", [BSplinePySim, SinusoidalPySim])
def test_multifeed_two_dipoles_in_phase(cls):
    hd = 0.962 * 22 / 4
    nsegs = 40
    wires = _two_dipoles(hd, spacing=2.0)
    sim = _mk_sim(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, 1.0 + 0.0j)],
    )
    z_per_feed, c = sim.compute_impedance()
    assert z_per_feed.shape == (2,)
    assert np.isfinite(c).all()
    assert abs(z_per_feed[0] - z_per_feed[1]) / abs(z_per_feed[0]) < 1e-4
    assert 30.0 < z_per_feed[0].real < 200.0


@pytest.mark.parametrize("cls", [BSplinePySim, SinusoidalPySim])
def test_multifeed_phase_shift_changes_driving_point(cls):
    hd = 0.962 * 22 / 4
    nsegs = 40
    wires = _two_dipoles(hd, spacing=2.0)
    z_inphase, _ = _mk_sim(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, 1.0 + 0.0j)],
    ).compute_impedance()
    z_anti, _ = _mk_sim(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, -1.0 + 0.0j)],
    ).compute_impedance()
    assert abs(z_inphase[0] - z_anti[0]) > 5.0


@pytest.mark.parametrize("cls", [BSplinePySim, SinusoidalPySim])
def test_multifeed_homogeneity_in_voltage(cls):
    # Driving-point impedance V/I is a ratio; scaling all voltages by a
    # common (complex) factor must leave Z_i unchanged. This is a
    # solver-agnostic linearity check that exercises the multi-feed RHS
    # plumbing without requiring an external port-Z reference.
    hd = 0.962 * 22 / 4
    nsegs = 40
    wires = _two_dipoles(hd, spacing=2.0)
    common = dict(wires=wires, n_per_edge_per_wire=[[nsegs], [nsegs]], nsegs=nsegs)
    V = np.array([1.0 + 0j, np.exp(1j * np.pi / 3)])
    z1, _ = _mk_sim(
        cls, **common, feeds=[(0, None, V[0]), (1, None, V[1])]
    ).compute_impedance()
    z2, _ = _mk_sim(
        cls,
        **common,
        feeds=[(0, None, 2.5 * V[0]), (1, None, 2.5 * V[1])],
    ).compute_impedance()
    assert np.allclose(z1, z2, rtol=1e-8, atol=1e-12)
    assert np.isfinite(z1).all()


@pytest.mark.parametrize("cls", [BSplinePySim, SinusoidalPySim])
def test_multifeed_swept_matches_single_k(cls):
    hd = 0.962 * 22 / 4
    nsegs = 30
    wires = _two_dipoles(hd, spacing=2.5)
    feeds = [(0, None, 1.0 + 0.0j), (1, None, np.exp(1j * np.pi / 4))]
    sim = _mk_sim(
        cls,
        wires=wires,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        feeds=feeds,
    )
    z_single, _ = sim.compute_impedance()
    z_swept = sim.compute_impedance_swept(np.array([sim.k]))
    assert z_swept.shape == (1, 2)
    assert np.allclose(z_swept[0], z_single, rtol=1e-8, atol=1e-12)


def test_triangular_bowtiearray_1x2_phased():
    # Simplified "bowtie-array 1x2" stand-in: two V-shaped (kinked-dipole)
    # elements side-by-side, each driven with its own complex voltage.
    # The point of this test is the multi-feed plumbing on a non-trivial
    # multi-wire / kinked geometry, not bowtie geometric fidelity.
    hd = 0.962 * 22 / 4
    nsegs = 30
    bend = 0.3 * hd  # z-droop at the tip — gives each element a kink
    del_y = 2.0
    elem_tmpl = np.array(
        [
            [0.0, -hd, -bend],
            [0.0, 0.0, 0.0],
            [0.0, hd, -bend],
        ]
    )
    left = elem_tmpl + np.array([0.0, -del_y, 0.0])
    right = elem_tmpl + np.array([0.0, +del_y, 0.0])

    sim = TriangularPySim(
        wires=[left, right],
        n_per_edge_per_wire=[[nsegs, nsegs], [nsegs, nsegs]],
        nsegs=nsegs,
        feeds=[(0, None, 1.0 + 0.0j), (1, None, np.exp(1j * np.pi / 2))],
    )
    z_per_feed, coeffs = sim.compute_impedance()
    assert z_per_feed.shape == (2,)
    assert np.isfinite(z_per_feed).all()
    assert np.isfinite(coeffs).all()
    # Per-feed Re(Z_i) can legitimately go negative when ports exchange
    # power through strong mutual coupling (the system as a whole still
    # radiates). Just sanity-check magnitudes stay bounded.
    assert abs(z_per_feed[0]) < 1000.0
    assert abs(z_per_feed[1]) < 1000.0
