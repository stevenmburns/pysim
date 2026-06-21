"""Tests for the hierarchical (H-matrix / ACA) B-spline MoM accelerator.

Phase 0: the on-demand block evaluator `HMatrixPySim.zblock(I, J)` must
reproduce, to machine precision, the corresponding sub-block of the exact
dense `BSplinePySim` impedance matrix — both the off-edge (far) path and the
same-edge analytic-overwrite (near) path.
"""

import numpy as np
import pytest

from pysim.bspline import BSplinePySim
from pysim.hmatrix import HMatrixPySim


def _dense_Z(sim):
    """The exact dense bspline Z the evaluator must match."""
    geom = sim._build_geometry()
    supp_seg, polys, _kcl_A, _wk, _wbg = sim._build_basis_polynomials(geom)
    J = sim._build_J_blocks(geom, sim.k)
    return sim._assemble_Z(J, supp_seg, polys, geom)


def _dipole(degree, nsegs):
    half = 0.962 * 22 / 4
    wire = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])
    return HMatrixPySim(
        wires=[wire],
        degree=degree,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        wavelength=22.0,
    )


def _bent_wire_with_junction(degree, nsegs):
    """Two wires meeting at a right-angle junction — exercises multiple edges
    and a junction directional basis (the same-edge overwrite + KCL path)."""
    h = 0.962 * 22 / 4
    w0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h]])
    w1 = np.array([[0.0, 0.0, h], [0.0, h, h]])
    junctions = [[(0, "end"), (1, "start")]]
    return HMatrixPySim(
        wires=[w0, w1],
        degree=degree,
        n_per_edge_per_wire=[[nsegs], [nsegs]],
        nsegs=nsegs,
        wavelength=22.0,
        junctions=junctions,
        feed_wire_index=0,
    )


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize(
    "builder", [_dipole, _bent_wire_with_junction], ids=["dipole", "junction"]
)
def test_zblock_matches_dense_full(builder, degree):
    sim = builder(degree, 16)
    n = sim._context()["n_basis"]
    Z = _dense_Z(sim)
    full = sim.zblock(np.arange(n), np.arange(n))
    rel = np.abs(full - Z).max() / np.abs(Z).max()
    assert rel < 1e-12, f"full-matrix rel err {rel:.2e}"


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize(
    "builder", [_dipole, _bent_wire_with_junction], ids=["dipole", "junction"]
)
def test_zblock_matches_dense_random_subblocks(builder, degree):
    sim = builder(degree, 18)
    n = sim._context()["n_basis"]
    Z = _dense_Z(sim)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(12):
        szI = int(rng.integers(1, n // 2 + 1))
        szJ = int(rng.integers(1, n // 2 + 1))
        I = rng.choice(n, size=szI, replace=False)
        J = rng.choice(n, size=szJ, replace=False)
        ref = Z[np.ix_(I, J)]
        blk = sim.zblock(I, J)
        worst = max(worst, np.abs(blk - ref).max() / (np.abs(ref).max() + 1e-30))
    # The near-band same-edge optimization computes same-edge moments over a
    # contiguous sub-range rather than the full edge block. The block is
    # mathematically identical, but the uniform static-moment accelerator has
    # a mild range-size dependence at the ~1e-13 level which the d=2 scalar-
    # potential 1/(ωε) term amplifies to ~1e-10 — far below any physical
    # tolerance. (The full-matrix case above stays <1e-12 since it spans the
    # whole edge.)
    assert worst < 1e-8, f"worst sub-block rel err {worst:.2e}"


def _long_wire(degree, nsegs):
    half = 2.0 * 0.962 * 22 / 4
    wire = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])
    return HMatrixPySim(
        wires=[wire],
        degree=degree,
        n_per_edge_per_wire=[[nsegs]],
        nsegs=nsegs,
        wavelength=22.0,
    )


@pytest.mark.parametrize("eta", [1.0, 2.0])
@pytest.mark.parametrize("leaf_size", [16, 32])
def test_block_partition_is_exact_cover(eta, leaf_size):
    """The far + near leaf blocks must tile the n x n index product with no
    gaps and no overlaps."""
    sim = _long_wire(1, 200)
    part = sim.build_partition(eta=eta, leaf_size=leaf_size)
    n = sim._context()["n_basis"]
    cover = np.zeros((n, n), dtype=np.int32)
    for s, t in part["far"] + part["near"]:
        cover[np.ix_(s.indices, t.indices)] += 1
    assert cover.min() == 1 and cover.max() == 1
    st = part["stats"]
    assert st["covered"] == st["total"] == n * n


def test_block_partition_compresses_far_field():
    """A multi-wavelength wire should put most of the matrix area into
    admissible (far) blocks."""
    sim = _long_wire(1, 300)
    part = sim.build_partition(eta=1.0, leaf_size=32)
    assert part["stats"]["far_frac"] > 0.6


def test_far_blocks_have_no_same_edge_pairs():
    """Admissibility must spatially separate clusters, so no far block may
    contain a self-pair (a basis paired with itself)."""
    sim = _long_wire(1, 300)
    part = sim.build_partition(eta=1.0, leaf_size=32)
    for s, t in part["far"]:
        assert np.intersect1d(s.indices, t.indices).size == 0


@pytest.mark.parametrize("tol", [1e-3, 1e-5])
def test_hmatvec_matches_dense(tol):
    """H-matrix matvec must reproduce the dense Z @ x; ACA's block-local
    tolerance bounds the error well below 1."""
    sim = _long_wire(1, 250)
    Z = _dense_Z(sim)
    n = Z.shape[0]
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    H = sim.build_hmatrix(tol=tol)
    rel = np.linalg.norm(H.matvec(x) - Z @ x) / np.linalg.norm(Z @ x)
    assert rel < 50 * tol


def test_hmatmat_equals_columnwise_matvec():
    """The batched H-matrix matmat must equal per-column matvec (the contract
    the block-GMRES multi-RHS solve depends on)."""
    sim = _long_wire(1, 250)
    H = sim.build_hmatrix(tol=1e-5)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((H.n, 4)) + 1j * rng.standard_normal((H.n, 4))
    Y = H.matmat(X)
    Ycol = np.column_stack([H.matvec(X[:, j]) for j in range(X.shape[1])])
    # GEMM (matmat) vs GEMV (matvec) round differently at ~1e-15 relative.
    assert np.linalg.norm(Y - Ycol) / np.linalg.norm(Ycol) < 1e-12


def test_hmatrix_to_dense_reconstruction():
    sim = _long_wire(1, 200)
    Z = _dense_Z(sim)
    H = sim.build_hmatrix(tol=1e-6)
    rel = np.abs(H.to_dense() - Z).max() / np.abs(Z).max()
    assert rel < 1e-3


def test_hmatrix_compresses_storage():
    """A multi-wavelength wire must store strictly fewer scalars than dense,
    and the far-field ranks must stay small."""
    sim = _long_wire(1, 400)
    H = sim.build_hmatrix(tol=1e-4)
    st = H.stats()
    assert st["compression"] < 0.7
    assert st["max_rank"] <= 32


@pytest.mark.parametrize("degree", [1, 2])
def test_hmatrix_accel_matches_numpy_path(degree):
    """The fused C++ off-edge block assembler must produce an H-matrix
    identical (to machine precision) to the pure-numpy zblock fill."""
    sim_a = _long_wire(degree, 200)
    sim_a.hmatrix_use_accel = True
    sim_n = _long_wire(degree, 200)
    sim_n.hmatrix_use_accel = False
    Ha = sim_a.build_hmatrix(tol=1e-5)
    Hn = sim_n.build_hmatrix(tol=1e-5)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(Ha.n) + 1j * rng.standard_normal(Ha.n)
    ya, yn = Ha.matvec(x), Hn.matvec(x)
    assert np.linalg.norm(ya - yn) / np.linalg.norm(yn) < 1e-12


def test_hmatrix_handles_junction_geometry():
    sim = _bent_wire_with_junction(1, 60)
    Z = _dense_Z(sim)
    n = Z.shape[0]
    rng = np.random.default_rng(2)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    H = sim.build_hmatrix(tol=1e-5)
    rel = np.linalg.norm(H.matvec(x) - Z @ x) / np.linalg.norm(Z @ x)
    assert rel < 1e-3


def _matched_pair(wires, *, degree, n_per_edge_per_wire, **kw):
    """A BSplinePySim and an HMatrixPySim built identically (same mesh), so
    the only difference is the dense vs hierarchical solve path."""
    common = dict(
        wires=wires,
        degree=degree,
        n_per_edge_per_wire=n_per_edge_per_wire,
        wavelength=22.0,
        **kw,
    )
    return BSplinePySim(**common), HMatrixPySim(aca_tol=1e-7, **common)


@pytest.mark.parametrize("degree", [1, 2])
def test_compute_impedance_matches_dense_dipole(degree):
    half = 0.962 * 22 / 4
    wires = [np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])]
    dense, hmat = _matched_pair(wires, degree=degree, n_per_edge_per_wire=[[80]])
    zd, _ = dense.compute_impedance()
    zh, _ = hmat.compute_impedance()
    assert abs(zh - zd) / abs(zd) < 1e-4


def test_compute_impedance_swept_matches_dense():
    """The H-matrix frequency sweep must run (it overrides the dense base
    sweep, whose `same_edge_prep` batching arg the accelerated
    `compute_impedance` doesn't accept — a bare inherit `TypeError`s) and match
    the dense bspline swept impedance."""
    half = 0.962 * 22 / 4
    wires = [np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])]
    dense, hmat = _matched_pair(wires, degree=2, n_per_edge_per_wire=[[60]])
    k0 = 2 * np.pi / 22.0
    k_array = np.linspace(0.9 * k0, 1.1 * k0, 5)
    zd = dense.compute_impedance_swept(k_array)
    zh = hmat.compute_impedance_swept(k_array)
    assert zh.shape == zd.shape == (5,)
    assert np.max(np.abs(zh - zd) / np.abs(zd)) < 1e-4


@pytest.mark.parametrize("degree", [1, 2])
def test_compute_y_matrix_matches_dense_junction(degree):
    h = 0.962 * 22 / 4
    wires = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h]]),
        np.array([[0.0, 0.0, h], [0.0, h, h]]),
    ]
    junctions = [[(0, "end"), (1, "start")]]
    npe = [[40], [40]]
    dense, hmat = _matched_pair(
        wires,
        degree=degree,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        feed_wire_index=0,
    )
    yd = dense.compute_y_matrix()
    yh = hmat.compute_y_matrix()
    assert np.abs(yh - yd).max() / np.abs(yd).max() < 1e-4


def test_first_ring_preconditioner_reduces_iterations():
    """The first-ring preconditioner (default precond_eta < aca_eta) must give
    the same answer as the plain near-field preconditioner but in no more —
    and in practice fewer — GMRES iterations."""
    half = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])]
    common = dict(
        wires=wires,
        degree=1,
        n_per_edge_per_wire=[[600]],
        wavelength=22.0,
        aca_tol=1e-5,
        aca_eta=2.0,
    )
    plain = HMatrixPySim(precond_eta=2.0, **common)  # ring disabled
    ring = HMatrixPySim(**common)  # default ring (0.5*aca_eta)
    zp, _ = plain.compute_impedance()
    zr, _ = ring.compute_impedance()
    assert abs(zr - zp) / abs(zp) < 1e-4
    assert ring._hmatrix.precond_extra  # the ring is non-empty
    assert max(ring._last_solve_iters) <= max(plain._last_solve_iters)


def test_solve_converges_in_few_iterations():
    """The near-field preconditioner should give single/low-double-digit
    GMRES iteration counts — the point of the hierarchical solve."""
    half = 2 * 0.962 * 22 / 4
    wires = [np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])]
    hmat = HMatrixPySim(
        wires=wires, degree=1, n_per_edge_per_wire=[[300]], wavelength=22.0
    )
    hmat.compute_impedance()
    assert max(hmat._last_solve_iters) < 50


def test_zblock_off_edge_skips_same_edge_path():
    """A block between two well-separated single basis functions must contain
    no same-edge pairs, so it is computed purely off-edge — and still matches
    the dense reference."""
    sim = _dipole(2, 24)
    Z = _dense_Z(sim)
    # endpoints of the basis index range are far apart along the wire
    I = np.array([1])
    J = np.array([sim._context()["n_basis"] - 2])
    blk = sim.zblock(I, J)
    ref = Z[np.ix_(I, J)]
    assert np.abs(blk - ref).max() / (np.abs(ref).max() + 1e-30) < 1e-12


# ---- PEC ground: per-block image term folded into near + far blocks --------


def _dense_Z_ground(sim):
    """The exact dense bspline Z under PEC ground (free-space minus the image
    assembly) the hierarchical operator must reproduce."""
    geom = sim._build_geometry()
    supp_seg, polys, _a, _wk, _wbg = sim._build_basis_polynomials(geom)
    Z = sim._assemble_Z(sim._build_J_blocks(geom, sim.k), supp_seg, polys, geom)
    J_img = sim._build_J_image_blocks(geom, sim.k)
    td_img = sim._image_tangent_dot(geom["tangents"])
    return Z - sim._assemble_Z(J_img, supp_seg, polys, geom, td_all=td_img)


def test_ground_hmatrix_to_dense_matches_dense_pec():
    """The grounded H-matrix (image folded into near + far blocks) reconstructs
    the dense PEC Z."""
    half = 0.962 * 22 / 4
    wires = [np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0 + 2 * half]])]
    sim = HMatrixPySim(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[[200]],
        wavelength=22.0,
        feeds=[(0, None, 1.0 + 0.0j)],
        ground_z=0.0,
        aca_tol=1e-7,
    )
    Z = _dense_Z_ground(sim)
    H = sim.build_hmatrix()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(Z.shape[0]) + 1j * rng.standard_normal(Z.shape[0])
    assert np.linalg.norm(H.matvec(x) - Z @ x) / np.linalg.norm(Z @ x) < 1e-4
    assert np.abs(H.to_dense() - Z).max() / np.abs(Z).max() < 1e-4


@pytest.mark.parametrize("degree", [1, 2])
def test_ground_compute_impedance_matches_dense_pec(degree):
    """HMatrixPySim + PEC ground matches dense BSplinePySim + PEC ground."""
    half = 0.962 * 22 / 4
    wires = [np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 1.5 + 2 * half]])]
    dense, hmat = _matched_pair(
        wires, degree=degree, n_per_edge_per_wire=[[100]], ground_z=0.0
    )
    zd, _ = dense.compute_impedance()
    zh, _ = hmat.compute_impedance()
    assert abs(zh - zd) / abs(zd) < 1e-4


def test_ground_compute_y_matrix_matches_dense_pec_junction():
    """Grounded H-matrix with a junction geometry — exercises the KCL saddle
    rows and the near-field preconditioner under the image term."""
    h = 0.962 * 22 / 4
    wires = [
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0 + h]]),
        np.array([[0.0, 0.0, 1.0 + h], [0.0, h, 1.0 + h]]),
    ]
    junctions = [[(0, "end"), (1, "start")]]
    dense, hmat = _matched_pair(
        wires,
        degree=2,
        n_per_edge_per_wire=[[40], [40]],
        junctions=junctions,
        feed_wire_index=0,
        ground_z=0.0,
    )
    yd = dense.compute_y_matrix()
    yh = hmat.compute_y_matrix()
    assert np.abs(yh - yd).max() / np.abs(yd).max() < 1e-4


def test_ground_iteration_count_near_free_space():
    """The per-block image term doesn't degrade the near-field preconditioner:
    GMRES under PEC ground converges in about as many iterations as free
    space."""
    half = 2 * 0.962 * 22 / 4
    free = HMatrixPySim(
        wires=[np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])],
        degree=1,
        n_per_edge_per_wire=[[300]],
        wavelength=22.0,
    )
    grnd = HMatrixPySim(
        wires=[np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0 + 2 * half]])],
        degree=1,
        n_per_edge_per_wire=[[300]],
        wavelength=22.0,
        ground_z=0.0,
    )
    free.compute_impedance()
    grnd.compute_impedance()
    assert abs(max(grnd._last_solve_iters) - max(free._last_solve_iters)) <= 2


def test_ground_hmatrix_still_compresses():
    """For a normal-height antenna the grounded H-matrix still compresses vs
    dense (the image stays low-rank — reflection only increases cluster
    separation), and a *low* antenna stays correct even if it compresses less
    (the dense-fallback guard makes the degradation graceful, not wrong)."""
    half = 2 * 0.962 * 22 / 4
    # normal height: image is far, compression close to free space
    tall = HMatrixPySim(
        wires=[np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0 + 2 * half]])],
        degree=1,
        n_per_edge_per_wire=[[400]],
        wavelength=22.0,
        ground_z=0.0,
        aca_tol=1e-6,
    )
    tall.compute_impedance()
    assert tall._hmatrix.stats()["compression"] < 0.9  # genuinely compressed

    # low horizontal wire: image is near, but the answer is still correct
    low = HMatrixPySim(
        wires=[np.array([[0.0, -half, 0.1], [0.0, half, 0.1]])],
        degree=1,
        n_per_edge_per_wire=[[400]],
        wavelength=22.0,
        ground_z=0.0,
        aca_tol=1e-6,
    )
    dense_low = BSplinePySim(
        wires=[np.array([[0.0, -half, 0.1], [0.0, half, 0.1]])],
        degree=1,
        n_per_edge_per_wire=[[400]],
        wavelength=22.0,
        ground_z=0.0,
    )
    zl, _ = low.compute_impedance()
    zd, _ = dense_low.compute_impedance()
    assert abs(zl - zd) / abs(zd) < 1e-3


def test_ground_does_not_break_free_space_path():
    """Building a grounded then a free-space H-matrix of the same geometry must
    give the free-space dense answer for the free-space one (the image term is
    gated strictly on ground_z, no leakage)."""
    half = 0.962 * 22 / 4
    wires = [np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0 + 2 * half]])]
    HMatrixPySim(
        wires=wires,
        degree=2,
        n_per_edge_per_wire=[[80]],
        wavelength=22.0,
        ground_z=0.0,
    ).compute_impedance()
    dense, hmat = _matched_pair(wires, degree=2, n_per_edge_per_wire=[[80]])
    zd, _ = dense.compute_impedance()
    zh, _ = hmat.compute_impedance()
    assert abs(zh - zd) / abs(zd) < 1e-4
