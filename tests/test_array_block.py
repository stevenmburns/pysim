"""Tests for the element-aware array-block solver (P0: grouping + shapes).

Self-contained: synthetic arrays are built from translated dipoles so the
expected element/shape structure is known without antenna_designer designs.
The real array designs are exercised by scripts/array_block_verify.py.
"""

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver
from momwire.array_block import (
    ArrayBlockSolver,
    cache_stats,
    element_groups,
    reset_array_caches,
)


def _dense_Z(sim):
    """The exact dense bspline Z the block decomposition must reproduce."""
    geom = sim._build_geometry()
    supp_seg, polys, _kcl_A, _wk, _wbg = sim._build_basis_polynomials(geom)
    J = sim._build_J_blocks(geom, sim.k)
    return sim._assemble_Z(J, supp_seg, polys, geom)


def _dipole_wire(half, y=0.0, z=0.0):
    return np.array([[0.0, y, z - half], [0.0, y, z + half]])


def _array_sim(offsets, halves, nsegs=12, degree=2):
    """One straight dipole per (offset, half-length): identical when the
    half-lengths match, a distinct shape when they differ. `offsets` are
    (y, z) element centres far enough apart to be electrically separate."""
    wires = [_dipole_wire(h, y=y, z=z) for (y, z), h in zip(offsets, halves)]
    return ArrayBlockSolver(
        wires=wires,
        degree=degree,
        n_per_edge_per_wire=[[nsegs]] * len(wires),
        nsegs=nsegs,
        wavelength=22.0,
        feeds=[(i, None, 1.0 + 0.0j) for i in range(len(wires))],
    )


def _array_solver(offsets, halves, voltages, solver, nsegs=14, degree=2):
    """Build `solver` (ArrayBlockSolver or BSplineSolver) for a dipole array with
    explicit per-feed voltages (so a 'phase sweep' is just changing voltages on
    a fixed geometry)."""
    wires = [_dipole_wire(h, y=y, z=z) for (y, z), h in zip(offsets, halves)]
    return solver(
        wires=wires,
        degree=degree,
        n_per_edge_per_wire=[[nsegs]] * len(wires),
        wavelength=22.0,
        feeds=[(i, None, v) for i, v in enumerate(voltages)],
    )


def test_grouping_separates_elements():
    """Three well-separated identical dipoles → three elements, all bases
    accounted for exactly once, equal sizes."""
    half = 0.962 * 22 / 4
    offsets = [(-6.0, 0.0), (0.0, 0.0), (6.0, 0.0)]
    sim = _array_sim(offsets, [half] * 3)
    part = element_groups(sim)
    assert part.n_elem == 3
    assert len(set(part.sizes.tolist())) == 1  # equal sizes
    # exact partition of [0, n_basis): disjoint union, every basis once
    allb = np.concatenate(part.groups)
    assert np.array_equal(np.sort(allb), np.arange(part.n_basis))
    # every basis maps back to its group
    for e, g in enumerate(part.groups):
        assert np.all(part.elem_of_basis[g] == e)


def test_identical_elements_one_shape():
    half = 0.962 * 22 / 4
    sim = _array_sim([(-6.0, 0.0), (0.0, 0.0), (6.0, 0.0)], [half] * 3)
    part = element_groups(sim)
    assert part.n_shapes == 1
    assert part.shape_of_elem.tolist() == [0, 0, 0]


def test_distinct_lengths_distinct_shapes():
    """Two long + two short dipoles, interleaved → two shape classes that
    track length, not position."""
    long_h = 0.962 * 22 / 4
    short_h = 0.7 * long_h
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    sim = _array_sim(offsets, [long_h, short_h, long_h, short_h])
    part = element_groups(sim)
    assert part.n_elem == 4
    assert part.n_shapes == 2
    # elements 0,2 share a shape; 1,3 share the other
    s = part.shape_of_elem
    assert s[0] == s[2] and s[1] == s[3] and s[0] != s[1]
    assert part.shape_representatives() == [0, 1]


def test_self_blocks_identical_within_shape_class():
    """The load-bearing P0 fact: free-space self-blocks of same-shape
    elements match to ~1e-12 in consistent ordering."""
    half = 0.962 * 22 / 4
    sim = _array_sim([(-6.0, 0.0), (0.0, 0.0), (6.0, 0.0)], [half] * 3)
    part = element_groups(sim)
    blocks = [sim.zblock(g, g) for g in part.groups]
    ref = blocks[0]
    rnorm = np.linalg.norm(ref)
    for b in blocks[1:]:
        assert np.linalg.norm(b - ref) / rnorm < 1e-10


def test_coupling_is_weak_and_low_rank():
    """Off-diagonal element blocks are a small fraction of the self-block and
    numerically low rank — the two facts the block solver exploits."""
    half = 0.962 * 22 / 4
    sim = _array_sim([(-6.0, 0.0), (0.0, 0.0), (6.0, 0.0)], [half] * 3)
    part = element_groups(sim)
    S00 = sim.zblock(part.groups[0], part.groups[0])
    C01 = sim.zblock(part.groups[0], part.groups[1])
    assert np.linalg.norm(C01) / np.linalg.norm(S00) < 0.1
    s = np.linalg.svd(C01, compute_uv=False)
    rank = int(np.count_nonzero(s > 0.01 * s[0]))
    assert rank <= 8


def test_single_structure_is_one_element():
    """A lone dipole is a degenerate 1-element array (no structure to
    exploit, but well-defined)."""
    half = 0.962 * 22 / 4
    sim = HMatrixSolver(
        wires=[_dipole_wire(half)],
        degree=2,
        n_per_edge_per_wire=[[16]],
        wavelength=22.0,
    )
    part = element_groups(sim)
    assert part.n_elem == 1
    assert part.n_shapes == 1
    assert part.sizes.tolist() == [part.n_basis]


# ---- P1: block decomposition + matvec ---------------------------------------


@pytest.mark.parametrize("degree", [1, 2])
def test_array_block_matvec_matches_dense(degree):
    """The ArrayBlock matvec must reproduce the dense Z @ x: the P x P element
    grid tiles Z exactly, self-blocks are exact dense, coupling is ACA to tol."""
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    sim = _array_sim(offsets, [half] * 4, nsegs=14, degree=degree)
    Z = _dense_Z(sim)
    n = Z.shape[0]
    AB = sim.build_array_blocks(tol=1e-7)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    rel = np.linalg.norm(AB.matvec(x) - Z @ x) / np.linalg.norm(Z @ x)
    assert rel < 1e-4


def test_array_block_matmat_equals_columnwise_matvec():
    """The batched matmat must equal applying matvec to each column — the
    correctness contract the block-GMRES solve relies on."""
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    sim = _array_sim(offsets, [half] * 4, nsegs=14)
    AB = sim.build_array_blocks(tol=1e-7)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((AB.n, 5)) + 1j * rng.standard_normal((AB.n, 5))
    Y = AB.matmat(X)
    Ycol = np.column_stack([AB.matvec(X[:, j]) for j in range(X.shape[1])])
    # GEMM (matmat) vs GEMV (matvec) round differently at ~1e-15 relative.
    assert np.linalg.norm(Y - Ycol) / np.linalg.norm(Ycol) < 1e-12


def test_array_block_to_dense_reconstruction():
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    sim = _array_sim(offsets, [half] * 4, nsegs=14)
    Z = _dense_Z(sim)
    AB = sim.build_array_blocks(tol=1e-7)
    rel = np.abs(AB.to_dense() - Z).max() / np.abs(Z).max()
    assert rel < 1e-3


def test_array_block_reuses_one_self_block_per_shape():
    """Identical elements ⇒ a single dense self-block shared by all of them."""
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    sim = _array_sim(offsets, [half] * 4, nsegs=14)
    AB = sim.build_array_blocks()
    assert len(AB.shape_blocks) == 1  # one shape ⇒ one stored self-block
    assert AB.stats()["n_shapes"] == 1


def test_array_block_symmetry_uses_transposed_factors():
    """Z is complex-symmetric, so block (b,a) is stored as the transpose of
    (a,b)'s factors — both directions present, reconstructing Z[b,a]=Z[a,b]^T."""
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0)]
    sim = _array_sim(offsets, [half] * 3, nsegs=14)
    AB = sim.build_array_blocks(tol=1e-8)
    pairs = {(a, b) for a, b, _, _ in AB.coupling}
    P = len(AB.groups)
    assert pairs == {(a, b) for a in range(P) for b in range(P) if a != b}
    # spot-check Z[b,a] == Z[a,b]^T from the stored factors
    blk = {(a, b): U @ V for a, b, U, V in AB.coupling}
    for a in range(P):
        for b in range(P):
            if a != b:
                assert np.allclose(blk[(b, a)], blk[(a, b)].T, atol=1e-10)


# ---- P2: block-Jacobi GMRES solve -------------------------------------------


def _bent_element(h, y=0.0):
    """Two wires meeting at a right-angle junction (an L), offset in y. Gives
    each element an internal junction → exercises the KCL saddle path."""
    w0 = np.array([[0.0, y, 0.0], [0.0, y, h]])
    w1 = np.array([[0.0, y, h], [0.0, y + h, h]])
    return [w0, w1]


def _bent_array(n_elem, h, dy=20.0, nsegs=12, degree=2):
    wires = []
    junctions = []
    feeds = []
    for e in range(n_elem):
        w0, w1 = _bent_element(h, y=e * dy)
        base = len(wires)
        wires += [w0, w1]
        junctions.append([(base, "end"), (base + 1, "start")])
        feeds.append((base, None, 1.0 + 0.0j))
    common = dict(
        wires=wires,
        degree=degree,
        n_per_edge_per_wire=[[nsegs]] * len(wires),
        wavelength=22.0,
        junctions=junctions,
        feeds=feeds,
    )
    return common


def test_compute_y_matrix_matches_dense_no_junction():
    """Multi-dipole array (no junctions) — plain block-Jacobi GMRES on Z."""
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    dense = _array_sim(offsets, [half] * 4, nsegs=16)
    arr = _array_sim(offsets, [half] * 4, nsegs=16)
    # the dense reference is the base solver on the same mesh
    yd = BSplineSolver(
        wires=[w for w in dense.wires_polylines],
        degree=2,
        n_per_edge_per_wire=[[16]] * 4,
        wavelength=22.0,
        feeds=[(i, None, 1.0 + 0.0j) for i in range(4)],
    ).compute_y_matrix()
    ya = arr.compute_y_matrix()
    assert np.abs(ya - yd).max() / np.abs(yd).max() < 1e-4
    assert max(arr._last_solve_iters) < 30


@pytest.mark.parametrize("degree", [1, 2])
def test_compute_y_matrix_matches_dense_with_junctions(degree):
    """Array of L-shaped elements with internal junctions — exercises the
    block-Jacobi preconditioner augmented with the KCL saddle rows."""
    h = 0.962 * 22 / 4
    common = _bent_array(3, h, nsegs=14, degree=degree)
    yd = BSplineSolver(**common).compute_y_matrix()
    ya = ArrayBlockSolver(**common).compute_y_matrix()
    assert np.abs(ya - yd).max() / np.abs(yd).max() < 1e-4


@pytest.mark.parametrize("with_junctions", [False, True])
def test_block_jacobi_precond_matches_sparse(with_junctions):
    """The per-element block-Jacobi preconditioner inverts the *same* augmented
    matrix as the generic sparse-LU preconditioner — it just exploits the
    element block-diagonal structure — so applying either to the same vector
    must agree (and hence GMRES convergence is identical)."""
    from momwire.array_block import _BlockJacobiAugPrecond
    from momwire.hmatrix import _SparseAugPrecond

    half = 0.962 * 22 / 4
    if with_junctions:
        sim = ArrayBlockSolver(**_bent_array(3, half, nsegs=14))
    else:
        sim = _array_sim([(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0)], [half] * 3, nsegs=14)
    H = sim.build_array_blocks()
    kcl = sim._context()["kcl_A"]
    nc = kcl.shape[0]
    N = H.n + nc
    rng = np.random.default_rng(0)
    R = rng.standard_normal((N, 3)) + 1j * rng.standard_normal((N, 3))
    sparse = _SparseAugPrecond(sim._near_sparse(H, H.n), kcl)
    bjac = _BlockJacobiAugPrecond(H, kcl)
    ref = sparse.solve(R)
    assert np.linalg.norm(bjac.solve(R) - ref) / np.linalg.norm(ref) < 1e-10


def test_solve_converges_in_few_iterations():
    """Weak inter-element coupling ⇒ the block-Jacobi preconditioner gives a
    small, RHS-flat GMRES iteration count."""
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    arr = _array_sim(offsets, [half] * 4, nsegs=20)
    arr.compute_y_matrix()
    iters = arr._last_solve_iters
    assert max(iters) < 25
    assert max(iters) - min(iters) <= 2  # ~flat across RHS


def test_compute_impedance_swept_matches_dense():
    """The array-block frequency sweep must run (overriding the dense base
    sweep, whose `same_edge_prep` arg the accelerated compute_impedance doesn't
    accept) and match the dense bspline swept impedance per port."""
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    arr = _array_sim(offsets, [half] * 4, nsegs=16)
    dense = BSplineSolver(
        wires=list(arr.wires_polylines),
        degree=2,
        n_per_edge_per_wire=[[16]] * 4,
        wavelength=22.0,
        feeds=[(i, None, 1.0 + 0.0j) for i in range(4)],
    )
    k0 = 2 * np.pi / 22.0
    k_array = np.linspace(0.9 * k0, 1.1 * k0, 4)
    za = arr.compute_impedance_swept(k_array)
    zd = dense.compute_impedance_swept(k_array)
    assert za.shape == zd.shape == (4, 4)
    assert np.max(np.abs(za - zd) / np.abs(zd)) < 1e-3


# ---- P3: identical-element + block-Toeplitz coupling reuse -------------------


def test_toeplitz_coupling_reuse_uniform_grid():
    """A uniform line array of identical elements has only as many distinct
    coupling blocks as there are distinct displacements: for 4 equally spaced
    elements the displacements are {1,2,3}·spacing, so ACA runs 3 times (not
    12), the rest reused by displacement + complex-symmetry transpose."""
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    sim = _array_sim(offsets, [half] * 4, nsegs=16)
    sim.build_array_blocks()
    assert sim._last_n_coupling_aca == 3


def test_coupling_reuse_preserves_answer():
    """Reused coupling blocks must give the same dense reconstruction as a
    fresh per-pair computation would (displacement equivalence is exact)."""
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    sim = _array_sim(offsets, [half] * 4, nsegs=16)
    Z = _dense_Z(sim)
    AB = sim.build_array_blocks(tol=1e-7)
    # fewer ACA solves than ordered pairs, yet still reconstructs Z
    assert sim._last_n_coupling_aca < 4 * 3
    assert np.abs(AB.to_dense() - Z).max() / np.abs(Z).max() < 1e-3


def test_one_self_block_per_shape_reused_across_elements():
    """Two shape classes in a 4-element line ⇒ exactly two stored self-blocks,
    each shared by its same-shape elements."""
    long_h = 0.962 * 22 / 4
    short_h = 0.7 * long_h
    offsets = [(-9.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (9.0, 0.0)]
    sim = _array_sim(offsets, [long_h, short_h, long_h, short_h], nsegs=14)
    AB = sim.build_array_blocks()
    assert len(AB.shape_blocks) == 2
    # element 0 and 2 (same shape) reference the same stored block object
    s0 = AB.shape_blocks[int(AB.shape_of_elem[0])]
    s2 = AB.shape_blocks[int(AB.shape_of_elem[2])]
    assert s0 is s2


# ---- P4: animation factor-cache --------------------------------------------


def test_phase_sweep_reuses_operator_and_factorization():
    """A phase/excitation sweep holds geometry fixed and only changes the feed
    voltages, so the assembled operator and its factorization are reused across
    frames (only the RHS back-substitution re-runs)."""
    reset_array_caches()
    half = 0.962 * 22 / 4
    offsets = [(-6.0, 0.0), (6.0, 0.0)]
    s0 = float(cache_stats()["operator_build"])
    z_frames = []
    for ph_deg in (0.0, 45.0, 90.0):
        v1 = np.exp(1j * np.deg2rad(ph_deg))
        sim = _array_solver(offsets, [half, half], [1.0 + 0j, v1], ArrayBlockSolver)
        z_frames.append(np.atleast_1d(sim.compute_impedance()[0]))
    st = cache_stats()
    assert st["operator_build"] - s0 == 1  # built once
    assert st["operator_hit"] >= 2  # reused on the later frames
    # the factorization is cached on the operator (reused, not refactored)
    op = ArrayBlockSolver(
        wires=[_dipole_wire(half, *o) for o in [(-6, 0), (6, 0)]],
        degree=2,
        n_per_edge_per_wire=[[14], [14]],
        wavelength=22.0,
        feeds=[(0, None, 1.0 + 0j), (1, None, 1.0 + 0j)],
    )._build_operator()
    assert hasattr(op, "_factored")


def test_phase_sweep_cached_result_matches_dense():
    """Reuse must not change the answer: each cached phase frame matches a
    from-scratch dense bspline solve at the same excitation."""
    reset_array_caches()
    half = 0.962 * 22 / 4
    offsets = [(-6.0, 0.0), (6.0, 0.0)]
    for ph_deg in (0.0, 60.0, 120.0):
        v = [1.0 + 0j, np.exp(1j * np.deg2rad(ph_deg))]
        za = _array_solver(
            offsets, [half, half], v, ArrayBlockSolver
        ).compute_impedance()[0]
        zd = _array_solver(offsets, [half, half], v, BSplineSolver).compute_impedance()[
            0
        ]
        za, zd = np.atleast_1d(za), np.atleast_1d(zd)
        assert np.max(np.abs(za - zd) / np.abs(zd)) < 1e-3


def test_spacing_sweep_reuses_self_blocks():
    """A spacing sweep moves identical elements, so the dense self-block
    assembly is reused across frames (the operator rebuilds for the new
    coupling, but the self-blocks are cache hits) and stays correct vs dense."""
    reset_array_caches()
    half = 0.962 * 22 / 4
    for spacing in (6.0, 8.0, 10.0):
        offs = [(-spacing, 0.0), (spacing, 0.0)]
        v = [1.0 + 0j, 1.0 + 0j]
        ya = _array_solver(offs, [half, half], v, ArrayBlockSolver).compute_y_matrix()
        yd = _array_solver(offs, [half, half], v, BSplineSolver).compute_y_matrix()
        assert np.abs(ya - yd).max() / np.abs(yd).max() < 1e-3
    st = cache_stats()
    # one shape, built once, then reused on the two later spacings
    assert st["self_block_build"] == 1
    assert st["self_block_hit"] >= 2
    assert st["operator_build"] == 3  # new geometry each frame


def test_reset_array_caches_clears_state():
    reset_array_caches()
    half = 0.962 * 22 / 4
    _array_solver(
        [(-6.0, 0.0), (6.0, 0.0)], [half, half], [1, 1], ArrayBlockSolver
    ).compute_y_matrix()
    assert sum(cache_stats().values()) > 0
    reset_array_caches()
    assert all(v == 0 for v in cache_stats().values())


# ---- PEC ground: per-block image term --------------------------------------


def _ground_array(offsets, halves, solver, ground_z=0.0, nsegs=14, degree=2):
    """`solver` for a dipole array above a PEC plane at `ground_z`. `offsets`
    are (y, z) element centres; raising z above the plane gives a near-field
    self-image. Mirrors `_array_solver` but with `ground_z` set."""
    wires = [_dipole_wire(h, y=y, z=z) for (y, z), h in zip(offsets, halves)]
    return solver(
        wires=wires,
        degree=degree,
        n_per_edge_per_wire=[[nsegs]] * len(wires),
        wavelength=22.0,
        feeds=[(i, None, 1.0 + 0.0j) for i in range(len(wires))],
        ground_z=ground_z,
    )


def _dense_Z_ground(sim):
    """The exact dense bspline Z under PEC ground (free-space minus the image
    assembly) the array-block decomposition must reproduce."""
    geom = sim._build_geometry()
    supp_seg, polys, _a, _wk, _wbg = sim._build_basis_polynomials(geom)
    Z = sim._assemble_Z(sim._build_J_blocks(geom, sim.k), supp_seg, polys, geom)
    J_img = sim._build_J_image_blocks(geom, sim.k)
    td_img = sim._image_tangent_dot(geom["tangents"])
    return Z - sim._assemble_Z(J_img, supp_seg, polys, geom, td_all=td_img)


def test_ground_array_block_matvec_matches_dense():
    """The grounded array-block operator (self-image folded into the self-blocks,
    real+image folded into each coupling block) reproduces the dense PEC Z @ x."""
    reset_array_caches()
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 3.0), (-3.0, 3.0), (3.0, 3.0), (9.0, 3.0)]
    sim = _ground_array(offsets, [half] * 4, ArrayBlockSolver, nsegs=16)
    Z = _dense_Z_ground(sim)
    AB = sim.build_array_blocks(tol=1e-7)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(Z.shape[0]) + 1j * rng.standard_normal(Z.shape[0])
    assert np.linalg.norm(AB.matvec(x) - Z @ x) / np.linalg.norm(Z @ x) < 1e-4
    assert np.abs(AB.to_dense() - Z).max() / np.abs(Z).max() < 1e-4


def test_ground_compute_y_matrix_matches_dense():
    """ArrayBlock + PEC ground matches the dense bspline + PEC ground Y."""
    reset_array_caches()
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 3.0), (-3.0, 3.0), (3.0, 3.0), (9.0, 3.0)]
    ya = _ground_array(
        offsets, [half] * 4, ArrayBlockSolver, nsegs=16
    ).compute_y_matrix()
    yd = _ground_array(offsets, [half] * 4, BSplineSolver, nsegs=16).compute_y_matrix()
    assert np.abs(ya - yd).max() / np.abs(yd).max() < 1e-4


def test_ground_compute_impedance_matches_dense():
    """ArrayBlock + PEC ground matches the dense bspline + PEC ground impedance."""
    reset_array_caches()
    half = 0.962 * 22 / 4
    offsets = [(-6.0, 3.0), (6.0, 3.0)]
    za = np.atleast_1d(
        _ground_array(offsets, [half] * 2, ArrayBlockSolver).compute_impedance()[0]
    )
    zd = np.atleast_1d(
        _ground_array(offsets, [half] * 2, BSplineSolver).compute_impedance()[0]
    )
    assert np.max(np.abs(za - zd) / np.abs(zd)) < 1e-3


def test_ground_iteration_count_near_free_space():
    """The per-block image term doesn't degrade the block-Jacobi conditioning:
    GMRES under PEC ground converges in about the same number of iterations as
    free space (the self-image stays inside the per-shape self-block, so the
    preconditioner remains near-exact)."""
    reset_array_caches()
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 3.0), (-3.0, 3.0), (3.0, 3.0), (9.0, 3.0)]
    free = _array_sim([(y, 0.0) for y, _z in offsets], [half] * 4, nsegs=16)
    free.compute_y_matrix()
    grnd = _ground_array(offsets, [half] * 4, ArrayBlockSolver, nsegs=16)
    grnd.compute_y_matrix()
    assert abs(max(grnd._last_solve_iters) - max(free._last_solve_iters)) <= 2


def test_ground_single_height_grid_reuses_one_block_per_shape():
    """A single-height grid of identical elements keeps the grid reuse intact
    under ground: one self-block for the shape and the displacement-keyed
    coupling reuse still collapses the pairs (the height insight)."""
    reset_array_caches()
    half = 0.962 * 22 / 4
    offsets = [(-9.0, 3.0), (-3.0, 3.0), (3.0, 3.0), (9.0, 3.0)]
    sim = _ground_array(offsets, [half] * 4, ArrayBlockSolver, nsegs=16)
    AB = sim.build_array_blocks()
    assert len(AB.shape_blocks) == 1  # one shape, one height
    assert sim._last_n_coupling_aca == 3  # displacements {1,2,3}·spacing


def test_ground_mixed_height_refines_blocks_and_stays_correct():
    """Elements of one geometric shape at two heights need two distinct
    self-blocks under ground (the self-image depends on height), and the result
    still matches the dense PEC solve."""
    reset_array_caches()
    half = 0.962 * 22 / 4
    offsets = [(-6.0, 3.0), (6.0, 3.0), (-6.0, 9.0), (6.0, 9.0)]
    sim = _ground_array(offsets, [half] * 4, ArrayBlockSolver, nsegs=14)
    AB = sim.build_array_blocks()
    # one geometric shape, but two heights ⇒ two block-shape classes
    assert len(AB.shape_blocks) == 2
    ya = _ground_array(
        offsets, [half] * 4, ArrayBlockSolver, nsegs=14
    ).compute_y_matrix()
    yd = _ground_array(offsets, [half] * 4, BSplineSolver, nsegs=14).compute_y_matrix()
    assert np.abs(ya - yd).max() / np.abs(yd).max() < 1e-4


def test_ground_and_free_self_blocks_do_not_alias():
    """The self-block cache key folds in ground height, so a free-space build and
    a grounded build of the same geometry never reuse each other's self-block."""
    reset_array_caches()
    half = 0.962 * 22 / 4
    offsets = [(-6.0, 3.0), (6.0, 3.0)]
    free = _array_sim([(y, z) for y, z in offsets], [half] * 2, nsegs=14)
    free.build_array_blocks()
    grnd = _ground_array(offsets, [half] * 2, ArrayBlockSolver, nsegs=14)
    grnd.build_array_blocks()
    # the grounded build must assemble its own self-block, not reuse the
    # free-space one cached under the same translation-invariant signature
    assert cache_stats()["self_block_build"] == 2
