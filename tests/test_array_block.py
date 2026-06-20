"""Tests for the element-aware array-block solver (P0: grouping + shapes).

Self-contained: synthetic arrays are built from translated dipoles so the
expected element/shape structure is known without antenna_designer designs.
The real array designs are exercised by scripts/array_block_verify.py.
"""

import numpy as np
import pytest

from pysim.bspline import BSplinePySim
from pysim.hmatrix import HMatrixPySim
from pysim.array_block import ArrayBlockPySim, element_groups


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
    return ArrayBlockPySim(
        wires=wires,
        degree=degree,
        n_per_edge_per_wire=[[nsegs]] * len(wires),
        nsegs=nsegs,
        wavelength=22.0,
        feeds=[(i, None, 1.0 + 0.0j) for i in range(len(wires))],
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
    sim = HMatrixPySim(
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
    yd = BSplinePySim(
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
    yd = BSplinePySim(**common).compute_y_matrix()
    ya = ArrayBlockPySim(**common).compute_y_matrix()
    assert np.abs(ya - yd).max() / np.abs(yd).max() < 1e-4


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
