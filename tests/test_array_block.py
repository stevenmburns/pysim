"""Tests for the element-aware array-block solver (P0: grouping + shapes).

Self-contained: synthetic arrays are built from translated dipoles so the
expected element/shape structure is known without antenna_designer designs.
The real array designs are exercised by scripts/array_block_verify.py.
"""

import numpy as np

from pysim.hmatrix import HMatrixPySim
from pysim.array_block import element_groups


def _dipole_wire(half, y=0.0, z=0.0):
    return np.array([[0.0, y, z - half], [0.0, y, z + half]])


def _array_sim(offsets, halves, nsegs=12, degree=2):
    """One straight dipole per (offset, half-length): identical when the
    half-lengths match, a distinct shape when they differ. `offsets` are
    (y, z) element centres far enough apart to be electrically separate."""
    wires = [_dipole_wire(h, y=y, z=z) for (y, z), h in zip(offsets, halves)]
    return HMatrixPySim(
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
