"""Element-aware block-low-rank solver for antenna arrays.

`ArrayBlockPySim` is a structural accelerator for the B-spline MoM, a sibling
of `HMatrixPySim`. Where the generic H-matrix partitions the impedance matrix
with a geometry-blind binary space-partition cluster tree, this solver uses the
*structural* partition an array hands it for free: a `P`-element array is a
`P x P` grid of blocks — strong dense self-blocks on the diagonal, weak
low-rank coupling blocks off it.

See `docs/array_block_solver_plan.md` for the design and the validated
measurements behind it. This file is being built up phase by phase:

  * P0 (this commit): element grouping (the key enabler) + the assumption
    verification harness. `element_groups` maps each basis to its array
    element via connected components of the wire graph; `ArrayPartition`
    bundles the grouping with the per-element basis ranges. The verification
    routines confirm the design's load-bearing measurements (identical
    self-blocks, weak + low-rank coupling) before the solver is built on top.

The grouping reuses BSplinePySim's geometry/basis build verbatim; nothing
here touches the MoM kernel.
"""

import numpy as np


# ----------------------------------------------------------------------
# Element grouping (P0 — the key enabler)
# ----------------------------------------------------------------------


def _basis_to_wire(supp_seg, wire_basis_global):
    """Map each global basis index to the polyline (wire) that owns it.

    Bases are emitted wire-by-wire in `_build_basis_polynomials` (the global
    index increments through each wire's kept bases in order), so every wire
    owns a contiguous range of basis indices. Returns an (n_basis,) int array.
    """
    n_basis = supp_seg.shape[0]
    b2w = np.empty(n_basis, dtype=np.int64)
    m = 0
    for w_idx, (kept, _local_to_global) in enumerate(wire_basis_global):
        nb = len(kept)
        b2w[m : m + nb] = w_idx
        m += nb
    assert m == n_basis, f"basis count {m} != n_basis {n_basis}"
    return b2w


def _wire_to_element(wires_polylines, tol=1e-6):
    """Group polylines into electrically connected elements by shared anchors.

    Two polylines belong to the same element when any of their anchor points
    coincide (within `tol`): wires inside one array element meet end-to-end at
    junction nodes, while distinct elements are spatially separated. Pure
    geometry — no junction list or builder metadata needed.

    Returns (wire_elem, n_elem) where `wire_elem[w]` is the element id of wire
    `w`, with element ids assigned in order of first appearance (so wire 0 is
    always in element 0).
    """
    n_w = len(wires_polylines)
    parent = list(range(n_w))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Map each rounded anchor coordinate to the wires that touch it, unioning
    # every wire that shares a node. Rounding to a `tol` grid merges anchors
    # that are coincident to ~1e-12 (consecutive segments / junctions) while
    # keeping distinct elements (separated by >> tol) apart.
    coord_to_wire = {}
    for w, pl in enumerate(wires_polylines):
        for pt in np.asarray(pl, dtype=float):
            key = tuple(np.round(pt / tol).astype(np.int64))
            prev = coord_to_wire.get(key)
            if prev is None:
                coord_to_wire[key] = w
            else:
                union(prev, w)

    # Relabel component roots to dense element ids in order of first appearance.
    roots = [find(w) for w in range(n_w)]
    label_of_root = {}
    wire_elem = np.empty(n_w, dtype=np.int64)
    for w, r in enumerate(roots):
        if r not in label_of_root:
            label_of_root[r] = len(label_of_root)
        wire_elem[w] = label_of_root[r]
    return wire_elem, len(label_of_root)


def _element_segment_groups(geom, wire_elem, n_elem):
    """Segments owned by each element, as a sorted index array per element.

    Segments are laid out contiguously per wire (`geom["seg_offsets"]`), and an
    element is a set of whole wires, so each element owns a clean union of
    per-wire segment ranges — no basis-support padding to filter out.
    """
    seg_off = geom["seg_offsets"]
    seg_groups = [[] for _ in range(n_elem)]
    for w, e in enumerate(wire_elem):
        seg_groups[e].append(np.arange(seg_off[w], seg_off[w + 1], dtype=np.int64))
    return [np.concatenate(s) if s else np.zeros(0, dtype=np.int64) for s in seg_groups]


class ArrayPartition:
    """The structural partition of an array's impedance matrix into elements.

    Attributes
    ----------
    n_basis : int
        Total number of bases.
    n_elem : int
        Number of array elements found.
    elem_of_basis : (n_basis,) int
        Element id of each basis.
    groups : list[np.ndarray]
        `groups[e]` is the sorted array of basis indices in element `e`.
        Within an element bases are sorted ascending; because the array builder
        emits each element from the same element generator with identical
        segmentation, that ascending order corresponds segment-for-segment
        across elements of the same shape (the property the identical-
        self-block reuse relies on).
    sizes : (n_elem,) int
        `len(groups[e])`.
    elem_of_wire : (n_wires,) int
        Element id of each polyline.
    seg_groups : list[np.ndarray]
        `seg_groups[e]` is the sorted array of segment indices in element `e`.
    shape_of_elem : (n_elem,) int
        Shape-class id of each element: elements that are translates of one
        another (identical geometry up to a rigid shift) share a class.
        Generally up to 4 classes for a 2x4 array (itop/ibot/otop/obot), up to
        2 for a 2x2 (top/bot), 1 for a 1x2; fewer when params coincide (e.g.
        the default 2x4 has inner==outer, collapsing 4 nominal shapes to 2).
    """

    def __init__(
        self, n_basis, elem_of_basis, groups, elem_of_wire, seg_groups, shape_of_elem
    ):
        self.n_basis = n_basis
        self.n_elem = len(groups)
        self.elem_of_basis = elem_of_basis
        self.groups = groups
        self.sizes = np.array([g.size for g in groups], dtype=np.int64)
        self.elem_of_wire = elem_of_wire
        self.seg_groups = seg_groups
        self.shape_of_elem = shape_of_elem
        self.n_shapes = int(shape_of_elem.max()) + 1 if len(shape_of_elem) else 0

    def shape_representatives(self):
        """First element id of each shape class, in class-id order."""
        reps = {}
        for e, s in enumerate(self.shape_of_elem):
            if s not in reps:
                reps[int(s)] = e
        return [reps[s] for s in range(self.n_shapes)]

    def __repr__(self):
        return (
            f"ArrayPartition(n_basis={self.n_basis}, n_elem={self.n_elem}, "
            f"n_shapes={self.n_shapes}, sizes={self.sizes.tolist()})"
        )


def _shape_classes(geom, seg_groups, tol=1e-6):
    """Cluster elements into geometric shape classes (translation-invariant).

    Each element's signature is its segment midpoints recentred on the element
    centroid, sorted lexicographically and rounded to `tol`. Two elements are
    translates of one another iff their signatures match, so this finds the
    true number of distinct shapes from geometry alone — 4 for a 2x4 with
    distinct inner/outer/top/bot params, fewer when params coincide. (These
    arrays differ only by translation, not rotation/reflection; a rotated
    element would land in its own class, which is the safe behaviour.)
    """
    seg_l, seg_r = geom["seg_l"], geom["seg_r"]
    sigs = []
    for segs in seg_groups:
        mids = 0.5 * (seg_l[segs] + seg_r[segs])
        mids = mids - mids.mean(axis=0)
        key = np.round(mids / tol).astype(np.int64)
        order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
        sigs.append(key[order].tobytes() + bytes(str(key.shape), "ascii"))
    label_of_sig = {}
    shape_of_elem = np.empty(len(seg_groups), dtype=np.int64)
    for e, s in enumerate(sigs):
        if s not in label_of_sig:
            label_of_sig[s] = len(label_of_sig)
        shape_of_elem[e] = label_of_sig[s]
    return shape_of_elem


def element_groups(sim, tol=1e-6):
    """Partition `sim`'s bases into array elements.

    Composes the exact basis→wire map (contiguous, from the basis build) with
    a wire→element connected-components grouping (shared anchors), then labels
    the elements by geometric shape class. Returns an `ArrayPartition`. A
    single connected structure yields one element of all bases — correct
    degenerate behaviour, just no array structure to exploit.
    """
    geom = sim._build_geometry()
    supp_seg, _polys, _kcl_A, _wk, wire_basis_global = sim._build_basis_polynomials(
        geom
    )
    n_basis = supp_seg.shape[0]
    b2w = _basis_to_wire(supp_seg, wire_basis_global)
    wire_elem, n_elem = _wire_to_element(sim.wires_polylines, tol=tol)
    elem_of_basis = wire_elem[b2w]
    groups = [
        np.sort(np.nonzero(elem_of_basis == e)[0].astype(np.int64))
        for e in range(n_elem)
    ]
    seg_groups = _element_segment_groups(geom, wire_elem, n_elem)
    shape_of_elem = _shape_classes(geom, seg_groups, tol=tol)
    return ArrayPartition(
        n_basis, elem_of_basis, groups, wire_elem, seg_groups, shape_of_elem
    )
