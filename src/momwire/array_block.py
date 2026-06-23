"""Element-aware block-low-rank solver for antenna arrays.

`ArrayBlockSolver` is a structural accelerator for the B-spline MoM, a sibling
of `HMatrixSolver`. Where the generic H-matrix partitions the impedance matrix
with a geometry-blind binary space-partition cluster tree, this solver uses the
*structural* partition an array hands it for free: a `P`-element array is a
`P x P` grid of blocks — strong dense self-blocks on the diagonal, weak
low-rank coupling blocks off it.

See `docs/array_block_solver_plan.md` for the design and the validated
measurements behind it. This file is being built up phase by phase:

  * P0: element grouping (the key enabler) + the assumption verification
    harness. `element_groups` maps each basis to its array element via
    connected components of the wire graph and labels elements by geometric
    shape class; `ArrayPartition` bundles the grouping. The verification
    routines confirmed the design's load-bearing measurements (self-blocks
    identical within a shape class, weak + low-rank coupling).

  * P1: `ArrayBlockSolver.build_array_blocks()` assembles the impedance matrix
    as an `ArrayBlock` — one dense self-block per distinct shape class (reused
    across same-shape elements; the P0-verified ~2e-12 identity) plus an
    ACA-compressed low-rank coupling block per element pair, with the
    complex-symmetry `Z_ba = Z_ab^T` halving the ACA work. The container has a
    fast `matvec` that reproduces the dense `Z @ x`.

  * P2: the constrained solve. `ArrayBlock` exposes its dense self-blocks as
    `.near`, so `ArrayBlockSolver` runs the inherited `_solve_hmatrix`
    augmented-GMRES verbatim — the block-diagonal self-blocks become the
    block-Jacobi preconditioner and the KCL junction constraints go in the
    saddle rows. Because coupling is ~1e-4 of the self-blocks, GMRES converges
    in a handful of iterations (5 on `invveearray`, 9 on `bowtiearray2x4`);
    impedance/Y match dense `BSplineSolver` to ~1e-5.

  * P3: identical-element + block-Toeplitz reuse. Self-blocks are one-per-shape.
    Coupling blocks are deduplicated by `(shape_a, shape_b, displacement)`:
    free-space translation invariance makes all pairs with that key the same
    block, so ACA runs once per unique key (+complex-symmetry transpose),
    collapsing the `P(P-1)` pairs to a handful of displacements on a regular
    grid — 56→13 on `bowtiearray2x4`, 12→5 on `invveearray`, 12→3 on a uniform
    4-element line.

  * P4 (this commit): the animation factor-cache. Two module-level caches let
    an animation sweep reuse the expensive work across frames:
      - operator cache (keyed by full geometry + k + tol): a *phase/excitation*
        sweep holds geometry fixed, so Z and its factored preconditioner are
        reused wholesale — each frame is just new-RHS back-substitutions.
      - self-block cache (keyed by an element's translation-invariant geometry
        signature + k + radius): a *spacing* sweep keeps identical elements, so
        the dense self-block assembly is reused across frames and only the
        cheap coupling blocks recompute.
    The factorisation itself is cached on the operator (see
    `HMatrixSolver._factored_solve`), so a reused operator never refactors.

  * Solve internals (post-profiling): the constrained solve is a
    left-preconditioned *block* GMRES over all RHS at once (batched `matmat` +
    batched preconditioner apply, BLAS-3), and the preconditioner is a
    per-element block-Jacobi (`_BlockJacobiAugPrecond`): because elements are
    electrically separate the augmented saddle is block-diagonal, so it factors
    once per shape with a dense LU and applies as one wide `lu_solve` per shape.
    Both are exact reformulations — convergence and accuracy are unchanged — and
    cut the dominant repeated-apply costs the profiler flagged (~2× on the
    preconditioner apply for `bowtiearray2x4`).

`ArrayBlockSolver` subclasses `HMatrixSolver`, reusing its `_context`, `zblock`,
the C++ off-edge block assembler, and `aca_partial` verbatim; the grouping
reuses BSplineSolver's geometry/basis build. Nothing here touches the kernel.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ._aca import aca_partial
from .hmatrix import (
    _HAVE_OFFEDGE_BLOCK_ACCEL,
    _OFFEDGE_BLOCK_ACCEL_MAX_D,
    HMatrixSolver,
)


# ----------------------------------------------------------------------
# Animation factor-cache (P4): module-level reuse across solves/frames
# ----------------------------------------------------------------------

# Whole-operator cache (keyed by geometry + k + tol): a phase/excitation sweep
# holds geometry fixed, so the assembled ArrayBlock and its factored
# preconditioner are reused wholesale and each frame only re-solves the RHS.
_ARRAY_OP_CACHE: dict = {}
_ARRAY_OP_CACHE_MAX = 8

# Self-block cache (keyed by an element's translation-invariant geometry
# signature + k + radius + basis): a spacing sweep keeps identical elements, so
# the dense self-block assembly is reused while only coupling recomputes.
_SELF_BLOCK_CACHE: dict = {}
_SELF_BLOCK_CACHE_MAX = 64

# Instrumentation so tests (and the demo script) can prove reuse happened.
_CACHE_STATS = {
    "operator_build": 0,
    "operator_hit": 0,
    "self_block_build": 0,
    "self_block_hit": 0,
}


def reset_array_caches():
    """Clear the animation caches and zero the hit/build counters."""
    _ARRAY_OP_CACHE.clear()
    _SELF_BLOCK_CACHE.clear()
    for key in _CACHE_STATS:
        _CACHE_STATS[key] = 0


def cache_stats():
    """A copy of the cache hit/build counters."""
    return dict(_CACHE_STATS)


def _cache_put(cache, key, value, max_size):
    """Insert with a crude FIFO bound so long sweeps don't grow unbounded."""
    if len(cache) >= max_size:
        cache.pop(next(iter(cache)))
    cache[key] = value


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


# ----------------------------------------------------------------------
# Block container + matvec (P1)
# ----------------------------------------------------------------------


class ArrayBlock:
    """Element-block decomposition of the impedance matrix with a fast matvec.

    The `P x P` grid of element blocks exactly tiles `Z` (the element groups
    partition all bases): diagonal blocks are dense self-blocks, off-diagonal
    blocks are low-rank coupling blocks.

    groups : list[np.ndarray]
        Per-element basis indices (from `ArrayPartition`).
    shape_of_elem : (P,) int
        Shape class of each element.
    shape_blocks : dict[int, np.ndarray]
        One dense `(N_s, N_s)` self-block per distinct shape class, applied to
        every element of that shape (same-shape self-blocks are identical to
        ~2e-12 in free space — the P0-verified reuse).
    coupling : list[(a, b, U, V)]
        Low-rank factors for every ordered off-diagonal pair: block `(a, b)`
        is `U @ V`. Stored for both directions (the `(b, a)` entry reuses the
        transposed factors of `(a, b)`).
    """

    def __init__(self, n, groups, shape_of_elem, shape_blocks, coupling):
        self.n = n
        self.groups = groups
        self.shape_of_elem = shape_of_elem
        self.shape_blocks = shape_blocks
        self.coupling = coupling
        # Dense self-blocks as (I, J, D) triples, so the block decomposition is
        # a drop-in for `HMatrixSolver._solve_hmatrix`: its near-field
        # preconditioner becomes the block-diagonal of Z (block-Jacobi), which
        # — because coupling is ~1e-4 of the self-blocks — drives GMRES to a
        # handful of iterations. `precond_extra` (the H-matrix's first-ring
        # strengthening) has no analogue here, so it is empty.
        self.near = [
            (g, g, shape_blocks[int(shape_of_elem[e])]) for e, g in enumerate(groups)
        ]
        self.precond_extra = []

    def matvec(self, x):
        x = np.asarray(x)
        y = np.zeros(self.n, dtype=np.complex128)
        for e, g in enumerate(self.groups):
            y[g] += self.shape_blocks[int(self.shape_of_elem[e])] @ x[g]
        for a, b, U, V in self.coupling:
            y[self.groups[a]] += U @ (V @ x[self.groups[b]])
        return y

    def matmat(self, X):
        """Apply the operator to all columns of X (n, nrhs) at once.

        Same block decomposition as `matvec`, but every block product becomes a
        BLAS-3 matrix-matrix multiply (`(N_s, N_s) @ (N_s, nrhs)`), which is far
        more cache- and overhead-efficient than nrhs separate matvecs — the
        point of the batched multi-RHS solve."""
        X = np.asarray(X)
        Y = np.zeros((self.n, X.shape[1]), dtype=np.complex128)
        for e, g in enumerate(self.groups):
            Y[g] += self.shape_blocks[int(self.shape_of_elem[e])] @ X[g]
        for a, b, U, V in self.coupling:
            Y[self.groups[a]] += U @ (V @ X[self.groups[b]])
        return Y

    def storage(self):
        """Complex scalars stored (distinct self-blocks + coupling factors)."""
        s = sum(D.size for D in self.shape_blocks.values())
        s += sum(U.size + V.size for _, _, U, V in self.coupling)
        return s

    def stats(self):
        ranks = [U.shape[1] for _, _, U, _ in self.coupling]
        return {
            "n": self.n,
            "n_elem": len(self.groups),
            "n_shapes": len(self.shape_blocks),
            "n_coupling": len(self.coupling),
            "storage": self.storage(),
            "dense_storage": self.n * self.n,
            "compression": self.storage() / (self.n * self.n),
            "max_rank": max(ranks) if ranks else 0,
            "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        }

    def to_dense(self):
        """Reconstruct the full dense matrix (validation / small n only)."""
        Z = np.zeros((self.n, self.n), dtype=np.complex128)
        for e, g in enumerate(self.groups):
            Z[np.ix_(g, g)] = self.shape_blocks[int(self.shape_of_elem[e])]
        for a, b, U, V in self.coupling:
            Z[np.ix_(self.groups[a], self.groups[b])] = U @ V
        return Z


class _BlockJacobiAugPrecond:
    """Block-Jacobi factorisation of the augmented near-field preconditioner
    `[Zn A^T; A 0]` for an `ArrayBlock`.

    Array elements are electrically separate, so the block-diagonal self-blocks
    `Zn` *and* the KCL constraint rows `A` are both block-diagonal by element:
    each junction's directional bases live in a single element (the grouping is
    connected-components on exactly those shared nodes). The augmented saddle is
    therefore itself block-diagonal — one small dense saddle
    `[S_e A_e^T; A_e 0]` per element. This is the *same* matrix the generic
    sparse-LU preconditioner factors, so GMRES convergence is identical; but it
    is factored once per distinct shape (`S_e` and `A_e` coincide across
    same-shape elements) with a dense LU. The apply stacks all same-shape
    elements' right-hand sides into one wide solve per shape (BLAS-3), so a
    `P`-element array costs `n_shapes` dense `lu_solve` calls per Krylov step
    instead of `P` separate ones. `.solve(R)` applies `M^{-1}` to an augmented
    block R (N, nrhs).
    """

    def __init__(self, ablock, kcl_A):
        self.n = ablock.n
        nc = kcl_A.shape[0] if kcl_A is not None else 0
        self.nc = nc
        groups = ablock.groups
        shape_of_elem = ablock.shape_of_elem
        shape_blocks = ablock.shape_blocks

        # Group elements that share an augmented factorisation (same shape and
        # same local KCL structure) so their solves batch into one wide RHS.
        by_key = {}  # (shape_id, A_e bytes) -> dict(fac, Ns, members=[(g, rows)])
        claimed = 0
        for e, g in enumerate(groups):
            Ns = g.size
            if nc > 0:
                rows = np.nonzero(np.abs(kcl_A[:, g]).sum(axis=1) > 0)[0].astype(
                    np.int64
                )
                A_e = (
                    kcl_A[np.ix_(rows, g)].astype(np.complex128) if rows.size else None
                )
            else:
                rows = np.empty(0, dtype=np.int64)
                A_e = None
            claimed += rows.size
            sid = int(shape_of_elem[e])
            key = (sid, A_e.tobytes() if A_e is not None else b"")
            grp = by_key.get(key)
            if grp is None:
                S = shape_blocks[sid]
                if A_e is not None:
                    nce = rows.size
                    M = np.zeros((Ns + nce, Ns + nce), dtype=np.complex128)
                    M[:Ns, :Ns] = S
                    M[:Ns, Ns:] = A_e.T
                    M[Ns:, :Ns] = A_e
                else:
                    M = np.ascontiguousarray(S)
                grp = {"fac": lu_factor(M), "Ns": Ns, "members": []}
                by_key[key] = grp
            grp["members"].append((g, rows))
        self._shapes = list(by_key.values())
        # Every constraint row must belong to exactly one element (no junction
        # bridges elements — that is what the connectivity grouping guarantees).
        assert claimed == nc, f"KCL rows {claimed} claimed != {nc} (cross-element?)"

    def solve(self, R):
        R = np.asarray(R)
        out = np.empty_like(R)
        n = self.n
        s = R.shape[1]
        for grp in self._shapes:
            fac, Ns, members = grp["fac"], grp["Ns"], grp["members"]
            k = len(members)
            nce = fac[0].shape[0] - Ns
            rhs = np.empty((Ns + nce, k * s), dtype=np.complex128)
            for j, (g, rows) in enumerate(members):
                rhs[:Ns, j * s : (j + 1) * s] = R[g]
                if nce:
                    rhs[Ns:, j * s : (j + 1) * s] = R[n + rows]
            sol = lu_solve(fac, rhs, check_finite=False, overwrite_b=True)
            for j, (g, rows) in enumerate(members):
                out[g] = sol[:Ns, j * s : (j + 1) * s]
                if nce:
                    out[n + rows] = sol[Ns:, j * s : (j + 1) * s]
        return out


class ArrayBlockSolver(HMatrixSolver):
    """Element-aware block-low-rank accelerator for arrays of identical (or
    few-shape) elements. Drop-in for `HMatrixSolver` (same constructor).

    P1 adds `array_partition()` (cached element grouping) and
    `build_array_blocks()` (the `ArrayBlock` assembly + matvec). The solve and
    `compute_impedance` / `compute_y_matrix` overrides arrive in P2; until then
    they resolve to the dense `BSplineSolver` path via the base class.
    """

    def _hmatrix_unsupported(self):
        """ArrayBlock supports PEC ground via the per-block image term (the
        free-space block reuse survives the image method for a grid array — see
        `build_array_blocks`), so it only falls back to the dense path for
        singular enrichment, which still belongs there. This narrows the base
        `HMatrixSolver` gate, which keeps excluding ground for the generic
        hierarchical solver."""
        return self.use_singular_enrichment

    def array_partition(self, tol=1e-6):
        """Element/shape partition of the bases (cached)."""
        cached = getattr(self, "_array_partition", None)
        if cached is None:
            cached = element_groups(self, tol=tol)
            self._array_partition = cached
        return cached

    def _self_block_key(self, ctx, segs, k):
        """Content-addressed key for an element's dense self-block: the
        element's segment endpoints recentred on its own centroid (so it is
        translation-invariant — identical elements at different array positions
        share a key), rounded and canonically ordered, plus the parameters the
        self-impedance depends on (k, wire radius, degree, quadrature).

        Under PEC ground the self-block also carries the self-image term, which
        depends on the element's *height above the ground plane* (the image sits
        at ``2·ground_z − z``), so the centroid height ``cen_z − ground_z`` joins
        the key — same-height translates still share a block (the grid case),
        while a free-space block (``ground_z is None``) never aliases a grounded
        one and elements at different heights get distinct blocks."""
        seg_l, seg_r = ctx["seg_l"][segs], ctx["seg_r"][segs]
        cen = 0.5 * (seg_l + seg_r).mean(axis=0)
        rel = np.hstack([seg_l - cen, seg_r - cen])
        keyarr = np.round(rel / 1e-6).astype(np.int64)
        mid = 0.5 * (keyarr[:, :3] + keyarr[:, 3:])
        order = np.lexsort((mid[:, 2], mid[:, 1], mid[:, 0]))
        sig = keyarr[order].tobytes()
        gkey = (
            None
            if self.ground_z is None
            else int(np.round((cen[2] - self.ground_z) / 1e-6))
        )
        return (
            sig,
            float(k),
            float(self.wire_radius),
            self.degree,
            self.n_qp_pair,
            gkey,
        )

    def _build_operator(self):
        """Build the array-block operator (cached) for the constrained solve.

        `compute_impedance` / `compute_y_matrix` (inherited from
        `HMatrixSolver`) run the same GMRES on it via `_solve_hmatrix`, with the
        block-diagonal self-blocks as the block-Jacobi preconditioner.

        The assembled operator is cached at module scope keyed by the full
        geometry + k + tol, so an animation *phase/excitation* sweep — geometry
        fixed, only the RHS changes — reuses both the operator and the
        factorisation cached on it (`HMatrixSolver._factored_solve`), making each
        frame a cheap multi-RHS back-substitution."""
        key = (
            self._geometry_cache_key(),
            float(self.k),
            float(self.wire_radius),
            self.degree,
            self.n_qp_pair,
            float(self.aca_tol),
            None if self.ground_z is None else float(self.ground_z),
        )
        op = _ARRAY_OP_CACHE.get(key)
        if op is None:
            op = self.build_array_blocks()
            op._n_coupling_aca = self._last_n_coupling_aca
            _cache_put(_ARRAY_OP_CACHE, key, op, _ARRAY_OP_CACHE_MAX)
            _CACHE_STATS["operator_build"] += 1
        else:
            _CACHE_STATS["operator_hit"] += 1
        self._last_n_coupling_aca = op._n_coupling_aca
        return op

    def _make_preconditioner(self, H, kcl_A):
        """Per-element block-Jacobi factorisation of the augmented near-field
        preconditioner — the same matrix the generic sparse-LU path factors,
        but block-diagonal by element so it factors once per shape and applies
        as batched dense solves (see `_BlockJacobiAugPrecond`)."""
        return _BlockJacobiAugPrecond(H, kcl_A)

    def _coupling_aca(self, ctx, I, J, k, tol, use_accel):
        """ACA low-rank factors (U, V) of the off-diagonal element block
        Z[I][:, J]. The two elements share no segments, so the block is purely
        off-edge (no same-edge analytic overwrite) and well separated ⇒ low
        rank. Reuses the shared off-edge ACA evaluators, which fold in the PEC
        image term (``Z_free − Z_image``) when ground is on — one factor pair
        per pair, so the matvec and block-Jacobi machinery are unchanged."""
        get_row, get_col, _dense = self._offedge_aca_evaluators(ctx, I, J, k, use_accel)
        U, V = aca_partial(get_row, get_col, I.size, J.size, tol=tol)
        return U, V

    def build_array_blocks(self, tol=None, k=None):
        """Assemble the impedance matrix as an `ArrayBlock`.

        One dense self-block per shape class (built once from a representative
        element and reused across same-shape elements), plus a low-rank
        coupling block per element pair.

        Coupling reuse (block-Toeplitz, generalised): a coupling block depends
        only on the two elements' shapes and their relative displacement (the
        free-space kernel is translation-invariant), so all pairs sharing a
        `(shape_a, shape_b, displacement)` key are the *same* block — ACA runs
        once per unique key. On a regular grid this collapses the `P(P-1)`
        pairs to a handful of displacements. Complex symmetry `Z_ba = Z_ab^T`
        is folded in too: a key whose reverse is already cached reuses the
        transposed factors instead of a fresh ACA. `self._last_n_coupling_aca`
        records how many ACA solves actually ran.
        """
        if tol is None:
            tol = self.aca_tol
        if k is None:
            k = self.k
        part = self.array_partition()
        ctx = self._context()
        n = ctx["n_basis"]

        # Element centroids, from each element's own segment midpoints (translated
        # elements differ by exactly the displacement, so rounding to the grouping
        # tol is safe). Computed from seg_groups, not ctx["basis_centroid"] — the
        # latter is polluted by boundary-basis support padding, which would
        # perturb the displacement/height keys.
        seg_mid = 0.5 * (ctx["seg_l"] + ctx["seg_r"])
        cen = np.array([seg_mid[sg].mean(axis=0) for sg in part.seg_groups])
        disp_tol = 1e-6

        # Block-shape classes. In free space a block depends only on an element's
        # (translation-invariant) geometric shape. Under PEC ground both the
        # self-image and the coupling-image terms also depend on the element's
        # height above the plane (the image sits at 2·ground_z − z), so refine the
        # shape classes by height: same-shape elements at the same height share a
        # block — the single-height grid case (e.g. bowtiearray2x4), full reuse —
        # while different heights get distinct blocks (correct; reuse degrades
        # gracefully, the solve stays fast). `shp[e]` is element e's dense
        # block-class id; free space leaves it equal to the geometric shape id.
        if self.ground_z is None:
            shp = np.asarray(part.shape_of_elem)
            reps = part.shape_representatives()
        else:
            label_of = {}
            reps = []
            shp = np.empty(part.n_elem, dtype=np.int64)
            for e in range(part.n_elem):
                hk = int(np.round((cen[e][2] - self.ground_z) / disp_tol))
                key = (int(part.shape_of_elem[e]), hk)
                lab = label_of.get(key)
                if lab is None:
                    lab = len(label_of)
                    label_of[key] = lab
                    reps.append(e)
                shp[e] = lab

        # Dense self-block per distinct block-shape, from a representative element.
        # Cached by the element's translation-invariant geometry signature (plus
        # height under ground) so a spacing sweep reuses the assembly.
        shape_blocks = {}
        for s, e in enumerate(reps):
            g = part.groups[e]
            sb_key = self._self_block_key(ctx, part.seg_groups[e], k)
            blk = _SELF_BLOCK_CACHE.get(sb_key)
            if blk is None:
                blk = self.zblock(g, g, k=k)
                if self.ground_z is not None:
                    # Self-image reaction: the element against its own mirror. One
                    # block per (shape, height) class, so the block-Jacobi
                    # preconditioner stays (near-)exact and the
                    # one-factorisation-per-class cost is unchanged.
                    blk = blk - self._zblock_image(g, g, k=k)
                _cache_put(_SELF_BLOCK_CACHE, sb_key, blk, _SELF_BLOCK_CACHE_MAX)
                _CACHE_STATS["self_block_build"] += 1
            else:
                _CACHE_STATS["self_block_hit"] += 1
            shape_blocks[s] = blk

        use_accel = (
            _HAVE_OFFEDGE_BLOCK_ACCEL
            and self.degree <= _OFFEDGE_BLOCK_ACCEL_MAX_D
            and self.hmatrix_use_accel
        )

        # Coupling reuse: a block depends only on the two elements' block-shapes
        # and their relative displacement (the free-space kernel is
        # translation-invariant; under ground the block-shape already encodes the
        # height the image term needs, so the same key keys both the free and the
        # image contribution). All pairs sharing `(block_shape_a, block_shape_b,
        # displacement)` are the same block — ACA runs once per unique key. On a
        # single-height grid this collapses the P(P-1) pairs to a handful of
        # displacements. Complex symmetry Z_ab = Z_ba^T survives the image method
        # (both free and image blocks are symmetric), so a key whose reverse is
        # cached reuses the transposed factors instead of a fresh ACA.
        coupling = []
        cache = {}  # (block_shape_a, block_shape_b, disp_key) -> (U, V)
        n_aca = 0
        P = part.n_elem
        for a in range(P):
            for b in range(P):
                if a == b:
                    continue
                sa, sb = int(shp[a]), int(shp[b])
                dkey = tuple(np.round((cen[b] - cen[a]) / disp_tol).astype(np.int64))
                key = (sa, sb, dkey)
                hit = cache.get(key)
                if hit is None:
                    rkey = (sb, sa, tuple(-d for d in dkey))
                    rhit = cache.get(rkey)
                    if rhit is not None:
                        # Z_ab = Z_ba^T = (U_r V_r)^T = V_r^T U_r^T.
                        hit = (rhit[1].T.copy(), rhit[0].T.copy())
                    else:
                        hit = self._coupling_aca(
                            ctx, part.groups[a], part.groups[b], k, tol, use_accel
                        )
                        n_aca += 1
                    cache[key] = hit
                coupling.append((a, b, hit[0], hit[1]))

        self._last_n_coupling_aca = n_aca
        return ArrayBlock(n, part.groups, shp, shape_blocks, coupling)
