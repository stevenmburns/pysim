"""Hierarchical (H-matrix / ACA) B-spline Galerkin MoM solver.

`HMatrixPySim` is a distance-based accelerator built on top of
`BSplinePySim`. It reuses BSplinePySim's geometry build, basis-polynomial
extraction, kernels, source vectors, and KCL machinery verbatim; the only
thing it replaces is the *dense* O(N²) impedance-matrix assembly + dense LU
solve.

The plan (phased — see project notes):

  * Phase 0 (this file, to start): an on-demand block evaluator
    `zblock(I, J)` that computes any rectangular sub-block Z[I][:, J] of the
    exact dense bspline Z *without* materialising the full (d+1, d+1, N, N)
    moment tensor. Well-separated (admissible) blocks contain no same-edge
    segment pairs, so they need only the off-edge full-kernel quadrature;
    near blocks additionally overwrite same-edge pairs with the analytic
    static + regularised split, identical to `BSplinePySim._build_J_blocks`.
    This evaluator is the foundation everything else stands on: ACA can only
    be as correct as the entries it samples.

  * Phase 1: binary-space-partition cluster tree over basis centroids;
    admissibility min(diam(s), diam(t)) <= eta * dist(s, t); recursive
    block tree → {admissible far, dense near} leaves.

  * Phase 2: partial-pivoted ACA low-rank approximation of admissible
    blocks; dense near blocks; fast H-matvec.

  * Phase 3: GMRES (LinearOperator) + near-field preconditioner, KCL
    junction constraints in the augmented system.

Why distance matters here: the moment integral kernel
G = exp(-jkR)/(4πR) is smooth and asymptotically smooth once the two
B-spline basis supports are well separated, so the corresponding Z block is
numerically low rank and ACA captures it from O(r·(m+n)) sampled entries
instead of the full m·n.
"""

import numpy as np

from .bspline import BSplinePySim
from ._bspline_kernels import (
    _seg_seg_full_moments_offedge,
    _seg_seg_reg_moments,
    _seg_seg_static_moments,
)
from ._aca import (
    build_block_tree,
    build_cluster_tree,
    partition_stats,
)


class HMatrixPySim(BSplinePySim):
    """Distance-based hierarchical accelerator for the B-spline MoM.

    Drop-in for `BSplinePySim` (same constructor); Phase 0 only adds the
    block-evaluator plumbing. `compute_impedance` / `compute_y_matrix` still
    resolve to the dense BSplinePySim path until later phases override them,
    so the class is usable and correct from the start.
    """

    # ------------------------------------------------------------------
    # Shared, k-independent geometry/basis context
    # ------------------------------------------------------------------

    def _context(self):
        """Build (and memoise) the k-independent geometry + basis tables and
        the per-segment edge map that the block evaluator needs.

        Returns a dict with:
          geom, supp_seg, polys, kcl_A, wire_knots, wire_basis_global,
          seg_l, seg_r, tangents, n_basis, n_segs,
          seg_edge_id   (N_segs,)   global edge index of each segment
          seg_edge_loc  (N_segs,)   within-edge local index of each segment
          edge_arc      list[ (edge_arc_edges array) ] indexed by edge id
          basis_centroid (n_basis, 3), basis_radius (n_basis,)
        """
        cached = getattr(self, "_hm_context", None)
        if cached is not None:
            return cached

        geom = self._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            self._build_basis_polynomials(geom)
        )

        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        tangents = geom["tangents"]
        n_segs = seg_l.shape[0]
        n_basis = supp_seg.shape[0]

        # Per-segment edge map: each (wire, edge) pair gets a global edge id;
        # record which edge every global segment lives on and its local index
        # within that edge's contiguous segment run. Used to overwrite
        # same-edge pairs with the analytic static + reg moments.
        seg_edge_id = np.full(n_segs, -1, dtype=np.int64)
        seg_edge_loc = np.full(n_segs, -1, dtype=np.int64)
        edge_arc = []
        per_wire = geom["per_wire"]
        seg_off = geom["seg_offsets"]
        eid = 0
        for w in range(len(per_wire)):
            pw = per_wire[w]
            ed_off = pw["edge_offsets"]
            ed_arc = pw["edge_arc_edges"]
            base = seg_off[w]
            for i_e in range(len(ed_off) - 1):
                lo = base + ed_off[i_e]
                hi = base + ed_off[i_e + 1]
                seg_edge_id[lo:hi] = eid
                seg_edge_loc[lo:hi] = np.arange(hi - lo, dtype=np.int64)
                edge_arc.append(np.asarray(ed_arc[i_e], dtype=float))
                eid += 1

        # Basis geometric extent: axis-aligned bounding box (and centroid)
        # of the union of the segment endpoints in the basis support. The
        # cluster tree partitions on the boxes so admissibility reflects the
        # true spatial extent of each basis function's current support.
        basis_lo = np.empty((n_basis, 3), dtype=float)
        basis_hi = np.empty((n_basis, 3), dtype=float)
        basis_centroid = np.empty((n_basis, 3), dtype=float)
        for m in range(n_basis):
            segs = np.unique(supp_seg[m])
            pts = np.vstack([seg_l[segs], seg_r[segs]])
            basis_lo[m] = pts.min(axis=0)
            basis_hi[m] = pts.max(axis=0)
            basis_centroid[m] = pts.mean(axis=0)

        ctx = {
            "geom": geom,
            "supp_seg": supp_seg,
            "polys": polys,
            "kcl_A": kcl_A,
            "wire_knots": wire_knots,
            "wire_basis_global": wire_basis_global,
            "seg_l": seg_l,
            "seg_r": seg_r,
            "tangents": tangents,
            "n_basis": n_basis,
            "n_segs": n_segs,
            "seg_edge_id": seg_edge_id,
            "seg_edge_loc": seg_edge_loc,
            "edge_arc": edge_arc,
            "basis_centroid": basis_centroid,
            "basis_lo": basis_lo,
            "basis_hi": basis_hi,
        }
        self._hm_context = ctx
        return ctx

    # ------------------------------------------------------------------
    # Same-edge analytic blocks (cached per k)
    # ------------------------------------------------------------------

    def _same_edge_block(self, ctx, edge_id, k):
        """Analytic static + regularised same-edge moment block for one edge,
        shape (d+1, d+1, N_e, N_e), indexed by within-edge local segment
        index. Cached per (edge_id, k); identical formula to
        `BSplinePySim._build_J_blocks`'s same-edge overwrite path.
        """
        cache = self._hm_se_cache
        key = (edge_id, k)
        blk = cache.get(key)
        if blk is not None:
            return blk
        a = self.wire_radius
        d = self.degree
        ed_arc = ctx["edge_arc"][edge_id]
        A_st = _seg_seg_static_moments(ed_arc, a, max_d=d)
        A_reg = _seg_seg_reg_moments(ed_arc, a, k, max_d=d, n_qp=self.n_qp_pair)
        blk = A_st + A_reg
        cache[key] = blk
        return blk

    # ------------------------------------------------------------------
    # On-demand moment sub-tensor
    # ------------------------------------------------------------------

    def _moment_subtensor(self, ctx, seg_I, seg_J, k):
        """Moment tensor J_pq between the global segment lists seg_I, seg_J,
        shape (d+1, d+1, |seg_I|, |seg_J|), matching the corresponding slice
        of `BSplinePySim._build_J_blocks`.

        Off-edge pairs use the full-kernel GL quadrature; pairs that share an
        edge are overwritten with the analytic static + reg block. For an
        admissible (well-separated) block there are no shared-edge pairs, so
        the overwrite loop is skipped entirely and no same-edge block is built
        — that is where the acceleration comes from.
        """
        d = self.degree
        a = self.wire_radius
        seg_l = ctx["seg_l"]
        seg_r = ctx["seg_r"]

        Jsub = _seg_seg_full_moments_offedge(
            seg_l[seg_I],
            seg_r[seg_I],
            seg_l[seg_J],
            seg_r[seg_J],
            a,
            k,
            d,
            self.n_qp_pair,
        )

        eid = ctx["seg_edge_id"]
        loc = ctx["seg_edge_loc"]
        eid_I = eid[seg_I]
        eid_J = eid[seg_J]
        shared = np.intersect1d(eid_I, eid_J)
        for e in shared:
            rows = np.nonzero(eid_I == e)[0]
            cols = np.nonzero(eid_J == e)[0]
            blk = self._same_edge_block(ctx, int(e), k)
            sub = blk[:, :, loc[seg_I[rows]][:, None], loc[seg_J[cols]][None, :]]
            Jsub[:, :, rows[:, None], cols[None, :]] = sub
        return Jsub

    # ------------------------------------------------------------------
    # Restricted Z assembly (numpy reference, mirrors _assemble_Z)
    # ------------------------------------------------------------------

    def _assemble_Z_block(
        self, Jsub, supp_I_local, polys_I, supp_J_local, polys_J, td_sub
    ):
        """Galerkin-assemble Z[I][:, J] from a local moment sub-tensor.

        `supp_*_local` are the wing→local-segment index tables (values index
        into Jsub's last two axes / td_sub), `polys_*` the per-basis poly
        coefficients, `td_sub` the tangent-dot table over the local segments.
        Bit-for-bit the same arithmetic as `BSplinePySim._assemble_Z`'s numpy
        fallback, restricted to the block.
        """
        d = self.degree
        nI = supp_I_local.shape[0]
        nJ = supp_J_local.shape[0]
        Z_A = np.zeros((nI, nJ), dtype=np.complex128)
        Z_Phi = np.zeros((nI, nJ), dtype=np.complex128)
        p_vec = np.arange(1, d + 1, dtype=np.float64) if d >= 1 else None

        for a in range(d + 1):
            sm = supp_I_local[:, a]
            for b in range(d + 1):
                sn = supp_J_local[:, b]
                J_blk = Jsub[:, :, sm[:, None], sn[None, :]]
                td_blk = td_sub[sm[:, None], sn[None, :]]
                inner_A = np.einsum(
                    "mp,pPmn,nP->mn", polys_I[:, a, :], J_blk, polys_J[:, b, :]
                )
                Z_A += td_blk * inner_A
                if d >= 1:
                    deriv_m = polys_I[:, a, 1:] * p_vec[None, :]
                    deriv_n = polys_J[:, b, 1:] * p_vec[None, :]
                    J_blk_lo = J_blk[:d, :d]
                    Z_Phi += np.einsum("mp,pPmn,nP->mn", deriv_m, J_blk_lo, deriv_n)

        Z_A = 1j * self.omega * self.mu * Z_A
        Z_Phi = Z_Phi / (1j * self.omega * self.eps)
        return Z_A + Z_Phi

    # ------------------------------------------------------------------
    # Public block evaluator
    # ------------------------------------------------------------------

    def zblock(self, I, J, k=None):
        """Return the dense sub-block Z[I][:, J] (shape (|I|, |J|), complex).

        `I`, `J` are 1-D integer arrays of *basis* indices. Computed on demand
        from only the segments in the supports of I and J — no full Z or full
        moment tensor is formed. Equivalent to slicing the dense bspline Z.
        """
        if k is None:
            k = self.k
        ctx = self._context()
        # Reset the per-k same-edge cache when k changes.
        if getattr(self, "_hm_se_cache_k", None) != k:
            self._hm_se_cache = {}
            self._hm_se_cache_k = k

        I = np.asarray(I, dtype=np.int64)
        J = np.asarray(J, dtype=np.int64)
        supp_seg = ctx["supp_seg"]
        polys = ctx["polys"]
        tangents = ctx["tangents"]

        # Union of segments touched by the two basis index sets, with a
        # global→local remap so the moment sub-tensor and td table are small.
        seg_I = np.unique(supp_seg[I].ravel())
        seg_J = np.unique(supp_seg[J].ravel())
        loc_of_I = {int(s): i for i, s in enumerate(seg_I)}
        loc_of_J = {int(s): i for i, s in enumerate(seg_J)}

        Jsub = self._moment_subtensor(ctx, seg_I, seg_J, k)

        supp_I_local = np.vectorize(loc_of_I.__getitem__)(supp_seg[I])
        supp_J_local = np.vectorize(loc_of_J.__getitem__)(supp_seg[J])
        td_sub = tangents[seg_I] @ tangents[seg_J].T

        return self._assemble_Z_block(
            Jsub, supp_I_local, polys[I], supp_J_local, polys[J], td_sub
        )

    # ------------------------------------------------------------------
    # Cluster / block tree (Phase 1)
    # ------------------------------------------------------------------

    def build_partition(self, eta=None, leaf_size=None):
        """Build (and memoise) the block-cluster partition of the n_basis x
        n_basis impedance matrix into far (admissible, compressible) and near
        (dense) leaf blocks.

        Returns a dict with the cluster-tree `root`, the `far`/`near` block
        lists (each a list of (Cluster, Cluster) pairs), and `stats`.
        """
        if eta is None:
            eta = self.aca_eta
        if leaf_size is None:
            leaf_size = self.aca_leaf_size

        cached = getattr(self, "_hm_partition", None)
        if (
            cached is not None
            and cached["eta"] == eta
            and (cached["leaf_size"] == leaf_size)
        ):
            return cached

        ctx = self._context()
        n = ctx["n_basis"]
        root = build_cluster_tree(
            np.arange(n), ctx["basis_lo"], ctx["basis_hi"], leaf_size=leaf_size
        )
        far, near = build_block_tree(root, root, eta)
        stats = partition_stats(n, far, near)
        part = {
            "root": root,
            "far": far,
            "near": near,
            "stats": stats,
            "eta": eta,
            "leaf_size": leaf_size,
        }
        self._hm_partition = part
        return part

    def __init__(self, *args, aca_eta=1.0, aca_leaf_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.aca_eta = float(aca_eta)
        self.aca_leaf_size = int(aca_leaf_size)
        self._hm_context = None
        self._hm_se_cache = {}
        self._hm_se_cache_k = None
        self._hm_partition = None
