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
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, gmres, splu

from .bspline import BSplinePySim
from ._bspline_kernels import (
    _seg_seg_full_moments_offedge,
    _seg_seg_reg_moments,
    _seg_seg_static_moments,
)
from ._aca import (
    HMatrix,
    aca_partial,
    admissible,
    build_block_tree,
    build_cluster_tree,
    partition_stats,
)
from ._quadrature import leggauss

try:
    from . import _accelerators as _acc

    _HAVE_OFFEDGE_BLOCK_ACCEL = hasattr(_acc, "bspline_assemble_offedge_block")
except ImportError:  # pragma: no cover
    _HAVE_OFFEDGE_BLOCK_ACCEL = False

_OFFEDGE_BLOCK_ACCEL_MAX_D = 2


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

    def _same_edge_band(self, ctx, edge_id, k, lo, hi):
        """Analytic static + regularised same-edge moments over the contiguous
        within-edge segment sub-range [lo, hi] (inclusive) of one edge. Shape
        (d+1, d+1, W, W) with W = hi - lo + 1, indexed by (local_idx - lo).

        Same formula as `BSplinePySim._build_J_blocks`'s same-edge overwrite,
        but restricted to the sub-range a block actually touches — O(W²)
        instead of the full O(N_edge²). A near block spans ~2·leaf_size
        segments, so W stays small and the same-edge fill is O(N·leaf) total
        rather than O(N²). The static moments are translation-invariant along a
        straight edge and the reg geometry depends only on arc differences, so
        the sub-range gives exactly the corresponding entries of the full
        edge block. Cached per (edge_id, lo, hi); the per-k cache is reset in
        `zblock` when k changes.
        """
        cache = self._hm_se_cache
        key = (edge_id, lo, hi)
        blk = cache.get(key)
        if blk is not None:
            return blk
        a = self.wire_radius
        d = self.degree
        sub_arc = ctx["edge_arc"][edge_id][lo : hi + 2]
        A_st = _seg_seg_static_moments(sub_arc, a, max_d=d)
        A_reg = _seg_seg_reg_moments(sub_arc, a, k, max_d=d, n_qp=self.n_qp_pair)
        blk = A_st + A_reg
        cache[key] = blk
        return blk

    # ------------------------------------------------------------------
    # On-demand moment sub-tensor
    # ------------------------------------------------------------------

    def _moment_subtensor(self, ctx, seg_I, seg_J, k, same_edge=True):
        """Moment tensor J_pq between the global segment lists seg_I, seg_J,
        shape (d+1, d+1, |seg_I|, |seg_J|), matching the corresponding slice
        of `BSplinePySim._build_J_blocks`.

        Off-edge pairs use the full-kernel GL quadrature. When `same_edge` is
        True, pairs that share an edge are overwritten with the analytic
        static + reg block (essential for the near-singular diagonal). When
        False, the overwrite is skipped entirely — valid for an *admissible*
        (well-separated) block, where every pair, even two segments on the
        same long wire, is far enough apart that the a²-regularised GL kernel
        is accurate (~1e-5). Skipping it is what makes far blocks cheap:
        no O(N_edge²) same-edge block is ever built for a single-wire mesh.
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

        if not same_edge:
            return Jsub

        eid = ctx["seg_edge_id"]
        loc = ctx["seg_edge_loc"]
        eid_I = eid[seg_I]
        eid_J = eid[seg_J]
        shared = np.intersect1d(eid_I, eid_J)
        for e in shared:
            rows = np.nonzero(eid_I == e)[0]
            cols = np.nonzero(eid_J == e)[0]
            li = loc[seg_I[rows]]
            lj = loc[seg_J[cols]]
            lo = int(min(li.min(), lj.min()))
            hi = int(max(li.max(), lj.max()))
            blk = self._same_edge_band(ctx, int(e), k, lo, hi)
            sub = blk[:, :, (li - lo)[:, None], (lj - lo)[None, :]]
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

    def zblock(self, I, J, k=None, same_edge=True):
        """Return the dense sub-block Z[I][:, J] (shape (|I|, |J|), complex).

        `I`, `J` are 1-D integer arrays of *basis* indices. Computed on demand
        from only the segments in the supports of I and J — no full Z or full
        moment tensor is formed.

        With `same_edge=True` (default) this equals the dense bspline Z slice
        exactly. With `same_edge=False` the same-edge analytic overwrite is
        skipped — correct (~1e-5) only for *admissible* far blocks, where it
        avoids ever materialising a same-edge block; used by the ACA fill.
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

        Jsub = self._moment_subtensor(ctx, seg_I, seg_J, k, same_edge=same_edge)

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

    # ------------------------------------------------------------------
    # H-matrix assembly (Phase 2): dense near blocks + ACA far blocks
    # ------------------------------------------------------------------

    def build_hmatrix(self, eta=None, leaf_size=None, tol=None, k=None):
        """Assemble the impedance matrix as an `HMatrix`: near blocks dense
        (via `zblock`, including the same-edge analytic path), far blocks
        compressed by partial-pivoted ACA on the off-edge kernel.

        A far block whose ACA factors would cost as much as the dense block
        falls back to dense storage, so the H-matvec is never worse than the
        dense block-by-block product.
        """
        if tol is None:
            tol = self.aca_tol
        if k is None:
            k = self.k
        part = self.build_partition(eta=eta, leaf_size=leaf_size)
        ctx = self._context()
        n = ctx["n_basis"]

        near_blocks = []
        for s, t in part["near"]:
            I, J = s.indices, t.indices
            near_blocks.append((I, J, self.zblock(I, J, k=k)))

        use_accel = (
            _HAVE_OFFEDGE_BLOCK_ACCEL
            and self.degree <= _OFFEDGE_BLOCK_ACCEL_MAX_D
            and self.hmatrix_use_accel
        )

        far_blocks = []
        precond_extra = []  # first-ring far blocks, dense, for the preconditioner
        p_eta = self.precond_eta
        for s, t in part["far"]:
            I, J = s.indices, t.indices
            mI, nJ = I.size, J.size

            if use_accel:
                get_row, get_col, dense = self._offedge_block_evaluators(ctx, I, J, k)
            else:

                def get_row(i, I=I, J=J):
                    return self.zblock(I[i : i + 1], J, k=k, same_edge=False).ravel()

                def get_col(j, I=I, J=J):
                    return self.zblock(I, J[j : j + 1], k=k, same_edge=False).ravel()

                def dense(I=I, J=J):
                    return self.zblock(I, J, k=k, same_edge=False)

            U, V = aca_partial(get_row, get_col, mI, nJ, tol=tol)
            r = U.shape[1]
            if r * (mI + nJ) >= mI * nJ:
                # No compression — store dense (off-edge kernel; the block is
                # admissible so the same-edge analytic path is unnecessary).
                near_blocks.append((I, J, dense()))
            else:
                far_blocks.append((I, J, U, V))
                # "First ring": leaf-scale blocks adjacent to the near band —
                # admissible for the operator but not at the tighter
                # precond_eta. Fold a dense reconstruction into the
                # preconditioner (free: reuses the low-rank factors). The
                # leaf-size cap keeps it a *thin* ring: large far blocks (big
                # well-separated clusters) are genuinely far and excluded, so
                # the preconditioner stays sparse.
                if (
                    p_eta < self.aca_eta
                    and max(mI, nJ) <= self.aca_leaf_size
                    and not admissible(s, t, p_eta)
                ):
                    precond_extra.append((I, J, U @ V))

        return HMatrix(n, near_blocks, far_blocks, precond_extra=precond_extra)

    def _gl01(self):
        """Gauss-Legendre nodes/weights mapped to [0, 1] (cached)."""
        cached = getattr(self, "_hm_gl01", None)
        if cached is None:
            xi, w = leggauss(self.n_qp_pair)
            cached = (
                np.ascontiguousarray(0.5 * (xi + 1.0)),
                np.ascontiguousarray(0.5 * w),
            )
            self._hm_gl01 = cached
        return cached

    def _offedge_block_evaluators(self, ctx, I, J, k):
        """Build (get_row, get_col, dense) closures for an admissible far block
        backed by the fused C++ off-edge assembler `bspline_assemble_offedge_block`.

        The block-wide I/J segment unions and local support maps are resolved
        once here; each row/column call passes only the single basis it needs
        on its own axis (so the C++ side never precomputes positions for unused
        segments) against the precomputed full opposite axis.
        """
        supp_seg = ctx["supp_seg"]
        polys = ctx["polys"]
        seg_l = ctx["seg_l"]
        seg_r = ctx["seg_r"]
        tangents = ctx["tangents"]
        d = self.degree
        a2 = self.wire_radius * self.wire_radius
        glt, glw = self._gl01()
        omega, eps, mu = self.omega, self.eps, self.mu

        segI = np.unique(supp_seg[I].ravel())
        segJ = np.unique(supp_seg[J].ravel())
        sIl = np.searchsorted(segI, supp_seg[I]).astype(np.int64)
        sJl = np.searchsorted(segJ, supp_seg[J]).astype(np.int64)
        pI = np.ascontiguousarray(polys[I])
        pJ = np.ascontiguousarray(polys[J])
        slI, srI, tI = seg_l[segI], seg_r[segI], tangents[segI]
        slJ, srJ, tJ = seg_l[segJ], seg_r[segJ], tangents[segJ]
        one_supp = np.arange(d + 1, dtype=np.int64)[None, :]

        def get_row(i):
            seg_i = supp_seg[I[i]]
            return _acc.bspline_assemble_offedge_block(
                one_supp,
                polys[I[i]][None],
                seg_l[seg_i],
                seg_r[seg_i],
                tangents[seg_i],
                sJl,
                pJ,
                slJ,
                srJ,
                tJ,
                a2,
                k,
                omega,
                eps,
                mu,
                d,
                glt,
                glw,
            ).ravel()

        def get_col(j):
            seg_j = supp_seg[J[j]]
            return _acc.bspline_assemble_offedge_block(
                sIl,
                pI,
                slI,
                srI,
                tI,
                one_supp,
                polys[J[j]][None],
                seg_l[seg_j],
                seg_r[seg_j],
                tangents[seg_j],
                a2,
                k,
                omega,
                eps,
                mu,
                d,
                glt,
                glw,
            ).ravel()

        def dense():
            return _acc.bspline_assemble_offedge_block(
                sIl,
                pI,
                slI,
                srI,
                tI,
                sJl,
                pJ,
                slJ,
                srJ,
                tJ,
                a2,
                k,
                omega,
                eps,
                mu,
                d,
                glt,
                glw,
            )

        return get_row, get_col, dense

    # ------------------------------------------------------------------
    # Iterative solve (Phase 3): GMRES on the H-matvec + near-field
    # preconditioner, KCL constraints via the augmented saddle system
    # ------------------------------------------------------------------

    def _near_sparse(self, H, n):
        """Assemble the near-field approximation of Z into one sparse (n, n)
        matrix for the GMRES preconditioner: the operator's dense near blocks
        plus the first-ring far blocks (H.precond_extra), the latter folded in
        as dense reconstructions of their low-rank factors to give a stronger
        preconditioner than the operator's own near band."""
        rows, cols, data = [], [], []
        for I, J, D in H.near + H.precond_extra:
            rr = np.repeat(I, J.size)
            cc = np.tile(J, I.size)
            rows.append(rr)
            cols.append(cc)
            data.append(D.ravel())
        if not rows:
            return sp.csc_matrix((n, n), dtype=np.complex128)
        return sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsc()

    def _solve_hmatrix(self, H, kcl_A, B):
        """Solve the constrained system  [Z A^T; A 0][x; λ] = [b; 0]  for each
        RHS column of B (n, nrhs), with Z applied via the H-matvec and a
        sparse-LU near-field preconditioner. Returns X (n, nrhs).

        With no junctions (kcl_A empty) this is a plain GMRES on Z.
        """
        n = H.n
        nc = kcl_A.shape[0] if kcl_A is not None else 0
        N = n + nc
        rtol = self.solve_tol

        # Near-field preconditioner, augmented with the KCL rows, factorised
        # once and reused across all RHS columns.
        Zn = self._near_sparse(H, n)
        if nc > 0:
            A_sp = sp.csr_matrix(kcl_A.astype(np.complex128))
            Saug = sp.bmat([[Zn, A_sp.T], [A_sp, None]], format="csc")
        else:
            Saug = Zn
        lu = splu(Saug)

        def aug_matvec(z):
            x = z[:n]
            out = np.empty(N, dtype=np.complex128)
            out[:n] = H.matvec(x)
            if nc > 0:
                lam = z[n:]
                out[:n] += kcl_A.T @ lam
                out[n:] = kcl_A @ x
            return out

        def prec(z):
            return lu.solve(z)

        Aug = LinearOperator((N, N), matvec=aug_matvec, dtype=np.complex128)
        Minv = LinearOperator((N, N), matvec=prec, dtype=np.complex128)

        nrhs = B.shape[1]
        X = np.zeros((n, nrhs), dtype=np.complex128)
        self._last_solve_iters = []
        for j in range(nrhs):
            rhs = np.zeros(N, dtype=np.complex128)
            rhs[:n] = B[:, j]
            x0 = prec(rhs)  # preconditioner solve is an excellent initial guess
            iters = [0]

            def _count(_xk):
                iters[0] += 1

            sol, info = gmres(
                Aug,
                rhs,
                M=Minv,
                x0=x0,
                rtol=rtol,
                atol=0.0,
                restart=min(N, 200),
                maxiter=2000,
                callback=_count,
                callback_type="pr_norm",
            )
            X[:, j] = sol[:n]
            self._last_solve_iters.append(iters[0])
        return X

    def _hmatrix_unsupported(self):
        """The H-matrix path is free-space, no-enrichment only for now."""
        return self.ground_z is not None or self.use_singular_enrichment

    def compute_y_matrix(self):
        if self._hmatrix_unsupported():
            return super().compute_y_matrix()
        ctx = self._context()
        geom = ctx["geom"]
        n = ctx["n_basis"]
        H = self.build_hmatrix()
        n_ports = len(self.feeds)
        B = np.zeros((n, n_ports), dtype=np.complex128)
        for j, (w_i, arc_i, _v) in enumerate(self.feeds):
            arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
            s_f_j = arc_i if arc_i is not None else arc_at_knot[-1] / 2.0
            B[:, j] = self._build_source_vector(
                geom,
                ctx["wire_knots"],
                ctx["wire_basis_global"],
                n,
                wi=w_i,
                s_f=s_f_j,
            )
        X = self._solve_hmatrix(H, ctx["kcl_A"], B)
        self._hmatrix = H
        return B.T @ X

    def compute_impedance(self):
        if self._hmatrix_unsupported():
            return super().compute_impedance()
        ctx = self._context()
        geom = ctx["geom"]
        n = ctx["n_basis"]
        H = self.build_hmatrix()

        v_per_feed = []
        for w_i, arc_i, _v in self.feeds:
            arc_at_knot = geom["per_wire"][w_i]["arc_at_knot"]
            s_f_i = arc_i if arc_i is not None else arc_at_knot[-1] / 2.0
            v_per_feed.append(
                self._build_source_vector(
                    geom,
                    ctx["wire_knots"],
                    ctx["wire_basis_global"],
                    n,
                    wi=w_i,
                    s_f=s_f_i,
                )
            )
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        v = np.zeros(n, dtype=np.complex128)
        for V_i, v_i in zip(voltages, v_per_feed):
            v += V_i * v_i

        coeffs = self._solve_hmatrix(H, ctx["kcl_A"], v[:, None])[:, 0]
        self._hmatrix = H

        currents = np.array([v_i @ coeffs for v_i in v_per_feed], dtype=np.complex128)
        z_per = voltages / currents
        z = z_per[0] if len(self.feeds) == 1 else z_per
        return z, coeffs

    def __init__(
        self,
        *args,
        aca_eta=1.0,
        aca_leaf_size=32,
        aca_tol=1e-4,
        solve_tol=1e-6,
        hmatrix_use_accel=True,
        precond_eta=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aca_eta = float(aca_eta)
        self.aca_leaf_size = int(aca_leaf_size)
        self.aca_tol = float(aca_tol)
        self.solve_tol = float(solve_tol)
        # Preconditioner near-field admissibility. The GMRES preconditioner
        # uses a *stronger* (tighter-eta) near-field than the operator: every
        # operator-far block that is inadmissible at `precond_eta` (the "first
        # ring" just outside the operator's near band) is folded into the
        # sparse preconditioner as a dense reconstruction of its already-
        # computed low-rank factors — no extra kernel work, operator stays
        # compressed. None ⇒ 0.5·aca_eta. Set equal to aca_eta to disable.
        # The MoM EFIE operator is non-normal, so spectral (coarse-space)
        # deflation does not help; a denser near-field does.
        self.precond_eta = (
            0.5 * self.aca_eta if precond_eta is None else float(precond_eta)
        )
        # Use the fused C++ off-edge block assembler for ACA far blocks when
        # available; set False to force the pure-numpy zblock path (testing).
        self.hmatrix_use_accel = bool(hmatrix_use_accel)
        self._hm_context = None
        self._hm_gl01 = None
        self._hm_se_cache = {}
        self._hm_se_cache_k = None
        self._hm_partition = None
