"""Hierarchical (H-matrix / ACA) B-spline Galerkin MoM solver.

`HMatrixSolver` is a distance-based accelerator built on top of
`BSplineSolver`. It reuses BSplineSolver's geometry build, basis-polynomial
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
    static + regularised split, identical to `BSplineSolver._build_J_blocks`.
    This evaluator is the foundation everything else stands on: ACA can only
    be as correct as the entries it samples.

  * Phase 1: binary-space-partition cluster tree over basis centroids;
    admissibility min(diam(s), diam(t)) <= eta * dist(s, t); recursive
    block tree → {admissible far, dense near} leaves.

  * Phase 2: partial-pivoted ACA low-rank approximation of admissible
    blocks; dense near blocks; fast H-matvec.

  * Phase 3: preconditioned block GMRES on the batched operator
    (`matmat`) + near-field preconditioner, KCL junction constraints in the
    augmented system. All RHS share one block Krylov space, so the operator
    and preconditioner applies are batched (BLAS-3) across the columns.

Why distance matters here: the moment integral kernel
G = exp(-jkR)/(4πR) is smooth and asymptotically smooth once the two
B-spline basis supports are well separated, so the corresponding Z block is
numerically low rank and ACA captures it from O(r·(m+n)) sampled entries
instead of the full m·n.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .bspline import BSplineSolver
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


class _SparseAugPrecond:
    """Generic single sparse-LU factorisation of the augmented near-field
    preconditioner `[Zn A^T; A 0]`. Used by the H-matrix path, where the near
    band has no exploitable block structure. `.solve(R)` applies `M^{-1}` to an
    augmented block R (N, nrhs) via one SuperLU back-substitution."""

    def __init__(self, Zn, kcl_A):
        nc = kcl_A.shape[0] if kcl_A is not None else 0
        if nc > 0:
            A_sp = sp.csr_matrix(kcl_A.astype(np.complex128))
            Saug = sp.bmat([[Zn, A_sp.T], [A_sp, None]], format="csc")
        else:
            Saug = Zn.tocsc() if sp.issparse(Zn) else Zn
        self.lu = splu(Saug)

    def solve(self, R):
        return self.lu.solve(R)


class _AugmentedFactoredSolve:
    """A factored augmented near-field preconditioner plus the preconditioned
    block GMRES on the operator `H`.

    The preconditioner is a pluggable object with a `.solve(R)` method (sparse
    LU for the H-matrix, per-element block-Jacobi for the array-block solver).
    Holds no reference to the solver instance, so it can be cached on the
    operator and reused across solves that share it: the factorisation is
    RHS-independent, so an animation phase/excitation sweep re-solves with
    cached back-substitutions (`X0 = M^{-1} B`) plus a handful of block-Krylov
    steps, never refactoring.
    """

    def __init__(self, H, kcl_A, precond):
        self.H = H
        self.kcl_A = kcl_A
        self.precond = precond
        n = H.n
        self.n = n
        nc = kcl_A.shape[0] if kcl_A is not None else 0
        self.nc = nc
        self.N = n + nc

    def _aug_matmat(self, Z):
        """Apply the augmented operator [Z A^T; A 0] to a block Z (N, nrhs)."""
        n, nc = self.n, self.nc
        out = np.empty((self.N, Z.shape[1]), dtype=np.complex128)
        out[:n] = self.H.matmat(Z[:n])
        if nc > 0:
            out[:n] += self.kcl_A.T @ Z[n:]
            out[n:] = self.kcl_A @ Z[:n]
        return out

    def solve(self, B, rtol):
        """Solve the augmented system for every RHS column of B (n, nrhs) at
        once with left-preconditioned block GMRES. Returns (X (n, nrhs),
        iters). All RHS share one block Krylov space and every operator/
        preconditioner apply is batched across the columns (BLAS-3), so the
        cost is ~one batched matmat + one batched back-substitution per Krylov
        step instead of nrhs separate matvec/solve pairs per step."""
        n, N = self.n, self.N
        Baug = np.zeros((N, B.shape[1]), dtype=np.complex128)
        Baug[:n] = B
        X, iters = self._block_gmres(Baug, rtol)
        return X[:n], iters

    def _block_gmres(self, Baug, rtol, restart=50, maxiter=2000):
        """Left-preconditioned restarted block GMRES on the augmented system.

        Solves M^{-1} A X = M^{-1} B for all columns simultaneously (M = the
        factored near-field preconditioner, A = the augmented operator). The
        preconditioned initial guess `X0 = M^{-1} B` is the same excellent
        starting point the per-RHS path used. Convergence is the per-column
        preconditioned-residual norm relative to ‖M^{-1} b_j‖ — matching SciPy
        gmres's left-preconditioned stopping test, so accuracy is unchanged.
        """
        prec = self.precond.solve
        Bt = prec(Baug)  # M^{-1} B
        bnorms = np.linalg.norm(Bt, axis=0)
        bnorms = np.where(bnorms == 0.0, 1.0, bnorms)
        X = Bt.copy()  # X0 = M^{-1} B
        s = Baug.shape[1]
        m = min(restart, self.N)
        total_iters = 0
        for _ in range(maxiter // m + 1):
            R = Bt - prec(self._aug_matmat(X))  # preconditioned residual
            if np.all(np.linalg.norm(R, axis=0) <= rtol * bnorms):
                break
            Q, beta = np.linalg.qr(R)  # R = Q @ beta; Q (N,s), beta (s,s)
            Vs = [Q]
            Hblocks = []  # Hblocks[k] = [H_{0,k}, …, H_{k+1,k}], each (s,s)
            Y = None
            converged = False
            for k in range(m):
                total_iters += 1
                W = prec(self._aug_matmat(Vs[k]))
                Hk = []
                for i in range(k + 1):  # block modified Gram-Schmidt
                    Hik = Vs[i].conj().T @ W
                    W = W - Vs[i] @ Hik
                    Hk.append(Hik)
                Qk, Hkk = np.linalg.qr(W)
                Hk.append(Hkk)
                Hblocks.append(Hk)
                Vs.append(Qk)
                kb = k + 1
                Hbar = np.zeros(((kb + 1) * s, kb * s), dtype=np.complex128)
                for col in range(kb):
                    for i in range(col + 2):
                        Hbar[i * s : (i + 1) * s, col * s : (col + 1) * s] = Hblocks[
                            col
                        ][i]
                E1b = np.zeros(((kb + 1) * s, s), dtype=np.complex128)
                E1b[:s, :] = beta
                Y, _res, _rank, _sv = np.linalg.lstsq(Hbar, E1b, rcond=None)
                rk = np.linalg.norm(E1b - Hbar @ Y, axis=0)
                if np.all(rk <= rtol * bnorms):
                    converged = True
                    break
            kb = len(Hblocks)
            X = X + np.concatenate(Vs[:kb], axis=1) @ Y
            if converged:
                break
        return X, [total_iters] * s


class HMatrixSolver(BSplineSolver):
    """Distance-based hierarchical accelerator for the B-spline MoM.

    Drop-in for `BSplineSolver` (same constructor); Phase 0 only adds the
    block-evaluator plumbing. `compute_impedance` / `compute_y_matrix` still
    resolve to the dense BSplineSolver path until later phases override them,
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

        Same formula as `BSplineSolver._build_J_blocks`'s same-edge overwrite,
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
        of `BSplineSolver._build_J_blocks`.

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
        Bit-for-bit the same arithmetic as `BSplineSolver._assemble_Z`'s numpy
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

    def _zblock_image(self, I, J, k=None):
        """Return the PEC-image sub-block for the basis pair (I, J): the real
        test bases I reacting against the trial bases J mirrored across
        ``z = ground_z`` (positions reflected, tangents z-flipped to (tx, ty,
        -tz)). The full impedance block under PEC ground is

            zblock(I, J) - _zblock_image(I, J)

        a single combined minus sign capturing both the image current's
        anti-parallel horizontal direction and the image charge sign flip (see
        `BSplineSolver.compute_impedance`). The mirror always separates the image
        from the real geometry, so every pair is off-edge — full GL quadrature
        throughout, no same-edge analytic overwrite (mirrors
        `BSplineSolver._build_J_image_blocks`)."""
        if k is None:
            k = self.k
        ctx = self._context()
        supp_seg = ctx["supp_seg"]
        polys = ctx["polys"]
        tangents = ctx["tangents"]
        seg_l = ctx["seg_l"]
        seg_r = ctx["seg_r"]
        a = self.wire_radius
        d = self.degree

        I = np.asarray(I, dtype=np.int64)
        J = np.asarray(J, dtype=np.int64)
        seg_I = np.unique(supp_seg[I].ravel())
        seg_J = np.unique(supp_seg[J].ravel())
        loc_of_I = {int(s): i for i, s in enumerate(seg_I)}
        loc_of_J = {int(s): i for i, s in enumerate(seg_J)}

        Jsub = _seg_seg_full_moments_offedge(
            seg_l[seg_I],
            seg_r[seg_I],
            self._image_positions(seg_l[seg_J]),
            self._image_positions(seg_r[seg_J]),
            a,
            k,
            d,
            self.n_qp_pair,
        )
        supp_I_local = np.vectorize(loc_of_I.__getitem__)(supp_seg[I])
        supp_J_local = np.vectorize(loc_of_J.__getitem__)(supp_seg[J])
        td_sub = tangents[seg_I] @ self._image_tangent_dot_cols(tangents[seg_J])
        return self._assemble_Z_block(
            Jsub, supp_I_local, polys[I], supp_J_local, polys[J], td_sub
        )

    @staticmethod
    def _image_tangent_dot_cols(tangents_J):
        """The (tx, ty, -tz)-flipped trial tangents as columns, so
        ``t_I @ _image_tangent_dot_cols(t_J)`` is the image tangent-dot table
        (rows = real test, cols = image trial) — the block form of
        `BSplineSolver._image_tangent_dot`."""
        return (tangents_J * np.array([1.0, 1.0, -1.0])).T

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

        Under PEC ground every block carries the per-block image term
        (`Z_free − Z_image`): near blocks subtract the dense `_zblock_image`,
        far blocks fold the image into the ACA target via
        `_offedge_aca_evaluators`. The `HMatrix` container, matvec, and
        preconditioner are ground-agnostic — they bake whatever block values
        they are handed — and the cluster partition is built on the real
        geometry, so it is reused unchanged.
        """
        if tol is None:
            tol = self.aca_tol
        if k is None:
            k = self.k
        part = self.build_partition(eta=eta, leaf_size=leaf_size)
        ctx = self._context()
        n = ctx["n_basis"]
        grounded = self.ground_z is not None

        near_blocks = []
        for s, t in part["near"]:
            I, J = s.indices, t.indices
            D = self.zblock(I, J, k=k)
            if grounded:
                D = D - self._zblock_image(I, J, k=k)
            near_blocks.append((I, J, D))

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

            get_row, get_col, dense = self._offedge_aca_evaluators(
                ctx, I, J, k, use_accel
            )

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

    def _offedge_block_evaluators(self, ctx, I, J, k, mirror_J=False):
        """Build (get_row, get_col, dense) closures for an admissible far block
        backed by the fused C++ off-edge assembler `bspline_assemble_offedge_block`.

        The block-wide I/J segment unions and local support maps are resolved
        once here; each row/column call passes only the single basis it needs
        on its own axis (so the C++ side never precomputes positions for unused
        segments) against the precomputed full opposite axis.

        With `mirror_J=True` the trial (J) segment endpoints are reflected across
        ``z = ground_z`` and their tangents z-flipped, so the kernel's internal R
        distances and tangent dot products reproduce the PEC-image reaction — the
        C++ counterpart of `_zblock_image` (the result is *subtracted* from the
        free-space block). The C++ assembler uses the trial tangents only through
        the dot product, so flipping their z is exactly the image-current sign
        flip.
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
        flip = np.array([1.0, 1.0, -1.0])

        def mirror_pos(p):
            return self._image_positions(p) if mirror_J else p

        def mirror_tan(t):
            return t * flip if mirror_J else t

        segI = np.unique(supp_seg[I].ravel())
        segJ = np.unique(supp_seg[J].ravel())
        sIl = np.searchsorted(segI, supp_seg[I]).astype(np.int64)
        sJl = np.searchsorted(segJ, supp_seg[J]).astype(np.int64)
        pI = np.ascontiguousarray(polys[I])
        pJ = np.ascontiguousarray(polys[J])
        slI, srI, tI = seg_l[segI], seg_r[segI], tangents[segI]
        slJ, srJ, tJ = (
            mirror_pos(seg_l[segJ]),
            mirror_pos(seg_r[segJ]),
            mirror_tan(tangents[segJ]),
        )
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
                mirror_pos(seg_l[seg_j]),
                mirror_pos(seg_r[seg_j]),
                mirror_tan(tangents[seg_j]),
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

    def _offedge_aca_evaluators(self, ctx, I, J, k, use_accel):
        """`(get_row, get_col, dense)` for the off-edge block Z[I][:, J] — the
        ACA-fill interface for an admissible far block (or a distinct array-
        element pair). "Off-edge" means no same-edge analytic overwrite, valid
        because the clusters are well separated; the C++ assembler is used when
        available, else the numpy `zblock(..., same_edge=False)` fallback.

        Under PEC ground the block is the *grounded* off-edge block,
        `Z_free − Z_image` (real cluster I against J, plus I against J's mirror
        image), with the image folded into every evaluator so a single ACA
        compresses the combined block — one factor pair, leaving the matvec and
        preconditioner unchanged. Rank rises modestly (real + image content); a
        block whose combined rank no longer compresses falls back to dense in
        the caller, which is correct (the image stays as low-rank as the real
        block for an antenna above the plane — reflection only increases the
        cluster separation)."""
        if use_accel:
            row_f, col_f, dense_f = self._offedge_block_evaluators(ctx, I, J, k)
        else:

            def row_f(i):
                return self.zblock(I[i : i + 1], J, k=k, same_edge=False).ravel()

            def col_f(j):
                return self.zblock(I, J[j : j + 1], k=k, same_edge=False).ravel()

            def dense_f():
                return self.zblock(I, J, k=k, same_edge=False)

        if self.ground_z is None:
            return row_f, col_f, dense_f

        if use_accel:
            row_i, col_i, dense_i = self._offedge_block_evaluators(
                ctx, I, J, k, mirror_J=True
            )
        else:

            def row_i(i):
                return self._zblock_image(I[i : i + 1], J, k=k).ravel()

            def col_i(j):
                return self._zblock_image(I, J[j : j + 1], k=k).ravel()

            def dense_i():
                return self._zblock_image(I, J, k=k)

        def get_row(i):
            return row_f(i) - row_i(i)

        def get_col(j):
            return col_f(j) - col_i(j)

        def dense():
            return dense_f() - dense_i()

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

    def _make_preconditioner(self, H, kcl_A):
        """Build the augmented near-field preconditioner for operator `H`. The
        generic H-matrix path uses a single sparse LU; subclasses with block
        structure (e.g. `ArrayBlockSolver`) override this with a cheaper
        block-wise factorisation."""
        return _SparseAugPrecond(self._near_sparse(H, H.n), kcl_A)

    def _factored_solve(self, H, kcl_A):
        """The augmented preconditioner factorisation for operator `H`, built
        once and cached on `H` itself. Because the factorisation depends only
        on `H` (its near blocks) and the KCL rows — not on the RHS or the
        excitation — caching it on `H` lets a *reused* operator (e.g. an
        animation phase sweep, where geometry and Z are fixed and only the RHS
        changes) skip the factorisation entirely. A freshly-built `H` (the
        generic H-matrix path) just factors once per solve as before."""
        fac = getattr(H, "_factored", None)
        if fac is None:
            fac = _AugmentedFactoredSolve(H, kcl_A, self._make_preconditioner(H, kcl_A))
            H._factored = fac
        return fac

    def _solve_hmatrix(self, H, kcl_A, B):
        """Solve the constrained system  [Z A^T; A 0][x; λ] = [b; 0]  for each
        RHS column of B (n, nrhs), with Z applied via the H-matvec and a
        sparse-LU near-field preconditioner. Returns X (n, nrhs).

        With no junctions (kcl_A empty) this is a plain GMRES on Z.
        """
        fac = self._factored_solve(H, kcl_A)
        X, self._last_solve_iters = fac.solve(B, self.solve_tol)
        return X

    def _hmatrix_unsupported(self):
        """The H-matrix path supports free space and PEC ground (the latter via
        the per-block image term folded into the near/far block fill — see
        `build_hmatrix` and `_offedge_aca_evaluators`). Only singular enrichment
        still falls back to the dense path (its image reaction isn't
        implemented; the constructor already forbids enrichment + ground)."""
        return self.use_singular_enrichment

    def _build_operator(self):
        """Build the fast operator the constrained solve runs GMRES on. The
        generic accelerator returns its H-matrix; subclasses with a different
        structural decomposition (e.g. `ArrayBlockSolver`) override this and
        feed the result through the same `_solve_hmatrix` machinery (the solve
        only needs `.n`, `.matvec`, and `.near`/`.precond_extra`)."""
        return self.build_hmatrix()

    def compute_y_matrix(self):
        if self._hmatrix_unsupported():
            return super().compute_y_matrix()
        ctx = self._context()
        geom = ctx["geom"]
        n = ctx["n_basis"]
        H = self._build_operator()
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
        H = self._build_operator()

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

    def compute_impedance_swept(self, k_array):
        """Frequency sweep on the accelerated operator: rebind k per point and
        reuse the fast `compute_impedance` (which assembles the H-matrix /
        array block for that k). Overrides the dense base sweep, whose
        `same_edge_prep` batching argument the accelerated `compute_impedance`
        doesn't accept — calling it would `TypeError`. Falls back to the dense
        base sweep (with its batched same-edge precompute) whenever the
        accelerator is unsupported for this configuration."""
        if self._hmatrix_unsupported():
            return super().compute_impedance_swept(k_array)
        k_array = np.asarray(k_array, dtype=float)
        n_feeds = len(self.feeds)
        if n_feeds == 1:
            z_out = np.zeros(k_array.shape[0], dtype=np.complex128)
        else:
            z_out = np.zeros((k_array.shape[0], n_feeds), dtype=np.complex128)
        k_save, wl_save, omega_save = self.k, self.wavelength, self.omega
        try:
            for i, kk in enumerate(k_array):
                self.k = float(kk)
                self.omega = self.k * self.c
                self.wavelength = self.c / (self.omega / (2 * np.pi))
                z, _ = self.compute_impedance()
                z_out[i] = z
        finally:
            self.k = k_save
            self.wavelength = wl_save
            self.omega = omega_save
        return z_out

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
