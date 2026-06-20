"""Kernel-agnostic hierarchical-matrix primitives: cluster tree, block
cluster tree (admissible/inadmissible partition), and (later) ACA.

Geometry only here — nothing about the MoM kernel. `HMatrixPySim` feeds in
per-basis axis-aligned bounding boxes (the extent of each basis function's
wire support) and gets back a partition of the index product [n]x[n] into
*admissible* (far, low-rank-able) and *inadmissible* (near, dense) leaf
blocks. The admissibility test is the standard

    min(diam(s), diam(t)) <= eta * dist(s, t)

with diam the bounding-box diagonal and dist the gap between boxes (0 when
they touch/overlap). Smaller eta => stricter => fewer/larger near blocks.
"""

import numpy as np


class Cluster:
    """A node of the binary space-partition cluster tree.

    `indices` are the basis indices contained in this cluster; `lo`/`hi` the
    axis-aligned bounding box (over the basis support boxes) of those indices.
    """

    __slots__ = ("indices", "lo", "hi", "left", "right")

    def __init__(self, indices, lo, hi, left=None, right=None):
        self.indices = indices
        self.lo = lo
        self.hi = hi
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None

    @property
    def size(self):
        return len(self.indices)

    @property
    def diam(self):
        return float(np.linalg.norm(self.hi - self.lo))

    def leaves(self):
        if self.is_leaf:
            yield self
        else:
            yield from self.left.leaves()
            yield from self.right.leaves()

    def n_nodes(self):
        if self.is_leaf:
            return 1
        return 1 + self.left.n_nodes() + self.right.n_nodes()


def build_cluster_tree(indices, box_lo, box_hi, leaf_size=32):
    """Recursively bisect `indices` along the longest box axis at the median
    of the per-basis box centers. `box_lo`/`box_hi` are global (N, 3) arrays
    indexed by basis id. Stops when a node holds <= `leaf_size` indices.
    """
    idx = np.asarray(indices, dtype=np.int64)
    lo = box_lo[idx].min(axis=0)
    hi = box_hi[idx].max(axis=0)
    node = Cluster(idx, lo, hi)
    if idx.size <= leaf_size:
        return node

    centers = 0.5 * (box_lo[idx] + box_hi[idx])
    axis = int(np.argmax(hi - lo))
    split = float(np.median(centers[:, axis]))
    left_mask = centers[:, axis] <= split
    # Degenerate median (many coincident centers): fall back to an
    # index-count bisection on the sorted axis so the recursion terminates.
    if left_mask.all() or not left_mask.any():
        order = np.argsort(centers[:, axis], kind="stable")
        left_mask = np.zeros(idx.size, dtype=bool)
        left_mask[order[: idx.size // 2]] = True

    node.left = build_cluster_tree(idx[left_mask], box_lo, box_hi, leaf_size)
    node.right = build_cluster_tree(idx[~left_mask], box_lo, box_hi, leaf_size)
    return node


def box_distance(a, b):
    """Euclidean gap between two axis-aligned boxes (0 if they overlap)."""
    delta = np.maximum(np.maximum(a.lo - b.hi, b.lo - a.hi), 0.0)
    return float(np.linalg.norm(delta))


def admissible(a, b, eta):
    d = box_distance(a, b)
    if d == 0.0:
        return False
    return min(a.diam, b.diam) <= eta * d


def build_block_tree(s, t, eta, leaf_size_stop=None):
    """Partition the product s.indices x t.indices into far (admissible) and
    near (dense) leaf blocks. Returns (far_blocks, near_blocks), each a list
    of (Cluster s, Cluster t) pairs.

    Standard block-cluster recursion: emit a far block as soon as the pair is
    admissible; recurse otherwise, subdividing whichever cluster(s) are not
    leaves; emit a near block when both are leaves and still inadmissible.
    """
    far = []
    near = []
    stack = [(s, t)]
    while stack:
        cs, ct = stack.pop()
        if admissible(cs, ct, eta):
            far.append((cs, ct))
        elif cs.is_leaf and ct.is_leaf:
            near.append((cs, ct))
        elif cs.is_leaf:
            stack.append((cs, ct.left))
            stack.append((cs, ct.right))
        elif ct.is_leaf:
            stack.append((cs.left, ct))
            stack.append((cs.right, ct))
        else:
            stack.append((cs.left, ct.left))
            stack.append((cs.left, ct.right))
            stack.append((cs.right, ct.left))
            stack.append((cs.right, ct.right))
    return far, near


def aca_partial(get_row, get_col, m, n, tol=1e-3, max_rank=None):
    """Adaptive Cross Approximation with partial pivoting.

    Builds a low-rank factorisation A ~ U @ V (U: (m, r), V: (r, n)) of an
    (m, n) block sampling only ~r full rows and r full columns of A via the
    callables `get_row(i) -> (n,)` and `get_col(j) -> (m,)`. Stops when the
    newest rank-1 update's Frobenius norm drops below `tol` times the running
    Frobenius norm of the accumulated approximation.

    Returns (U, V). Pure linear algebra — knows nothing about the kernel.
    """
    if max_rank is None:
        max_rank = min(m, n)
    max_rank = min(max_rank, m, n)

    U = []  # list of (m,) columns
    V = []  # list of (n,) rows
    used_rows = np.zeros(m, dtype=bool)
    used_cols = np.zeros(n, dtype=bool)
    approx_norm2 = 0.0
    i_star = 0
    rank = 0

    for _ in range(max_rank):
        row = np.asarray(get_row(i_star), dtype=np.complex128).copy()
        for k in range(rank):
            row -= U[k][i_star] * V[k]
        used_rows[i_star] = True

        absrow = np.abs(row)
        absrow[used_cols] = -1.0
        j_star = int(np.argmax(absrow))
        delta = row[j_star]
        if np.abs(delta) < 1e-300 or absrow[j_star] <= 0.0:
            # Residual row vanished: hop to an unused row and retry, else done.
            rem = np.flatnonzero(~used_rows)
            if rem.size == 0:
                break
            i_star = int(rem[0])
            continue

        v = row / delta
        col = np.asarray(get_col(j_star), dtype=np.complex128).copy()
        for k in range(rank):
            col -= V[k][j_star] * U[k]
        used_cols[j_star] = True
        u = col

        un = float(np.linalg.norm(u))
        vn = float(np.linalg.norm(v))
        cross = 0.0
        for k in range(rank):
            cross += np.real(np.vdot(U[k], u) * np.vdot(V[k], v))
        approx_norm2 += 2.0 * cross + (un * vn) ** 2

        U.append(u)
        V.append(v)
        rank += 1

        if approx_norm2 <= 0.0 or un * vn <= tol * np.sqrt(approx_norm2):
            break

        abscol = np.abs(col)
        abscol[used_rows] = -1.0
        if not (~used_rows).any():
            break
        i_star = int(np.argmax(abscol))

    if rank == 0:
        return (
            np.zeros((m, 0), dtype=np.complex128),
            np.zeros((0, n), dtype=np.complex128),
        )
    return np.array(U).T.copy(), np.array(V).copy()


class HMatrix:
    """Hierarchical matrix: a set of dense (near) and low-rank (far) blocks
    tiling an n x n matrix, with a fast matvec.

    near : list of (row_idx, col_idx, D)            D: (|row|, |col|)
    far  : list of (row_idx, col_idx, U, V)         U@V approximates the block
    precond_extra : optional list of (row_idx, col_idx, D) dense reconstructions
        of the first-ring far blocks — NOT part of the operator (the far list
        already represents them as low-rank); used only to strengthen the
        GMRES preconditioner's near-field.
    """

    def __init__(self, n, near, far, precond_extra=None):
        self.n = n
        self.near = near
        self.far = far
        self.precond_extra = precond_extra or []

    def matvec(self, x):
        x = np.asarray(x)
        y = np.zeros(self.n, dtype=np.complex128)
        for I, J, D in self.near:
            y[I] += D @ x[J]
        for I, J, U, V in self.far:
            y[I] += U @ (V @ x[J])
        return y

    def storage(self):
        """Number of complex scalars stored (vs n^2 for dense)."""
        s = sum(D.size for _, _, D in self.near)
        s += sum(U.size + V.size for _, _, U, V in self.far)
        return s

    def stats(self):
        near_area = sum(D.size for _, _, D in self.near)
        ranks = [U.shape[1] for _, _, U, _ in self.far]
        return {
            "n": self.n,
            "n_near": len(self.near),
            "n_far": len(self.far),
            "storage": self.storage(),
            "dense_storage": self.n * self.n,
            "compression": self.storage() / (self.n * self.n),
            "near_area": near_area,
            "max_rank": max(ranks) if ranks else 0,
            "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        }

    def to_dense(self):
        """Reconstruct the full dense matrix (validation / small n only)."""
        Z = np.zeros((self.n, self.n), dtype=np.complex128)
        for I, J, D in self.near:
            Z[np.ix_(I, J)] = D
        for I, J, U, V in self.far:
            Z[np.ix_(I, J)] = U @ V
        return Z


def partition_stats(n, far, near):
    """Summary of a block partition: counts and the fraction of the n x n
    matrix area that falls in far (compressible) vs near (dense) blocks.
    """
    far_area = sum(b[0].size * b[1].size for b in far)
    near_area = sum(b[0].size * b[1].size for b in near)
    total = n * n
    return {
        "n": n,
        "n_far": len(far),
        "n_near": len(near),
        "far_area": far_area,
        "near_area": near_area,
        "covered": far_area + near_area,
        "total": total,
        "far_frac": far_area / total if total else 0.0,
        "near_frac": near_area / total if total else 0.0,
    }
