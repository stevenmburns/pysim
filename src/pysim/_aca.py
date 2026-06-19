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
