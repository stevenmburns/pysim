"""Cached Gauss-Legendre quadrature nodes.

`numpy.polynomial.legendre.leggauss(n)` computes its nodes/weights via an
eigendecomposition of the Jacobi companion matrix — not free. The MoM
kernels call it once per same-edge block per wavenumber across a swept-k
solve (hundreds of times for a 41-point sweep), always with the same handful
of `n` values. Memoize on `n`.

The cached arrays are marked read-only so a shared entry can never be mutated
by a caller; every kernel here only ever reads them (e.g. ``0.5 * (xi + 1)``),
which allocates fresh arrays.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _leggauss_cached(n: int):
    xi, w = np.polynomial.legendre.leggauss(n)
    xi.setflags(write=False)
    w.setflags(write=False)
    return xi, w


def leggauss(n):
    """Memoized `numpy.polynomial.legendre.leggauss`. Returns read-only
    `(nodes, weights)` arrays — do not mutate; derive new arrays instead."""
    return _leggauss_cached(int(n))
