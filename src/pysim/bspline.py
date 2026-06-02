"""Higher-order B-spline Galerkin MoM solver (first cut).

Triangular basis (`TriangularPySim`) is degree-1 B-spline; this module
extends to arbitrary degree d, primarily so we can run degree-2 / degree-3
as an in-codebase arbiter for the hentenna question (see NEXT_STEPS.md
items 9, 13, 14): does the tent basis converge to the correct value, or
is it converged-to-the-wrong-place?

Scope of this first cut (deliberately narrow):
  * single straight single-edge wire (the polyline has 2 anchor points)
  * uniform N segments
  * free space, thin-wire kernel with a² regularization on every pair
    (no analytic same-edge static extraction yet — straight extension of
    the existing kernels, but unnecessary for the basic arbitration probe)
  * delta-gap "applied-E" source at the wire midpoint
  * degree d in {1, 2, 3}  (d=1 reproduces the tent basis up to quadrature)
  * no junctions, no multi-wire, no ground plane

The basis on each physical segment is a polynomial of degree d in the local
arc length u ∈ [0, h]. With C[m, w, p] = coefficient of u^p on wing w of
basis m and supp_seg[m, w] = global segment of that wing:

    Z_A[m,n]   = jωμ Σ_{a,b} (t_a · t_b) · Σ_{p,q} C[m,a,p] C[n,b,q]
                 · J_{pq}[supp_seg[m,a], supp_seg[n,b]]
    Z_Φ[m,n]   = (1/jωε) Σ_{a,b} Σ_{p≥1,q≥1} p·q · C[m,a,p] C[n,b,q]
                 · J_{p-1,q-1}[supp_seg[m,a], supp_seg[n,b]]

where

    J_{pq}[i,j] = ∫_{seg_i} ∫_{seg_j} u^p u'^q · exp(-jkR)/(4πR) du' du
    R = √(|r_i(u) - r_j(u')|² + a²)

Feed: pick a sample point s_f on the wire (default: midpoint). The Galerkin
source vector is v_m = Φ_m(s_f) (absorbing the V=1 convention). The current
at s_f is I(s_f) = Σ_n c_n Φ_n(s_f) = v^T c, and Z_drive = 1 / I(s_f).
"""

import numpy as np
import scipy.linalg
from scipy.interpolate import BSpline

from ._bspline_kernels import _seg_seg_reg_moments, _seg_seg_static_moments


class BSplinePySim:
    """Degree-d B-spline Galerkin MoM, single straight wire.

    Parameters
    ----------
    wires : list of (M, 3) polyline arrays
        Must contain exactly one polyline with exactly 2 anchor points.
    n_per_edge_per_wire : list, optional
        Per-wire segment count list. For the single-edge wire here, must be
        [[N]] or None (uses `nsegs`).
    degree : int
        B-spline degree d ≥ 1. d=1 reproduces the tent basis (up to the
        quadrature-vs-analytic difference in the same-segment kernel),
        d=2 / d=3 are the new arbiters.
    n_qp_pair : int
        Gauss-Legendre points per segment per axis in the J integrals.
    wavelength, wire_radius, nsegs : as in TriangularPySim.
    feed_arclength : arc length from wire start at which to inject the
        delta-gap source. Default: wire midpoint.
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        degree=2,
        feed_wire_index=0,
        feed_arclength=None,
        n_qp_pair=4,
        wavelength=22,
        halfdriver_factor=0.962,
        wire_radius=0.0005,
        nsegs=101,
    ):
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if not wires:
            raise ValueError("wires must be non-empty")
        if len(wires) != 1:
            raise NotImplementedError(
                f"BSplinePySim first cut: single-wire only (got {len(wires)} wires)"
            )
        pl = np.asarray(wires[0], dtype=float)
        if pl.ndim != 2 or pl.shape != (2, 3):
            raise NotImplementedError(
                "BSplinePySim first cut: single straight edge (2 anchors)"
                f", got polyline shape {pl.shape}"
            )

        self.degree = int(degree)
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.wire_radius = wire_radius
        self.nsegs = nsegs

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = self.omega / self.c
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4

        self.wires_polylines = [pl]
        if n_per_edge_per_wire is None:
            n_per_edge_per_wire = [None]
        if len(n_per_edge_per_wire) != 1:
            raise ValueError("n_per_edge_per_wire length must equal n_wires (1)")
        npe = n_per_edge_per_wire[0]
        if npe is None:
            npe = self.nsegs
        if np.isscalar(npe):
            npe = [int(npe)]
        npe = list(npe)
        if len(npe) != 1:
            raise ValueError("first cut: single edge expected")
        self.n_per_edge_per_wire = [npe]
        self.N = int(npe[0])

        if feed_wire_index != 0:
            raise ValueError("feed_wire_index must be 0 for the single-wire case")
        self.feed_wire_index = 0
        self.feed_arclength = feed_arclength
        self.n_qp_pair = int(n_qp_pair)

    # ------------------------------------------------------------------
    # Geometry build
    # ------------------------------------------------------------------

    def _build_geometry(self):
        pl = self.wires_polylines[0]
        p0, p1 = pl[0], pl[1]
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1e-15:
            raise ValueError("wire edge has zero length")
        tan = edge_vec / edge_len
        N = self.N
        h = edge_len / N

        seg_endpoints_arc = np.linspace(0.0, edge_len, N + 1)
        # 3D positions of segment endpoints
        seg_pts = p0[None, :] + seg_endpoints_arc[:, None] * tan[None, :]
        seg_l = seg_pts[:-1]
        seg_r = seg_pts[1:]

        return {
            "p0": p0,
            "p1": p1,
            "tan": tan,
            "edge_len": edge_len,
            "h": h,
            "N": N,
            "seg_l": seg_l,
            "seg_r": seg_r,
            "seg_endpoints_arc": seg_endpoints_arc,
        }

    # ------------------------------------------------------------------
    # B-spline basis polynomial extraction
    # ------------------------------------------------------------------

    def _build_basis_polynomials(self, geom):
        """Per-basis polynomial-coefficient table.

        Returns
        -------
        supp_seg : (n_basis, n_wings) int
            global segment index of wing w of basis m. n_wings = degree + 1.
            For boundary bases whose support has fewer than d+1 segments
            (clamped knot multiplicity), we still allocate n_wings slots but
            mark the unused ones with seg=0 and zero coefficients so they
            contribute nothing to the assembly.
        polys : (n_basis, n_wings, n_poly) float
            polys[m, w, p] = coefficient of u^p (u ∈ [0, h], local segment
            arc) on wing w of basis m. Inactive wings have all zeros.
        knots : (n_knots,) float
            The clamped knot vector (in arc length).
        n_basis_total : int
            Total bases on the clamped spline (= N + d). Includes the two
            boundary bases that are NOT in the returned tables (they're
            dropped for the zero-current Dirichlet BC at the wire ends).
        """
        d = self.degree
        N = self.N
        h = geom["h"]
        edge_len = geom["edge_len"]
        n_wings = d + 1

        interior_knots = np.linspace(0.0, edge_len, N + 1)
        knots = np.concatenate([np.full(d, 0.0), interior_knots, np.full(d, edge_len)])
        n_basis_total = len(knots) - d - 1  # = N + d

        # Drop bases j=0 and j=n_basis_total-1 (the two clamped-boundary
        # bases that have nonzero value at the wire endpoint).
        keep_idx = np.arange(1, n_basis_total - 1)
        n_basis = len(keep_idx)

        supp_seg = np.zeros((n_basis, n_wings), dtype=np.int64)
        polys = np.zeros((n_basis, n_wings, d + 1), dtype=np.float64)

        # Per-basis polynomial extraction: evaluate B_j at d+1 points
        # inside each segment in its support, solve a Vandermonde for the
        # polynomial coefficients in local-u coordinates.
        u_eval_local = np.linspace(0.0, h, d + 1)
        V = np.vander(u_eval_local, d + 1, increasing=True)  # (d+1, d+1)
        V_inv = np.linalg.inv(V)

        eps_seg = 1e-9 * h
        for m, j in enumerate(keep_idx):
            c_all = np.zeros(n_basis_total, dtype=np.float64)
            c_all[j] = 1.0
            bspl = BSpline(knots, c_all, d, extrapolate=False)

            support_lo = knots[j]
            support_hi = knots[j + d + 1]

            # Find segments overlapping the support
            wing = 0
            for seg in range(N):
                seg_l_arc = seg * h
                seg_r_arc = (seg + 1) * h
                if seg_r_arc < support_lo + eps_seg:
                    continue
                if seg_l_arc > support_hi - eps_seg:
                    break
                u_eval_global = seg_l_arc + u_eval_local
                vals = bspl(u_eval_global)
                coeffs = V_inv @ vals
                supp_seg[m, wing] = seg
                polys[m, wing, :] = coeffs
                wing += 1
            # remaining wings (if any) stay at zero coefficients

        return supp_seg, polys, knots, n_basis_total

    # ------------------------------------------------------------------
    # Polynomial-moment quadrature
    # ------------------------------------------------------------------

    def _J_moments(self, geom, k):
        """All polynomial moment integrals on every same-edge segment pair.

        Returns J of shape (d+1, d+1, N, N) complex with
            J[p, q, i, j] = ∫∫ u^p u'^q · exp(-jkR)/(4πR) du' du
            R = √((s_i(u) - s_j(u'))² + a²)

        Split as J = J_static + J_reg, where the static piece is integrated
        in closed form (essential on the same-segment diagonal where direct
        GL converges only logarithmically on the 1/R singularity) and the
        smooth piece (exp(-jkR) - 1)/R is integrated by Gauss-Legendre.
        """
        d = self.degree
        a = self.wire_radius
        # seg_endpoints_arc are arc lengths from edge start: 0, h, 2h, ..., L
        seg_endpoints_arc = geom["seg_endpoints_arc"]

        # The closed-form kernel uses (s - alpha_i)^p (s' - A_j)^q, where
        # alpha_i, A_j are segment-LEFT global arc positions. The local-u
        # coordinate u = s - alpha_i runs from 0 to h on segment i — so
        # (s - alpha_i)^p is exactly u^p. No reparametrization needed.
        J_static = _seg_seg_static_moments(
            seg_endpoints_arc, a, max_d=d
        )  # (d+1, d+1, N, N) real
        J_reg = _seg_seg_reg_moments(
            seg_endpoints_arc, a, k, max_d=d, n_qp=self.n_qp_pair
        )  # (d+1, d+1, N, N) complex
        return J_static + J_reg

    # ------------------------------------------------------------------
    # Z assembly
    # ------------------------------------------------------------------

    def _assemble_Z(self, J, supp_seg, polys, tangent_dot):
        """(n_basis, n_basis) complex Z from the polynomial-moment tensors."""
        d = self.degree
        n_basis, n_wings, n_poly = polys.shape
        assert n_wings == d + 1 and n_poly == d + 1

        Z_A = np.zeros((n_basis, n_basis), dtype=np.complex128)
        Z_Phi = np.zeros((n_basis, n_basis), dtype=np.complex128)

        for a in range(n_wings):
            sm = supp_seg[:, a]
            for b in range(n_wings):
                sn = supp_seg[:, b]
                # Block J[p, P, i, j] for (i, j) ∈ (sm, sn)
                # Shape: (d+1, d+1, n_basis, n_basis)
                J_blk = J[:, :, sm[:, None], sn[None, :]]
                td_blk = tangent_dot[sm[:, None], sn[None, :]]

                # Z_A: ∑_{p,q} polys[m,a,p] polys[n,b,q] J[p,q,m,n]
                inner_A = np.einsum(
                    "mp,pPmn,nP->mn", polys[:, a, :], J_blk, polys[:, b, :]
                )
                Z_A += td_blk * inner_A

                # Z_Phi: ∑_{p,q ≥ 1} p·q polys[m,a,p] polys[n,b,q] J[p-1,q-1,m,n]
                if d >= 1:
                    p_vec = np.arange(1, d + 1, dtype=np.float64)
                    deriv_m = polys[:, a, 1:] * p_vec[None, :]  # (n_basis, d)
                    deriv_n = polys[:, b, 1:] * p_vec[None, :]
                    J_blk_lo = J_blk[:d, :d]  # (d, d, n_basis, n_basis)
                    inner_Phi = np.einsum("mp,pPmn,nP->mn", deriv_m, J_blk_lo, deriv_n)
                    Z_Phi += inner_Phi

        Z_A = 1j * self.omega * self.mu * Z_A
        Z_Phi = Z_Phi / (1j * self.omega * self.eps)
        return Z_A + Z_Phi

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def _build_source_vector(self, geom, supp_seg, polys, knots, n_basis_total):
        """Galerkin delta-gap rhs: v_m = Φ_m(s_f) with s_f = feed arclength.

        Also returns the basis-value vector at s_f used to compute
        I(s_f) = Σ_n c_n Φ_n(s_f) = v^T c, so the driver impedance is
        Z = 1 / (v^T c) with V=1.
        """
        edge_len = geom["edge_len"]
        s_f = self.feed_arclength if self.feed_arclength is not None else edge_len / 2.0
        d = self.degree
        # evaluate ALL bases at s_f via design_matrix
        DM = BSpline.design_matrix(np.array([s_f]), knots, d).toarray()[0]
        # drop boundary bases (j=0 and j=n_basis_total-1)
        v_full = DM[1 : n_basis_total - 1]
        return v_full

    # ------------------------------------------------------------------
    # Driver impedance
    # ------------------------------------------------------------------

    def compute_impedance(self):
        geom = self._build_geometry()
        supp_seg, polys, knots, n_basis_total = self._build_basis_polynomials(geom)
        # tangent dot (single straight wire → all 1)
        N = geom["N"]
        tangent_dot = np.ones((N, N), dtype=np.float64)

        J = self._J_moments(geom, self.k)
        Z = self._assemble_Z(J, supp_seg, polys, tangent_dot)
        v = self._build_source_vector(
            geom, supp_seg, polys, knots, n_basis_total
        ).astype(np.complex128)
        coeffs = scipy.linalg.solve(Z, v)
        I_at_feed = v @ coeffs  # = Σ_n c_n Φ_n(s_f)
        driver_impedance = 1.0 / I_at_feed
        self.z = Z
        return driver_impedance, coeffs
