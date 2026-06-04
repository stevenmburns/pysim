"""NEC2-style sinusoidal-basis MoM for wires (Section III of the NEC2
Theory Manual, Burke & Poggio 1981 — see `docs/sinusoidal_basis_design.md`).

This is an OPTIONAL solver alongside `TriangularPySim` (the default). The
sinusoidal basis is what NEC2 / PyNEC / nec2c use; reproducing it in pysim
lets us isolate which parts of NEC's pulse-basis convergence behaviour are
intrinsic to the basis itself versus its kernel / source / junction
treatment.

Scope (deliberately narrow):
  * Free-space only — no Sommerfeld ground, no PEC image.
  * Thin-wire kernel (Eqs 73-79), no extended thin-wire / current-element.
  * Delta-gap "applied-E" source (Eq 187) on a single basis function.
  * Uniform wire radius across all wires.
  * Free wire ends use the X_i = 0 zero-current condition (the more
    physical J_1/J_0 end-cap condition is negligible for thin wires).
"""

import numpy as np
import scipy.linalg
import scipy.sparse

try:
    from pysim import _accelerators as _acc

    _HAVE_FIELD_TENSOR = hasattr(_acc, "sinusoidal_field_tensor")
except ImportError:
    _HAVE_FIELD_TENSOR = False

_EULER_GAMMA = 0.5772156649015329

# Threshold for dense vs sparse assembly in `_assemble_Z`. Below this N
# the BLAS overhead on a tiny matrix loses to dense matmul; above it the
# O(N³) zgemm cost on a mostly-zero matrix loses to CSC sparse matmul.
# Measured crossover on Kaby Lake R / OpenBLAS-pthreads ≈ 60.
_DENSE_ASSEMBLY_THRESHOLD = 60


class SinusoidalPySim:
    """NEC2's three-term (const + sin + cos) basis on each segment, with
    end-condition coefficients closed-form per Eqs 25-64.

    Constructor takes the same `wires` / `n_per_edge_per_wire` / `junctions`
    interface as `TriangularPySim` for drop-in comparison.
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        feed_wire_index=0,
        feed_arclength=None,
        wavelength=22,
        halfdriver_factor=0.962,
        wire_radius=0.0005,
        nsegs=101,
        ground_z=None,
        junctions=None,
        n_qp_const=8,
    ):
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.wire_radius = wire_radius
        self.nsegs = nsegs
        self.ground_z = ground_z

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = self.omega / self.c
        self.eta = float(np.sqrt(self.mu / self.eps))
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4
        # Gauss-Legendre nodes for the const-source self-integral are
        # k-independent; cache by n_qp so sweep loops don't pay for
        # repeated leggauss() calls.
        self._leggauss_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # `compute_impedance(...)` and `currents_at_knots(alpha)` are almost
        # always called as a pair from the UI; both internally rebuild geom
        # and basis-coefs from scratch (~5 ms/step on N=21 hentenna). The
        # geometry is purely a function of (wires, n_per_edge, junctions),
        # which are immutable after __init__; the basis-coefs are a function
        # of (geom, k, wire_radius). Cache both so the second call reuses
        # the work the first call already did. _basis_coefs validates the
        # cache by identity-checking the geom dict + value-comparing k and
        # wire_radius — so `compute_impedance_swept` (which mutates self.k
        # in a loop) still rebuilds basis-coefs per k, but reuses geom.
        self._cached_geometry: dict | None = None
        self._cached_basis: tuple[dict, float, float, list] | None = None

        if not wires:
            raise ValueError("wires must be non-empty")
        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")

        n_w = len(self.wires_polylines)
        if n_per_edge_per_wire is None:
            n_per_edge_per_wire = [None] * n_w
        if len(n_per_edge_per_wire) != n_w:
            raise ValueError(
                f"n_per_edge_per_wire length {len(n_per_edge_per_wire)} "
                f"!= number of wires {n_w}"
            )

        self.n_per_edge_per_wire = []
        for i, (pl, npe) in enumerate(zip(self.wires_polylines, n_per_edge_per_wire)):
            n_edges_w = pl.shape[0] - 1
            if npe is None:
                npe = self.nsegs
            if np.isscalar(npe):
                npe = [int(npe)] * n_edges_w
            npe = list(npe)
            if len(npe) != n_edges_w:
                raise ValueError(
                    f"wire {i}: n_per_edge length {len(npe)} "
                    f"!= number of edges {n_edges_w}"
                )
            self.n_per_edge_per_wire.append(npe)

        if not (0 <= feed_wire_index < n_w):
            raise ValueError(f"feed_wire_index {feed_wire_index} out of range")
        self.feed_wire_index = feed_wire_index
        self.feed_arclength = feed_arclength
        self.n_qp_const = n_qp_const

        self.junctions = []
        if junctions is not None:
            for j, jw in enumerate(junctions):
                if len(jw) < 2:
                    raise ValueError(
                        f"junction {j}: need >= 2 wire-ends, got {len(jw)}"
                    )
                normalized = []
                for w, end in jw:
                    if not (0 <= w < n_w):
                        raise ValueError(
                            f"junction {j}: wire_idx {w} out of range [0, {n_w})"
                        )
                    if end not in ("start", "end"):
                        raise ValueError(
                            f"junction {j}: end must be 'start' or 'end', got {end!r}"
                        )
                    normalized.append((int(w), end))
                self.junctions.append(normalized)

    def _leggauss_cached(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._leggauss_cache.get(n)
        if cached is not None:
            return cached
        gx, gw = np.polynomial.legendre.leggauss(n)
        gx = np.ascontiguousarray(gx, dtype=np.float64)
        gw = np.ascontiguousarray(gw, dtype=np.float64)
        self._leggauss_cache[n] = (gx, gw)
        return gx, gw

    # ------------------------------------------------------------------
    # Geometry build
    # ------------------------------------------------------------------

    def _build_geometry(self):
        """Discretize wires into segments and build the N^-/N^+ neighbor
        tables for every segment, with arc-flip σ signs.

        For a segment n at the head of a wire-segment sequence whose
        natural arc direction matches NEC's convention (segment's `end-2` =
        `seg_r` is at the junction node when treated as an N^- neighbour
        of another basis), σ = +1. When the natural tangent is reversed
        relative to NEC's expected arc, σ = -1.
        """
        if self._cached_geometry is not None:
            return self._cached_geometry
        # Build per-edge chunks (n_e segments per edge) in vectorized form,
        # then concatenate. Inner k_seg Python loop (171 iters × 5 list
        # appends + 4 numpy temporaries) was ~12% of py-spy samples at N=21.
        seg_l_chunks: list[np.ndarray] = []
        seg_r_chunks: list[np.ndarray] = []
        seg_c_chunks: list[np.ndarray] = []
        seg_t_chunks: list[np.ndarray] = []
        seg_h_chunks: list[np.ndarray] = []
        wire_first_seg: list[int] = []
        wire_last_seg: list[int] = []
        running_count = 0

        for w_idx, (pl, npe_list) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            wire_first = running_count
            for e_idx in range(pl.shape[0] - 1):
                p0 = pl[e_idx]
                p1 = pl[e_idx + 1]
                vec = p1 - p0
                edge_len = float(np.linalg.norm(vec))
                if edge_len < 1e-15:
                    raise ValueError(f"wire {w_idx} edge {e_idx} has zero length")
                tan = vec / edge_len
                n_e = npe_list[e_idx]
                h_e = edge_len / n_e
                # frac in [0, 1] sampled at n_e+1 points; consecutive points
                # bound each segment.
                frac = np.linspace(0.0, 1.0, n_e + 1)
                pts = p0 + frac[:, None] * vec  # (n_e+1, 3)
                pl_l_arr = pts[:-1]  # (n_e, 3)
                pl_r_arr = pts[1:]  # (n_e, 3)
                seg_l_chunks.append(pl_l_arr)
                seg_r_chunks.append(pl_r_arr)
                seg_c_chunks.append(0.5 * (pl_l_arr + pl_r_arr))
                seg_t_chunks.append(np.broadcast_to(tan, (n_e, 3)).copy())
                seg_h_chunks.append(np.full(n_e, h_e, dtype=np.float64))
                running_count += n_e
            wire_last_seg.append(running_count - 1)
            wire_first_seg.append(wire_first)

        seg_l = np.concatenate(seg_l_chunks, axis=0)
        seg_r = np.concatenate(seg_r_chunks, axis=0)
        seg_c = np.concatenate(seg_c_chunks, axis=0)
        seg_t = np.concatenate(seg_t_chunks, axis=0)
        seg_h = np.concatenate(seg_h_chunks)
        n_segs = seg_l.shape[0]

        # Per-segment N^- / N^+ neighbour lists.
        # nm[i] = list of (j, sigma) where j is a segment whose "end-2" in
        # NEC's arc convention coincides with i's end-1 (i's seg_l end).
        # np[i] = list of (j, sigma) where j's NEC "end-1" coincides with
        # i's end-2 (i's seg_r end).
        nm = [[] for _ in range(n_segs)]
        np_ = [[] for _ in range(n_segs)]

        # In-wire neighbours: adjacent segments in the same wire, both at
        # kinks and along straight edges, are connected through their
        # naturally adjacent seg_l/seg_r endpoints.
        for w_idx in range(len(self.wires_polylines)):
            first = wire_first_seg[w_idx]
            last = wire_last_seg[w_idx]
            for i in range(first, last + 1):
                if i > first:
                    # segment i-1 connects to i at i's end-1 (seg_l).
                    # (i-1)'s natural seg_r is at the junction → matches
                    # NEC's "end-2 of j" convention for N^- → σ = +1.
                    nm[i].append((i - 1, +1))
                if i < last:
                    # segment i+1 connects to i at i's end-2 (seg_r).
                    # (i+1)'s natural seg_l is at the junction → matches
                    # NEC's "end-1 of j" convention for N^+ → σ = +1.
                    np_[i].append((i + 1, +1))

        # Junction neighbours. For each junction node, every (wire, end)
        # listed contributes its end-segment. The end-segments at this
        # node are all mutually connected.
        for jn in self.junctions:
            # Collect (segment_idx, which_end_of_segment_is_at_node) for
            # every wire-end at this junction.
            members = []
            for w, end in jn:
                if end == "start":
                    seg_idx = wire_first_seg[w]
                    end_side = "L"  # seg_l of this segment is at node
                else:
                    seg_idx = wire_last_seg[w]
                    end_side = "R"  # seg_r of this segment is at node
                members.append((seg_idx, end_side))

            # Wire each pair of (i, j) members: j is a neighbour of i at
            # the node-side of i; figure out which list (nm/np_) of i it
            # goes into and what σ j contributes.
            for a in range(len(members)):
                i_seg, i_side = members[a]
                for b in range(len(members)):
                    if b == a:
                        continue
                    j_seg, j_side = members[b]
                    # i's side at the node — does the node live at i's
                    # end-1 (L) or end-2 (R)?
                    if i_side == "L":
                        # j is in N^-(i). NEC wants j's "end-2" at the
                        # node. j's natural seg_r is at the node iff
                        # j_side == "R" → σ = +1; else σ = -1.
                        sigma = +1 if j_side == "R" else -1
                        nm[i_seg].append((j_seg, sigma))
                    else:
                        # j is in N^+(i). NEC wants j's "end-1" at the
                        # node. j's natural seg_l is at the node iff
                        # j_side == "L" → σ = +1; else σ = -1.
                        sigma = +1 if j_side == "L" else -1
                        np_[i_seg].append((j_seg, sigma))

        # Feed segment index: the segment on the feed wire whose center is
        # closest to feed_arclength (default: midpoint of the wire).
        w_f = self.feed_wire_index
        first = wire_first_seg[w_f]
        last = wire_last_seg[w_f]
        feed_h = seg_h[first : last + 1]
        feed_arc_centers = np.cumsum(feed_h) - 0.5 * feed_h
        total_arc = float(np.sum(feed_h))
        feed_arc = (
            self.feed_arclength if self.feed_arclength is not None else 0.5 * total_arc
        )
        feed_seg = first + int(np.argmin(np.abs(feed_arc_centers - feed_arc)))

        self._cached_geometry = {
            "seg_l": seg_l,
            "seg_r": seg_r,
            "seg_centers": seg_c,
            "seg_tangents": seg_t,
            "seg_h": seg_h,
            "n_segs": n_segs,
            "nm": nm,
            "np": np_,
            "wire_first": wire_first_seg,
            "wire_last": wire_last_seg,
            "feed_seg": feed_seg,
        }
        return self._cached_geometry

    # ------------------------------------------------------------------
    # Basis-function coefficient computation
    # ------------------------------------------------------------------

    def _basis_coefs(self, geom, k):
        """Per-basis closed-form (A, B, C, σ) coefficients on every
        supporting segment, following Eqs 43-64 of the NEC2 Theory Manual.

        Returns a CSR-by-segment `seg_view` dict:

            seg_view["starts"][s:s+2] → range of entries for segment s in
                the flat per-segment arrays below.
            seg_view["jbasis"][k]     → which basis contributes entry k.
            seg_view["A"/"B"/"C"][k]  → that basis's coefficient on seg.
            seg_view["sigma"][k]      → σ sign relative to NEC arc.

        Writing the entries directly into seg-major position during the
        main per-basis loop (instead of building a list-of-lists `basis`
        and then transposing it) avoids a second Python pass over ~1700
        entries — the path that was costing ~1.2 ms/step at N=21 hentenna.
        It also lets `_assemble_Z`'s flat-array fill become vectorized
        numpy (np.repeat + element-wise multiply) rather than a Python
        scatter loop.

        Reciprocity lets us compute `starts[]` upfront from the geometry
        alone, without a first pass to count: a segment s appears as a
        support entry of basis s itself (the self entry) plus once per
        basis i in `nm[s] ∪ np_[s]` — i.e. once for every neighbour of s.
        So entries_per_seg[s] = 1 + len(nm[s]) + len(np_[s]). Each basis
        i contributes 1 + len(nm[i]) + len(np_[i]) entries on its own
        support, so the *basis-major* count is the same value indexed
        differently. We only need the seg-major layout for downstream
        consumers, so that's all we materialise.
        """
        a = self.wire_radius
        cached = self._cached_basis
        if (
            cached is not None
            and cached[0] is geom
            and cached[1] == k
            and cached[2] == a
        ):
            return cached[3]
        seg_h = geom["seg_h"]
        n_segs = geom["n_segs"]
        nm = geom["nm"]
        np_ = geom["np"]

        ka = k * a
        # a_i± from Eq 25 (same value for both ends since wire radius is
        # uniform; we name it a_const).
        a_const = 1.0 / (np.log(2.0 / ka) - _EULER_GAMMA)

        # Pre-allocate the seg-major flat arrays. counts[s] = 1 (self) +
        # len(nm[s]) + len(np_[s]) by reciprocity (see docstring).
        counts = np.empty(n_segs, dtype=np.int64)
        for s in range(n_segs):
            counts[s] = 1 + len(nm[s]) + len(np_[s])
        starts = np.empty(n_segs + 1, dtype=np.int64)
        starts[0] = 0
        np.cumsum(counts, out=starts[1:])
        total = int(starts[-1])
        jbasis_flat = np.empty(total, dtype=np.int64)
        A_flat = np.empty(total, dtype=np.complex128)
        B_flat = np.empty(total, dtype=np.complex128)
        C_flat = np.empty(total, dtype=np.complex128)
        sigma_flat = np.empty(total, dtype=np.int8)
        cursor = starts[:-1].copy()  # write head per segment

        # Pre-compute every per-segment trig used inside the main loop in
        # one vectorized pass. Replacing ~10 np.sin/np.cos ufunc calls per
        # iter (171 iters → ~1700 dispatches × ~2 µs) with array lookups
        # was the biggest remaining Python win at N=21 — py-spy attributed
        # ~25% of samples here. Also pre-compute the (1∓cos)/sin ratios
        # used by P_minus / P_plus accumulation.
        kd_arr = k * np.asarray(seg_h, dtype=np.float64)
        sin_kd = np.sin(kd_arr)
        cos_kd = np.cos(kd_arr)
        sin_kd_2 = np.sin(0.5 * kd_arr)
        cos_kd_2 = np.cos(0.5 * kd_arr)
        # P-sum atoms: (1-cos(kd_j))/sin(kd_j) * a_const for N⁻; flip sign
        # for N⁺. Same regardless of which basis i references segment j as
        # a neighbour, so we cache per-segment.
        P_minus_atom = (1.0 - cos_kd) / sin_kd * a_const
        P_plus_atom = -P_minus_atom  # (cos(kd_j) - 1)/sin(kd_j) * a_const

        for i in range(n_segs):
            sin_kdi = sin_kd[i]
            cos_kdi = cos_kd[i]
            sin_kdi_2 = sin_kd_2[i]
            cos_kdi_2 = cos_kd_2[i]

            N_minus = nm[i]
            N_plus = np_[i]
            has_minus = len(N_minus) > 0
            has_plus = len(N_plus) > 0

            # P_i± via Eqs 62, 63. With uniform a these sums are pure
            # geometric ratios times a_const — atom precomputed above.
            P_minus = 0.0
            for j, sig in N_minus:
                P_minus += P_minus_atom[j]
            P_plus = 0.0
            for j, sig in N_plus:
                P_plus += P_plus_atom[j]

            a_minus = a_const
            a_plus = a_const

            if has_minus and has_plus:
                # Interior basis (Eqs 49-53).
                D = (P_minus * P_plus + a_minus * a_plus) * sin_kdi + (
                    P_minus * a_plus - P_plus * a_minus
                ) * cos_kdi
                Q_minus = (a_plus * (1 - cos_kdi) - P_plus * sin_kdi) / D
                Q_plus = (a_minus * (cos_kdi - 1) - P_minus * sin_kdi) / D
                A_i0 = -1.0
                B_i0 = (a_minus * Q_minus + a_plus * Q_plus) * sin_kdi_2 / sin_kdi
                C_i0 = (a_minus * Q_minus - a_plus * Q_plus) * cos_kdi_2 / sin_kdi
            elif has_plus and not has_minus:
                # End segment with free wire end at end-1 (Eqs 54-57).
                # Zero-current free-end convention → X_i = 0.
                X_i = 0.0
                denom_x = cos_kdi - X_i * sin_kdi
                Q_plus = (cos_kdi - 1 - X_i * sin_kdi) / (
                    (a_plus + X_i * P_plus) * sin_kdi
                    + (a_plus * X_i - P_plus) * cos_kdi
                )
                Q_minus = 0.0  # no end-1 segments
                A_i0 = -1.0
                B_i0 = (
                    sin_kdi_2 / denom_x
                    + a_plus * Q_plus * (cos_kdi_2 - X_i * sin_kdi_2) / denom_x
                )
                C_i0 = (
                    cos_kdi_2 / denom_x
                    + a_plus * Q_plus * (sin_kdi_2 + X_i * cos_kdi_2) / denom_x
                )
            elif has_minus and not has_plus:
                # End segment with free wire end at end-2 (Eqs 58-61).
                X_i = 0.0
                denom_x = cos_kdi - X_i * sin_kdi
                Q_minus = (1 - cos_kdi + X_i * sin_kdi) / (
                    (a_minus - X_i * P_minus) * sin_kdi
                    + (P_minus + X_i * a_minus) * cos_kdi
                )
                Q_plus = 0.0
                A_i0 = -1.0
                B_i0 = (
                    -sin_kdi_2 / denom_x
                    + a_minus * Q_minus * (cos_kdi_2 - X_i * sin_kdi_2) / denom_x
                )
                C_i0 = (
                    cos_kdi_2 / denom_x
                    - a_minus * Q_minus * (sin_kdi_2 + X_i * cos_kdi_2) / denom_x
                )
            else:
                # Isolated single segment (both ends free); Eq 64.
                # f_i^0(s) = cos(k(s-s_i))/(cos(kΔ_i/2) - X_i sin(kΔ_i/2)) - 1.
                # → A_i0 = -1, B_i0 = 0, C_i0 = 1/(cos kΔ/2 - X sin kΔ/2)
                X_i = 0.0
                A_i0 = -1.0
                B_i0 = 0.0
                C_i0 = 1.0 / (cos_kdi_2 - X_i * sin_kdi_2)
                Q_minus = 0.0
                Q_plus = 0.0

            # Self entry of basis i — lands at seg i.
            p = cursor[i]
            jbasis_flat[p] = i
            A_flat[p] = A_i0
            B_flat[p] = B_i0
            C_flat[p] = C_i0
            sigma_flat[p] = +1
            cursor[i] = p + 1

            # N^- neighbours: Eqs 43-45. σ records the arc-flip relative
            # to the segment's natural tangent; coefficients are NEC's
            # (computed in the NEC-arc frame).
            for j, sig in N_minus:
                p = cursor[j]
                jbasis_flat[p] = i
                A_flat[p] = a_plus * Q_minus / sin_kd[j]
                B_flat[p] = a_plus * Q_minus / (2.0 * cos_kd_2[j])
                C_flat[p] = -a_plus * Q_minus / (2.0 * sin_kd_2[j])
                sigma_flat[p] = sig
                cursor[j] = p + 1

            # N^+ neighbours: Eqs 46-48.
            for j, sig in N_plus:
                p = cursor[j]
                jbasis_flat[p] = i
                A_flat[p] = -a_minus * Q_plus / sin_kd[j]
                B_flat[p] = a_minus * Q_plus / (2.0 * cos_kd_2[j])
                C_flat[p] = a_minus * Q_plus / (2.0 * sin_kd_2[j])
                sigma_flat[p] = sig
                cursor[j] = p + 1

        seg_view = {
            "starts": starts,
            "jbasis": jbasis_flat,
            "A": A_flat,
            "B": B_flat,
            "C": C_flat,
            "sigma": sigma_flat,
        }
        self._cached_basis = (geom, k, a, seg_view)
        return seg_view

    def _evaluate_basis_at_points(self, seg_view, eval_seg, eval_s, alpha):
        """Vectorized evaluation of Σ_j α_j · f_{j, seg}(s_local) at an
        array of (segment, s_local) pairs.

        Replaces the per-knot `eval_at` Python closure: 342 individual
        calls per N=21 hentenna step (1.5 ms/step of Python frame +
        per-segment list iteration) collapse to one sin/cos call over
        n_eval points, one ragged gather, and one scatter-add.

        Parameters
        ----------
        seg_view : dict
            CSR-format inverse index from `_basis_coefs`.
        eval_seg : (n_eval,) int64
            Segment index per evaluation point.
        eval_s : (n_eval,) float64
            Local arc from each segment's centre.
        alpha : (n_basis,) complex128
            Basis amplitudes (from the EFIE solve).

        Returns
        -------
        (n_eval,) complex128
        """
        n_eval = eval_seg.shape[0]
        if n_eval == 0:
            return np.zeros(0, dtype=np.complex128)
        starts = seg_view["starts"]
        starts_at = starts[eval_seg]  # (n_eval,)
        lengths = starts[eval_seg + 1] - starts_at  # entries per eval
        n_entries = int(lengths.sum())
        if n_entries == 0:
            return np.zeros(n_eval, dtype=np.complex128)
        # Precompute trig at each eval point.
        sin_ks = np.sin(self.k * eval_s)
        cos_ks = np.cos(self.k * eval_s)
        # Ragged-gather expansion: for each eval i with `lengths[i]` entries,
        # produce that many entry-level rows. `entry_eval_idx` maps each row
        # back to its source eval; `entry_global` gathers from `seg_view`.
        entry_eval_idx = np.repeat(np.arange(n_eval, dtype=np.int64), lengths)
        # within-segment offset of each entry within its eval's block:
        # arange(n_entries) - cumulative-start-per-eval-block
        cum_starts = np.empty(n_eval, dtype=np.int64)
        cum_starts[0] = 0
        if n_eval > 1:
            np.cumsum(lengths[:-1], out=cum_starts[1:])
        within = np.arange(n_entries, dtype=np.int64) - np.repeat(cum_starts, lengths)
        entry_global = np.repeat(starts_at, lengths) + within
        jb = seg_view["jbasis"][entry_global]
        A_e = seg_view["A"][entry_global]
        B_e = seg_view["B"][entry_global]
        C_e = seg_view["C"][entry_global]
        sigma_e = seg_view["sigma"][entry_global]
        sin_e = sin_ks[entry_eval_idx]
        cos_e = cos_ks[entry_eval_idx]
        contrib = alpha[jb] * (sigma_e * A_e + B_e * sin_e + sigma_e * C_e * cos_e)
        I_out = np.zeros(n_eval, dtype=np.complex128)
        np.add.at(I_out, entry_eval_idx, contrib)
        return I_out

    # ------------------------------------------------------------------
    # Field of elementary current segments (Eqs 76-79)
    # ------------------------------------------------------------------

    def _field_tensor(self, geom, k, src_centers=None, src_tangents=None):
        """Tangential-field tensor Φ of shape (3, N, N) where
        Φ[0, m, n] = ŝ_m · E^const_n(at center of m's surface),
        Φ[1, m, n] = ŝ_m · E^sin_n(at center of m's surface),
        Φ[2, m, n] = ŝ_m · E^cos_n(at center of m's surface).

        The source's local frame is centered on segment n with z-axis
        along n's natural tangent. The "sin"/"cos" sources are
        sin(k·z'_local)/cos(k·z'_local) with z'_local measured from n's
        center along n's natural tangent. σ accounting is the caller's
        job — the tensor is in NATURAL-arc convention.

        `src_centers` / `src_tangents` default to the geometry's segment
        centers and tangents (free-space build). The PEC image build
        passes mirrored versions so the same tensor formula computes
        the image-source field at the original observer points.

        Hot path uses the C++ accelerator `sinusoidal_field_tensor` (the
        70% bottleneck of single-k solves at N≳80); the pure-numpy
        formulation below is kept as a reference / fallback when the
        accelerator isn't available.
        """
        a = self.wire_radius
        seg_c = geom["seg_centers"]  # (N, 3) — observer centers
        seg_t = geom["seg_tangents"]  # (N, 3) — observer tangents
        seg_h = geom["seg_h"]  # (N,) full lengths
        N = geom["n_segs"]
        h_n = 0.5 * seg_h  # (N,) half-lengths

        src_c = src_centers if src_centers is not None else seg_c
        src_t = src_tangents if src_tangents is not None else seg_t

        if _HAVE_FIELD_TENSOR:
            gx, gw = self._leggauss_cached(self.n_qp_const)
            return _acc.sinusoidal_field_tensor(
                np.ascontiguousarray(seg_c, dtype=np.float64),
                np.ascontiguousarray(seg_t, dtype=np.float64),
                np.ascontiguousarray(src_c, dtype=np.float64),
                np.ascontiguousarray(src_t, dtype=np.float64),
                np.ascontiguousarray(seg_h, dtype=np.float64),
                float(a),
                float(k),
                float(self.eta),
                np.ascontiguousarray(gx, dtype=np.float64),
                np.ascontiguousarray(gw, dtype=np.float64),
            )

        # Pairwise vectors c_m - c_n: shape (M=obs, N=src, 3).
        # rvec_mn = seg_c[m] - src_c[n]
        rvec = seg_c[:, None, :] - src_c[None, :, :]  # (M, N, 3)
        t_src = src_t[None, :, :]  # (1, N, 3)
        t_obs = seg_t[:, None, :]  # (M, 1, 3)

        z_eval = np.einsum("mnd,nd->mn", rvec, src_t)  # (M, N)
        # Perpendicular component:
        rho_vec = rvec - z_eval[..., None] * t_src  # (M, N, 3)
        rho_axis = np.linalg.norm(rho_vec, axis=-1)  # (M, N)
        rho_eval = np.sqrt(rho_axis * rho_axis + a * a)  # (M, N), >= a

        # Tangent dot products
        td = np.einsum("mnd,mnd->mn", t_obs * np.ones_like(t_src), t_src)
        # t_obs is shape (M, 1, 3); broadcasting handles the singletons.
        # Re-do cleanly:
        td = (t_obs * t_src).sum(axis=-1)  # (M, N)
        # rho_vec · t_obs at the observer (rho_vec is the perpendicular
        # vector from source axis to obs-axis center)
        rho_dot_tobs = (rho_vec * t_obs).sum(axis=-1)  # (M, N)
        # NEC's prescription: tangential E_ρ component is (ρ·ŝ)/ρ' · E_ρ.
        rho_proj_factor = rho_dot_tobs / rho_eval  # (M, N)

        # Half-length per source segment broadcast to (M, N)
        H = np.broadcast_to(h_n[None, :], (N, N))

        # z values at source ends: z' = +H (z2) and z' = -H (z1).
        # Δz = z_eval - z' at the two endpoints.
        dz2 = z_eval - H  # at z' = +H
        dz1 = z_eval + H  # at z' = -H
        r0_2 = np.sqrt(rho_eval * rho_eval + dz2 * dz2)
        r0_1 = np.sqrt(rho_eval * rho_eval + dz1 * dz1)
        G0_2 = np.exp(-1j * k * r0_2) / r0_2
        G0_1 = np.exp(-1j * k * r0_1) / r0_1

        # Common scalar prefactors. λ = 2π/k → k²λ = 2πk → jη/(2k²λ) = jη/(4πk).
        # Eqs 76-79 carry a "-I_0/λ · jη/(2k²ρ)" (E_ρ) or "I_0/λ · jη/(2k²)"
        # (E_z) prefactor. For unit I_0 = 1, factor pulled out:
        #   E_ρ prefactor = -jη/(4πk·ρ_eval)
        #   E_z prefactor = +jη/(4πk)
        pref_rho = -1j * self.eta / (4.0 * np.pi * k * rho_eval)
        pref_z = 1j * self.eta / (4.0 * np.pi * k)

        # ---- Constant source (I = 1): Eqs 78, 79 ----
        # E_ρ^f = -I/λ · jη/(2k²) · [(1+jkr_0) ρ G_0 / r_0²]_{z1}^{z2}
        #       = pref_rho · ρ_eval · [(1+jkr_0) ρ_eval G_0 / r_0²]_{z1}^{z2}
        #         (pref_rho already has 1/ρ_eval; multiply back by ρ_eval to
        #          recover the form -jη·ρ_eval/(4πk·ρ_eval)= -jη/(4πk))
        # Reorganize for clarity:
        pref_rho_const = -1j * self.eta / (4.0 * np.pi * k)
        Erho_const = pref_rho_const * (
            (1.0 + 1j * k * r0_2) * rho_eval * G0_2 / (r0_2 * r0_2)
            - (1.0 + 1j * k * r0_1) * rho_eval * G0_1 / (r0_1 * r0_1)
        )
        # E_z^f = -I/λ · jη/(2k²) · { [(1+jkr_0)(z-z') G_0 / r_0²]_{z1}^{z2}
        #                              + k² ∫_{z1}^{z2} G_0 dz' }
        # Note sign: -I/λ · jη/(2k²) = -jη/(4πk) for our prefactor convention.
        # We have pref_z = +jη/(4πk); so we use -pref_z.
        # ∫G_0 dz' via singularity extraction. The plain integrand has a
        # 1/r_0 spike at z' = z_eval when ρ_eval is small (self / near-self
        # pairs in thin wires), and Gauss-Legendre with a small node count
        # under-resolves it. Split into closed-form singular + smooth regular:
        #   ∫G_0 dz' = ∫ 1/r_0 dz'              (closed form, arcsinh)
        #            + ∫ (G_0 - 1/r_0) dz'      (regular: tends to -jk as r_0→0)
        # 1/r_0 closed form: ∫_{-H}^{+H} 1/√(ρ²+(z-z')²) dz' = arcsinh((H-z)/ρ) - arcsinh((-H-z)/ρ).
        u2 = (H - z_eval) / rho_eval  # (M, N)
        u1 = (-H - z_eval) / rho_eval
        int_inv_r0 = np.arcsinh(u2) - np.arcsinh(u1)
        gx, gw = self._leggauss_cached(self.n_qp_const)
        z_qp = H[..., None] * gx[None, None, :]  # (M, N, n_qp)
        dz_qp = z_eval[..., None] - z_qp
        r0_qp = np.sqrt(rho_eval[..., None] ** 2 + dz_qp**2)
        G0_qp = np.exp(-1j * k * r0_qp) / r0_qp
        reg_qp = G0_qp - 1.0 / r0_qp
        int_reg = np.einsum("mnq,q->mn", reg_qp, gw) * H
        int_G0 = int_inv_r0 + int_reg  # (M, N)
        Ez_const_boundary = (1.0 + 1j * k * r0_2) * dz2 * G0_2 / (r0_2 * r0_2) - (
            1.0 + 1j * k * r0_1
        ) * dz1 * G0_1 / (r0_1 * r0_1)
        Ez_const = -pref_z * (Ez_const_boundary + k * k * int_G0)

        # ---- Sine source (I = sin(k·z'_local)): Eq 76, Eq 77 ----
        # Eq 71 (sinusoidal general) factored: E_ρ^f = pref_rho · G_0 ·
        # {k(z-z') cos(kz') + [1 - (z-z')²(1+jkr_0)/r_0²] sin(kz')} |_{z1}^{z2}
        # — note G_0 is an overall factor on the WHOLE bracket, evaluated
        # at each endpoint with its own r_0.
        sin2 = np.sin(k * H)  # sin(k · z'_2) where z'_2 = +H, so sin(kH)
        cos2 = np.cos(k * H)
        sin1 = np.sin(-k * H)  # at z'_1 = -H
        cos1 = np.cos(-k * H)
        bracket_sin_2 = G0_2 * (
            k * dz2 * cos2
            + (1.0 - dz2 * dz2 * (1.0 + 1j * k * r0_2) / (r0_2 * r0_2)) * sin2
        )
        bracket_sin_1 = G0_1 * (
            k * dz1 * cos1
            + (1.0 - dz1 * dz1 * (1.0 + 1j * k * r0_1) / (r0_1 * r0_1)) * sin1
        )
        Erho_sin = pref_rho * (bracket_sin_2 - bracket_sin_1)
        # E_z^f = pref_z · G_0 · {k cos(kz') - (1+jkr_0)(z-z')/r_0² sin(kz')}_{z1}^{z2}
        bracket_sin_z_2 = G0_2 * (
            k * cos2 - (1.0 + 1j * k * r0_2) * dz2 / (r0_2 * r0_2) * sin2
        )
        bracket_sin_z_1 = G0_1 * (
            k * cos1 - (1.0 + 1j * k * r0_1) * dz1 / (r0_1 * r0_1) * sin1
        )
        Ez_sin = pref_z * (bracket_sin_z_2 - bracket_sin_z_1)

        # ---- Cosine source (I = cos(k·z'_local)): same as Eqs 76, 77 with
        #      the "(cos kz'/-sin kz')" toggle picking the lower row, i.e.
        #      swap sin↔cos and negate the (sin-row → -sin) term.
        bracket_cos_2 = G0_2 * (
            -k * dz2 * sin2
            + (1.0 - dz2 * dz2 * (1.0 + 1j * k * r0_2) / (r0_2 * r0_2)) * cos2
        )
        bracket_cos_1 = G0_1 * (
            -k * dz1 * sin1
            + (1.0 - dz1 * dz1 * (1.0 + 1j * k * r0_1) / (r0_1 * r0_1)) * cos1
        )
        Erho_cos = pref_rho * (bracket_cos_2 - bracket_cos_1)
        bracket_cos_z_2 = G0_2 * (
            -k * sin2 - (1.0 + 1j * k * r0_2) * dz2 / (r0_2 * r0_2) * cos2
        )
        bracket_cos_z_1 = G0_1 * (
            -k * sin1 - (1.0 + 1j * k * r0_1) * dz1 / (r0_1 * r0_1) * cos1
        )
        Ez_cos = pref_z * (bracket_cos_z_2 - bracket_cos_z_1)

        # Project tangentially to obs segment: E_t = td · E_z + rho_proj · E_ρ
        Phi_const = td * Ez_const + rho_proj_factor * Erho_const
        Phi_sin = td * Ez_sin + rho_proj_factor * Erho_sin
        Phi_cos = td * Ez_cos + rho_proj_factor * Erho_cos
        return Phi_const, Phi_sin, Phi_cos

    # ------------------------------------------------------------------
    # Matrix assembly and solve
    # ------------------------------------------------------------------

    def _image_source_centers_tangents(self, geom):
        """Mirror source segments across z = ground_z and flip their tangent
        z-components, mirroring the convention TriangularPySim uses for the
        PEC image build. Same shape ((N, 3), (N, 3)) as the originals.
        """
        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        src_c_img = seg_c * np.array([1.0, 1.0, -1.0]) + np.array(
            [0.0, 0.0, 2.0 * self.ground_z]
        )
        src_t_img = seg_t * np.array([1.0, 1.0, -1.0])
        return src_c_img, src_t_img

    def _field_tensor_image(self, geom, k):
        """Field tensor for image sources at PEC ground. The image keeps the
        same per-segment half-length and basis shape; only the source center
        is mirrored and the source tangent z-component is flipped.
        """
        src_c_img, src_t_img = self._image_source_centers_tangents(geom)
        return self._field_tensor(
            geom, k, src_centers=src_c_img, src_tangents=src_t_img
        )

    def _assemble_Z(self, geom, k):
        Phi_c, Phi_s, Phi_co = self._field_tensor(geom, k)
        if self.ground_z is not None:
            # PEC image: subtract the sub-assembly built from the image
            # field tensor. The image source's mirrored geometry + flipped
            # z-tangent already encode both the anti-parallel horizontal
            # image current and the parallel vertical image current; the
            # combined image-current + image-charge sign flip reduces to
            # a single minus sign on the image-Z block (same as Triangular).
            Phi_c_i, Phi_s_i, Phi_co_i = self._field_tensor_image(geom, k)
            Phi_c = Phi_c - Phi_c_i
            Phi_s = Phi_s - Phi_s_i
            Phi_co = Phi_co - Phi_co_i

        seg_view = self._basis_coefs(geom, k)
        N = geom["n_segs"]
        # Build (N, N) coefficient matrices M_{A,B,C}[n, j] = effective
        # coefficient that basis j contributes at source segment n. With
        #   A_eff = σ * A, B_eff = B, C_eff = σ * C
        # (see docs/sinusoidal_basis_design.md), the per-basis loop reduces
        # to three N×N matmuls G = Phi_c @ M_A + Phi_s @ M_B + Phi_co @ M_C.
        #
        # seg_view is already CSR-by-segment, so the row coordinate n_idx
        # is just np.repeat(arange(N), counts) and σA / σC are element-wise
        # numpy products. No Python scatter loop here — the per-basis fill
        # already happened directly into seg_view inside _basis_coefs.
        #
        # Each basis has only ~3 entries (self + N⁻ + N⁺ neighbour), so M
        # is very sparse (~3N nonzeros). Two regimes:
        #   N < _DENSE_ASSEMBLY_THRESHOLD: dense matmul wins because the
        #     scipy.sparse constructor overhead dominates the BLAS call.
        #   N ≥ threshold: sparse matmul wins because BLAS zgemm pays the
        #     full O(N³) cost on a mostly-zero matrix, while CSC matmul
        #     pays O(N · 3N) = O(N²).
        # Crossover measured ≈ N=60 on Kaby Lake R / OpenBLAS-pthreads.
        starts = seg_view["starts"]
        n_idx_arr = np.repeat(np.arange(N, dtype=np.int64), starts[1:] - starts[:-1])
        j_idx_arr = seg_view["jbasis"]
        sigma_arr = seg_view["sigma"]
        A_eff = sigma_arr * seg_view["A"]
        B_eff = seg_view["B"]
        C_eff = sigma_arr * seg_view["C"]
        if N < _DENSE_ASSEMBLY_THRESHOLD:
            M_A = np.zeros((N, N), dtype=np.complex128)
            M_B = np.zeros((N, N), dtype=np.complex128)
            M_C = np.zeros((N, N), dtype=np.complex128)
            M_A[n_idx_arr, j_idx_arr] = A_eff
            M_B[n_idx_arr, j_idx_arr] = B_eff
            M_C[n_idx_arr, j_idx_arr] = C_eff
            G = Phi_c @ M_A + Phi_s @ M_B + Phi_co @ M_C
        else:
            M_A = scipy.sparse.csc_matrix((A_eff, (n_idx_arr, j_idx_arr)), shape=(N, N))
            M_B = scipy.sparse.csc_matrix((B_eff, (n_idx_arr, j_idx_arr)), shape=(N, N))
            M_C = scipy.sparse.csc_matrix((C_eff, (n_idx_arr, j_idx_arr)), shape=(N, N))
            G = (Phi_c @ M_A) + (Phi_s @ M_B) + (Phi_co @ M_C)
        return G, seg_view

    def compute_impedance(self):
        """Return (Z_drive, alpha) where alpha[j] is the amplitude of
        basis j and Z_drive = V_applied / I(feed-center) using V = 1 V.
        """
        geom = self._build_geometry()
        G, seg_view = self._assemble_Z(geom, self.k)
        feed = geom["feed_seg"]
        h_feed = geom["seg_h"][feed]
        # Eq 187 applied-E source: E_feed = V/Δ_feed, zero elsewhere.
        # EFIE boundary condition: ŝ · E^scat = -ŝ · E^applied, with G
        # filled as +ŝ · E^scat-due-to-unit-basis-amplitude → solve
        # G·α = -E_applied.  For V = 1 the RHS magnitude is -1/Δ_feed.
        v = np.zeros(geom["n_segs"], dtype=np.complex128)
        v[feed] = -1.0 / h_feed
        alpha = scipy.linalg.solve(G, v)

        # Current at centre of feed segment: I(s_local=0) = Σ_j α_j · σ_j ·
        # (A_jn + C_jn) summed over bases j whose support includes `feed`
        # (sin(k·0)=0 so B drops out). Pre-indexed in seg_view: every basis
        # entry whose `n_seg == feed` lives in the slice
        # seg_view[feed : feed+1]. Single numpy reduction replaces the
        # original double-Python-loop that filtered the full basis list.
        s = seg_view["starts"][feed]
        e = seg_view["starts"][feed + 1]
        I_feed = complex(
            (
                alpha[seg_view["jbasis"][s:e]]
                * seg_view["sigma"][s:e]
                * (seg_view["A"][s:e] + seg_view["C"][s:e])
            ).sum()
        )
        Z_drive = 1.0 / I_feed
        self.Z_matrix = G
        return Z_drive, alpha

    def compute_impedance_swept(self, k_array):
        """Loop over wavenumbers. Per-call work that doesn't depend on k
        (geometry build, source-vector index, the set of bases that touch
        the feed segment) is lifted out of the loop so the per-k cost
        reduces to field-tensor + basis-coefs + assembly + solve. Together
        with the assemble_Z vectorization and the C++ field-tensor
        accelerator, this brings the n=21 sweep from ~70 ms to ~30 ms.
        """
        k_array = np.asarray(k_array, dtype=float)
        z_out = np.zeros(k_array.shape[0], dtype=np.complex128)
        k_save = self.k
        wl_save = self.wavelength
        omega_save = self.omega
        geom = self._build_geometry()
        feed = geom["feed_seg"]
        h_feed = geom["seg_h"][feed]
        n_segs = geom["n_segs"]
        v = np.zeros(n_segs, dtype=np.complex128)
        v[feed] = -1.0 / h_feed
        for i, kk in enumerate(k_array):
            self.k = float(kk)
            self.omega = self.k * self.c
            self.wavelength = self.c / (self.omega / (2 * np.pi))
            G, seg_view = self._assemble_Z(geom, self.k)
            alpha = scipy.linalg.solve(G, v)
            s = seg_view["starts"][feed]
            e = seg_view["starts"][feed + 1]
            I_feed = complex(
                (
                    alpha[seg_view["jbasis"][s:e]]
                    * seg_view["sigma"][s:e]
                    * (seg_view["A"][s:e] + seg_view["C"][s:e])
                ).sum()
            )
            z_out[i] = 1.0 / I_feed
        self.k = k_save
        self.wavelength = wl_save
        self.omega = omega_save
        return z_out

    def currents_at_knots(self, alpha, s_array=None):
        """Per-wire complex current sampled at every mesh knot.

        Each basis j contributes (A_jn + B_jn sin(k·s_local) +
        C_jn cos(k·s_local))·σ_jn on every segment n in its support, with
        s_local measured from segment n's centre. The current at a knot
        between adjacent segments is the average of the right-edge value
        of the segment to its left and the left-edge value of the segment
        to its right (continuity makes them equal up to round-off; the
        average is the symmetric pick). Wire-endpoint knots use only the
        adjacent segment.

        When `s_array` is provided as a list of 1D arc-length arrays (one per
        wire), evaluates the basis sum at those arc positions instead of the
        mesh knots. Arc is measured from the wire's start (s=0) to its end
        (s=Σ h_seg). Samples that fall exactly on an interior knot return the
        symmetric average of the two adjacent segments (same as the default
        knot path); samples in the interior of a segment evaluate the basis
        directly on that segment.
        """
        alpha = np.asarray(alpha)
        geom = self._build_geometry()
        seg_view = self._basis_coefs(geom, self.k)
        seg_h = geom["seg_h"]
        n_wires = len(self.wires_polylines)

        # Pattern shared between both paths: each evaluation produces a
        # (segment, s_local) pair, a destination index in the flat output,
        # and a weight (1.0 for endpoints / interior samples, 0.5 each side
        # for interior-knot symmetric-average pairs). We batch them all up,
        # call `_evaluate_basis_at_points` once, then scatter-add into the
        # output. In the natural-tangent frame I = σA + B·sin(ks) + σC·cos(ks);
        # this lives inside `_evaluate_basis_at_points`. The earlier `σ` bug
        # (fixed 2026-06: multiplying the whole bracket by σ added a spurious
        # 2·B·sin(ks) at σ=−1 junction neighbours) is gone by construction.

        if s_array is None:
            # Per wire of M segs: M+1 knots. Each segment contributes two
            # edge evaluations — its left edge to the adjacent left knot,
            # its right edge to the adjacent right knot. The first seg's
            # left-edge and the last seg's right-edge hit wire endpoints
            # (weight 1.0); every other edge hits an interior knot shared
            # with another segment's edge, so each contributes weight 0.5.
            eval_seg_parts = []
            eval_s_parts = []
            eval_target_parts = []
            eval_weight_parts = []
            wire_knot_offsets = [0]
            for w_idx in range(n_wires):
                first = geom["wire_first"][w_idx]
                last = geom["wire_last"][w_idx]
                n_w_segs = last - first + 1
                h_w = seg_h[first : last + 1]
                base = wire_knot_offsets[-1]
                seg_arr = np.arange(first, last + 1, dtype=np.int64)
                # Left edges (s = -h/2) feed knot 0..M-1
                left_target = base + np.arange(n_w_segs, dtype=np.int64)
                left_weight = np.full(n_w_segs, 0.5)
                left_weight[0] = 1.0
                # Right edges (s = +h/2) feed knot 1..M
                right_target = base + np.arange(1, n_w_segs + 1, dtype=np.int64)
                right_weight = np.full(n_w_segs, 0.5)
                right_weight[-1] = 1.0
                eval_seg_parts.append(np.concatenate([seg_arr, seg_arr]))
                eval_s_parts.append(np.concatenate([-0.5 * h_w, +0.5 * h_w]))
                eval_target_parts.append(np.concatenate([left_target, right_target]))
                eval_weight_parts.append(np.concatenate([left_weight, right_weight]))
                wire_knot_offsets.append(base + n_w_segs + 1)

            eval_seg = np.concatenate(eval_seg_parts)
            eval_s = np.concatenate(eval_s_parts)
            eval_target = np.concatenate(eval_target_parts)
            eval_weight = np.concatenate(eval_weight_parts)

            I_evals = self._evaluate_basis_at_points(seg_view, eval_seg, eval_s, alpha)
            I_flat = np.zeros(wire_knot_offsets[-1], dtype=np.complex128)
            np.add.at(I_flat, eval_target, eval_weight * I_evals)
            return [
                I_flat[wire_knot_offsets[i] : wire_knot_offsets[i + 1]]
                for i in range(n_wires)
            ]

        # s_array path: per-wire, vectorized over the sample positions.
        # Each sample is either right on an interior knot (→ two evals,
        # weight 0.5 each, for the continuity-average) or interior to a
        # segment (→ one eval, weight 1.0).
        sampled = []
        for w_idx, sv in enumerate(s_array):
            sv = np.asarray(sv, dtype=np.float64)
            first = geom["wire_first"][w_idx]
            last = geom["wire_last"][w_idx]
            n_w_segs = last - first + 1
            wire_h = seg_h[first : last + 1]
            arc_at_knot = np.concatenate([[0.0], np.cumsum(wire_h)])
            wire_arc = float(arc_at_knot[-1])
            n_s = sv.shape[0]
            if n_s == 0:
                sampled.append(np.zeros(0, dtype=np.complex128))
                continue
            s_clip = np.clip(sv, 0.0, wire_arc)
            eps = 1e-12 * max(wire_arc, 1.0)
            knot_hit = np.searchsorted(arc_at_knot, s_clip)
            on_interior_knot = (
                (knot_hit > 0)
                & (knot_hit < n_w_segs)
                & (np.abs(s_clip - arc_at_knot[np.clip(knot_hit, 0, n_w_segs)]) <= eps)
            )
            non_knot_mask = ~on_interior_knot

            # Containing-segment evaluation for non-knot samples.
            seg_in_wire = np.searchsorted(arc_at_knot, s_clip, side="right") - 1
            seg_in_wire = np.clip(seg_in_wire, 0, n_w_segs - 1)
            nk_target = np.nonzero(non_knot_mask)[0]
            nk_seg_in_wire = seg_in_wire[non_knot_mask]
            nk_seg = first + nk_seg_in_wire
            nk_s = (s_clip[non_knot_mask] - arc_at_knot[nk_seg_in_wire]) - 0.5 * wire_h[
                nk_seg_in_wire
            ]
            nk_weight = np.ones(nk_seg.shape[0])

            # Two-sided evaluation for samples on an interior knot.
            k_target = np.nonzero(on_interior_knot)[0]
            k_idx = knot_hit[on_interior_knot]
            k_left_seg = first + k_idx - 1
            k_right_seg = first + k_idx
            k_left_s = +0.5 * seg_h[k_left_seg]
            k_right_s = -0.5 * seg_h[k_right_seg]
            k_weight = np.full(2 * k_target.shape[0], 0.5)

            all_seg = np.concatenate([nk_seg, k_left_seg, k_right_seg])
            all_s = np.concatenate([nk_s, k_left_s, k_right_s])
            all_target = np.concatenate([nk_target, k_target, k_target])
            all_weight = np.concatenate([nk_weight, k_weight])

            I_evals = self._evaluate_basis_at_points(seg_view, all_seg, all_s, alpha)
            I_out = np.zeros(n_s, dtype=np.complex128)
            np.add.at(I_out, all_target, all_weight * I_evals)
            sampled.append(I_out)
        return sampled
