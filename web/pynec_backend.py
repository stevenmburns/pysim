"""PyNEC drop-in backend for the web UI.

Mirrors the response shape of `web.server`'s pysim solver paths so the
frontend can swap between solvers via a `solver` field on every request.

PyNEC is optional: `HAVE_PYNEC` is False if the import fails, and the
server falls back to pysim with a one-time warning.
"""

from __future__ import annotations

import math
import time

import numpy as np

try:
    import PyNEC as nec  # type: ignore

    HAVE_PYNEC = True
except ImportError:
    HAVE_PYNEC = False
    nec = None

from .examples import REGISTRY as EXAMPLES


C_LIGHT = 299_792_458.0

# Typical "average" earth, matching antenna_designer/sim.py.
GROUND_DIELECTRIC = 10.0
GROUND_CONDUCTIVITY = 0.002


def _segment_centers_to_knot_currents(
    cur_per_seg: np.ndarray,
    n_knots: int,
    junction_at_start: bool = False,
    junction_at_end: bool = False,
) -> np.ndarray:
    """Map NEC's per-segment-center currents onto the (n_knots,)-knot array
    the UI expects.

    Interior knot k sits between segments k-1 and k, so we average. The
    boundary knots default to zero (open-wire BC), but at a junction with
    another wire the current is continuous through the endpoint — pass
    junction_at_start/end=True to carry the adjacent segment-center value
    onto the boundary knot instead.
    """
    full = np.zeros(n_knots, dtype=np.complex128)
    if cur_per_seg.shape[0] != n_knots - 1:
        raise RuntimeError(
            f"segment-current length {cur_per_seg.shape[0]} doesn't match "
            f"n_knots-1 = {n_knots - 1}"
        )
    full[1:-1] = 0.5 * (cur_per_seg[:-1] + cur_per_seg[1:])
    if junction_at_start:
        full[0] = cur_per_seg[0]
    if junction_at_end:
        full[-1] = cur_per_seg[-1]
    return full


def _run_solve(
    c,
    n_seg_total: int,
    feed_seg: int,
    freq_mhz: float,
    ground: bool = False,
    feed_tag: int = 1,
    ground_fast: bool = False,
):
    if ground:
        # ITYPE=2: Sommerfeld-Norton, full kernel integration — accurate for
        # antennas a fraction of a wavelength above ground, ~100x slower per
        # solve than free space.
        # ITYPE=0: reflection-coefficient approximation — applies a Fresnel
        # reflection on the image current rather than evaluating Sommerfeld
        # integrals. Cheap (~10x faster than ITYPE=2) and accurate enough
        # away from grazing angles; degrades for very low antennas.
        itype = 0 if ground_fast else 2
        c.gn_card(itype, 0, GROUND_DIELECTRIC, GROUND_CONDUCTIVITY, 0, 0, 0, 0)
    else:
        c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)  # free space
    c.ex_card(0, feed_tag, feed_seg, 0, 1.0, 0.0, 0, 0, 0, 0)
    c.fr_card(0, 1, freq_mhz, 0)
    c.xq_card(0)
    sc = c.get_structure_currents(0)
    cur_arr = np.asarray(sc.get_current(), dtype=np.complex128)
    tag_arr = np.asarray(sc.get_current_segment_tag())
    return cur_arr, tag_arr




def _polyline_knots(path, npe_list) -> np.ndarray:
    """Concatenated per-edge knot positions, deduping shared corners."""
    parts = []
    for i, n_e in enumerate(npe_list):
        seg = np.linspace(path[i], path[i + 1], n_e + 1)
        parts.append(seg if i == 0 else seg[1:])
    return np.vstack(parts)




_HENTENNA_FEED_GAP = 0.05  # meters; half-gap (eps) between feed knots T and S


def _build_hentenna(req: dict):
    """Build the PyNEC context + geometry for a single-band hentenna.

    Five logical wires (9 NEC wire cards total) following the pysim layout:
      tag 1: feed gap T->S                    (3 segs, EX on middle seg)
      tag 2: cross-bar right S->B             (n_per_wire)
      tags 3..5: upper rectangle B->A->C->D   (n_per_wire each)
      tag 6: cross-bar left T->D              (n_per_wire)
      tags 7..9: lower rectangle D->E->F->B   (n_per_wire each)
    NEC connects wire ends by coordinate match, so the K=2 (S,T) and K=3
    (B,D) junctions are implicit.
    """
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    width_factor = float(req.get("width_factor", 0.1378))
    top_height_factor = float(req.get("top_height_factor", 0.5081))
    mid_height_factor = float(req.get("mid_height_factor", 0.1094))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    ground_fast = bool(req.get("ground_fast", False))
    height_m = float(req.get("height_m", 0.0))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    half_w = wavelength_design * width_factor / 2.0
    eps_feed = _HENTENNA_FEED_GAP
    z_offset = height_m if ground else 0.0
    z_mid = wavelength_design * (mid_height_factor - top_height_factor) + z_offset
    z_top = z_offset
    z_bot = -wavelength_design * top_height_factor + z_offset

    A = (0.0, half_w, z_top)
    B = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    Cc = (0.0, -half_w, z_top)
    D = (0.0, -half_w, z_mid)
    E = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)

    # NEC2's pulse basis puts the source as a delta-gap at a segment
    # *centre*, so an ODD count is required to place the source at the
    # gap's geometric centre. Even n_feed offsets the source by half a
    # segment length, biasing the impedance by ~1 Ω on R.
    nf = max(1, n_per_wire // 7)
    n_feed = nf if nf % 2 == 1 else nf + 1
    feed_seg = (n_feed + 1) // 2  # middle segment, 1-indexed in NEC

    feed_path = [T, S]
    cross_r_path = [S, B]
    upper_path = [B, A, Cc, D]
    cross_l_path = [T, D]
    lower_path = [D, E, F, B]

    npe_feed = [n_feed]
    npe_cross_r = [n_per_wire]
    npe_upper = [n_per_wire, n_per_wire, n_per_wire]
    npe_cross_l = [n_per_wire]
    npe_lower = [n_per_wire, n_per_wire, n_per_wire]

    c = nec.nec_context()
    geo = c.get_geometry()
    # Tag 1: feed gap T->S
    geo.wire(1, n_feed, *T, *S, wire_radius, 1.0, 1.0)
    # Tag 2: cross-bar right S->B
    geo.wire(2, n_per_wire, *S, *B, wire_radius, 1.0, 1.0)
    # Tags 3..5: upper rectangle B->A->C->D
    for i, (p0, p1) in enumerate(zip(upper_path[:-1], upper_path[1:])):
        geo.wire(3 + i, n_per_wire, *p0, *p1, wire_radius, 1.0, 1.0)
    # Tag 6: cross-bar left T->D
    geo.wire(6, n_per_wire, *T, *D, wire_radius, 1.0, 1.0)
    # Tags 7..9: lower rectangle D->E->F->B
    for i, (p0, p1) in enumerate(zip(lower_path[:-1], lower_path[1:])):
        geo.wire(7 + i, n_per_wire, *p0, *p1, wire_radius, 1.0, 1.0)
    c.geometry_complete(0)

    return {
        "context": c,
        "feed_tag": 1,
        "feed_seg": feed_seg,
        "n_per_wire": n_per_wire,
        "n_feed": n_feed,
        "wires_meta": [
            ("feed", feed_path, npe_feed, [(1,)]),
            ("cross_right", cross_r_path, npe_cross_r, [(2,)]),
            ("upper", upper_path, npe_upper, [(3,), (4,), (5,)]),
            ("cross_left", cross_l_path, npe_cross_l, [(6,)]),
            ("lower", lower_path, npe_lower, [(7,), (8,), (9,)]),
        ],
        "wavelength_design": wavelength_design,
        "design_freq_mhz": design_freq_mhz,
        "half_width_m": half_w,
        "top_height_m": wavelength_design * top_height_factor,
        "mid_offset_m": z_mid - z_top,
        "ground": ground,
        "ground_fast": ground_fast,
        "z_offset": z_offset,
    }


def solve_hentenna(req: dict) -> dict:
    """Single-band hentenna via PyNEC."""
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 28.47))
    )
    b = _build_hentenna(req)
    c = b["context"]

    n_seg_total = b["n_feed"] + 7 * b["n_per_wire"]

    t0_clock = time.perf_counter()
    cur_arr, tag_arr = _run_solve(
        c,
        n_seg_total,
        b["feed_seg"],
        meas_freq_mhz,
        ground=b["ground"],
        feed_tag=b["feed_tag"],
        ground_fast=b["ground_fast"],
    )
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    feed_idx_in_tag = np.where(tag_arr == b["feed_tag"])[0]
    fed_global = feed_idx_in_tag[b["feed_seg"] - 1]
    z_in = complex(1.0 / cur_arr[fed_global])

    # Every wire end in this topology is at a junction (K=2 or K=3), so
    # current is continuous at all boundary knots — carry the adjacent
    # segment-center value onto the boundary instead of zeroing it.
    wire_records = []
    for label, path, npe, tag_groups in b["wires_meta"]:
        knots = _polyline_knots(path, npe)
        # Concatenate per-segment currents in path order across this wire's
        # NEC tags. Each tag_group is a (tag,) tuple — one NEC card per edge.
        cur_wire = np.concatenate(
            [cur_arr[np.where(tag_arr == tg[0])[0]] for tg in tag_groups]
        )
        knot_cur = _segment_centers_to_knot_currents(
            cur_wire,
            knots.shape[0],
            junction_at_start=True,
            junction_at_end=True,
        )
        wire_records.append(
            {
                "label": label,
                "knot_positions": knots.tolist(),
                "knot_currents_re": knot_cur.real.tolist(),
                "knot_currents_im": knot_cur.imag.tolist(),
            }
        )

    # feed_knot_index on wire 0: the knot immediately after the fed segment.
    # With n_feed segments the fed segment is feed_seg (1-indexed), so the
    # right-side knot is feed_seg (0-indexed) — interior so long as n_feed >= 2.
    feed_knot_index = b["feed_seg"]

    return {
        "geometry": "hentenna",
        "wires": wire_records,
        "feed_wire_index": 0,
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": b["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": b["wavelength_design"],
        "half_width_m": b["half_width_m"],
        "top_height_m": b["top_height_m"],
        "mid_offset_m": b["mid_offset_m"],
        "solve_ms": solve_ms,
        "solver": "pynec",
        "ground": b["ground"],
        "ground_fast": b["ground_fast"],
        "height_m": b["z_offset"],
        "ground_eps_r": GROUND_DIELECTRIC,
        "ground_sigma": GROUND_CONDUCTIVITY,
    }


_BOWTIE_EPS = 0.05  # meters; half-gap at the feed and at the top pinch,
# matches the pysim/web/server.py side so the two backends solve the same
# geometry on the bowtie 1×2 array.


def _bowtie_element_polylines(
    slope: float, length: float, y_offset: float, z_offset: float
):
    """Return the 4 polylines of one bowtie element in the y-z plane (x=0)
    at the given y_offset / z_offset. Same shapes as the pysim helper.
    """
    eps_b = _BOWTIE_EPS
    y = 0.5 * length / (slope + math.sqrt(1.0 + slope * slope))
    z = slope * y

    def P(yy, zz):
        return (0.0, yy + y_offset, zz + z_offset)

    w0 = [P(-y, 0), P(-y, z), P(-eps_b, eps_b), P(eps_b, eps_b), P(y, z), P(y, 0)]
    w1 = [P(-y, 0), P(-y, -z), P(-eps_b, -eps_b)]
    w2 = [P(eps_b, -eps_b), P(y, -z), P(y, 0)]
    w3 = [P(-eps_b, -eps_b), P(eps_b, -eps_b)]
    return w0, w1, w2, w3, y, z


def _build_bowtie(req: dict):
    """Build the PyNEC context + geometry for the bowtie 1×2 phased array.

    Two bowtie elements at y = ±del_y_m. Per element: 10 NEC wire cards
    (one per polyline edge — 5 + 2 + 2 + 1), tags grouped per polyline so
    the per-polyline current concatenation matches the pysim wire shape
    the frontend already renders. Two EX cards apply the multi-feed
    voltage drive: V_0 = 1+0j on element 0's feed wire, V_1 =
    exp(j·π·phase_lr_deg/180) on element 1's feed wire.

    Returns the same per-element metadata for both elements so
    solve_bowtie can pack wires + feeds for the response.
    """
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    slope = float(req.get("slope", 0.5376))
    length_factor = float(req.get("length_factor", 0.515))
    del_y_m = float(req.get("del_y_m", 4.0))
    phase_lr_deg = float(req.get("phase_lr_deg", 0.0))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    ground_fast = bool(req.get("ground_fast", False))
    height_m = float(req.get("height_m", 0.0))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    length_m = length_factor * wavelength_design
    z_offset = height_m if ground else 0.0

    # Segment counts: long edges get n_per_wire; short eps-scale edges (top
    # pinch and feed gap) get a small odd count so the NEC EX card lands
    # on the segment whose centre is at the gap's geometric centre.
    n_long = max(2, n_per_wire)
    n_short_even = max(2, n_long // 7)
    n_short_odd = n_short_even if n_short_even % 2 == 1 else n_short_even + 1
    # `n_short_odd` is odd so the feed-gap card has a single centre segment;
    # the top-pinch card also uses an odd count for visual symmetry.
    feed_seg_local = (n_short_odd + 1) // 2  # 1-indexed middle of feed wire

    c = nec.nec_context()
    geo = c.get_geometry()

    elements = []
    tag = 0
    for elem_idx, (y_off, v_complex) in enumerate(
        [
            (-del_y_m, complex(1.0, 0.0)),
            (
                +del_y_m,
                complex(
                    math.cos(math.pi * phase_lr_deg / 180.0),
                    math.sin(math.pi * phase_lr_deg / 180.0),
                ),
            ),
        ]
    ):
        w0, w1, w2, w3, y, z = _bowtie_element_polylines(
            slope, length_m, y_off, z_offset
        )

        # Per-polyline segment counts and tag groupings. Each polyline
        # produces one NEC wire card per edge — same tag for each edge of
        # the polyline isn't possible (NEC tags are per-card), so we group
        # by polyline in `tag_groups` and concatenate currents at response
        # time.
        polyline_specs = [
            ("top_arc", w0, [n_long, n_long, n_short_odd, n_long, n_long]),
            ("bot_left_arc", w1, [n_long, n_long]),
            ("bot_right_arc", w2, [n_long, n_long]),
            ("feed", w3, [n_short_odd]),
        ]
        polyline_meta = []
        feed_tag_for_elem: int | None = None
        for label, path, npe in polyline_specs:
            tag_group = []
            for i in range(len(npe)):
                tag += 1
                p0, p1 = path[i], path[i + 1]
                geo.wire(tag, npe[i], *p0, *p1, wire_radius, 1.0, 1.0)
                tag_group.append(tag)
            polyline_meta.append((label, path, npe, tag_group))
            if label == "feed":
                feed_tag_for_elem = tag_group[0]  # single-edge polyline

        elements.append(
            {
                "y_offset": y_off,
                "v_complex": v_complex,
                "polylines": polyline_meta,
                "feed_tag": feed_tag_for_elem,
                "feed_seg": feed_seg_local,
                "y_m": y,
                "z_m": z,
            }
        )
    c.geometry_complete(0)

    n_seg_total = tag  # each tag carries its own segments — sum below
    n_seg_total = sum(
        npe_i
        for elem in elements
        for (_label, _path, npe, _tg) in elem["polylines"]
        for npe_i in npe
    )

    return {
        "context": c,
        "elements": elements,
        "n_per_wire": n_per_wire,
        "n_seg_total": n_seg_total,
        "wavelength_design": wavelength_design,
        "design_freq_mhz": design_freq_mhz,
        "length_m": length_m,
        "del_y_m": del_y_m,
        "phase_lr_deg": phase_lr_deg,
        "slope": slope,
        "ground": ground,
        "ground_fast": ground_fast,
        "z_offset": z_offset,
    }


def _run_solve_bowtie(
    c,
    n_seg_total: int,
    elements: list[dict],
    freq_mhz: float,
    ground: bool,
    ground_fast: bool,
):
    """Multi-feed NEC2 solve. Each element gets one EX card at its feed
    tag's centre segment with its prescribed complex voltage. NEC
    superposes the excitations; the returned current vector reflects the
    combined drive.
    """
    if ground:
        itype = 0 if ground_fast else 2
        c.gn_card(itype, 0, GROUND_DIELECTRIC, GROUND_CONDUCTIVITY, 0, 0, 0, 0)
    else:
        c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)
    for elem in elements:
        v = elem["v_complex"]
        c.ex_card(0, elem["feed_tag"], elem["feed_seg"], 0, v.real, v.imag, 0, 0, 0, 0)
    c.fr_card(0, 1, freq_mhz, 0)
    c.xq_card(0)
    sc = c.get_structure_currents(0)
    cur_arr = np.asarray(sc.get_current(), dtype=np.complex128)
    tag_arr = np.asarray(sc.get_current_segment_tag())
    return cur_arr, tag_arr


def _bowtie_per_element_wires(
    elem: dict, cur_arr: np.ndarray, tag_arr: np.ndarray, half_label: str
) -> list[dict]:
    """Pack one element's polylines into wire records the frontend renders.
    Concatenates segment currents across each polyline's tag group and maps
    them onto knot positions; every bowtie polyline endpoint is at a K=2
    junction so junction_at_start/end are True throughout.
    """
    out = []
    for label, path, npe, tag_group in elem["polylines"]:
        knots = _polyline_knots(path, npe)
        cur_wire = np.concatenate(
            [cur_arr[np.where(tag_arr == tg)[0]] for tg in tag_group]
        )
        knot_cur = _segment_centers_to_knot_currents(
            cur_wire,
            knots.shape[0],
            junction_at_start=True,
            junction_at_end=True,
        )
        out.append(
            {
                "label": f"{half_label}_{label}",
                "knot_positions": knots.tolist(),
                "knot_currents_re": knot_cur.real.tolist(),
                "knot_currents_im": knot_cur.imag.tolist(),
            }
        )
    return out


def solve_bowtie(req: dict) -> dict:
    """Bowtie 1×2 phased array via PyNEC."""
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 28.47))
    )
    b = _build_bowtie(req)

    t0 = time.perf_counter()
    cur_arr, tag_arr = _run_solve_bowtie(
        b["context"],
        b["n_seg_total"],
        b["elements"],
        meas_freq_mhz,
        ground=b["ground"],
        ground_fast=b["ground_fast"],
    )
    solve_ms = (time.perf_counter() - t0) * 1e3

    wire_records: list[dict] = []
    feeds: list[dict] = []
    for elem_idx, (elem, half_label) in enumerate(zip(b["elements"], ("L", "R"))):
        wires_before = len(wire_records)
        wire_records.extend(
            _bowtie_per_element_wires(elem, cur_arr, tag_arr, half_label)
        )
        # Feed wire is always the 4th polyline (index 3) of each element.
        feed_wire_global = wires_before + 3
        # Single-edge polyline with n_short_odd segments → middle segment
        # carries the EX source; its right-side knot is the feed marker
        # (index = feed_seg, since knot k sits between segments k-1 and k).
        feed_seg = elem["feed_seg"]
        feed_knot_index = feed_seg
        # Per-feed current at the fed segment, taken from this element's
        # feed_tag at feed_seg (1-indexed within the tag).
        feed_idx_in_tag = np.where(tag_arr == elem["feed_tag"])[0]
        i_feed = complex(cur_arr[feed_idx_in_tag[feed_seg - 1]])
        v = elem["v_complex"]
        z_i = v / i_feed
        feeds.append(
            {
                "wire_index": feed_wire_global,
                "knot_index": feed_knot_index,
                "z_re": float(z_i.real),
                "z_im": float(z_i.imag),
                "v_re": float(v.real),
                "v_im": float(v.imag),
            }
        )

    primary = feeds[0]
    return {
        "geometry": "bowtie",
        "wires": wire_records,
        "feeds": feeds,
        "feed_wire_index": primary["wire_index"],
        "feed_knot_index": primary["knot_index"],
        "z_in_re": primary["z_re"],
        "z_in_im": primary["z_im"],
        "design_freq_mhz": b["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": b["wavelength_design"],
        "y_m": b["elements"][0]["y_m"],
        "z_m": b["elements"][0]["z_m"],
        "length_m": b["length_m"],
        "slope": b["slope"],
        "del_y_m": b["del_y_m"],
        "phase_lr_deg": b["phase_lr_deg"],
        "z0_ohms": 100.0,
        "solve_ms": solve_ms,
        "solver": "pynec",
        "ground": b["ground"],
        "ground_fast": b["ground_fast"],
        "height_m": b["z_offset"],
        "ground_eps_r": GROUND_DIELECTRIC,
        "ground_sigma": GROUND_CONDUCTIVITY,
    }


_FANDIPOLE_FEED_GAP = 0.01  # meters; half-gap, matches antenna_designer eps


def _fandipole_ring(k_bands):
    """K cone-direction ring positions evenly distributed at 360°/K around
    the cone axis. K=2 places the two bands at opposite ends of a diameter
    (180° apart), K=3 at the vertices of an equilateral triangle, etc.
    Matches a physical K-spreader fan dipole where the bands fan
    symmetrically around the central feed axis.
    """
    step = 360.0 / k_bands
    return [
        (
            math.cos(math.radians(i * step)),
            math.sin(math.radians(i * step)),
        )
        for i in range(k_bands)
    ]


def _build_fandipole(req: dict):
    """Cone-arrangement fan dipole. Up to 5 bands, each a two-edge arm on
    each side (S->A_i->B_i, mirrored T->Ay_i->By_i), all bands sharing the
    T->S feed gap. Mirrors antenna_designer/.../fandipole.py.
    """
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    n_bands = int(req.get("n_bands", 2))
    if not 1 <= n_bands <= 5:
        raise ValueError(f"n_bands must be in [1, 5], got {n_bands}")
    band_lengths_m = list(req.get("band_lengths_m", [10.2551, 5.2691]))
    if len(band_lengths_m) < n_bands:
        raise ValueError(
            f"band_lengths_m has {len(band_lengths_m)} entries, need {n_bands}"
        )
    band_lengths_m = band_lengths_m[:n_bands]
    band_freqs_mhz = list(req.get("band_freqs_mhz", []))[:n_bands]
    slope = float(req.get("slope", 0.5))
    cone_radius_m = float(req.get("cone_radius_m", 0.12))
    t0_factor = float(req.get("t0_factor", math.sqrt(2.0)))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    ground_fast = bool(req.get("ground_fast", False))
    height_m = float(req.get("height_m", 0.0))

    eps_feed = _FANDIPOLE_FEED_GAP
    t0 = cone_radius_m * t0_factor
    Zc = 1.0 / math.sqrt(1.0 + slope * slope)
    Zs = slope * Zc
    z_offset = height_m if ground else 0.0

    def ry(p):
        return (p[0], -p[1], p[2])

    S = (0.0, eps_feed, z_offset)
    T = ry(S)
    C = (S[0], S[1] + t0 * Zc, S[2] - t0 * Zs)
    lst = _fandipole_ring(n_bands)

    A_pos = [
        (
            C[0] + cone_radius_m * x,
            C[1] + cone_radius_m * y * Zs,
            C[2] + cone_radius_m * y * Zc,
        )
        for (x, y) in lst
    ]
    # dist(S, A_i) is independent of i: sqrt(radius^2 + t0^2). Each arm's
    # axial leg makes up the remainder of the half-band-length.
    ls = []
    for i, (q, a) in enumerate(zip(band_lengths_m, A_pos)):
        dsa = math.sqrt(sum((s_ - a_) ** 2 for s_, a_ in zip(S, a)))
        l_i = q / 2.0 - dsa
        if l_i <= 0:
            raise ValueError(
                f"band {i}: cone geometry leaves no axial leg "
                f"(band_length={q:.3f} m, radial leg={dsa:.3f} m)"
            )
        ls.append(l_i)
    B_pos = [(a[0], a[1] + l * Zc, a[2] - l * Zs) for (l, a) in zip(ls, A_pos)]
    A_neg = [ry(a) for a in A_pos]
    B_neg = [ry(b) for b in B_pos]

    c = nec.nec_context()
    geo = c.get_geometry()
    # Tag 1: feed gap T->S (1 segment, delta-gap source location).
    geo.wire(1, 1, *T, *S, wire_radius, 1.0, 1.0)
    next_tag = 2
    band_tags = []
    for i in range(n_bands):
        t_sr, t_sa, t_tr, t_ta = next_tag, next_tag + 1, next_tag + 2, next_tag + 3
        geo.wire(t_sr, n_per_wire, *S, *A_pos[i], wire_radius, 1.0, 1.0)
        geo.wire(t_sa, n_per_wire, *A_pos[i], *B_pos[i], wire_radius, 1.0, 1.0)
        geo.wire(t_tr, n_per_wire, *T, *A_neg[i], wire_radius, 1.0, 1.0)
        geo.wire(t_ta, n_per_wire, *A_neg[i], *B_neg[i], wire_radius, 1.0, 1.0)
        band_tags.append((t_sr, t_sa, t_tr, t_ta))
        next_tag += 4
    c.geometry_complete(0)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    return {
        "context": c,
        "feed_tag": 1,
        "feed_seg": 1,
        "n_per_wire": n_per_wire,
        "n_bands": n_bands,
        "band_tags": band_tags,
        "band_lengths_m": band_lengths_m,
        "band_freqs_mhz": band_freqs_mhz,
        "S": S,
        "T": T,
        "A_pos": A_pos,
        "B_pos": B_pos,
        "A_neg": A_neg,
        "B_neg": B_neg,
        "slope": slope,
        "cone_radius_m": cone_radius_m,
        "t0_m": t0,
        "wavelength_design": wavelength_design,
        "design_freq_mhz": design_freq_mhz,
        "ground": ground,
        "ground_fast": ground_fast,
        "z_offset": z_offset,
    }


def _fandipole_band_label(i: int, freqs_mhz: list[float], length_m: float) -> str:
    if i < len(freqs_mhz):
        return f"{freqs_mhz[i]:.2f} MHz"
    return f"band {i} ({length_m:.2f} m)"


def solve_fandipole(req: dict) -> dict:
    """Multi-band fan dipole via PyNEC. All band wires share the T->S feed
    gap; PyNEC's segment-endpoint junctions stitch the geometry together
    automatically.
    """
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )
    b = _build_fandipole(req)
    c = b["context"]
    n_per = b["n_per_wire"]
    n_total = 1 + 4 * b["n_bands"] * n_per

    t0_clock = time.perf_counter()
    cur_arr, tag_arr = _run_solve(
        c,
        n_total,
        b["feed_seg"],
        meas_freq_mhz,
        ground=b["ground"],
        feed_tag=b["feed_tag"],
        ground_fast=b["ground_fast"],
    )
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    feed_idx_in_tag = np.where(tag_arr == b["feed_tag"])[0]
    fed_global = feed_idx_in_tag[b["feed_seg"] - 1]
    z_in = complex(1.0 / cur_arr[fed_global])

    wires = []
    # Synthetic 3-knot feed record (T, midpoint, S) — gives the UI an
    # interior knot to anchor feed_knot_index on. NEC's per-segment
    # current goes onto the midpoint; the two end-knots are zero so the
    # render's open-wire convention holds.
    T, S = b["T"], b["S"]
    mid = tuple(0.5 * (a + s_) for a, s_ in zip(T, S))
    feed_cur = complex(cur_arr[fed_global])
    feed_knots = np.array([T, mid, S], dtype=float)
    feed_currents = np.array([0.0 + 0.0j, feed_cur, 0.0 + 0.0j], dtype=np.complex128)
    wires.append(
        {
            "label": "feed",
            "knot_positions": feed_knots.tolist(),
            "knot_currents_re": feed_currents.real.tolist(),
            "knot_currents_im": feed_currents.imag.tolist(),
        }
    )

    for i, (t_sr, t_sa, t_tr, t_ta) in enumerate(b["band_tags"]):
        label = _fandipole_band_label(i, b["band_freqs_mhz"], b["band_lengths_m"][i])
        path_pos = [S, b["A_pos"][i], b["B_pos"][i]]
        knots_pos = _polyline_knots(path_pos, [n_per, n_per])
        cur_pos = np.concatenate(
            [
                cur_arr[np.where(tag_arr == t_sr)[0]],
                cur_arr[np.where(tag_arr == t_sa)[0]],
            ]
        )
        knot_cur_pos = _segment_centers_to_knot_currents(
            cur_pos, knots_pos.shape[0], junction_at_start=True
        )
        wires.append(
            {
                "label": f"{label} +y",
                "knot_positions": knots_pos.tolist(),
                "knot_currents_re": knot_cur_pos.real.tolist(),
                "knot_currents_im": knot_cur_pos.imag.tolist(),
            }
        )
        path_neg = [T, b["A_neg"][i], b["B_neg"][i]]
        knots_neg = _polyline_knots(path_neg, [n_per, n_per])
        cur_neg = np.concatenate(
            [
                cur_arr[np.where(tag_arr == t_tr)[0]],
                cur_arr[np.where(tag_arr == t_ta)[0]],
            ]
        )
        knot_cur_neg = _segment_centers_to_knot_currents(
            cur_neg, knots_neg.shape[0], junction_at_start=True
        )
        wires.append(
            {
                "label": f"{label} -y",
                "knot_positions": knots_neg.tolist(),
                "knot_currents_re": knot_cur_neg.real.tolist(),
                "knot_currents_im": knot_cur_neg.imag.tolist(),
            }
        )

    return {
        "geometry": "fan_dipole",
        "wires": wires,
        "feed_wire_index": 0,
        "feed_knot_index": 1,  # midpoint of the synthetic 3-knot feed wire
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": b["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": b["wavelength_design"],
        "n_bands": b["n_bands"],
        "band_lengths_m": list(b["band_lengths_m"]),
        "band_freqs_mhz": list(b["band_freqs_mhz"]),
        "slope": b["slope"],
        "cone_radius_m": b["cone_radius_m"],
        "t0_m": b["t0_m"],
        "solve_ms": solve_ms,
        "solver": "pynec",
        "ground": b["ground"],
        "ground_fast": b["ground_fast"],
        "height_m": b["z_offset"],
        "ground_eps_r": GROUND_DIELECTRIC,
        "ground_sigma": GROUND_CONDUCTIVITY,
    }


def solve(req: dict) -> dict:
    geometry = req.get("geometry", "inverted_v")
    if geometry == "fan_dipole":
        return solve_fandipole(req)
    if geometry == "hentenna":
        return solve_hentenna(req)
    if geometry == "bowtie":
        return solve_bowtie(req)
    ex = EXAMPLES.get(geometry) or EXAMPLES["inverted_v"]
    if ex.pynec_solve is None:
        raise ValueError(f"PyNEC solve not implemented for geometry {ex.name!r}")
    return ex.pynec_solve(req)


def pattern(req: dict) -> dict:
    """NEC's `rp_card`-computed gain pattern over the upper hemisphere.

    Returns a (n_theta × n_phi) gain grid in dBi at θ ∈ [0°, 90°], full φ.
    With ground off, the lower hemisphere is symmetric to the upper for the
    flat geometries supported here, but we only ship the upper half — the
    UI mirrors as needed.
    """
    geometry = req.get("geometry", "inverted_v")
    if geometry == "fan_dipole":
        b = _build_fandipole(req)
    elif geometry == "hentenna":
        b = _build_hentenna(req)
    elif geometry == "bowtie":
        b = _build_bowtie(req)
    else:
        ex = EXAMPLES.get(geometry) or EXAMPLES["inverted_v"]
        if ex.pynec_build is None:
            raise ValueError(
                f"PyNEC pattern not implemented for geometry {ex.name!r}"
            )
        b = ex.pynec_build(req)
    c = b["context"]
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )

    t0 = time.perf_counter()
    if geometry == "bowtie":
        # Multi-feed solve path: 2 EX cards, one per element. The pattern
        # then reflects the combined-excitation radiation, matching what
        # the bowtie's `feeds` were prescribed in `solve_bowtie`.
        _run_solve_bowtie(
            c,
            b["n_seg_total"],
            b["elements"],
            meas_freq_mhz,
            ground=b["ground"],
            ground_fast=b["ground_fast"],
        )
    else:
        feed_seg = b["feed_seg"]
        n_per_wire = b["n_per_wire"]
        feed_tag = b.get("feed_tag", 1)
        _run_solve(
            c,
            2 * n_per_wire,
            feed_seg,
            meas_freq_mhz,
            ground=b["ground"],
            feed_tag=feed_tag,
            ground_fast=b["ground_fast"],
        )

    # 2°×5° grid: 46 thetas (0..90), 73 phis (0..360 inclusive). At ~3.4k
    # rays this runs in tens of ms — fine for a debounced overlay request.
    n_theta = 46
    n_phi = 73
    del_theta = 90.0 / (n_theta - 1)
    del_phi = 360.0 / (n_phi - 1)
    c.rp_card(0, n_theta, n_phi, 0, 5, 0, 0, 0.0, 0.0, del_theta, del_phi, 0.0, 0.0)
    gains = [
        [float(c.get_gain(0, ti, pi)) for pi in range(n_phi)] for ti in range(n_theta)
    ]
    pattern_ms = (time.perf_counter() - t0) * 1e3

    return {
        "available": True,
        "geometry": geometry,
        "ground": b["ground"],
        "ground_fast": b["ground_fast"],
        "height_m": b["z_offset"],
        "measurement_freq_mhz": meas_freq_mhz,
        "theta_deg": [ti * del_theta for ti in range(n_theta)],
        "phi_deg": [pi * del_phi for pi in range(n_phi)],
        "gain_dbi": gains,
        "pattern_ms": pattern_ms,
    }


def _sweep_at(req: dict, freq_mhz: float) -> complex:
    """Single-frequency Z via PyNEC, used to build the swept Z array.

    Returns the primary feed's Z for back-compat. Multi-feed callers that
    need per-feed data should use `_sweep_at_multifeed`.
    """
    req2 = dict(req)
    req2["measurement_freq_mhz"] = freq_mhz
    res = solve(req2)
    return complex(res["z_in_re"], res["z_in_im"])


def _sweep_at_multifeed(req: dict, freq_mhz: float):
    """Single-frequency multi-feed sweep point. Returns (primary_z,
    feeds_z_list) — feeds_z_list is the per-feed driving-point Z list
    matching the multi-feed `feeds[]` order.

    Used for the bowtie 1×2 array sweep so the streamed NDJSON can carry
    per-feed Z alongside the legacy `z_re` / `z_im` (primary) fields.
    """
    req2 = dict(req)
    req2["measurement_freq_mhz"] = freq_mhz
    res = solve(req2)
    primary = complex(res["z_in_re"], res["z_in_im"])
    feeds_z = [complex(f["z_re"], f["z_im"]) for f in res.get("feeds", [])]
    return primary, feeds_z


def sweep(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    """Loop-based sweep. PyNEC has no batched API, so we run one solve per
    frequency. At N=30 each solve is ~1.5 ms — 41 points * 1.5 ms = ~60 ms,
    fine for an interactive sweep."""
    z_re, z_im = [], []
    for f in freqs_mhz:
        z = _sweep_at(req, f)
        z_re.append(float(z.real))
        z_im.append(float(z.imag))
    return z_re, z_im
