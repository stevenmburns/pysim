"""Bowtie 1×2 phased array: two bowtie elements at y = ±del_y_m sharing a
multi-feed delta-gap drive.

Element 0 (left, y < 0) is fed with V = 1+0j; element 1 (right, y > 0) is
fed with V = exp(j·π·phase_lr_deg/180). The response carries per-feed
driving-point impedances in feeds[]; the primary feed (element 0) is also
exposed on the legacy z_in_re / z_in_im / feed_wire_index / feed_knot_index
keys so single-feed-aware frontends keep working without change.

This is the only example so far that uses the multi-feed AntennaExample
fields (`multi_feed=True`, custom `pynec_pattern_excite`). The shapes it
adds:
  - momwire_sweep returns the 4-tuple (primary_re, primary_im, feeds_re,
    feeds_im); the /sweep endpoint reads multi_feed to know to expect it.
  - pynec_pattern_excite is the multi-source NEC drive (two ex_card calls
    superposed) used in pattern().
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import register
from ._base import AntennaExample, ParamSpec, ResultFieldSpec

# Half-gap (m) at the feed and at the top-of-bowtie pinch.
_BOWTIE_EPS = 0.05


# ---------------------------------------------------------------------------
# momwire path
# ---------------------------------------------------------------------------


def _element_wires(
    slope: float,
    length: float,
    n_per_long_edge: int,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
) -> dict:
    """Build a single bowtie's 4 polylines + per-edge segment counts.

    Replicates antenna_designer/designs/bowtie.py. The antenna lies in the
    y-z plane (x = 0), centred on (y, z) = (0, 0):

        y = (length/2) / (slope + sqrt(1 + slope^2))
        z = slope * y

    The two triangles' tips are at (±y, 0); their open sides flare to
    (±y, ±z); each side closes to within an `eps` half-gap on the central
    axis at (±eps, ±eps). The feed sits across the bottom gap
    (-eps, -eps) → (eps, -eps).

    4 polylines connected at 4 K=2 junctions where the loops meet:

      W0  top arc:        (-y, 0) -> (-y, z) -> (-eps,  eps) ->
                          ( eps, eps) -> ( y, z) -> ( y, 0)
      W1  bot-left arc:   (-y, 0) -> (-y,-z) -> (-eps, -eps)
      W2  bot-right arc:  ( eps,-eps) -> ( y,-z) -> ( y, 0)
      W3  feed wire:      (-eps,-eps) -> ( eps,-eps)
    """
    eps_b = _BOWTIE_EPS
    y = 0.5 * length / (slope + np.sqrt(1.0 + slope * slope))
    z = slope * y
    yo = y_offset
    zo = z_offset

    def P(yy, zz):
        return (0.0, yy + yo, zz + zo)

    w0 = np.array(
        [P(-y, 0), P(-y, z), P(-eps_b, eps_b), P(eps_b, eps_b), P(y, z), P(y, 0)],
        dtype=float,
    )
    w1 = np.array([P(-y, 0), P(-y, -z), P(-eps_b, -eps_b)], dtype=float)
    w2 = np.array([P(eps_b, -eps_b), P(y, -z), P(y, 0)], dtype=float)
    w3 = np.array([P(-eps_b, -eps_b), P(eps_b, -eps_b)], dtype=float)

    n_long = max(2, int(n_per_long_edge))
    n_short = max(2, n_long // 7)
    if n_short % 2 == 1:
        n_short += 1  # even count puts the feed's interior knot at gap centre

    long_edge_ref = float(np.linalg.norm(w0[1] - w0[0]))  # (-y,0)→(-y,z) length

    def npe(anchors: np.ndarray) -> list[int]:
        out = []
        for i in range(anchors.shape[0] - 1):
            edge_len = float(np.linalg.norm(anchors[i + 1] - anchors[i]))
            if edge_len < 2 * eps_b * 1.01:
                out.append(n_short)
            else:
                # Scale long edges to the reference flare length so the
                # diagonal hypotenuse gets proportionally more segments
                # than the short vertical leg.
                out.append(max(2, int(round(n_long * edge_len / long_edge_ref))))
        return out

    wires = [w0, w1, w2, w3]
    n_per_edge_per_wire = [npe(w) for w in wires]
    # Junction wire-indices LOCAL to this element (0..3); the array wrapper
    # renumbers to global wire indices.
    junctions = [
        [(0, "start"), (1, "start")],
        [(0, "end"), (2, "end")],
        [(1, "end"), (3, "start")],
        [(3, "end"), (2, "start")],
    ]

    return {
        "wires": wires,
        "n_per_edge_per_wire": n_per_edge_per_wire,
        "junctions": junctions,
        "feed_wire_local": 3,
        "feed_arclength": eps_b,
        "y_m": y,
        "z_m": z,
        "eps_m": eps_b,
    }


def _array_geometry(
    slope: float,
    length: float,
    n_per_long_edge: int,
    del_y: float,
    phase_lr_deg: float,
    z_offset: float = 0.0,
) -> dict:
    """Concatenate two bowtie elements at y_offset = -del_y and +del_y.

    Element 1's junction wire-indices are shifted by n_wires_per_element
    (=4 for the bowtie). The returned `feeds` list is what
    TriangularSolver(feeds=...) consumes.
    """
    elem_l = _element_wires(
        slope, length, n_per_long_edge, y_offset=-del_y, z_offset=z_offset
    )
    elem_r = _element_wires(
        slope, length, n_per_long_edge, y_offset=+del_y, z_offset=z_offset
    )
    n_wpe = len(elem_l["wires"])  # 4

    wires = elem_l["wires"] + elem_r["wires"]
    n_per_edge_per_wire = elem_l["n_per_edge_per_wire"] + elem_r["n_per_edge_per_wire"]
    junctions = list(elem_l["junctions"]) + [
        [(w + n_wpe, end) for (w, end) in jw] for jw in elem_r["junctions"]
    ]

    phase_lr_rad = np.pi * phase_lr_deg / 180.0
    feeds = [
        (elem_l["feed_wire_local"], elem_l["feed_arclength"], 1.0 + 0.0j),
        (
            elem_r["feed_wire_local"] + n_wpe,
            elem_r["feed_arclength"],
            complex(np.cos(phase_lr_rad), np.sin(phase_lr_rad)),
        ),
    ]

    return {
        "wires": wires,
        "n_per_edge_per_wire": n_per_edge_per_wire,
        "junctions": junctions,
        "feeds": feeds,
        "y_m": elem_l["y_m"],
        "z_m": elem_l["z_m"],
        "eps_m": elem_l["eps_m"],
        "del_y_m": del_y,
    }


def _request_args(req: dict) -> dict:
    """Defaults match antenna_designer's canonical 28.47 MHz bowtiearray1x2."""
    return {
        "n_per_wire": int(req.get("n_per_wire", 21)),
        "slope": float(req.get("slope", 0.5376)),
        # length_factor = (antenna_designer "length") / λ_design.
        # 0.515 ≈ 5.42 m at 28.47 MHz.
        "length_factor": float(req.get("length_factor", 0.515)),
        # del_y_m: half the centre-to-centre spacing between the two
        # bowtie elements, matching Array1x2Builder's `del_y`.
        "del_y_m": float(req.get("del_y_m", 4.0)),
        # Phase shift on element 1 (the +y bowtie) in degrees; 0 = in-phase,
        # 180 = anti-phase.
        "phase_lr_deg": float(req.get("phase_lr_deg", 0.0)),
        "wire_radius": float(req.get("wire_radius", 0.0005)),
    }


def _feed_knot_index(
    feed_wire_global: int, feed_arclength: float, knots_per_wire: list[np.ndarray]
) -> int:
    """Interior knot of `feed_wire_global` whose arc-length from the wire's
    start is closest to `feed_arclength`. Mirrors TriangularSolver's
    feed-basis-index convention so the frontend marker matches the actual
    delta-gap source location."""
    feed_knots = knots_per_wire[feed_wire_global]
    arc_at_knot = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(feed_knots, axis=0), axis=1))]
    )
    interior_arc = arc_at_knot[1:-1]
    if len(interior_arc) == 0:
        return 0
    return int(np.argmin(np.abs(interior_arc - feed_arclength))) + 1


def momwire_solve(req: dict) -> dict:
    from web.server import (
        C_LIGHT,
        _PEC_GROUND_EPS_R,
        _PEC_GROUND_SIGMA,
        _make_momwire_sim,
        _pack_momwire_wires,
        _polyline_knots,
        _read_ground,
    )

    args = _request_args(req)
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    ground_on, _, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    length_m = args["length_factor"] * wavelength_design

    geom = _array_geometry(
        args["slope"],
        length_m,
        args["n_per_wire"],
        args["del_y_m"],
        args["phase_lr_deg"],
        z_offset=z_offset,
    )

    sim = _make_momwire_sim(
        req,
        wires=geom["wires"],
        n_per_edge_per_wire=geom["n_per_edge_per_wire"],
        feeds=geom["feeds"],
        wavelength=wavelength_meas,
        nsegs=args["n_per_wire"],
        ground_z=0.0 if ground_on else None,
        junctions=geom["junctions"],
    )
    sim.wire_radius = args["wire_radius"]

    t0 = time.perf_counter()
    z_per_feed, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0) * 1e3

    knots_per_wire = [
        _polyline_knots(w, npe)
        for w, npe in zip(geom["wires"], geom["n_per_edge_per_wire"])
    ]
    wire_labels = [
        f"{half}_{part}"
        for half in ("L", "R")
        for part in ("top_arc", "bot_left_arc", "bot_right_arc", "feed")
    ]
    wire_records = _pack_momwire_wires(sim, coeffs, knots_per_wire, wire_labels)

    z_arr = np.atleast_1d(z_per_feed)
    feed_entries = []
    for i, (w_global, arc, v_complex) in enumerate(geom["feeds"]):
        kidx = _feed_knot_index(w_global, arc, knots_per_wire)
        zi = complex(z_arr[i])
        feed_entries.append(
            {
                "wire_index": int(w_global),
                "knot_index": int(kidx),
                "z_re": float(zi.real),
                "z_im": float(zi.imag),
                "v_re": float(v_complex.real),
                "v_im": float(v_complex.imag),
            }
        )

    primary = feed_entries[0]
    return {
        "geometry": "bowtie",
        "wires": wire_records,
        "feeds": feed_entries,
        "feed_wire_index": primary["wire_index"],
        "feed_knot_index": primary["knot_index"],
        "z_in_re": primary["z_re"],
        "z_in_im": primary["z_im"],
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": wavelength_design,
        "y_m": geom["y_m"],
        "z_m": geom["z_m"],
        "length_m": length_m,
        "slope": args["slope"],
        "del_y_m": args["del_y_m"],
        "phase_lr_deg": args["phase_lr_deg"],
        # 1×2 array is designed for 100 Ω feedlines per element; surface as
        # the Smith-chart / SWR reference instead of the default 50 Ω.
        "z0_ohms": 100.0,
        "solve_ms": solve_ms,
        "ground": ground_on,
        "height_m": z_offset,
        "ground_eps_r": _PEC_GROUND_EPS_R,
        "ground_sigma": _PEC_GROUND_SIGMA,
    }


def momwire_sweep(
    req: dict, freqs_mhz: list[float]
) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    """Multi-feed sweep returning the 4-tuple (primary_re, primary_im,
    feeds_re, feeds_im) where feeds_* is (n_freqs × n_feeds)."""
    from web.server import C_LIGHT, _make_momwire_sim, _read_ground

    args = _request_args(req)
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    ground_on, _, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    length_m = args["length_factor"] * wavelength_design

    geom = _array_geometry(
        args["slope"],
        length_m,
        args["n_per_wire"],
        args["del_y_m"],
        args["phase_lr_deg"],
        z_offset=z_offset,
    )
    sim = _make_momwire_sim(
        req,
        wires=geom["wires"],
        n_per_edge_per_wire=geom["n_per_edge_per_wire"],
        feeds=geom["feeds"],
        wavelength=wavelength_design,
        nsegs=args["n_per_wire"],
        ground_z=0.0 if ground_on else None,
        junctions=geom["junctions"],
    )
    sim.wire_radius = args["wire_radius"]

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = np.atleast_2d(sim.compute_impedance_swept(k_array))  # (n_k, n_f)
    feeds_re = z_array.real.tolist()
    feeds_im = z_array.imag.tolist()
    primary_re = [row[0] for row in feeds_re]
    primary_im = [row[0] for row in feeds_im]
    return primary_re, primary_im, feeds_re, feeds_im


# ---------------------------------------------------------------------------
# pynec path
# ---------------------------------------------------------------------------


def _pynec_element_polylines(
    slope: float, length: float, y_offset: float, z_offset: float
):
    """Return the 4 polylines of one bowtie element in the y-z plane (x=0)
    at the given y_offset / z_offset. Same shapes as the momwire helper but
    as plain tuples (no numpy arrays) so they round-trip into PyNEC's
    geo.wire() args by splat."""
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


def pynec_build(req: dict) -> dict:
    """Build the PyNEC context + geometry for the bowtie 1×2 phased array.

    Two bowtie elements at y = ±del_y_m. Per element: 10 NEC wire cards
    (one per polyline edge — 5 + 2 + 2 + 1), tags grouped per polyline so
    the per-polyline current concatenation matches the momwire wire shape
    the frontend already renders. Multi-feed drive applies V_0 = 1+0j on
    element 0's feed and V_1 = exp(j·π·phase_lr_deg/180) on element 1's
    feed — see `pynec_excite()` below.
    """
    from web.pynec_backend import C_LIGHT, nec

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

    # Segment counts: long edges get n_per_wire; short eps-scale edges get
    # a small odd count so the NEC EX card lands on the segment whose
    # centre is at the gap's geometric centre.
    n_long = max(2, n_per_wire)
    n_short_even = max(2, n_long // 7)
    n_short_odd = n_short_even if n_short_even % 2 == 1 else n_short_even + 1
    feed_seg_local = (n_short_odd + 1) // 2  # 1-indexed middle of feed wire

    c = nec.nec_context()
    geo = c.get_geometry()

    elements = []
    tag = 0
    for _elem_idx, (y_off, v_complex) in enumerate(
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
        w0, w1, w2, w3, y, z = _pynec_element_polylines(
            slope, length_m, y_off, z_offset
        )
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
                feed_tag_for_elem = tag_group[0]

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


def _run_solve(b: dict, freq_mhz: float):
    """Multi-feed NEC2 solve. Each element gets one EX card at its feed
    tag's centre segment with its prescribed complex voltage; NEC
    superposes the excitations and the returned current vector reflects
    the combined drive."""
    from web.pynec_backend import (
        GROUND_CONDUCTIVITY,
        GROUND_DIELECTRIC,
    )

    c = b["context"]
    if b["ground"]:
        itype = 0 if b["ground_fast"] else 2
        c.gn_card(itype, 0, GROUND_DIELECTRIC, GROUND_CONDUCTIVITY, 0, 0, 0, 0)
    else:
        c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)
    for elem in b["elements"]:
        v = elem["v_complex"]
        c.ex_card(0, elem["feed_tag"], elem["feed_seg"], 0, v.real, v.imag, 0, 0, 0, 0)
    c.fr_card(0, 1, freq_mhz, 0)
    c.xq_card(0)
    sc = c.get_structure_currents(0)
    cur_arr = np.asarray(sc.get_current(), dtype=np.complex128)
    tag_arr = np.asarray(sc.get_current_segment_tag())
    return cur_arr, tag_arr


def pynec_pattern_excite(b: dict, freq_mhz: float) -> None:
    """Excite the NEC context for pattern computation. Hands off to
    _run_solve so pattern() gets the combined-excitation radiation,
    matching what `solve_bowtie` reports in `feeds`."""
    _run_solve(b, freq_mhz)


def _path_knots(path, npe_list) -> np.ndarray:
    parts = []
    for i, n_e in enumerate(npe_list):
        seg = np.linspace(path[i], path[i + 1], n_e + 1)
        parts.append(seg if i == 0 else seg[1:])
    return np.vstack(parts)


def _per_element_wires(
    elem: dict, cur_arr: np.ndarray, tag_arr: np.ndarray, half_label: str
) -> list[dict]:
    """Pack one element's polylines into wire records the frontend renders.
    Every bowtie polyline endpoint is at a K=2 junction so the carry-over
    `junction_at_start/end` apply throughout."""
    from web.pynec_backend import _segment_centers_to_knot_currents

    out = []
    for label, path, npe, tag_group in elem["polylines"]:
        knots = _path_knots(path, npe)
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


def pynec_solve(req: dict) -> dict:
    from web.pynec_backend import GROUND_CONDUCTIVITY, GROUND_DIELECTRIC

    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 28.47))
    )
    b = pynec_build(req)

    t0 = time.perf_counter()
    cur_arr, tag_arr = _run_solve(b, meas_freq_mhz)
    solve_ms = (time.perf_counter() - t0) * 1e3

    wire_records: list[dict] = []
    feeds: list[dict] = []
    for _elem_idx, (elem, half_label) in enumerate(zip(b["elements"], ("L", "R"))):
        wires_before = len(wire_records)
        wire_records.extend(_per_element_wires(elem, cur_arr, tag_arr, half_label))
        # Feed wire is always the 4th polyline (index 3) of each element.
        feed_wire_global = wires_before + 3
        feed_seg = elem["feed_seg"]
        # knot k sits between segments k-1 and k → right-side knot of the
        # fed segment is at index feed_seg.
        feed_knot_index = feed_seg
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


EXAMPLE = register(
    AntennaExample(
        name="bowtie",
        label="Bowtie 1×2 array",
        default_view="yz",
        momwire_solve=momwire_solve,
        momwire_sweep=momwire_sweep,
        pynec_build=pynec_build,
        pynec_solve=pynec_solve,
        pynec_pattern_excite=pynec_pattern_excite,
        multi_feed=True,
        param_schema=(
            ParamSpec(
                name="length_factor",
                label="length factor",
                default=0.515,
                min=0.30,
                max=0.80,
                step=0.0005,
                precision=4,
                unit="L/λ",
            ),
            ParamSpec(
                name="slope",
                label="slope",
                default=0.5376,
                min=0.0,
                max=1.5,
                step=0.001,
                precision=4,
            ),
            ParamSpec(
                name="del_y_m",
                label="element spacing",
                default=4.0,
                min=1.0,
                max=10.0,
                step=0.05,
                precision=2,
                unit="m",
            ),
            ParamSpec(
                name="phase_lr_deg",
                label="phase L/R",
                default=0.0,
                min=-180.0,
                max=180.0,
                step=1.0,
                precision=0,
                unit="°",
            ),
        ),
        result_schema=(
            ResultFieldSpec(
                field="y_m", label="arm half-length", precision=3, unit=" m"
            ),
            ResultFieldSpec(field="z_m", label="tip droop z", precision=3, unit=" m"),
            ResultFieldSpec(
                field="del_y_m", label="spacing del_y", precision=3, unit=" m"
            ),
            ResultFieldSpec(
                field="phase_lr_deg", label="phase_lr", precision=1, unit="°"
            ),
        ),
    )
)
