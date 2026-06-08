"""Hentenna: tall narrow rectangular loop with a horizontal cross-bar feed.

Sliced into 5 logical wires that meet at K=2 (S, T) and K=3 (B, D)
junctions. The feed sits in a small gap (T, S) at the centre of the
cross-bar.

    C----------------------------A
    |                            |
    D------------T--S------------B
    |                            |
    E----------------------------F
"""

from __future__ import annotations

import time

import numpy as np

from . import register
from ._base import AntennaExample, ParamSpec, ResultFieldSpec

_FEED_GAP = 0.05  # meters; half-gap between feed knots T and S


def _geometry(
    width_factor: float,
    top_height_factor: float,
    mid_height_factor: float,
    wavelength_design: float,
    n_per_long_edge: int,
    z_offset: float = 0.0,
) -> dict:
    """Build the 5 pysim wires, per-edge segment counts, and the K=2/K=3
    junction descriptors that TriangularPySim needs."""
    half_w = wavelength_design * width_factor / 2.0
    z_mid = wavelength_design * (mid_height_factor - top_height_factor) + z_offset
    z_top = z_offset
    z_bot = -wavelength_design * top_height_factor + z_offset
    eps_feed = _FEED_GAP

    A = (0.0, half_w, z_top)
    B = (0.0, half_w, z_mid)
    F = (0.0, half_w, z_bot)
    S = (0.0, eps_feed, z_mid)
    C = (0.0, -half_w, z_top)
    D = (0.0, -half_w, z_mid)
    E = (0.0, -half_w, z_bot)
    T = (0.0, -eps_feed, z_mid)

    wires = [
        np.array([T, S], dtype=float),
        np.array([S, B], dtype=float),
        np.array([B, A, C, D], dtype=float),
        np.array([T, D], dtype=float),
        np.array([D, E, F, B], dtype=float),
    ]

    # Uniform N segments per non-feed edge. Edge lengths differ ~6× between
    # cross-bar half and verticals, but length-scaled segmentation would
    # either undersample the cross-bar or wildly oversample the verticals —
    # uniform-per-edge matches the reference NEC card layout.
    #
    # Feed-wire parity: pysim's tent basis carries the source on an interior
    # knot. For that knot to sit exactly on the gap's geometric centre the
    # segment count must be EVEN; n_feed=2 puts the single interior knot at
    # z=0. Minimum 2 (n_feed=1 leaves no interior knot to feed on).
    def npe(anchors: np.ndarray) -> list[int]:
        out = []
        for i in range(anchors.shape[0] - 1):
            edge_len = float(np.linalg.norm(anchors[i + 1] - anchors[i]))
            if edge_len < 2 * eps_feed * 1.01:
                nf = max(2, n_per_long_edge // 7)
                if nf % 2 == 1:
                    nf += 1
                out.append(nf)
            else:
                out.append(max(2, n_per_long_edge))
        return out

    n_per_edge_per_wire = [npe(w) for w in wires]

    junctions = [
        [(0, "end"), (1, "start")],  # at S (K=2)
        [(0, "start"), (3, "start")],  # at T (K=2)
        [(1, "end"), (2, "start"), (4, "end")],  # at B (K=3)
        [(2, "end"), (3, "end"), (4, "start")],  # at D (K=3)
    ]

    return {
        "wires": wires,
        "n_per_edge_per_wire": n_per_edge_per_wire,
        "junctions": junctions,
        "feed_wire_index": 0,
        "feed_arclength": eps_feed,
        "half_width_m": half_w,
        "top_height_m": wavelength_design * top_height_factor,
        "mid_offset_m": z_mid - z_top,
    }


def _derive(req: dict, z_offset: float):
    from web.server import C_LIGHT

    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    width_factor = float(req.get("width_factor", 0.1378))
    top_height_factor = float(req.get("top_height_factor", 0.5081))
    mid_height_factor = float(req.get("mid_height_factor", 0.1094))
    wire_radius = float(req.get("wire_radius", 0.0005))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    geom = _geometry(
        width_factor,
        top_height_factor,
        mid_height_factor,
        wavelength_design,
        n_per_wire,
        z_offset=z_offset,
    )
    return {
        "n_per_wire": n_per_wire,
        "design_freq_mhz": design_freq_mhz,
        "wire_radius": wire_radius,
        "wavelength_design": wavelength_design,
        "geom": geom,
    }


def pysim_solve(req: dict) -> dict:
    from web.server import (
        C_LIGHT,
        _PEC_GROUND_EPS_R,
        _PEC_GROUND_SIGMA,
        _make_pysim_sim,
        _pack_pysim_wires,
        _polyline_knots,
        _read_ground,
    )

    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 28.47))
    )
    ground_on, _, z_offset = _read_ground(req)
    d = _derive(req, z_offset)
    geom = d["geom"]
    n_per_wire = d["n_per_wire"]
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)

    sim = _make_pysim_sim(
        req,
        wires=geom["wires"],
        n_per_edge_per_wire=geom["n_per_edge_per_wire"],
        feed_wire_index=geom["feed_wire_index"],
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_meas,
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
        junctions=geom["junctions"],
    )
    sim.wire_radius = d["wire_radius"]

    t0_clock = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    knots_per_wire = [
        _polyline_knots(w, npe)
        for w, npe in zip(geom["wires"], geom["n_per_edge_per_wire"])
    ]
    wire_labels = ["feed", "cross_right", "upper", "cross_left", "lower"]
    wire_records = _pack_pysim_wires(sim, coeffs, knots_per_wire, wire_labels)

    feed_knots = knots_per_wire[geom["feed_wire_index"]]
    arc_at_knot = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(feed_knots, axis=0), axis=1))]
    )
    interior_arc = arc_at_knot[1:-1]
    if len(interior_arc) > 0:
        feed_basis_local = int(np.argmin(np.abs(interior_arc - geom["feed_arclength"])))
        feed_knot_index = feed_basis_local + 1
    else:
        feed_knot_index = 0

    return {
        "geometry": "hentenna",
        "wires": wire_records,
        "feed_wire_index": geom["feed_wire_index"],
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": d["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": d["wavelength_design"],
        "half_width_m": geom["half_width_m"],
        "top_height_m": geom["top_height_m"],
        "mid_offset_m": geom["mid_offset_m"],
        "solve_ms": solve_ms,
        "ground": ground_on,
        "height_m": z_offset,
        "ground_eps_r": _PEC_GROUND_EPS_R,
        "ground_sigma": _PEC_GROUND_SIGMA,
    }


def pysim_sweep(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    from web.server import C_LIGHT, _make_pysim_sim, _read_ground

    ground_on, _, z_offset = _read_ground(req)
    d = _derive(req, z_offset)
    geom = d["geom"]
    n_per_wire = d["n_per_wire"]

    sim = _make_pysim_sim(
        req,
        wires=geom["wires"],
        n_per_edge_per_wire=geom["n_per_edge_per_wire"],
        feed_wire_index=geom["feed_wire_index"],
        feed_arclength=geom["feed_arclength"],
        wavelength=d["wavelength_design"],
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
        junctions=geom["junctions"],
    )
    sim.wire_radius = d["wire_radius"]

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def pynec_build(req: dict) -> dict:
    """9 NEC wire cards organised into 5 logical wires.

      tag 1: feed gap T->S                    (n_feed segs, EX on middle seg)
      tag 2: cross-bar right S->B             (n_per_wire)
      tags 3..5: upper rectangle B->A->C->D   (n_per_wire each)
      tag 6: cross-bar left T->D              (n_per_wire)
      tags 7..9: lower rectangle D->E->F->B   (n_per_wire each)

    NEC connects wire ends by coordinate match, so the K=2 (S,T) and K=3
    (B,D) junctions are implicit. NEC2's pulse basis requires an ODD
    feed-segment count so the source lands at the gap's geometric centre;
    even n_feed offsets the source by half a segment and biases R by ~1 Ω.
    """
    from web.pynec_backend import C_LIGHT, nec

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
    eps_feed = _FEED_GAP
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
    geo.wire(1, n_feed, *T, *S, wire_radius, 1.0, 1.0)
    geo.wire(2, n_per_wire, *S, *B, wire_radius, 1.0, 1.0)
    for i, (p0, p1) in enumerate(zip(upper_path[:-1], upper_path[1:])):
        geo.wire(3 + i, n_per_wire, *p0, *p1, wire_radius, 1.0, 1.0)
    geo.wire(6, n_per_wire, *T, *D, wire_radius, 1.0, 1.0)
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


def _path_knots(path, npe_list) -> np.ndarray:
    parts = []
    for i, n_e in enumerate(npe_list):
        seg = np.linspace(path[i], path[i + 1], n_e + 1)
        parts.append(seg if i == 0 else seg[1:])
    return np.vstack(parts)


def pynec_solve(req: dict) -> dict:
    from web.pynec_backend import (
        GROUND_CONDUCTIVITY,
        GROUND_DIELECTRIC,
        _run_solve,
        _segment_centers_to_knot_currents,
    )

    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 28.47))
    )
    b = pynec_build(req)
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

    # Every wire end in this topology sits at a junction (K=2 or K=3), so
    # current is continuous at all boundary knots — carry the adjacent
    # segment-center value onto the boundary instead of zeroing it.
    wire_records = []
    for label, path, npe, tag_groups in b["wires_meta"]:
        knots = _path_knots(path, npe)
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


EXAMPLE = register(
    AntennaExample(
        name="hentenna",
        label="Hentenna",
        default_view="yz",
        pysim_solve=pysim_solve,
        pysim_sweep=pysim_sweep,
        pynec_build=pynec_build,
        pynec_solve=pynec_solve,
        param_schema=(
            ParamSpec(
                name="width_factor",
                label="width factor",
                default=0.1378,
                min=0.05,
                max=0.30,
                step=0.0005,
                precision=4,
            ),
            ParamSpec(
                name="top_height_factor",
                label="top height factor",
                default=0.5081,
                min=0.30,
                max=0.70,
                step=0.0005,
                precision=4,
            ),
            ParamSpec(
                name="mid_height_factor",
                label="mid height factor",
                default=0.1094,
                min=0.03,
                max=0.30,
                step=0.0005,
                precision=4,
            ),
        ),
        result_schema=(
            ResultFieldSpec(
                field="half_width_m", label="half width", precision=3, unit=" m"
            ),
            ResultFieldSpec(
                field="top_height_m", label="top height", precision=3, unit=" m"
            ),
            ResultFieldSpec(
                field="mid_offset_m", label="mid offset", precision=3, unit=" m"
            ),
        ),
    )
)
