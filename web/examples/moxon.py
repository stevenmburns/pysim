"""Moxon: driver + reflector, both bent into a stretched rectangle.

The driver is a 5-edge polyline (G→H→T→S→A→B); the reflector is a 3-edge
polyline (C→D→E→F). The feed sits on the 1-segment T→S edge at the centre
of the gap. Defaults match the antenna_designer single-band reference
geometry.
"""

from __future__ import annotations

import time

import numpy as np

from . import register
from ._base import AntennaExample, ParamSpec

_FEED_GAP = 0.05  # meters; half-gap between feed knots T and S


def _polylines(
    halfdriver: float,
    aspect_ratio: float,
    tipspacer_factor: float,
    t0_factor: float,
    n_per_long_edge: int,
    z_offset: float = 0.0,
) -> dict:
    """Build moxon driver + reflector polylines, per-edge segment counts,
    and the feed arclength on the driver wire."""
    long_ = 2 * halfdriver / (1 + 2 * aspect_ratio * t0_factor)
    short_ = aspect_ratio * long_
    tipspacer = short_ * tipspacer_factor
    t0 = short_ * t0_factor
    eps_feed = _FEED_GAP

    def rx(p):
        return (-p[0], p[1], p[2])

    def ry(p):
        return (p[0], -p[1], p[2])

    S = (short_ / 2, eps_feed, z_offset)
    A = (S[0], long_ / 2, z_offset)
    B = (A[0] - t0, A[1], z_offset)
    C = (B[0] - tipspacer, B[1], z_offset)
    D = rx(A)
    E = ry(D)
    F = ry(C)
    G = ry(B)
    H = ry(A)
    T = ry(S)

    driver = np.array([G, H, T, S, A, B], dtype=float)
    reflector = np.array([C, D, E, F], dtype=float)

    long_edge_ref = long_ / 2 - eps_feed

    def npe(anchors: np.ndarray) -> list[int]:
        out = []
        for i in range(anchors.shape[0] - 1):
            edge_len = float(np.linalg.norm(anchors[i + 1] - anchors[i]))
            if edge_len < 2 * eps_feed * 1.01:
                out.append(1)
            else:
                out.append(
                    max(2, int(round(n_per_long_edge * edge_len / long_edge_ref)))
                )
        return out

    return {
        "driver": driver,
        "reflector": reflector,
        "npe_driver": npe(driver),
        "npe_reflector": npe(reflector),
        "feed_arclength": halfdriver,
        "long_m": long_,
        "short_m": short_,
        "tipspacer_m": tipspacer,
        "t0_m": t0,
    }


def _derive(req: dict, z_offset: float):
    from web.server import C_LIGHT

    design_freq_mhz = float(req.get("design_freq_mhz", 28.57))
    n_per_wire = int(req.get("n_per_wire", 21))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    aspect_ratio = float(req.get("aspect_ratio", 0.3646))
    tipspacer_factor = float(req.get("tipspacer_factor", 0.0773))
    t0_factor = float(req.get("t0_factor", 0.4078))
    wire_radius = float(req.get("wire_radius", 0.0005))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0
    geom = _polylines(
        halfdriver,
        aspect_ratio,
        tipspacer_factor,
        t0_factor,
        n_per_wire,
        z_offset=z_offset,
    )
    return {
        "design_freq_mhz": design_freq_mhz,
        "n_per_wire": n_per_wire,
        "halfdriver_factor": halfdriver_factor,
        "wire_radius": wire_radius,
        "wavelength_design": wavelength_design,
        "halfdriver": halfdriver,
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
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 28.57))
    )
    ground_on, _, z_offset = _read_ground(req)
    d = _derive(req, z_offset)
    geom = d["geom"]
    n_per_wire = d["n_per_wire"]
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)

    sim = _make_pysim_sim(
        req,
        wires=[geom["driver"], geom["reflector"]],
        n_per_edge_per_wire=[geom["npe_driver"], geom["npe_reflector"]],
        feed_wire_index=0,
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_meas,
        halfdriver_factor=d["halfdriver_factor"],
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = d["wire_radius"]

    t0_clock = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    driver_knots = _polyline_knots(geom["driver"], geom["npe_driver"])
    refl_knots = _polyline_knots(geom["reflector"], geom["npe_reflector"])

    arc_at_knot = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(driver_knots, axis=0), axis=1))]
    )
    interior_arc = arc_at_knot[1:-1]
    feed_basis_local = int(np.argmin(np.abs(interior_arc - geom["feed_arclength"])))
    feed_knot_index = feed_basis_local + 1

    return {
        "geometry": "moxon",
        "wires": _pack_pysim_wires(
            sim, coeffs, [driver_knots, refl_knots], ["driver", "reflector"]
        ),
        "feed_wire_index": 0,
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": d["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": d["wavelength_design"],
        "halfdriver_m": d["halfdriver"],
        "long_m": geom["long_m"],
        "short_m": geom["short_m"],
        "tipspacer_m": geom["tipspacer_m"],
        "t0_m": geom["t0_m"],
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
        wires=[geom["driver"], geom["reflector"]],
        n_per_edge_per_wire=[geom["npe_driver"], geom["npe_reflector"]],
        feed_wire_index=0,
        feed_arclength=geom["feed_arclength"],
        wavelength=d["wavelength_design"],
        halfdriver_factor=d["halfdriver_factor"],
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = d["wire_radius"]

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def pynec_build(req: dict) -> dict:
    """PyNEC moxon: 5 driver wires (tags 1..5) + 3 reflector wires
    (tags 6..8). Feed on the 1-segment T→S edge (tag 3)."""
    from web.pynec_backend import C_LIGHT, nec

    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.57))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    aspect_ratio = float(req.get("aspect_ratio", 0.3646))
    tipspacer_factor = float(req.get("tipspacer_factor", 0.0773))
    t0_factor = float(req.get("t0_factor", 0.4078))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    ground_fast = bool(req.get("ground_fast", False))
    height_m = float(req.get("height_m", 0.0))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0
    long_ = 2 * halfdriver / (1 + 2 * aspect_ratio * t0_factor)
    short_ = aspect_ratio * long_
    tipspacer = short_ * tipspacer_factor
    t0 = short_ * t0_factor
    eps_feed = _FEED_GAP
    z_offset = height_m if ground else 0.0

    def rx(p):
        return (-p[0], p[1], p[2])

    def ry(p):
        return (p[0], -p[1], p[2])

    S = (short_ / 2, eps_feed, z_offset)
    A = (S[0], long_ / 2, z_offset)
    B = (A[0] - t0, A[1], z_offset)
    Cc = (B[0] - tipspacer, B[1], z_offset)
    D = rx(A)
    E = ry(D)
    F = ry(Cc)
    G = ry(B)
    H = ry(A)
    T = ry(S)

    long_edge_ref = long_ / 2 - eps_feed

    def npe_edge(p, q) -> int:
        edge_len = float(np.linalg.norm(np.subtract(q, p)))
        if edge_len < 2 * eps_feed * 1.01:
            return 1
        return max(2, int(round(n_per_wire * edge_len / long_edge_ref)))

    npe_d = [
        npe_edge(G, H),
        npe_edge(H, T),
        1,
        npe_edge(S, A),
        npe_edge(A, B),
    ]
    npe_r = [npe_edge(Cc, D), npe_edge(D, E), npe_edge(E, F)]

    driver_path = [G, H, T, S, A, B]
    reflector_path = [Cc, D, E, F]

    c = nec.nec_context()
    geo = c.get_geometry()
    for i in range(5):
        p0, p1 = driver_path[i], driver_path[i + 1]
        geo.wire(i + 1, npe_d[i], *p0, *p1, wire_radius, 1.0, 1.0)
    for i in range(3):
        p0, p1 = reflector_path[i], reflector_path[i + 1]
        geo.wire(6 + i, npe_r[i], *p0, *p1, wire_radius, 1.0, 1.0)
    c.geometry_complete(0)

    return {
        "context": c,
        "feed_tag": 3,  # T->S edge
        "feed_seg": 1,
        "n_per_wire": n_per_wire,
        "npe_d": npe_d,
        "npe_r": npe_r,
        "driver_path": driver_path,
        "reflector_path": reflector_path,
        "wavelength_design": wavelength_design,
        "design_freq_mhz": design_freq_mhz,
        "halfdriver_m": halfdriver,
        "long_m": long_,
        "short_m": short_,
        "tipspacer_m": tipspacer,
        "t0_m": t0,
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
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 28.57))
    )
    b = pynec_build(req)
    c = b["context"]

    t0_clock = time.perf_counter()
    cur_arr, tag_arr = _run_solve(
        c,
        sum(b["npe_d"]) + sum(b["npe_r"]),
        b["feed_seg"],
        meas_freq_mhz,
        ground=b["ground"],
        feed_tag=b["feed_tag"],
        ground_fast=b["ground_fast"],
    )
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    feed_idx_in_tag3 = np.where(tag_arr == b["feed_tag"])[0]
    fed_global = feed_idx_in_tag3[b["feed_seg"] - 1]
    z_in = complex(1.0 / cur_arr[fed_global])

    cur_driver = np.concatenate(
        [cur_arr[np.where(tag_arr == t)[0]] for t in range(1, 6)]
    )
    cur_refl = np.concatenate([cur_arr[np.where(tag_arr == t)[0]] for t in range(6, 9)])

    driver_knots = _path_knots(b["driver_path"], b["npe_d"])
    refl_knots = _path_knots(b["reflector_path"], b["npe_r"])
    driver_knot_cur = _segment_centers_to_knot_currents(
        cur_driver, driver_knots.shape[0]
    )
    refl_knot_cur = _segment_centers_to_knot_currents(cur_refl, refl_knots.shape[0])

    # T is at index sum(npe_d[:2]); S is one more (the T->S edge has 1 segment).
    feed_knot_index = sum(b["npe_d"][:2])

    return {
        "geometry": "moxon",
        "wires": [
            {
                "label": "driver",
                "knot_positions": driver_knots.tolist(),
                "knot_currents_re": driver_knot_cur.real.tolist(),
                "knot_currents_im": driver_knot_cur.imag.tolist(),
            },
            {
                "label": "reflector",
                "knot_positions": refl_knots.tolist(),
                "knot_currents_re": refl_knot_cur.real.tolist(),
                "knot_currents_im": refl_knot_cur.imag.tolist(),
            },
        ],
        "feed_wire_index": 0,
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": b["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": b["wavelength_design"],
        "halfdriver_m": b["halfdriver_m"],
        "long_m": b["long_m"],
        "short_m": b["short_m"],
        "tipspacer_m": b["tipspacer_m"],
        "t0_m": b["t0_m"],
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
        name="moxon",
        label="Moxon",
        pysim_solve=pysim_solve,
        pysim_sweep=pysim_sweep,
        pynec_build=pynec_build,
        pynec_solve=pynec_solve,
        param_schema=(
            ParamSpec(
                name="halfdriver_factor",
                label="halfdriver factor",
                default=0.962,
                min=0.8,
                max=1.1,
                step=0.001,
                precision=3,
            ),
            ParamSpec(
                name="aspect_ratio",
                label="aspect ratio",
                default=0.3646,
                min=0.2,
                max=0.6,
                step=0.001,
                precision=4,
            ),
            ParamSpec(
                name="tipspacer_factor",
                label="tip spacer factor",
                default=0.0773,
                min=0.02,
                max=0.20,
                step=0.0005,
                precision=4,
            ),
            ParamSpec(
                name="t0_factor",
                label="t0 factor",
                default=0.4078,
                min=0.15,
                max=0.6,
                step=0.001,
                precision=4,
            ),
        ),
    )
)
