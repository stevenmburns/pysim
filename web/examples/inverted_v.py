"""Inverted-V dipole: apex at z=z_offset, arms drooping in the yz plane.

Migrated from web/server.py (_inverted_v_polyline, _solve_inverted_v,
_sweep_inverted_v) and web/pynec_backend.py (_build_inverted_v,
solve_inverted_v) as the pilot for the registry-based example layout.

Helpers (_make_pysim_sim, _read_ground, _pack_pysim_wires, _polyline_knots,
_run_solve, _segment_centers_to_knot_currents, etc.) are imported lazily
inside each function to break the import cycle: the dispatchers in
web.server import this package, so this module can't pull from web.server
at import time.
"""

from __future__ import annotations

import time

import numpy as np

from . import register
from ._base import AntennaExample


def _polyline(arm_len: float, angle_deg: float, z_offset: float = 0.0) -> np.ndarray:
    """Three-point polyline (left tip → apex → right tip).

    Arms run along ±y so the broadside null axis is ±x — matching the
    Yagi/moxon/hexbeam convention where the main lobe peaks at azimuth 0°.
    angle_deg is each arm's droop from horizontal: 0 = flat dipole.
    """
    alpha = np.deg2rad(angle_deg)
    cos_a, sin_a = float(np.cos(alpha)), float(np.sin(alpha))
    left = np.array([0.0, -arm_len * cos_a, z_offset - arm_len * sin_a])
    apex = np.array([0.0, 0.0, z_offset])
    right = np.array([0.0, arm_len * cos_a, z_offset - arm_len * sin_a])
    return np.vstack([left, apex, right])


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

    angle_deg = float(req.get("angle_deg", 30.0))
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    arm_len = halfdriver_factor * wavelength_design / 4.0

    polyline = _polyline(arm_len, angle_deg, z_offset=z_offset)
    sim = _make_pysim_sim(
        req,
        wires=[polyline],
        n_per_edge_per_wire=[[n_per_wire, n_per_wire]],
        feed_wire_index=0,
        wavelength=wavelength_meas,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = wire_radius

    t0 = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0) * 1e3

    knots = _polyline_knots(polyline, [n_per_wire, n_per_wire])
    feed_knot_index = n_per_wire  # apex (midpoint of polyline)

    return {
        "geometry": "inverted_v",
        "wires": _pack_pysim_wires(sim, coeffs, [knots], ["wire"]),
        "feed_wire_index": 0,
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": wavelength_design,
        "arm_len_m": arm_len,
        "solve_ms": solve_ms,
        "ground": ground_on,
        "height_m": z_offset,
        "ground_eps_r": _PEC_GROUND_EPS_R,
        "ground_sigma": _PEC_GROUND_SIGMA,
    }


def pysim_sweep(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    """Batched sweep using the pysim model's compute_impedance_swept."""
    from web.server import C_LIGHT, _make_pysim_sim, _read_ground

    angle_deg = float(req.get("angle_deg", 30.0))
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    arm_len = halfdriver_factor * wavelength_design / 4.0

    sim = _make_pysim_sim(
        req,
        wires=[_polyline(arm_len, angle_deg, z_offset=z_offset)],
        n_per_edge_per_wire=[[n_per_wire, n_per_wire]],
        feed_wire_index=0,
        wavelength=wavelength_design,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = wire_radius

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def pynec_build(req: dict) -> dict:
    """Build the PyNEC context + geometry for the inverted V. Returns the
    context, feed segment, and derived geometry that callers need for
    response formatting."""
    from web.pynec_backend import C_LIGHT, nec

    angle_deg = float(req.get("angle_deg", 30.0))
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    ground_fast = bool(req.get("ground_fast", False))
    height_m = float(req.get("height_m", 0.0))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    arm_len = halfdriver_factor * wavelength_design / 4.0
    alpha = np.deg2rad(angle_deg)
    cos_a, sin_a = float(np.cos(alpha)), float(np.sin(alpha))
    # When ground is on, lift the entire geometry by height_m so the arms
    # stay above z=0 (NEC rejects segments at/below the ground plane).
    z_offset = height_m if ground else 0.0
    left = (0.0, -arm_len * cos_a, z_offset - arm_len * sin_a)
    apex = (0.0, 0.0, z_offset)
    right = (0.0, arm_len * cos_a, z_offset - arm_len * sin_a)

    c = nec.nec_context()
    geo = c.get_geometry()
    geo.wire(1, n_per_wire, *left, *apex, wire_radius, 1.0, 1.0)
    geo.wire(2, n_per_wire, *apex, *right, wire_radius, 1.0, 1.0)
    c.geometry_complete(0)

    return {
        "context": c,
        "feed_seg": n_per_wire,  # last segment of wire 1 (touches the apex)
        "n_per_wire": n_per_wire,
        "left": left,
        "apex": apex,
        "right": right,
        "arm_len_m": arm_len,
        "wavelength_design": wavelength_design,
        "design_freq_mhz": design_freq_mhz,
        "ground": ground,
        "ground_fast": ground_fast,
        "z_offset": z_offset,
    }


def pynec_solve(req: dict) -> dict:
    """Inverted V via two PyNEC wires meeting at the apex."""
    from web.pynec_backend import (
        GROUND_CONDUCTIVITY,
        GROUND_DIELECTRIC,
        _run_solve,
        _segment_centers_to_knot_currents,
    )

    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )
    b = pynec_build(req)
    c = b["context"]
    n_per_wire = b["n_per_wire"]
    feed_seg = b["feed_seg"]

    t0 = time.perf_counter()
    cur_arr, tag_arr = _run_solve(
        c,
        2 * n_per_wire,
        feed_seg,
        meas_freq_mhz,
        ground=b["ground"],
        ground_fast=b["ground_fast"],
    )
    solve_ms = (time.perf_counter() - t0) * 1e3

    wire1_idx = np.where(tag_arr == 1)[0]
    fed_global_idx = wire1_idx[feed_seg - 1]
    z_in = complex(1.0 / cur_arr[fed_global_idx])

    # Knot positions and currents, matching pysim's response shape: one
    # continuous wire (left arm + right arm), feed at apex.
    arm1_knots = np.linspace(b["left"], b["apex"], n_per_wire + 1)
    arm2_knots = np.linspace(b["apex"], b["right"], n_per_wire + 1)
    knots = np.vstack([arm1_knots[:-1], arm2_knots])  # (2N+1, 3)

    wire2_idx = np.where(tag_arr == 2)[0]
    cur_wire1 = cur_arr[wire1_idx]
    cur_wire2 = cur_arr[wire2_idx]
    cur_all = np.concatenate([cur_wire1, cur_wire2])
    knot_currents = _segment_centers_to_knot_currents(cur_all, knots.shape[0])
    feed_knot_index = n_per_wire  # apex

    return {
        "geometry": "inverted_v",
        "wires": [
            {
                "label": "wire",
                "knot_positions": knots.tolist(),
                "knot_currents_re": knot_currents.real.tolist(),
                "knot_currents_im": knot_currents.imag.tolist(),
            }
        ],
        "feed_wire_index": 0,
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": b["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": b["wavelength_design"],
        "arm_len_m": b["arm_len_m"],
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
        name="inverted_v",
        label="Inverted V",
        pysim_solve=pysim_solve,
        pysim_sweep=pysim_sweep,
        pynec_build=pynec_build,
        pynec_solve=pynec_solve,
    )
)
