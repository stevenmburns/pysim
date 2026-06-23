"""Yagi: driver + reflector + N optional uniformly-spaced directors.

Canonical layout: boom along +x (driver at x=0, reflector at x=-spacing,
directors at x=+i·dir_spacing), elements along ±y, beam direction +x —
matching moxon/hexbeam so the UI keeps a consistent beam axis.
"""

from __future__ import annotations

import time

import numpy as np

from . import register
from ._base import AntennaExample, ParamSpec, ResultFieldSpec


def _polylines(
    h_driver: float,
    h_refl: float,
    spacing_m: float,
    n_directors: int,
    dir_spacing_m: float,
    h_dir: float,
    z_offset: float = 0.0,
) -> list[np.ndarray]:
    """Driver + reflector + n_directors straight wires as 2-anchor polylines."""
    polylines = [
        np.array([(0.0, -h_driver, z_offset), (0.0, h_driver, z_offset)]),
        np.array([(-spacing_m, -h_refl, z_offset), (-spacing_m, h_refl, z_offset)]),
    ]
    for i in range(n_directors):
        x = (i + 1) * dir_spacing_m
        polylines.append(np.array([(x, -h_dir, z_offset), (x, h_dir, z_offset)]))
    return polylines


def _derive(req: dict, ground_z_offset: float):
    """Pull the request knobs and compute the derived geometry quantities
    that both momwire_solve and momwire_sweep need."""
    from web.server import C_LIGHT

    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    driver_factor = float(req.get("driver_length_factor", 0.962))
    refl_factor_abs = float(req.get("reflector_length_factor", 1.01))
    spacing_wavelengths = float(req.get("spacing_wavelengths", 0.15))
    wire_radius = float(req.get("wire_radius", 0.0005))
    n_directors = int(req.get("n_directors", 0))
    dir_spacing_wl = float(req.get("director_spacing_wavelengths", 0.2))
    dir_size_factor = float(req.get("director_size_factor", 0.95))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    h_driver = driver_factor * wavelength_design / 4.0
    h_refl = refl_factor_abs * wavelength_design / 4.0
    spacing_m = spacing_wavelengths * wavelength_design
    dir_spacing_m = dir_spacing_wl * wavelength_design
    h_dir = dir_size_factor * h_driver

    return {
        "n_per_wire": n_per_wire,
        "design_freq_mhz": design_freq_mhz,
        "driver_factor": driver_factor,
        "wire_radius": wire_radius,
        "n_directors": n_directors,
        "wavelength_design": wavelength_design,
        "h_driver": h_driver,
        "h_refl": h_refl,
        "spacing_m": spacing_m,
        "dir_spacing_m": dir_spacing_m,
        "h_dir": h_dir,
        "polylines": _polylines(
            h_driver,
            h_refl,
            spacing_m,
            n_directors,
            dir_spacing_m,
            h_dir,
            z_offset=ground_z_offset,
        ),
    }


def momwire_solve(req: dict) -> dict:
    from web.server import (
        C_LIGHT,
        _PEC_GROUND_EPS_R,
        _PEC_GROUND_SIGMA,
        _make_momwire_sim,
        _pack_momwire_wires,
        _read_ground,
    )

    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )
    ground_on, _, z_offset = _read_ground(req)
    d = _derive(req, z_offset)
    n_per_wire = d["n_per_wire"]
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)

    sim = _make_momwire_sim(
        req,
        wires=d["polylines"],
        n_per_edge_per_wire=[[n_per_wire]] * len(d["polylines"]),
        feed_wire_index=0,
        wavelength=wavelength_meas,
        halfdriver_factor=d["driver_factor"],
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = d["wire_radius"]

    t0 = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0) * 1e3

    N = n_per_wire
    h_driver = d["h_driver"]
    h_refl = d["h_refl"]
    h_dir = d["h_dir"]
    spacing_m = d["spacing_m"]
    dir_spacing_m = d["dir_spacing_m"]
    n_directors = d["n_directors"]

    def _knots_at(x_pos: float, half_len: float) -> np.ndarray:
        return np.column_stack(
            [
                np.full(N + 1, x_pos),
                np.linspace(-half_len, half_len, N + 1),
                np.full(N + 1, z_offset),
            ]
        )

    knot_arrays = [_knots_at(0.0, h_driver), _knots_at(-spacing_m, h_refl)]
    labels = ["driver", "reflector"]
    for i in range(n_directors):
        knot_arrays.append(_knots_at((i + 1) * dir_spacing_m, h_dir))
        labels.append(f"director {i + 1}" if n_directors > 1 else "director")
    wires = _pack_momwire_wires(sim, coeffs, knot_arrays, labels)

    # Feed: TriangularSolver picks the interior knot of the driver closest to
    # the wire midpoint (= h_driver in arc length); for N segments along
    # [-h_driver, +h_driver] that's the middle interior knot at index N//2.
    feed_knot_index = N // 2

    return {
        "geometry": "yagi",
        "wires": wires,
        "feed_wire_index": 0,
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": d["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": d["wavelength_design"],
        "driver_length_m": 2 * h_driver,
        "reflector_length_m": 2 * h_refl,
        "spacing_m": spacing_m,
        "n_directors": n_directors,
        "director_length_m": 2 * h_dir if n_directors > 0 else None,
        "director_spacing_m": dir_spacing_m if n_directors > 0 else None,
        "solve_ms": solve_ms,
        "ground": ground_on,
        "height_m": z_offset,
        "ground_eps_r": _PEC_GROUND_EPS_R,
        "ground_sigma": _PEC_GROUND_SIGMA,
    }


def momwire_sweep(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    """Batched sweep using the momwire model's compute_impedance_swept."""
    from web.server import C_LIGHT, _make_momwire_sim, _read_ground

    ground_on, _, z_offset = _read_ground(req)
    d = _derive(req, z_offset)
    n_per_wire = d["n_per_wire"]

    sim = _make_momwire_sim(
        req,
        wires=d["polylines"],
        n_per_edge_per_wire=[[n_per_wire]] * len(d["polylines"]),
        feed_wire_index=0,
        wavelength=d["wavelength_design"],
        halfdriver_factor=d["driver_factor"],
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = d["wire_radius"]

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def pynec_build(req: dict) -> dict:
    """Build the PyNEC context + geometry for the Yagi (driver + reflector
    + n_directors uniformly-spaced directors)."""
    from web.pynec_backend import C_LIGHT, nec

    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    driver_factor = float(req.get("driver_length_factor", 0.962))
    refl_factor_abs = float(req.get("reflector_length_factor", 1.01))
    spacing_wavelengths = float(req.get("spacing_wavelengths", 0.15))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    ground_fast = bool(req.get("ground_fast", False))
    height_m = float(req.get("height_m", 0.0))
    n_directors = int(req.get("n_directors", 0))
    # Uniform director spacing between consecutive elements, in wavelengths.
    # Director length: size factor times the driver's halflength.
    dir_spacing_wl = float(req.get("director_spacing_wavelengths", 0.2))
    dir_size_factor = float(req.get("director_size_factor", 0.95))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    h_driver = driver_factor * wavelength_design / 4.0
    h_refl = refl_factor_abs * wavelength_design / 4.0
    spacing_m = spacing_wavelengths * wavelength_design
    dir_spacing_m = dir_spacing_wl * wavelength_design
    h_dir = dir_size_factor * h_driver
    z_offset = height_m if ground else 0.0

    c = nec.nec_context()
    geo = c.get_geometry()
    # Elements run along y. Driver (tag 1) at x=0, reflector (tag 2) at
    # x=-spacing, directors (tags 3..) at x = +i·dir_spacing toward the
    # beam direction (+x).
    geo.wire(
        1,
        n_per_wire,
        0.0,
        -h_driver,
        z_offset,
        0.0,
        h_driver,
        z_offset,
        wire_radius,
        1.0,
        1.0,
    )
    geo.wire(
        2,
        n_per_wire,
        -spacing_m,
        -h_refl,
        z_offset,
        -spacing_m,
        h_refl,
        z_offset,
        wire_radius,
        1.0,
        1.0,
    )
    for i in range(n_directors):
        x = (i + 1) * dir_spacing_m
        geo.wire(
            3 + i,
            n_per_wire,
            x,
            -h_dir,
            z_offset,
            x,
            h_dir,
            z_offset,
            wire_radius,
            1.0,
            1.0,
        )
    c.geometry_complete(0)

    return {
        "context": c,
        "feed_seg": (n_per_wire + 1) // 2,  # driver center segment
        "n_per_wire": n_per_wire,
        "h_driver": h_driver,
        "h_refl": h_refl,
        "h_dir": h_dir,
        "spacing_m": spacing_m,
        "dir_spacing_m": dir_spacing_m,
        "n_directors": n_directors,
        "wavelength_design": wavelength_design,
        "design_freq_mhz": design_freq_mhz,
        "ground": ground,
        "ground_fast": ground_fast,
        "z_offset": z_offset,
    }


def pynec_solve(req: dict) -> dict:
    """Yagi via PyNEC — elements along y, boom along x."""
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
    h_driver = b["h_driver"]
    h_refl = b["h_refl"]
    h_dir = b["h_dir"]
    spacing_m = b["spacing_m"]
    dir_spacing_m = b["dir_spacing_m"]
    n_directors = b["n_directors"]
    z_offset = b["z_offset"]
    n_wires_total = 2 + n_directors

    t0 = time.perf_counter()
    cur_arr, tag_arr = _run_solve(
        c,
        n_wires_total * n_per_wire,
        feed_seg,
        meas_freq_mhz,
        ground=b["ground"],
        ground_fast=b["ground_fast"],
    )
    solve_ms = (time.perf_counter() - t0) * 1e3

    driver_idx = np.where(tag_arr == 1)[0]
    fed_global = driver_idx[feed_seg - 1]
    z_in = complex(1.0 / cur_arr[fed_global])

    N = n_per_wire

    def _wire_record(x_pos: float, half_len: float, tag: int, label: str) -> dict:
        knots = np.column_stack(
            [
                np.full(N + 1, x_pos),
                np.linspace(-half_len, half_len, N + 1),
                np.full(N + 1, z_offset),
            ]
        )
        idx = np.where(tag_arr == tag)[0]
        cur = _segment_centers_to_knot_currents(cur_arr[idx], knots.shape[0])
        return {
            "label": label,
            "knot_positions": knots.tolist(),
            "knot_currents_re": cur.real.tolist(),
            "knot_currents_im": cur.imag.tolist(),
        }

    wires = [
        _wire_record(0.0, h_driver, 1, "driver"),
        _wire_record(-spacing_m, h_refl, 2, "reflector"),
    ]
    for i in range(n_directors):
        wires.append(
            _wire_record(
                (i + 1) * dir_spacing_m,
                h_dir,
                3 + i,
                f"director {i + 1}" if n_directors > 1 else "director",
            )
        )

    # Feed marker: middle interior knot of driver (matches TriangularSolver).
    interior_arc = np.linspace(0.0, 2 * h_driver, N + 1)[1:-1]
    m_center_interior = int(np.argmin(np.abs(interior_arc - h_driver)))
    feed_knot_index = m_center_interior + 1

    return {
        "geometry": "yagi",
        "wires": wires,
        "feed_wire_index": 0,
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": b["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": b["wavelength_design"],
        "driver_length_m": 2 * h_driver,
        "reflector_length_m": 2 * h_refl,
        "spacing_m": spacing_m,
        "n_directors": n_directors,
        "director_length_m": 2 * h_dir if n_directors > 0 else None,
        "director_spacing_m": dir_spacing_m if n_directors > 0 else None,
        "solve_ms": solve_ms,
        "solver": "pynec",
        "ground": b["ground"],
        "ground_fast": b["ground_fast"],
        "height_m": z_offset,
        "ground_eps_r": GROUND_DIELECTRIC,
        "ground_sigma": GROUND_CONDUCTIVITY,
    }


EXAMPLE = register(
    AntennaExample(
        name="yagi",
        label="Yagi",
        momwire_solve=momwire_solve,
        momwire_sweep=momwire_sweep,
        pynec_build=pynec_build,
        pynec_solve=pynec_solve,
        param_schema=(
            ParamSpec(
                name="driver_length_factor",
                label="driver length factor",
                default=0.962,
                min=0.5,
                max=1.2,
                step=0.001,
                precision=3,
            ),
            ParamSpec(
                name="reflector_length_factor",
                label="reflector length factor",
                default=1.01,
                min=0.5,
                max=1.2,
                step=0.001,
                precision=3,
            ),
            ParamSpec(
                name="spacing_wavelengths",
                label="spacing",
                default=0.15,
                min=0.05,
                max=0.5,
                step=0.001,
                precision=3,
                unit="λ",
            ),
            ParamSpec(
                name="n_directors",
                label="# directors",
                default=0,
                kind="int",
                min=0,
                max=8,
                step=1,
                precision=0,
            ),
            ParamSpec(
                name="director_spacing_wavelengths",
                label="director spacing",
                default=0.2,
                min=0.05,
                max=0.5,
                step=0.001,
                precision=3,
                unit="λ",
                visible_when={"name": "n_directors", "op": "gt", "value": 0},
            ),
            ParamSpec(
                name="director_size_factor",
                label="director size factor",
                default=0.95,
                min=0.5,
                max=1.2,
                step=0.001,
                precision=3,
                visible_when={"name": "n_directors", "op": "gt", "value": 0},
            ),
        ),
        result_schema=(
            ResultFieldSpec(
                field="driver_length_m", label="driver L", precision=3, unit=" m"
            ),
            ResultFieldSpec(
                field="reflector_length_m", label="reflector L", precision=3, unit=" m"
            ),
            ResultFieldSpec(field="spacing_m", label="spacing", precision=3, unit=" m"),
        ),
    )
)
