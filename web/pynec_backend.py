"""PyNEC drop-in backend for the web UI.

Mirrors the response shape of `web.server`'s pysim solver paths so the
frontend can swap between solvers via a `solver` field on every request.

PyNEC is optional: `HAVE_PYNEC` is False if the import fails, and the
server falls back to pysim with a one-time warning.
"""

from __future__ import annotations

import time

import numpy as np

try:
    import PyNEC as nec  # type: ignore

    HAVE_PYNEC = True
except ImportError:
    HAVE_PYNEC = False
    nec = None


C_LIGHT = 299_792_458.0

# Typical "average" earth, matching antenna_designer/sim.py.
GROUND_DIELECTRIC = 10.0
GROUND_CONDUCTIVITY = 0.002


def _segment_centers_to_knot_currents(
    cur_per_seg: np.ndarray, n_knots: int
) -> np.ndarray:
    """Map NEC's per-segment-center currents onto the (n_knots,)-knot array
    the UI expects. Currents at the two end-knots are zero (open-wire BC);
    interior knot k sits between segments k-1 and k, so we average."""
    full = np.zeros(n_knots, dtype=np.complex128)
    # cur_per_seg has length n_knots - 1 (one current per segment).
    if cur_per_seg.shape[0] != n_knots - 1:
        raise RuntimeError(
            f"segment-current length {cur_per_seg.shape[0]} doesn't match "
            f"n_knots-1 = {n_knots - 1}"
        )
    full[1:-1] = 0.5 * (cur_per_seg[:-1] + cur_per_seg[1:])
    return full


def _run_solve(
    c, n_seg_total: int, feed_seg: int, freq_mhz: float, ground: bool = False
):
    if ground:
        # Sommerfeld-Norton finite ground (ITYPE=2) with average-earth constants.
        c.gn_card(2, 0, GROUND_DIELECTRIC, GROUND_CONDUCTIVITY, 0, 0, 0, 0)
    else:
        c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)  # free space
    c.ex_card(0, 1, feed_seg, 0, 1.0, 0.0, 0, 0, 0, 0)
    c.fr_card(0, 1, freq_mhz, 0)
    c.xq_card(0)
    sc = c.get_structure_currents(0)
    cur_arr = np.asarray(sc.get_current(), dtype=np.complex128)
    tag_arr = np.asarray(sc.get_current_segment_tag())
    return cur_arr, tag_arr


def _build_inverted_v(req: dict):
    """Build the PyNEC context + geometry for the inverted V. Returns the
    context, feed segment, and derived geometry that callers need for
    response formatting."""
    angle_deg = float(req.get("angle_deg", 30.0))
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    height_m = float(req.get("height_m", 0.0))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    arm_len = halfdriver_factor * wavelength_design / 4.0
    alpha = np.deg2rad(angle_deg)
    cos_a, sin_a = float(np.cos(alpha)), float(np.sin(alpha))
    # When ground is on, lift the entire geometry by height_m so the arms
    # stay above z=0 (NEC rejects segments at/below the ground plane).
    z_offset = height_m if ground else 0.0
    left = (-arm_len * cos_a, 0.0, z_offset - arm_len * sin_a)
    apex = (0.0, 0.0, z_offset)
    right = (arm_len * cos_a, 0.0, z_offset - arm_len * sin_a)

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
        "z_offset": z_offset,
    }


def solve_inverted_v(req: dict) -> dict:
    """Inverted V via two PyNEC wires meeting at the apex."""
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )
    b = _build_inverted_v(req)
    c = b["context"]
    n_per_wire = b["n_per_wire"]
    feed_seg = b["feed_seg"]

    t0 = time.perf_counter()
    cur_arr, tag_arr = _run_solve(
        c, 2 * n_per_wire, feed_seg, meas_freq_mhz, ground=b["ground"]
    )
    solve_ms = (time.perf_counter() - t0) * 1e3

    # z_in = 1 / current at the fed segment center.
    wire1_idx = np.where(tag_arr == 1)[0]
    fed_global_idx = wire1_idx[feed_seg - 1]
    z_in = complex(1.0 / cur_arr[fed_global_idx])

    # Build knot positions and currents, matching pysim's response shape:
    # one continuous wire (left arm reversed + right arm), feed at apex.
    arm1_knots = np.linspace(b["left"], b["apex"], n_per_wire + 1)
    arm2_knots = np.linspace(b["apex"], b["right"], n_per_wire + 1)
    knots = np.vstack([arm1_knots[:-1], arm2_knots])  # (2N+1, 3)

    # Per-wire segment currents.
    wire2_idx = np.where(tag_arr == 2)[0]
    cur_wire1 = cur_arr[wire1_idx]
    cur_wire2 = cur_arr[wire2_idx]
    # Concatenate currents in the same direction as knots.
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
        "height_m": b["z_offset"],
        "ground_eps_r": GROUND_DIELECTRIC,
        "ground_sigma": GROUND_CONDUCTIVITY,
    }


def _build_yagi(req: dict):
    """Build the PyNEC context + geometry for the Yagi (driver + reflector
    + n_directors uniformly-spaced directors)."""
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    driver_factor = float(req.get("driver_length_factor", 0.962))
    refl_factor_abs = float(req.get("reflector_length_factor", 1.01))
    spacing_wavelengths = float(req.get("spacing_wavelengths", 0.15))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    height_m = float(req.get("height_m", 0.0))
    n_directors = int(req.get("n_directors", 0))
    # Uniform director spacing (between consecutive elements) in wavelengths.
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
    # Driver (tag 1) at y=0, reflector (tag 2) at y=-spacing, directors
    # (tags 3..) at y = +i·dir_spacing toward the beam direction.
    geo.wire(
        1,
        n_per_wire,
        -h_driver,
        0.0,
        z_offset,
        h_driver,
        0.0,
        z_offset,
        wire_radius,
        1.0,
        1.0,
    )
    geo.wire(
        2,
        n_per_wire,
        -h_refl,
        -spacing_m,
        z_offset,
        h_refl,
        -spacing_m,
        z_offset,
        wire_radius,
        1.0,
        1.0,
    )
    for i in range(n_directors):
        y = (i + 1) * dir_spacing_m
        geo.wire(
            3 + i,
            n_per_wire,
            -h_dir,
            y,
            z_offset,
            h_dir,
            y,
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
        "z_offset": z_offset,
    }


def solve_yagi(req: dict) -> dict:
    """Yagi (driver + reflector + n_directors) — driver along x, reflector at
    -y, directors at increasing +y."""
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )
    b = _build_yagi(req)
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
        c, n_wires_total * n_per_wire, feed_seg, meas_freq_mhz, ground=b["ground"]
    )
    solve_ms = (time.perf_counter() - t0) * 1e3

    driver_idx = np.where(tag_arr == 1)[0]
    fed_global = driver_idx[feed_seg - 1]
    z_in = complex(1.0 / cur_arr[fed_global])

    N = n_per_wire

    def _wire_record(y_pos: float, half_len: float, tag: int, label: str) -> dict:
        knots = np.column_stack(
            [
                np.linspace(-half_len, half_len, N + 1),
                np.full(N + 1, y_pos),
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

    # Feed marker: middle interior knot of driver (matches TriangularYagiPySim).
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
        "height_m": z_offset,
        "ground_eps_r": GROUND_DIELECTRIC,
        "ground_sigma": GROUND_CONDUCTIVITY,
    }


def solve(req: dict) -> dict:
    geometry = req.get("geometry", "inverted_v")
    if geometry == "yagi":
        return solve_yagi(req)
    return solve_inverted_v(req)


def pattern(req: dict) -> dict:
    """NEC's `rp_card`-computed gain pattern over the upper hemisphere.

    Returns a (n_theta × n_phi) gain grid in dBi at θ ∈ [0°, 90°], full φ.
    With ground off, the lower hemisphere is symmetric to the upper for the
    flat geometries supported here, but we only ship the upper half — the
    UI mirrors as needed.
    """
    geometry = req.get("geometry", "inverted_v")
    b = _build_yagi(req) if geometry == "yagi" else _build_inverted_v(req)
    c = b["context"]
    feed_seg = b["feed_seg"]
    n_per_wire = b["n_per_wire"]
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )

    t0 = time.perf_counter()
    _run_solve(c, 2 * n_per_wire, feed_seg, meas_freq_mhz, ground=b["ground"])

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
        "height_m": b["z_offset"],
        "measurement_freq_mhz": meas_freq_mhz,
        "theta_deg": [ti * del_theta for ti in range(n_theta)],
        "phi_deg": [pi * del_phi for pi in range(n_phi)],
        "gain_dbi": gains,
        "pattern_ms": pattern_ms,
    }


def _sweep_at(req: dict, freq_mhz: float) -> complex:
    """Single-frequency Z via PyNEC, used to build the swept Z array."""
    req2 = dict(req)
    req2["measurement_freq_mhz"] = freq_mhz
    res = solve(req2)
    return complex(res["z_in_re"], res["z_in_im"])


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
