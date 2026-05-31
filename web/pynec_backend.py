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
    ground_fast = bool(req.get("ground_fast", False))
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
        "ground_fast": ground_fast,
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
        c,
        2 * n_per_wire,
        feed_seg,
        meas_freq_mhz,
        ground=b["ground"],
        ground_fast=b["ground_fast"],
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
        "ground_fast": b["ground_fast"],
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
    ground_fast = bool(req.get("ground_fast", False))
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


def solve_yagi(req: dict) -> dict:
    """Yagi (driver + reflector + n_directors) — elements along y, boom
    along x. Driver at x=0, reflector at x=-spacing, directors at
    increasing +x. Beam direction +x."""
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
        "ground_fast": b["ground_fast"],
        "height_m": z_offset,
        "ground_eps_r": GROUND_DIELECTRIC,
        "ground_sigma": GROUND_CONDUCTIVITY,
    }


_MOXON_FEED_GAP = 0.05  # meters, half-gap between feed knots T and S


def _build_moxon(req: dict):
    """Build the PyNEC context + geometry for the moxon.

    Driver path (5 NEC wires, tags 1..5): G->H, H->T, T->S (feed),
    S->A, A->B. Reflector (3 NEC wires, tags 6..8): C->D, D->E, E->F.
    The T->S edge carries a 1-segment feed; its segment center is the
    delta-gap source.
    """
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
    eps_feed = _MOXON_FEED_GAP
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
        "feed_seg": 1,  # its only segment
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


def _polyline_knots(path, npe_list) -> np.ndarray:
    """Concatenated per-edge knot positions, deduping shared corners."""
    parts = []
    for i, n_e in enumerate(npe_list):
        seg = np.linspace(path[i], path[i + 1], n_e + 1)
        parts.append(seg if i == 0 else seg[1:])
    return np.vstack(parts)


def solve_moxon(req: dict) -> dict:
    """Moxon via PyNEC: 5 driver wires + 3 reflector wires, fed on T->S."""
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 28.57))
    )
    b = _build_moxon(req)
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

    # z_in from the current at the fed segment's center.
    feed_idx_in_tag3 = np.where(tag_arr == b["feed_tag"])[0]
    fed_global = feed_idx_in_tag3[b["feed_seg"] - 1]
    z_in = complex(1.0 / cur_arr[fed_global])

    # Combine driver (tags 1..5) and reflector (tags 6..8) into one polyline
    # of currents each, then map onto knot positions with open-wire BC at
    # the actual physical wire ends (G/B for driver, C/F for reflector).
    cur_driver = np.concatenate(
        [cur_arr[np.where(tag_arr == t)[0]] for t in range(1, 6)]
    )
    cur_refl = np.concatenate([cur_arr[np.where(tag_arr == t)[0]] for t in range(6, 9)])

    driver_knots = _polyline_knots(b["driver_path"], b["npe_d"])
    refl_knots = _polyline_knots(b["reflector_path"], b["npe_r"])
    driver_knot_cur = _segment_centers_to_knot_currents(
        cur_driver, driver_knots.shape[0]
    )
    refl_knot_cur = _segment_centers_to_knot_currents(cur_refl, refl_knots.shape[0])

    # Feed knot on the driver: the knot between the T->S edge's start and
    # end. driver_path[2] = T, driver_path[3] = S. After deduping, T is at
    # index sum(npe_d[:2]); S is at sum(npe_d[:3]) (= same + 1 since
    # the T->S edge has 1 segment).
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


_HEXBEAM_FEED_GAP = 0.05


def _build_hexbeam(req: dict):
    """Build the PyNEC context + geometry for a single-band hexbeam.

    10 NEC wires total: driver edges (tags 1..5) follow the polyline
    II -> J -> T -> S -> A -> B, with the feed on the 1-segment T->S
    edge (tag 3). Reflector edges (tags 6..10) follow C -> D -> E -> F
    -> G -> H.
    """
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    halfdriver_factor = float(req.get("halfdriver_factor", 1.071))
    tipspacer_factor = float(req.get("tipspacer_factor", 0.1312))
    t0_factor = float(req.get("t0_factor", 0.1243))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    ground_fast = bool(req.get("ground_fast", False))
    height_m = float(req.get("height_m", 0.0))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0
    radius = halfdriver / (2 - t0_factor - tipspacer_factor)
    tipspacer = radius * tipspacer_factor
    t0 = radius * t0_factor
    t1 = radius - tipspacer - t0
    eps_feed = _HEXBEAM_FEED_GAP
    z_offset = height_m if ground else 0.0
    cos30 = float(np.sqrt(3) / 2)
    sin30 = 0.5

    def rx(p):
        return (-p[0], p[1], p[2])

    def ry(p):
        return (p[0], -p[1], p[2])

    A = (radius * cos30, radius * sin30, z_offset)
    B = (A[0] - t1 * cos30, A[1] + t1 * sin30, z_offset)
    D = (0.0, radius, z_offset)
    Cc = (D[0] + t0 * cos30, D[1] - t0 * sin30, z_offset)
    E = rx(A)
    F = ry(E)
    G = ry(D)
    H = ry(Cc)
    I_ = ry(B)
    J = ry(A)
    S = (eps_feed * cos30, eps_feed * sin30, z_offset)
    T = ry(S)

    def npe_edge(p, q) -> int:
        edge_len = float(np.linalg.norm(np.subtract(q, p)))
        if edge_len < 2 * eps_feed * 1.01:
            return 1
        return max(2, int(round(n_per_wire * edge_len / radius)))

    npe_d = [
        npe_edge(I_, J),
        npe_edge(J, T),
        1,
        npe_edge(S, A),
        npe_edge(A, B),
    ]
    npe_r = [
        npe_edge(Cc, D),
        npe_edge(D, E),
        npe_edge(E, F),
        npe_edge(F, G),
        npe_edge(G, H),
    ]

    driver_path = [I_, J, T, S, A, B]
    reflector_path = [Cc, D, E, F, G, H]

    c = nec.nec_context()
    geo = c.get_geometry()
    for i in range(5):
        p0, p1 = driver_path[i], driver_path[i + 1]
        geo.wire(i + 1, npe_d[i], *p0, *p1, wire_radius, 1.0, 1.0)
    for i in range(5):
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
        "radius_m": radius,
        "t0_m": t0,
        "t1_m": t1,
        "tipspacer_m": tipspacer,
        "ground": ground,
        "ground_fast": ground_fast,
        "z_offset": z_offset,
    }


def solve_hexbeam(req: dict) -> dict:
    """Single-band hexbeam via PyNEC."""
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 28.47))
    )
    b = _build_hexbeam(req)
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
    cur_refl = np.concatenate(
        [cur_arr[np.where(tag_arr == t)[0]] for t in range(6, 11)]
    )

    driver_knots = _polyline_knots(b["driver_path"], b["npe_d"])
    refl_knots = _polyline_knots(b["reflector_path"], b["npe_r"])
    driver_knot_cur = _segment_centers_to_knot_currents(
        cur_driver, driver_knots.shape[0]
    )
    refl_knot_cur = _segment_centers_to_knot_currents(cur_refl, refl_knots.shape[0])

    feed_knot_index = sum(b["npe_d"][:2])

    return {
        "geometry": "hexbeam",
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
        "radius_m": b["radius_m"],
        "t0_m": b["t0_m"],
        "t1_m": b["t1_m"],
        "tipspacer_m": b["tipspacer_m"],
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
    if geometry == "yagi":
        return solve_yagi(req)
    if geometry == "moxon":
        return solve_moxon(req)
    if geometry == "hexbeam":
        return solve_hexbeam(req)
    return solve_inverted_v(req)


def pattern(req: dict) -> dict:
    """NEC's `rp_card`-computed gain pattern over the upper hemisphere.

    Returns a (n_theta × n_phi) gain grid in dBi at θ ∈ [0°, 90°], full φ.
    With ground off, the lower hemisphere is symmetric to the upper for the
    flat geometries supported here, but we only ship the upper half — the
    UI mirrors as needed.
    """
    geometry = req.get("geometry", "inverted_v")
    if geometry == "yagi":
        b = _build_yagi(req)
    elif geometry == "moxon":
        b = _build_moxon(req)
    elif geometry == "hexbeam":
        b = _build_hexbeam(req)
    else:
        b = _build_inverted_v(req)
    c = b["context"]
    feed_seg = b["feed_seg"]
    n_per_wire = b["n_per_wire"]
    feed_tag = b.get("feed_tag", 1)
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )

    t0 = time.perf_counter()
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
