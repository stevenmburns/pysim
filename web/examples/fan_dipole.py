"""Fan dipole: multi-band cone-spread dipole sharing one feed.

Up to 5 bands, each a two-edge arm on each side (S→A_i→B_i mirrored to
T→A_neg_i→B_neg_i). All bands share the T→S feed gap. The cone geometry
spreads bands radially around the central feed axis; band lengths and
optional band frequencies come from the request.
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import register
from ._base import AntennaExample, ParamGroupSpec, ParamSpec, ResultFieldSpec  # noqa: F401

_FEED_GAP = 0.01  # meters; half-gap, matches antenna_designer eps


def _ring(k_bands: int) -> list[tuple[float, float]]:
    """K cone-direction ring positions evenly distributed at 360°/K around
    the cone axis. K=2 places the two bands at opposite ends of a diameter,
    K=3 at the vertices of an equilateral triangle, etc."""
    step = 360.0 / k_bands
    return [
        (math.cos(math.radians(i * step)), math.sin(math.radians(i * step)))
        for i in range(k_bands)
    ]


def _band_label(i: int, freqs_mhz: list[float], length_m: float) -> str:
    if i < len(freqs_mhz):
        return f"{freqs_mhz[i]:.2f} MHz"
    return f"band {i} ({length_m:.2f} m)"


def _bands_from_request(req: dict) -> tuple[list[float], list[float]]:
    """Extract per-band (freq_mhz, length_m) from the request.

    Schema-driven shape (preferred): `bands: [{freq, length_factor,
    ...ignored}, ...]`. n_bands is derived from len(bands), capped at
    the explicit n_bands field if present, then at 5.

    Legacy shape (test fixtures, antenna_designer parity tests): three
    parallel arrays `band_freqs_mhz` / `band_lengths_m` /
    `band_halfdriver_factors` with `n_bands` an explicit length. Used
    when `bands` is absent.
    """
    from web.server import C_LIGHT

    bands = req.get("bands")
    if bands:
        n_bands = int(req.get("n_bands", len(bands)))
        n_bands = min(n_bands, len(bands), 5)
        freqs = [float(b["freq"]) for b in bands[:n_bands]]
        factors = [float(b.get("length_factor", 0.962)) for b in bands[:n_bands]]
        # band_length_m = halfdriver_factor × λ / 2 (mirrors the JS
        # bandLengthFromFreqFactor at App.tsx:434).
        lengths = [
            f * (C_LIGHT / (freq * 1e6)) / 2.0 for freq, f in zip(freqs, factors)
        ]
        return freqs, lengths

    n_bands = int(req.get("n_bands", 2))
    band_lengths_m = list(req.get("band_lengths_m", [10.2551, 5.2691]))
    if len(band_lengths_m) < n_bands:
        raise ValueError(
            f"band_lengths_m has {len(band_lengths_m)} entries, need {n_bands}"
        )
    lengths = band_lengths_m[:n_bands]
    freqs = list(req.get("band_freqs_mhz", []))[:n_bands]
    return freqs, lengths


def _geometry(req: dict, z_offset: float):
    """Compute the cone-projected anchor points for all bands.

    Pulls just the geometric knobs from the request — band counts, slope,
    cone radius, lengths — and returns the S/T feed anchors plus the
    per-band (A, B) anchors on the +y and -y sides. Both pysim and pynec
    paths share this so the two backends stay geometrically locked.
    """
    n_per_wire = int(req.get("n_per_wire", 21))
    band_freqs_mhz, band_lengths_m = _bands_from_request(req)
    n_bands = len(band_lengths_m)
    if not 1 <= n_bands <= 5:
        raise ValueError(f"n_bands must be in [1, 5], got {n_bands}")
    slope = float(req.get("slope", 0.5))
    cone_radius_m = float(req.get("cone_radius_m", 0.12))
    t0_factor = float(req.get("t0_factor", math.sqrt(2.0)))

    eps_feed = _FEED_GAP
    t0 = cone_radius_m * t0_factor
    Zc = 1.0 / math.sqrt(1.0 + slope * slope)
    Zs = slope * Zc

    def ry(p):
        return (p[0], -p[1], p[2])

    S = (0.0, eps_feed, z_offset)
    T = ry(S)
    C = (S[0], S[1] + t0 * Zc, S[2] - t0 * Zs)
    lst = _ring(n_bands)

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

    return {
        "n_per_wire": n_per_wire,
        "n_bands": n_bands,
        "band_lengths_m": band_lengths_m,
        "band_freqs_mhz": band_freqs_mhz,
        "slope": slope,
        "cone_radius_m": cone_radius_m,
        "t0_m": t0,
        "S": S,
        "T": T,
        "A_pos": A_pos,
        "B_pos": B_pos,
        "A_neg": A_neg,
        "B_neg": B_neg,
    }


def _pysim_pack_wires(g, sim, coeffs, polyline_knots, pack_pysim_wires):
    n_bands = g["n_bands"]
    n_per = g["n_per_wire"]
    T, S = g["T"], g["S"]

    knot_arrays = [polyline_knots(np.array([T, S], dtype=float), [2])]
    labels = ["feed"]
    for side in ("+y", "-y"):
        for i in range(n_bands):
            if side == "+y":
                path = [g["S"], g["A_pos"][i], g["B_pos"][i]]
            else:
                path = [g["T"], g["A_neg"][i], g["B_neg"][i]]
            knot_arrays.append(polyline_knots(np.array(path), [n_per, n_per]))
            labels.append(
                f"{_band_label(i, g['band_freqs_mhz'], g['band_lengths_m'][i])} {side}"
            )
    return pack_pysim_wires(sim, coeffs, knot_arrays, labels)


def _pysim_build_sim_args(req: dict, z_offset: float, ground_on: bool):
    """Build the pysim wire/junction structures from the cone geometry."""
    g = _geometry(req, z_offset)
    n_per_wire = g["n_per_wire"]
    n_bands = g["n_bands"]

    # Wires:
    #   0:                       feed wire T -> S (2 segments)
    #   1..n_bands:              +y arms S -> A_i -> B_i
    #   n_bands+1..2*n_bands:    -y arms T -> A_neg_i -> B_neg_i
    wires = [np.array([g["T"], g["S"]], dtype=float)]
    n_per_edge = [[2]]
    for i in range(n_bands):
        wires.append(np.array([g["S"], g["A_pos"][i], g["B_pos"][i]], dtype=float))
        n_per_edge.append([n_per_wire, n_per_wire])
    for i in range(n_bands):
        wires.append(np.array([g["T"], g["A_neg"][i], g["B_neg"][i]], dtype=float))
        n_per_edge.append([n_per_wire, n_per_wire])

    # Junctions: K = 1 + n_bands wires meeting at each of S and T.
    j_S = [(0, "end")] + [(1 + i, "start") for i in range(n_bands)]
    j_T = [(0, "start")] + [(1 + n_bands + i, "start") for i in range(n_bands)]

    return {
        "g": g,
        "wires": wires,
        "n_per_edge": n_per_edge,
        "junctions": [j_S, j_T],
        "feed_arclength": _FEED_GAP,
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

    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, z_offset = _read_ground(req)
    build = _pysim_build_sim_args(req, z_offset, ground_on)
    g = build["g"]

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)

    sim = _make_pysim_sim(
        req,
        wires=build["wires"],
        n_per_edge_per_wire=build["n_per_edge"],
        feed_wire_index=0,
        feed_arclength=build["feed_arclength"],
        wavelength=wavelength_meas,
        nsegs=g["n_per_wire"],
        ground_z=0.0 if ground_on else None,
        junctions=build["junctions"],
    )
    sim.wire_radius = wire_radius

    t0_clock = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    return {
        "geometry": "fan_dipole",
        "wires": _pysim_pack_wires(g, sim, coeffs, _polyline_knots, _pack_pysim_wires),
        "feed_wire_index": 0,
        "feed_knot_index": 1,  # midpoint of the 3-knot feed wire record
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": wavelength_design,
        "n_bands": g["n_bands"],
        "band_lengths_m": list(g["band_lengths_m"]),
        "band_freqs_mhz": list(g["band_freqs_mhz"]),
        "slope": g["slope"],
        "cone_radius_m": g["cone_radius_m"],
        "t0_m": g["t0_m"],
        "solve_ms": solve_ms,
        "ground": ground_on,
        "height_m": z_offset,
        "ground_eps_r": _PEC_GROUND_EPS_R,
        "ground_sigma": _PEC_GROUND_SIGMA,
    }


def pysim_sweep(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    from web.server import C_LIGHT, _make_pysim_sim, _read_ground

    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, z_offset = _read_ground(req)
    build = _pysim_build_sim_args(req, z_offset, ground_on)
    g = build["g"]

    sim = _make_pysim_sim(
        req,
        wires=build["wires"],
        n_per_edge_per_wire=build["n_per_edge"],
        feed_wire_index=0,
        feed_arclength=build["feed_arclength"],
        wavelength=C_LIGHT / (design_freq_mhz * 1e6),
        nsegs=g["n_per_wire"],
        ground_z=0.0 if ground_on else None,
        junctions=build["junctions"],
    )
    sim.wire_radius = wire_radius

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def pynec_build(req: dict) -> dict:
    """Cone-arrangement fan dipole. Up to 5 bands, each a two-edge arm on
    each side (S->A_i->B_i mirrored to T->A_neg_i->B_neg_i). All bands
    share the T->S feed gap."""
    from web.pynec_backend import C_LIGHT, nec

    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    ground_fast = bool(req.get("ground_fast", False))
    height_m = float(req.get("height_m", 0.0))
    z_offset = height_m if ground else 0.0

    g = _geometry(req, z_offset)
    n_per_wire = g["n_per_wire"]
    n_bands = g["n_bands"]

    c = nec.nec_context()
    geo = c.get_geometry()
    # Tag 1: feed gap T->S (1 segment, delta-gap source location).
    geo.wire(1, 1, *g["T"], *g["S"], wire_radius, 1.0, 1.0)
    next_tag = 2
    band_tags = []
    for i in range(n_bands):
        t_sr, t_sa, t_tr, t_ta = next_tag, next_tag + 1, next_tag + 2, next_tag + 3
        geo.wire(t_sr, n_per_wire, *g["S"], *g["A_pos"][i], wire_radius, 1.0, 1.0)
        geo.wire(
            t_sa, n_per_wire, *g["A_pos"][i], *g["B_pos"][i], wire_radius, 1.0, 1.0
        )
        geo.wire(t_tr, n_per_wire, *g["T"], *g["A_neg"][i], wire_radius, 1.0, 1.0)
        geo.wire(
            t_ta, n_per_wire, *g["A_neg"][i], *g["B_neg"][i], wire_radius, 1.0, 1.0
        )
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
        "band_lengths_m": g["band_lengths_m"],
        "band_freqs_mhz": g["band_freqs_mhz"],
        "S": g["S"],
        "T": g["T"],
        "A_pos": g["A_pos"],
        "B_pos": g["B_pos"],
        "A_neg": g["A_neg"],
        "B_neg": g["B_neg"],
        "slope": g["slope"],
        "cone_radius_m": g["cone_radius_m"],
        "t0_m": g["t0_m"],
        "wavelength_design": wavelength_design,
        "design_freq_mhz": design_freq_mhz,
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
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )
    b = pynec_build(req)
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
    # interior knot to anchor feed_knot_index on. NEC's per-segment current
    # goes on the midpoint; the two end-knots are zero so the render's
    # open-wire convention holds.
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
        label = _band_label(i, b["band_freqs_mhz"], b["band_lengths_m"][i])
        path_pos = [S, b["A_pos"][i], b["B_pos"][i]]
        knots_pos = _path_knots(path_pos, [n_per, n_per])
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
        knots_neg = _path_knots(path_neg, [n_per, n_per])
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
        "feed_knot_index": 1,
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


# Amateur HF bands the per-band picker exposes. The keys are reused by
# the frontend's range_from_enum_option / on_change_set mechanisms — a
# sibling freq slider reads `freq_min`/`freq_max` from the active band
# entry, and snaps to `freq_default` when the band pulldown changes.
_BANDS_ENUM = (
    {
        "value": "20m",
        "label": "20m",
        "freq_min": 14.000,
        "freq_max": 14.350,
        "freq_default": 14.300,
    },
    {
        "value": "17m",
        "label": "17m",
        "freq_min": 18.068,
        "freq_max": 18.168,
        "freq_default": 18.1575,
    },
    {
        "value": "15m",
        "label": "15m",
        "freq_min": 21.000,
        "freq_max": 21.450,
        "freq_default": 21.383,
    },
    {
        "value": "12m",
        "label": "12m",
        "freq_min": 24.890,
        "freq_max": 24.990,
        "freq_default": 24.970,
    },
    {
        "value": "10m",
        "label": "10m",
        "freq_min": 28.000,
        "freq_max": 29.700,
        "freq_default": 28.470,
    },
)


EXAMPLE = register(
    AntennaExample(
        name="fan_dipole",
        label="Fan Dipole",
        pysim_solve=pysim_solve,
        pysim_sweep=pysim_sweep,
        pynec_build=pynec_build,
        pynec_solve=pynec_solve,
        # The input controls now fit the schema (the first user of
        # ParamGroupSpec — per-band repeat group with enum + dynamic
        # range + on_change side effect). Result panel still uses a
        # per-band repeat group on band_lengths_m[] / band_freqs_mhz[]
        # which isn't yet schema-modelled, so the legacy result block
        # in App.tsx stays for now.
        legacy_results=True,
        param_schema=(
            ParamSpec(
                name="n_bands",
                label="# bands",
                default=2,
                kind="int",
                min=1,
                max=5,
                step=1,
                precision=0,
            ),
            ParamGroupSpec(
                name="bands",
                label_template="band {i}",
                repeat_count="n_bands",
                max_repeats=5,
                # Touching any knob in band i → measFreq follows to
                # bands[i].freq when linkMeas is on, so the live solve
                # tracks whichever band the user is currently tuning.
                link_meas_freq_to_param="freq",
                # Per-instance defaults seed the 5 slots with the
                # historical FAN_BAND_IDS_DEFAULT order from App.tsx so
                # existing sessions look identical after the cutover.
                # Each band_id default carries the matching freq via
                # the enum's freq_default key (the on_change_set
                # mechanism wouldn't fire on initial seeding; instead
                # the frontend pre-resolves freq when seeding from a
                # default_override that names an enum value).
                default_overrides=(
                    {"band_id": "20m", "freq": 14.300},
                    {"band_id": "10m", "freq": 28.470},
                    {"band_id": "17m", "freq": 18.1575},
                    {"band_id": "12m", "freq": 24.970},
                    {"band_id": "15m", "freq": 21.383},
                ),
                params=(
                    ParamSpec(
                        name="band_id",
                        label="band",
                        default="20m",
                        kind="enum",
                        enum_options=_BANDS_ENUM,
                        on_change_set={"set": "freq", "from_enum_key": "freq_default"},
                    ),
                    ParamSpec(
                        name="freq",
                        label="freq",
                        default=14.300,
                        step=0.001,
                        precision=3,
                        unit=" MHz",
                        range_from_enum_option={
                            "param": "band_id",
                            "min_key": "freq_min",
                            "max_key": "freq_max",
                        },
                        # The first band's freq drives the global design
                        # frequency, replacing the standalone slider.
                        linked_to_design_freq=True,
                    ),
                    ParamSpec(
                        name="length_factor",
                        label="length factor",
                        default=0.962,
                        min=0.85,
                        max=1.05,
                        step=0.001,
                        precision=3,
                    ),
                ),
            ),
            ParamSpec(
                name="slope",
                label="cone slope",
                default=0.5,
                min=0.0,
                max=1.5,
                step=0.01,
                precision=3,
            ),
            ParamSpec(
                name="cone_radius_m",
                label="cone radius",
                default=0.12,
                min=0.05,
                max=0.5,
                step=0.005,
                precision=3,
                unit=" m",
            ),
        ),
    )
)
