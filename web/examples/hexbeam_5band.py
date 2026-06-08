"""5-band hexbeam stack: N concentric hexbeams staggered along z, each
with its own feed.

Each band carries the same shape as the single-band hexbeam (driver +
reflector hexagonal layout with t0/t1 tip segments and a 1-segment T→S
feed gap), sized to its own band's wavelength via per-band
halfdriver_factor / tipspacer_factor / t0_factor. The N bands stack at
i × z_spacing_m along z so they share the same support footprint in
xy but don't physically intersect.

Multi-feed: each band's feed is driven independently with V = 1+0j.
The response carries one entry in `feeds[]` per band, exposing the per-
band driving-point impedance — same shape as the bowtie 1×2 array but
with up to 5 ports instead of 2.

This is the first multi-band-AND-multi-feed antenna registered. It
exercises both ParamGroupSpec (the per-band repeat group) and the
multi-feed Smith-chart + per-feed Z table machinery in one example.
"""

from __future__ import annotations

import time

import numpy as np

from . import register
from ._base import AntennaExample, ParamGroupSpec, ParamSpec, ResultFieldSpec

_FEED_GAP = 0.05  # meters; half-gap between feed knots T and S


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


# ---------------------------------------------------------------------------
# Geometry — one band
# ---------------------------------------------------------------------------


def _band_polylines(
    halfdriver: float,
    tipspacer_factor: float,
    t0_factor: float,
    n_per_long_edge: int,
    z_offset: float,
) -> dict:
    """Build one band's hexbeam (driver + reflector) at the given z.

    Lifted from web/examples/hexbeam.py's _polylines so each example
    file stays self-contained for easy deletion. If a future antenna
    needs a third shared copy, factor it into a helper module.
    """
    radius = halfdriver / (2 - t0_factor - tipspacer_factor)
    tipspacer = radius * tipspacer_factor
    t0 = radius * t0_factor
    t1 = radius - tipspacer - t0
    eps_feed = _FEED_GAP
    cos30 = float(np.sqrt(3) / 2)
    sin30 = 0.5

    def rx(p):
        return (-p[0], p[1], p[2])

    def ry(p):
        return (p[0], -p[1], p[2])

    A = (radius * cos30, radius * sin30, z_offset)
    B = (A[0] - t1 * cos30, A[1] + t1 * sin30, z_offset)
    D = (0.0, radius, z_offset)
    C = (D[0] + t0 * cos30, D[1] - t0 * sin30, z_offset)
    E = rx(A)
    F = ry(E)
    G = ry(D)
    H = ry(C)
    I_ = ry(B)
    J = ry(A)
    S = (eps_feed * cos30, eps_feed * sin30, z_offset)
    T = ry(S)

    driver = np.array([I_, J, T, S, A, B], dtype=float)
    reflector = np.array([C, D, E, F, G, H], dtype=float)

    def npe(anchors: np.ndarray) -> list[int]:
        out = []
        for i in range(anchors.shape[0] - 1):
            edge_len = float(np.linalg.norm(anchors[i + 1] - anchors[i]))
            if edge_len < 2 * eps_feed * 1.01:
                out.append(1)
            else:
                out.append(max(2, int(round(n_per_long_edge * edge_len / radius))))
        return out

    return {
        "driver": driver,
        "reflector": reflector,
        "npe_driver": npe(driver),
        "npe_reflector": npe(reflector),
        "feed_arclength": halfdriver,
        "radius_m": radius,
        "t0_m": t0,
        "t1_m": t1,
        "tipspacer_m": tipspacer,
        "driver_path_anchors": [I_, J, T, S, A, B],
        "reflector_path_anchors": [C, D, E, F, G, H],
    }


# ---------------------------------------------------------------------------
# Request unpacking
# ---------------------------------------------------------------------------


def _request_bands(req: dict) -> list[dict]:
    """Extract per-band parameter rows from the schema-driven request.

    `bands` in the request body is a list of dicts (one per active
    band) carrying freq / halfdriver_factor / tipspacer_factor /
    t0_factor — exactly what _band_polylines() needs. `n_bands` caps
    the list. Defaults track the single-band hexbeam's canonical
    28.47 MHz design knobs.
    """
    bands_in = req.get("bands") or []
    n_bands = int(req.get("n_bands", len(bands_in) or 1))
    n_bands = max(1, min(n_bands, len(bands_in) or n_bands, 5))
    out = []
    for i in range(n_bands):
        b = bands_in[i] if i < len(bands_in) else {}
        out.append(
            {
                "freq_mhz": float(b.get("freq", 14.300)),
                "halfdriver_factor": float(b.get("halfdriver_factor", 1.071)),
                "tipspacer_factor": float(b.get("tipspacer_factor", 0.1312)),
                "t0_factor": float(b.get("t0_factor", 0.1243)),
            }
        )
    return out


def _build_per_band_geometry(req: dict, z_offset_global: float) -> dict:
    """Common driver: pull bands + global knobs, build per-band polylines,
    and return the bundle both pysim and pynec paths consume.
    """
    from web.server import C_LIGHT

    bands = _request_bands(req)
    n_per_wire = int(req.get("n_per_wire", 21))
    z_spacing_m = float(req.get("z_spacing_m", 1.0))

    per_band = []
    for i, band in enumerate(bands):
        wavelength = C_LIGHT / (band["freq_mhz"] * 1e6)
        halfdriver = band["halfdriver_factor"] * wavelength / 4.0
        z = z_offset_global + i * z_spacing_m
        geom = _band_polylines(
            halfdriver,
            band["tipspacer_factor"],
            band["t0_factor"],
            n_per_wire,
            z,
        )
        geom["freq_mhz"] = band["freq_mhz"]
        geom["wavelength_design"] = wavelength
        geom["halfdriver_m"] = halfdriver
        geom["z_m"] = z
        per_band.append(geom)

    return {
        "n_per_wire": n_per_wire,
        "n_bands": len(bands),
        "z_spacing_m": z_spacing_m,
        "per_band": per_band,
    }


# ---------------------------------------------------------------------------
# pysim path
# ---------------------------------------------------------------------------


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

    design_freq_mhz = float(req.get("design_freq_mhz", 14.300))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, z_offset = _read_ground(req)

    g = _build_per_band_geometry(req, z_offset)
    per_band = g["per_band"]
    n_bands = g["n_bands"]

    # Flat wire list: band 0 (driver, reflector), band 1 (driver,
    # reflector), ... Band i's driver lives at wire index 2*i and
    # contributes one feed at its feed_arclength.
    wires: list = []
    n_per_edge: list = []
    feeds: list = []
    for i, band in enumerate(per_band):
        wires.append(band["driver"])
        wires.append(band["reflector"])
        n_per_edge.append(band["npe_driver"])
        n_per_edge.append(band["npe_reflector"])
        feeds.append((2 * i, band["feed_arclength"], 1.0 + 0.0j))

    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    sim = _make_pysim_sim(
        req,
        wires=wires,
        n_per_edge_per_wire=n_per_edge,
        feeds=feeds,
        wavelength=wavelength_meas,
        nsegs=g["n_per_wire"],
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = wire_radius

    t0_clock = time.perf_counter()
    z_per_feed, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    knots_per_wire = []
    wire_labels = []
    for i, band in enumerate(per_band):
        knots_per_wire.append(_polyline_knots(band["driver"], band["npe_driver"]))
        knots_per_wire.append(_polyline_knots(band["reflector"], band["npe_reflector"]))
        wire_labels.append(f"band {i} driver ({band['freq_mhz']:.2f} MHz)")
        wire_labels.append(f"band {i} reflector ({band['freq_mhz']:.2f} MHz)")
    wire_records = _pack_pysim_wires(sim, coeffs, knots_per_wire, wire_labels)

    z_arr = np.atleast_1d(z_per_feed)
    feed_entries = []
    for i, band in enumerate(per_band):
        # Feed knot is the interior knot of the driver wire closest to
        # feed_arclength — mirror the single-band hexbeam convention.
        driver_knots = knots_per_wire[2 * i]
        arc_at_knot = np.concatenate(
            [
                [0.0],
                np.cumsum(np.linalg.norm(np.diff(driver_knots, axis=0), axis=1)),
            ]
        )
        interior_arc = arc_at_knot[1:-1]
        feed_basis_local = int(np.argmin(np.abs(interior_arc - band["feed_arclength"])))
        zi = complex(z_arr[i])
        feed_entries.append(
            {
                "wire_index": 2 * i,
                "knot_index": feed_basis_local + 1,
                "z_re": float(zi.real),
                "z_im": float(zi.imag),
                "v_re": 1.0,
                "v_im": 0.0,
            }
        )

    primary = feed_entries[0]
    return {
        "geometry": "hexbeam_5band",
        "wires": wire_records,
        "feeds": feed_entries,
        "feed_wire_index": primary["wire_index"],
        "feed_knot_index": primary["knot_index"],
        "z_in_re": primary["z_re"],
        "z_in_im": primary["z_im"],
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": C_LIGHT / (design_freq_mhz * 1e6),
        "n_bands": n_bands,
        "z_spacing_m": g["z_spacing_m"],
        "solve_ms": solve_ms,
        "ground": ground_on,
        "height_m": z_offset,
        "ground_eps_r": _PEC_GROUND_EPS_R,
        "ground_sigma": _PEC_GROUND_SIGMA,
    }


def pysim_sweep(
    req: dict, freqs_mhz: list[float]
) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    """Multi-feed sweep returning the 4-tuple shape (primary_re,
    primary_im, feeds_re, feeds_im).

    Sweeps the geometry built at the *current* per-band design freqs
    over the requested measurement frequencies. Each frequency in the
    sweep is the measurement freq applied to the same physical antenna
    — moving a freq slider in the band group rebuilds geometry; this
    sweep just runs the existing geometry across many meas freqs.
    """
    from web.server import C_LIGHT, _make_pysim_sim, _read_ground

    design_freq_mhz = float(req.get("design_freq_mhz", 14.300))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, z_offset = _read_ground(req)

    g = _build_per_band_geometry(req, z_offset)
    per_band = g["per_band"]
    wires: list = []
    n_per_edge: list = []
    feeds: list = []
    for i, band in enumerate(per_band):
        wires.append(band["driver"])
        wires.append(band["reflector"])
        n_per_edge.append(band["npe_driver"])
        n_per_edge.append(band["npe_reflector"])
        feeds.append((2 * i, band["feed_arclength"], 1.0 + 0.0j))

    sim = _make_pysim_sim(
        req,
        wires=wires,
        n_per_edge_per_wire=n_per_edge,
        feeds=feeds,
        wavelength=C_LIGHT / (design_freq_mhz * 1e6),
        nsegs=g["n_per_wire"],
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = wire_radius

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


def pynec_build(req: dict) -> dict:
    """Build the PyNEC context: 10 wire cards per band (5 driver edges +
    5 reflector edges), each band's feed sits on its T→S edge.

    Tag numbering: band i uses tags `10*i + 1 .. 10*i + 10`. The
    driver edges are tags `10*i + 1..5` (in order I→J, J→T, T→S, S→A,
    A→B); reflector edges are `10*i + 6..10` (C→D, D→E, E→F, F→G,
    G→H). The feed sits on tag `10*i + 3` (the T→S edge).
    """
    from web.pynec_backend import C_LIGHT, nec

    n_per_wire = int(req.get("n_per_wire", 21))
    z_spacing_m = float(req.get("z_spacing_m", 1.0))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    ground_fast = bool(req.get("ground_fast", False))
    height_m = float(req.get("height_m", 0.0))
    z_offset_global = height_m if ground else 0.0
    design_freq_mhz = float(req.get("design_freq_mhz", 14.300))

    bands = _request_bands(req)
    c = nec.nec_context()
    geo = c.get_geometry()

    elements = []  # one per band; mirrors bowtie's elements[] convention
    tag = 0
    for band_idx, band in enumerate(bands):
        wavelength = C_LIGHT / (band["freq_mhz"] * 1e6)
        halfdriver = band["halfdriver_factor"] * wavelength / 4.0
        z = z_offset_global + band_idx * z_spacing_m
        bg = _band_polylines(
            halfdriver,
            band["tipspacer_factor"],
            band["t0_factor"],
            n_per_wire,
            z,
        )

        driver_anchors = bg["driver_path_anchors"]
        reflector_anchors = bg["reflector_path_anchors"]
        npe_d = bg["npe_driver"]
        npe_r = bg["npe_reflector"]

        driver_tags = []
        for i in range(5):
            tag += 1
            p0, p1 = driver_anchors[i], driver_anchors[i + 1]
            geo.wire(tag, npe_d[i], *p0, *p1, wire_radius, 1.0, 1.0)
            driver_tags.append(tag)
        reflector_tags = []
        for i in range(5):
            tag += 1
            p0, p1 = reflector_anchors[i], reflector_anchors[i + 1]
            geo.wire(tag, npe_r[i], *p0, *p1, wire_radius, 1.0, 1.0)
            reflector_tags.append(tag)

        elements.append(
            {
                "v_complex": complex(1.0, 0.0),
                "freq_mhz": band["freq_mhz"],
                "halfdriver_m": halfdriver,
                "wavelength_design": wavelength,
                "z_m": z,
                "driver_path": driver_anchors,
                "reflector_path": reflector_anchors,
                "npe_d": npe_d,
                "npe_r": npe_r,
                "driver_tags": driver_tags,
                "reflector_tags": reflector_tags,
                # T→S edge is the 3rd driver edge (index 2). Its single
                # segment carries the EX source.
                "feed_tag": driver_tags[2],
                "feed_seg": 1,
                "radius_m": bg["radius_m"],
                "t0_m": bg["t0_m"],
                "t1_m": bg["t1_m"],
                "tipspacer_m": bg["tipspacer_m"],
            }
        )
    c.geometry_complete(0)

    n_seg_total = sum(n for elem in elements for n in (*elem["npe_d"], *elem["npe_r"]))

    return {
        "context": c,
        "elements": elements,
        "n_per_wire": n_per_wire,
        "n_seg_total": n_seg_total,
        "z_spacing_m": z_spacing_m,
        "design_freq_mhz": design_freq_mhz,
        "wavelength_design": C_LIGHT / (design_freq_mhz * 1e6),
        "ground": ground,
        "ground_fast": ground_fast,
        "z_offset": z_offset_global,
    }


def _run_solve(b: dict, freq_mhz: float):
    """Multi-feed NEC solve. One EX card per band's T→S edge."""
    from web.pynec_backend import GROUND_CONDUCTIVITY, GROUND_DIELECTRIC

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
    """Drive the NEC context with all N feeds active so pattern()
    reflects the combined-excitation radiation, matching what
    pynec_solve() reports per feed."""
    _run_solve(b, freq_mhz)


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
        _segment_centers_to_knot_currents,
    )

    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.300))
    )
    b = pynec_build(req)

    t0 = time.perf_counter()
    cur_arr, tag_arr = _run_solve(b, meas_freq_mhz)
    solve_ms = (time.perf_counter() - t0) * 1e3

    wire_records: list[dict] = []
    feeds: list[dict] = []
    for band_idx, elem in enumerate(b["elements"]):
        # Driver record: concatenate currents across the 5 driver tags
        # in path order, then map to knots. Endpoints are at open
        # wire ends (no junctions in the single-band hexbeam topology),
        # so the default zero-boundary BC is correct.
        driver_knots = _path_knots(elem["driver_path"], elem["npe_d"])
        driver_cur = np.concatenate(
            [cur_arr[np.where(tag_arr == t)[0]] for t in elem["driver_tags"]]
        )
        driver_knot_cur = _segment_centers_to_knot_currents(
            driver_cur, driver_knots.shape[0]
        )
        wire_records.append(
            {
                "label": f"band {band_idx} driver ({elem['freq_mhz']:.2f} MHz)",
                "knot_positions": driver_knots.tolist(),
                "knot_currents_re": driver_knot_cur.real.tolist(),
                "knot_currents_im": driver_knot_cur.imag.tolist(),
            }
        )

        refl_knots = _path_knots(elem["reflector_path"], elem["npe_r"])
        refl_cur = np.concatenate(
            [cur_arr[np.where(tag_arr == t)[0]] for t in elem["reflector_tags"]]
        )
        refl_knot_cur = _segment_centers_to_knot_currents(refl_cur, refl_knots.shape[0])
        wire_records.append(
            {
                "label": f"band {band_idx} reflector ({elem['freq_mhz']:.2f} MHz)",
                "knot_positions": refl_knots.tolist(),
                "knot_currents_re": refl_knot_cur.real.tolist(),
                "knot_currents_im": refl_knot_cur.imag.tolist(),
            }
        )

        # Per-feed Z from V / I at the fed segment of this band's
        # driver. Each band's wire_records appears at index 2*band_idx
        # in the wires list (driver) / 2*band_idx+1 (reflector).
        feed_idx_in_tag = np.where(tag_arr == elem["feed_tag"])[0]
        i_feed = complex(cur_arr[feed_idx_in_tag[elem["feed_seg"] - 1]])
        v = elem["v_complex"]
        z_i = v / i_feed
        # The feed knot is between the T→S edge's start and end. With
        # 5 driver edges concatenated and the first 2 contributing
        # `npe_d[0] + npe_d[1]` segments, T sits at knot index
        # `sum(npe_d[:2])` and S at `sum(npe_d[:3])` (1 segment apart).
        feed_knot_index = sum(elem["npe_d"][:2])
        feeds.append(
            {
                "wire_index": 2 * band_idx,
                "knot_index": feed_knot_index,
                "z_re": float(z_i.real),
                "z_im": float(z_i.imag),
                "v_re": float(v.real),
                "v_im": float(v.imag),
            }
        )

    primary = feeds[0]
    return {
        "geometry": "hexbeam_5band",
        "wires": wire_records,
        "feeds": feeds,
        "feed_wire_index": primary["wire_index"],
        "feed_knot_index": primary["knot_index"],
        "z_in_re": primary["z_re"],
        "z_in_im": primary["z_im"],
        "design_freq_mhz": b["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": b["wavelength_design"],
        "n_bands": len(b["elements"]),
        "z_spacing_m": b["z_spacing_m"],
        "solve_ms": solve_ms,
        "solver": "pynec",
        "ground": b["ground"],
        "ground_fast": b["ground_fast"],
        "height_m": b["z_offset"],
        "ground_eps_r": GROUND_DIELECTRIC,
        "ground_sigma": GROUND_CONDUCTIVITY,
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


EXAMPLE = register(
    AntennaExample(
        name="hexbeam_5band",
        label="Hexbeam 5-band",
        pysim_solve=pysim_solve,
        pysim_sweep=pysim_sweep,
        pynec_build=pynec_build,
        pynec_solve=pynec_solve,
        pynec_pattern_excite=pynec_pattern_excite,
        multi_feed=True,
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
            ParamSpec(
                name="z_spacing_m",
                label="z spacing",
                default=1.0,
                min=0.2,
                max=3.0,
                step=0.05,
                precision=2,
                unit=" m",
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
                        # Band 0's freq drives the global design freq —
                        # same convention as fan_dipole.
                        linked_to_design_freq=True,
                    ),
                    ParamSpec(
                        name="halfdriver_factor",
                        label="halfdriver factor",
                        default=1.071,
                        min=0.9,
                        max=1.25,
                        step=0.001,
                        precision=3,
                    ),
                    ParamSpec(
                        name="tipspacer_factor",
                        label="tip spacer factor",
                        default=0.1312,
                        min=0.04,
                        max=0.25,
                        step=0.0005,
                        precision=4,
                    ),
                    ParamSpec(
                        name="t0_factor",
                        label="t0 factor",
                        default=0.1243,
                        min=0.04,
                        max=0.30,
                        step=0.001,
                        precision=4,
                    ),
                ),
            ),
        ),
        # Per-band geometry quantities (radius, t0, t1, tipspacer) would
        # need ResultGroupSpec to render — that's the next schema lift.
        # For now the result panel just shows the stack-wide scalars;
        # the per-band sizes are derivable from the band sliders + JS
        # if the user wants them visible.
        result_schema=(
            ResultFieldSpec(field="n_bands", label="# bands", precision=0),
            ResultFieldSpec(
                field="z_spacing_m", label="z spacing", precision=2, unit=" m"
            ),
        ),
    )
)
