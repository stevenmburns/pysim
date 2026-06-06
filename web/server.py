"""FastAPI server for the interactive antenna UI.

Supports four geometries (inverted_v, yagi, moxon, hexbeam), all solved by
the single TriangularPySim backend: each geometry just builds a list of
polyline wires + per-edge segment counts and feeds them in.

The response shape is uniform across geometries — each wire is a sequence of
knots with per-knot complex currents and the feed lives on one of the wires —
so the frontend draws every geometry the same way.

Run: uvicorn web.server:app --reload
"""

from __future__ import annotations

# Configure BLAS/OpenMP thread counts BEFORE numpy/scipy/PyNEC import — each
# library snapshots the env at its own import time and ignores later changes.
#
# OPENBLAS_NUM_THREADS=1: numpy/scipy bring their own OpenBLAS thread pool
#   that sits idle most of the request lifetime but contends with PyNEC's
#   MKL/OpenBLAS-LAPACKE pool for cores. vtune confirmed this was costing
#   ~8% wall time at NP=4 on the gather-scatter fill.
#
# OMP_NUM_THREADS / MKL_NUM_THREADS: with the gather-scatter matrix fill
#   (see PR #21) the per-source parallel-for inside cmset() and MKL/OpenBLAS'
#   zgetrf both want available cores. Default to the physical-core count
#   (not logical / HT count) — see _physical_cpu_count(); the FP-vector-
#   saturated quadrature inner loops gain nothing from HT siblings and
#   actually slow down ~15% from execution-unit contention on KBL-class
#   chips. An operator can override via the env to share with other
#   workloads on the same host.
#
# Older comment explaining why we used to pin everything to 1: the interactive
# workload is many small solves (≤ 250×250 dense complex matrices), and on
# the pre-gather-scatter code path thread orchestration costs dwarfed the
# per-call work — a 2-director live solve went from 220 ms (8 threads) to
# 67 ms (1 thread) on an 8-core box. That regression is no longer reproducible
# with the current build: matrix fill itself parallelizes, the OMP team is
# spawned once per cmset(), and OpenBLAS contention is removed by the
# OPENBLAS_NUM_THREADS=1 pin above.
import os


def _physical_cpu_count() -> int:
    """Number of physical cores (not logical / HT siblings).

    Our quadrature kernels are FP-vector-saturated (libmvec AVX2 sin/cos
    inner loops, no spare FU bandwidth), so two HT siblings on one physical
    core contend for execution units rather than overlap. Ad-hoc bench on
    KBL-R 4C/8T showed 4-thread runs ~15% faster than 8-thread runs of the
    swept-ground hot path. Pin to physical-core count to skip that loss.
    """
    try:
        cores = set()
        phys, coreid = None, None
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if key == "physical id":
                    phys = val
                elif key == "core id":
                    coreid = val
                elif not line.strip() and phys is not None and coreid is not None:
                    cores.add((phys, coreid))
                    phys, coreid = None, None
        if phys is not None and coreid is not None:
            cores.add((phys, coreid))
        if cores:
            return len(cores)
    except OSError:
        pass
    # Fallback: assume 2 HT siblings per core on x86. Wrong on chips without
    # HT, but in that case the caller can override via the env var.
    return max(1, (os.cpu_count() or 1) // 2)


_NPROC = str(_physical_cpu_count())
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", _NPROC)
os.environ.setdefault("MKL_NUM_THREADS", _NPROC)
# Between OMP parallel regions the workers default to busy-spinning on the
# team barrier (GOMP_SPINCOUNT ~300k, ~80 ms on KBL). Each pysim solve runs
# only ~1 ms of C++ kernel then ~10–20 ms of Python serial work (basis-coef
# build, sparse matmul, LU solve); the workers spin through all of that
# Python time on every solve. On the N=21 hentenna width-sweep harness
# (`scripts/vtune_hentenna_width_sweep.py`) VTune attributed ~63% of sin's
# CPU and ~32% of pynec's CPU to `libgomp` barrier-wait under this default.
# Make workers park immediately so the spin time goes away — wall-clock
# drops ~4× on both solvers at N=21 (sin 78 → 19 ms/step, pynec 67 → 9 ms).
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")

# ruff: noqa: E402 — imports below must follow the env-var setup above so
# OpenBLAS picks up the thread count at its own import time.
import json
import time

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pysim.bspline import BSplinePySim
from pysim.sinusoidal import SinusoidalPySim
from pysim.triangular import TriangularPySim

from . import pynec_backend


# Per-model option allowlist. Frontend sends `pysim_model` + `model_options`
# (a flat dict); we forward only the kwargs each class accepts, so an
# unrecognised option from a stale client never raises. Defaults match the
# class signatures so unset options behave identically to the old code.
_PYSIM_MODEL_KEYS = {
    "triangular": ("n_qp_reg", "n_qp_off"),
    "sinusoidal": ("n_qp_const",),
    "bspline": (
        "degree",
        "n_qp_pair",
        "n_qp_source",
        "feed_smoothing_factor",
        "use_singular_enrichment",
        "n_qp_sing",
        "enrichment_min_k",
        "enrichment_variant",
        "tikhonov_lambda",
        "auto_tap_ratio_threshold",
    ),
}
_PYSIM_MODELS = {
    "triangular": TriangularPySim,
    "sinusoidal": SinusoidalPySim,
    "bspline": BSplinePySim,
}


_PYSIM_MODELS_WITH_GROUND = {"triangular", "bspline", "sinusoidal"}


def _make_pysim_sim(req: dict, **base_kwargs):
    """Instantiate the PySim model the request selected.

    base_kwargs are the geometry-derived constructor kwargs every model
    accepts (wires, n_per_edge_per_wire, feed_*, wavelength, halfdriver_factor,
    nsegs, junctions). All three pysim models now accept ground_z; the set
    is kept as an allowlist so future models that don't support it can be
    excluded by name. model_options entries are filtered through the
    per-model allowlist.
    """
    model = req.get("pysim_model", "triangular")
    if model not in _PYSIM_MODELS:
        model = "triangular"
    cls = _PYSIM_MODELS[model]
    allowed = _PYSIM_MODEL_KEYS[model]

    opts = req.get("model_options") or {}
    extra = {k: opts[k] for k in allowed if k in opts}

    if model not in _PYSIM_MODELS_WITH_GROUND:
        base_kwargs.pop("ground_z", None)

    return cls(**base_kwargs, **extra)


# Target per-chunk wall time for the adaptive pysim /sweep chunking. The
# chunk size is tuned each iteration so a batch takes roughly this long —
# enough to amortise per-call overhead and benefit from numpy batching,
# small enough that an aborted fetch only wastes ~this much CPU before the
# next disconnect check kicks in.
_CHUNK_TARGET_MS = 500


app = FastAPI(title="pysim interactive")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


C_LIGHT = 299_792_458.0  # m/s, matches TriangularPySim's eps*mu derivation to ~1e-9


def _compute_directivity_norm(out: dict, n_theta: int = 45, n_phi: int = 90) -> None:
    """Attach `directivity_norm` = 4π / ∫|M_perp|² dΩ to the response.

    Multiplying this by the frontend's azimuth-cut |M_perp(π/2, φ)|² yields
    absolute directivity D(φ) (linear); 10·log10(D) is dBi.

    With ground enabled, integrates only the upper hemisphere and adds the
    Fresnel-reflected contribution from the geometric image so the
    normalization matches what the JS far-field code displays.
    """
    k = 2 * np.pi * out["measurement_freq_mhz"] * 1e6 / C_LIGHT
    ground_on = bool(out.get("ground", False))

    mids, drs, i_mids = [], [], []
    for w in out["wires"]:
        # Prefer the finer-grained sample arrays (knot + segment-midpoint)
        # when the model produced them, so non-tent bases get their intra-
        # segment curvature integrated. Falls back to knot arrays for any
        # backend that only ships knot data (PyNEC).
        if "sample_positions" in w:
            pts = np.asarray(w["sample_positions"], dtype=np.float64)
            cur = np.asarray(
                w["sample_currents_re"], dtype=np.float64
            ) + 1j * np.asarray(w["sample_currents_im"], dtype=np.float64)
        else:
            pts = np.asarray(w["knot_positions"], dtype=np.float64)
            cur = np.asarray(w["knot_currents_re"], dtype=np.float64) + 1j * np.asarray(
                w["knot_currents_im"], dtype=np.float64
            )
        drs.append(pts[1:] - pts[:-1])
        mids.append(0.5 * (pts[1:] + pts[:-1]))
        i_mids.append(0.5 * (cur[1:] + cur[:-1]))
    mid = np.concatenate(mids, axis=0)  # (Nseg, 3)
    dr = np.concatenate(drs, axis=0)  # (Nseg, 3)
    i_mid = np.concatenate(i_mids, axis=0)  # (Nseg,)

    # Cell-centered grid. With ground, sample only the upper hemisphere so
    # the integral is over the half-space the antenna actually radiates into.
    if ground_on:
        theta = (np.arange(n_theta) + 0.5) * (np.pi / 2 / n_theta)
        dtheta = np.pi / 2 / n_theta
    else:
        theta = (np.arange(n_theta) + 0.5) * (np.pi / n_theta)
        dtheta = np.pi / n_theta
    phi = np.arange(n_phi) * (2 * np.pi / n_phi)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)

    rx = sin_t[:, None] * cos_p[None, :]
    ry = sin_t[:, None] * sin_p[None, :]
    rz = np.broadcast_to(cos_t[:, None], (n_theta, n_phi))
    rhat = np.stack([rx, ry, rz], axis=-1)  # (nθ, nφ, 3)

    phase = k * np.einsum("ijc,nc->ijn", rhat, mid)  # (nθ, nφ, Nseg)
    expp = np.exp(1j * phase)
    weighted = i_mid[:, None] * dr  # (Nseg, 3)
    M = np.einsum("ijn,nc->ijc", expp, weighted)  # (nθ, nφ, 3)
    m_dot_r = np.sum(M * rhat, axis=-1)
    M_perp = M - m_dot_r[..., None] * rhat

    if ground_on:
        # PEC-image method, then Fresnel-correct the reflected wave per-ray.
        # Image current: horizontal components flipped, vertical preserved.
        # This reproduces PEC reflection when ρ_h=-1, ρ_v=+1, and lets us
        # apply the actual finite-ground coefficients to that same image.
        mid_img = mid * np.array([1.0, 1.0, -1.0])
        dr_img = dr * np.array([-1.0, -1.0, 1.0])
        weighted_img = i_mid[:, None] * dr_img
        phase_img = k * np.einsum("ijc,nc->ijn", rhat, mid_img)
        expp_img = np.exp(1j * phase_img)
        M_img = np.einsum("ijn,nc->ijc", expp_img, weighted_img)
        m_img_dot_r = np.sum(M_img * rhat, axis=-1)
        M_img_perp = M_img - m_img_dot_r[..., None] * rhat

        # Polarization basis at each ray: ĥ = ẑ × r̂ (perp to plane of
        # incidence), v̂ = r̂ × ĥ (in plane of incidence, perp to r̂).
        s = np.sqrt(rx * rx + ry * ry)
        s_safe = np.where(s > 1e-12, s, 1.0)
        h_hat = np.stack([-ry / s_safe, rx / s_safe, np.zeros_like(rx)], axis=-1)
        v_hat = np.stack([-rx * rz / s_safe, -ry * rz / s_safe, s], axis=-1)

        M_img_h = np.sum(M_img_perp * h_hat, axis=-1)  # complex (nθ, nφ)
        M_img_v = np.sum(M_img_perp * v_hat, axis=-1)

        eps0 = 8.854187817e-12
        omega = 2 * np.pi * out["measurement_freq_mhz"] * 1e6
        eps_c = out["ground_eps_r"] - 1j * out["ground_sigma"] / (omega * eps0)
        cos_ti = rz
        sin2_ti = s * s
        Q = np.sqrt(eps_c - sin2_ti)
        rho_h = (cos_ti - Q) / (cos_ti + Q)
        rho_v = (eps_c * cos_ti - Q) / (eps_c * cos_ti + Q)

        # Reflected: ρ_v on the v-pol component, −ρ_h on the h-pol component
        # (the minus sign folds the PEC image's pre-applied horizontal flip
        # back out so ρ_h=−1 recovers the PEC limit exactly).
        M_refl = (rho_v * M_img_v)[..., None] * v_hat - (rho_h * M_img_h)[
            ..., None
        ] * h_hat
        M_perp = M_perp + M_refl

    mag2 = np.sum((M_perp.real**2 + M_perp.imag**2), axis=-1)  # (nθ, nφ)

    dphi = 2 * np.pi / n_phi
    p_rad = float(np.sum(mag2 * sin_t[:, None]) * dtheta * dphi)
    out["directivity_norm"] = (4 * np.pi / p_rad) if p_rad > 0 else 0.0


def _wire_record(
    knots: np.ndarray,
    currents: np.ndarray,
    label: str,
    sample_currents: np.ndarray | None = None,
) -> dict:
    """Package one wire's record for the JSON response. `currents` is a
    length-M_w complex array (one per mesh knot) as produced by each
    model's `currents_at_knots(coeffs)` method.

    When `sample_currents` is provided, additional `sample_positions` /
    `sample_currents_re` / `sample_currents_im` arrays are attached at
    knots-and-midpoints interleaved (2*N_seg + 1 entries per wire). This is
    what `_compute_directivity_norm` and the frontend renderers consume to
    resolve intra-segment basis curvature (B-spline d=2, sinusoidal three-
    term) and the B-spline enrichment shape that vanishes at every knot.
    """
    currents = np.asarray(currents, dtype=np.complex128)
    if currents.shape[0] != knots.shape[0]:
        raise ValueError(
            f"_wire_record: currents/knots length mismatch "
            f"({currents.shape[0]} vs {knots.shape[0]})"
        )
    out = {
        "label": label,
        "knot_positions": knots.tolist(),
        "knot_currents_re": currents.real.tolist(),
        "knot_currents_im": currents.imag.tolist(),
    }
    if sample_currents is not None:
        sample_currents = np.asarray(sample_currents, dtype=np.complex128)
        n_seg = knots.shape[0] - 1
        expected = 2 * n_seg + 1
        if sample_currents.shape[0] != expected:
            raise ValueError(
                f"_wire_record: sample_currents length {sample_currents.shape[0]} "
                f"!= expected 2*N_seg+1 = {expected}"
            )
        sample_positions = np.empty((expected, 3), dtype=np.float64)
        sample_positions[0::2] = knots
        sample_positions[1::2] = 0.5 * (knots[:-1] + knots[1:])
        out["sample_positions"] = sample_positions.tolist()
        out["sample_currents_re"] = sample_currents.real.tolist()
        out["sample_currents_im"] = sample_currents.imag.tolist()
    return out


def _sample_arc_for_wire(knots: np.ndarray) -> np.ndarray:
    """Build interleaved (knot_arc, midpoint_arc, knot_arc, ...) array from a
    wire's 3D knot positions. Segment lengths come from successive-knot
    distances along the polyline.
    """
    knots = np.asarray(knots, dtype=np.float64)
    h_seg = np.linalg.norm(knots[1:] - knots[:-1], axis=1)
    arc_at_knot = np.concatenate([[0.0], np.cumsum(h_seg)])
    mid_arc = 0.5 * (arc_at_knot[:-1] + arc_at_knot[1:])
    sample_arc = np.empty(2 * h_seg.shape[0] + 1, dtype=np.float64)
    sample_arc[0::2] = arc_at_knot
    sample_arc[1::2] = mid_arc
    return sample_arc


def _pack_pysim_wires(sim, coeffs, knot_arrays, labels) -> list[dict]:
    """Build wire records for every pysim wire with both knot-level currents
    AND finer-grained mid-segment samples (one extra sample per segment).

    Calls `sim.currents_at_knots(coeffs)` once for the knot values and once
    more with an `s_array` of per-wire interleaved knot-and-midpoint arcs.
    The model's basis is then evaluated exactly at the midpoints — including
    the B-spline enrichment basis Φ_sing, which is zero at the knots but
    non-zero in the interior.
    """
    sample_arcs = [_sample_arc_for_wire(k) for k in knot_arrays]
    knot_currents = sim.currents_at_knots(coeffs)
    sample_currents = sim.currents_at_knots(coeffs, s_array=sample_arcs)
    return [
        _wire_record(
            np.asarray(knot_arrays[i]),
            knot_currents[i],
            labels[i],
            sample_currents=sample_currents[i],
        )
        for i in range(len(knot_arrays))
    ]


def _yagi_polylines(
    h_driver: float,
    h_refl: float,
    spacing_m: float,
    n_directors: int,
    dir_spacing_m: float,
    h_dir: float,
    z_offset: float = 0.0,
) -> list[np.ndarray]:
    """Driver + reflector + n_directors directors, all y-directed straight
    wires expressed as 2-anchor polylines for TriangularPySim.
    """
    polylines = [
        np.array([(0.0, -h_driver, z_offset), (0.0, h_driver, z_offset)]),
        np.array([(-spacing_m, -h_refl, z_offset), (-spacing_m, h_refl, z_offset)]),
    ]
    for i in range(n_directors):
        x = (i + 1) * dir_spacing_m
        polylines.append(np.array([(x, -h_dir, z_offset), (x, h_dir, z_offset)]))
    return polylines


def _inverted_v_polyline(
    arm_len: float, angle_deg: float, z_offset: float = 0.0
) -> np.ndarray:
    """Inverted-V with apex at z = z_offset and arms drooping in the yz plane.

    Arms run along ±y so the broadside null axis is ±x — matching the
    Yagi/moxon/hexbeam convention where the main lobe peaks at azimuth 0°
    (along +x). angle_deg is each arm's droop from horizontal: 0 = flat
    dipole, larger = more closed V.
    """
    alpha = np.deg2rad(angle_deg)
    cos_a, sin_a = float(np.cos(alpha)), float(np.sin(alpha))
    left = np.array([0.0, -arm_len * cos_a, z_offset - arm_len * sin_a])
    apex = np.array([0.0, 0.0, z_offset])
    right = np.array([0.0, arm_len * cos_a, z_offset - arm_len * sin_a])
    return np.vstack([left, apex, right])


# Pysim PEC ground: pass these to the response so the frontend's Fresnel
# far-field code treats the surface as a perfect electric conductor
# (ρ_h → −1, ρ_v → +1 in the eps_r → ∞ limit).
_PEC_GROUND_EPS_R = 1.0e10
_PEC_GROUND_SIGMA = 0.0


def _read_ground(req: dict) -> tuple[bool, float, float]:
    """Common request parsing: returns (ground_on, height_m, z_offset).

    height_m is the antenna height above ground when ground_on=True; z_offset
    is what each geometry helper adds to its native (z=0) coordinates.
    """
    ground_on = bool(req.get("ground", False))
    height_m = float(req.get("height_m", 0.0))
    z_offset = height_m if ground_on else 0.0
    return ground_on, height_m, z_offset


def _solve_inverted_v(req: dict) -> dict:
    angle_deg = float(req.get("angle_deg", 30.0))
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, height_m, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    arm_len = halfdriver_factor * wavelength_design / 4.0

    polyline = _inverted_v_polyline(arm_len, angle_deg, z_offset=z_offset)
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


def _solve_yagi(req: dict) -> dict:
    """Yagi (driver + reflector + optional directors), all parallel straight
    wires built as one-edge polylines and fed to TriangularPySim.

    Canonical layout for the UI:
        boom / spacing axis: +x (driver at x=0, reflector at x=-spacing,
                                  directors at x = +i·dir_spacing)
        element direction:   +y (each element runs from -half_len to +half_len
                                  along y)
        beam direction:      +x (away from reflector)
        z = 0 everywhere
    Aligns the Yagi convention with moxon and hexbeam (also +x beam).
    """
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    driver_factor = float(req.get("driver_length_factor", 0.962))
    refl_factor_abs = float(req.get("reflector_length_factor", 1.01))
    # Spacing in wavelengths of the design freq.
    spacing_wavelengths = float(req.get("spacing_wavelengths", 0.15))
    wire_radius = float(req.get("wire_radius", 0.0005))
    n_directors = int(req.get("n_directors", 0))
    dir_spacing_wl = float(req.get("director_spacing_wavelengths", 0.2))
    dir_size_factor = float(req.get("director_size_factor", 0.95))
    ground_on, height_m, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    h_driver = driver_factor * wavelength_design / 4.0
    h_refl = refl_factor_abs * wavelength_design / 4.0
    spacing_m = spacing_wavelengths * wavelength_design
    dir_spacing_m = dir_spacing_wl * wavelength_design
    h_dir = dir_size_factor * h_driver

    wires_polylines = _yagi_polylines(
        h_driver,
        h_refl,
        spacing_m,
        n_directors,
        dir_spacing_m,
        h_dir,
        z_offset=z_offset,
    )

    sim = _make_pysim_sim(
        req,
        wires=wires_polylines,
        n_per_edge_per_wire=[[n_per_wire]] * len(wires_polylines),
        feed_wire_index=0,
        wavelength=wavelength_meas,
        halfdriver_factor=driver_factor,
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = wire_radius

    t0 = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0) * 1e3

    N = n_per_wire

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
    wires = _pack_pysim_wires(sim, coeffs, knot_arrays, labels)

    # Feed: TriangularPySim picks the interior knot of the driver closest to
    # the wire midpoint (= h_driver in arc length). For N segments / N+1 knots
    # along [-h_driver, +h_driver], that's the middle interior knot at full-
    # list index N//2.
    feed_knot_index = N // 2

    return {
        "geometry": "yagi",
        "wires": wires,
        "feed_wire_index": 0,
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": wavelength_design,
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


_MOXON_FEED_GAP = 0.05  # meters; half-gap (eps) between feed knots T and S


def _moxon_polylines(
    halfdriver: float,
    aspect_ratio: float,
    tipspacer_factor: float,
    t0_factor: float,
    n_per_long_edge: int,
    z_offset: float = 0.0,
) -> dict:
    """Build the two moxon wires + per-edge segment counts.

    Driver polyline (5 edges, 6 anchors): bottom-tip -> bottom-corner ->
    feed-bot -> feed-top -> top-corner -> top-tip. Reflector polyline
    (3 edges, 4 anchors): top-tip -> top-corner -> bottom-corner -> bottom-tip.

    Segment counts: long vertical edges get `n_per_long_edge`; shorter edges
    scale proportionally with edge length (min 2). The feed gap gets 1.

    Returns: dict with driver/reflector polylines, per-edge segment counts,
    the feed arc-length on the driver wire (midpoint = T-S gap), and the
    derived rectangular dimensions.
    """
    long_ = 2 * halfdriver / (1 + 2 * aspect_ratio * t0_factor)
    short_ = aspect_ratio * long_
    tipspacer = short_ * tipspacer_factor
    t0 = short_ * t0_factor
    eps_feed = _MOXON_FEED_GAP

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

    long_edge_ref = long_ / 2 - eps_feed  # length of H->T / S->A

    def npe(anchors: np.ndarray) -> list[int]:
        out = []
        for i in range(anchors.shape[0] - 1):
            edge_len = float(np.linalg.norm(anchors[i + 1] - anchors[i]))
            # T->S feed gap is 2*eps_feed; mark it with 1 segment so the feed
            # sits on its own basis-pair boundary.
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
        # Total driver arc length is 2*t0 + 2*long_edge_ref + 2*eps_feed = 2*halfdriver;
        # the midpoint sits at the centre of the T-S feed gap.
        "feed_arclength": halfdriver,
        "long_m": long_,
        "short_m": short_,
        "tipspacer_m": tipspacer,
        "t0_m": t0,
    }


def _polyline_knots(polyline: np.ndarray, npe_list: list[int]) -> np.ndarray:
    """Concatenated per-edge knot positions, with shared corners deduped."""
    parts = []
    for i, n_e in enumerate(npe_list):
        seg = np.linspace(polyline[i], polyline[i + 1], n_e + 1)
        parts.append(seg if i == 0 else seg[1:])
    return np.vstack(parts)


def _solve_moxon(req: dict) -> dict:
    """Moxon (driver + reflector, both bent rectangular) via TriangularPySim."""
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.57))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    aspect_ratio = float(req.get("aspect_ratio", 0.3646))
    tipspacer_factor = float(req.get("tipspacer_factor", 0.0773))
    t0_factor = float(req.get("t0_factor", 0.4078))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, height_m, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0

    geom = _moxon_polylines(
        halfdriver,
        aspect_ratio,
        tipspacer_factor,
        t0_factor,
        n_per_wire,
        z_offset=z_offset,
    )

    sim = _make_pysim_sim(
        req,
        wires=[geom["driver"], geom["reflector"]],
        n_per_edge_per_wire=[geom["npe_driver"], geom["npe_reflector"]],
        feed_wire_index=0,
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_meas,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = wire_radius

    t0_clock = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    driver_knots = _polyline_knots(geom["driver"], geom["npe_driver"])
    refl_knots = _polyline_knots(geom["reflector"], geom["npe_reflector"])

    # Feed knot index on the driver wire: the interior knot the solver picked
    # (closest to feed_arclength). Recompute here to mark it in the response.
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
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": wavelength_design,
        "halfdriver_m": halfdriver,
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


_HEXBEAM_FEED_GAP = 0.05  # meters; half-gap (eps) between feed knots T and S


def _hexbeam_polylines(
    halfdriver: float,
    tipspacer_factor: float,
    t0_factor: float,
    n_per_long_edge: int,
    z_offset: float = 0.0,
) -> dict:
    """Build hexbeam driver + reflector polylines + per-edge segment counts.

    Hexagon has 6 shoulders at 30°, 90°, 150°, 210°, 270°, 330° from +x.
    Driver lives on the +x side: polyline II -> J -> T -> S -> A -> B
    (5 edges, 6 anchors). II/B are the bottom/top driver tips that point
    inward from the right shoulders J/A; T/S bracket the feed midway
    between J and A along the radius from the origin.
    Reflector wraps the -x side: polyline C -> D -> E -> F -> G -> H
    (5 edges, 6 anchors), with C/H pointing inward from the top/bottom
    apices D/G.

    All four "long" hexagon spokes are length `radius`; the t0/t1 tip
    pieces are shorter. Segment counts scale with edge length so segment
    density is roughly uniform; the feed gap gets 1 segment.

    By construction the driver's total arc length is 2 * halfdriver, so
    the feed sits at arc = halfdriver — equivalently the midpoint of
    the T-S edge.
    """
    radius = halfdriver / (2 - t0_factor - tipspacer_factor)
    tipspacer = radius * tipspacer_factor
    t0 = radius * t0_factor
    t1 = radius - tipspacer - t0
    eps_feed = _HEXBEAM_FEED_GAP
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
    }


def _solve_hexbeam(req: dict) -> dict:
    """Single-band hexbeam (driver + reflector, hexagonal layout)."""
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    halfdriver_factor = float(req.get("halfdriver_factor", 1.071))
    tipspacer_factor = float(req.get("tipspacer_factor", 0.1312))
    t0_factor = float(req.get("t0_factor", 0.1243))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, height_m, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0

    geom = _hexbeam_polylines(
        halfdriver,
        tipspacer_factor,
        t0_factor,
        n_per_wire,
        z_offset=z_offset,
    )

    sim = _make_pysim_sim(
        req,
        wires=[geom["driver"], geom["reflector"]],
        n_per_edge_per_wire=[geom["npe_driver"], geom["npe_reflector"]],
        feed_wire_index=0,
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_meas,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = wire_radius

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
        "geometry": "hexbeam",
        "wires": _pack_pysim_wires(
            sim, coeffs, [driver_knots, refl_knots], ["driver", "reflector"]
        ),
        "feed_wire_index": 0,
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": wavelength_design,
        "halfdriver_m": halfdriver,
        "radius_m": geom["radius_m"],
        "t0_m": geom["t0_m"],
        "t1_m": geom["t1_m"],
        "tipspacer_m": geom["tipspacer_m"],
        "solve_ms": solve_ms,
        "ground": ground_on,
        "height_m": z_offset,
        "ground_eps_r": _PEC_GROUND_EPS_R,
        "ground_sigma": _PEC_GROUND_SIGMA,
    }


_HENTENNA_FEED_GAP = 0.05  # meters; half-gap (eps) between feed knots T and S


def _hentenna_geometry(
    width_factor: float,
    top_height_factor: float,
    mid_height_factor: float,
    wavelength_design: float,
    n_per_long_edge: int,
    z_offset: float = 0.0,
) -> dict:
    """Build hentenna wires + per-edge segment counts + junctions.

    Single-band hentenna: a tall narrow rectangular loop with a horizontal
    cross-bar near the bottom, sliced into 5 wires that meet at K=2/K=3
    junctions. The feed sits in a small gap (T,S) at the centre of the
    cross-bar.

        C----------------------------A
        |                            |
        D------------T--S------------B
        |                            |
        E----------------------------F

    Five wires:
      0: T -> S                (feed gap)
      1: S -> B                (right half of cross-bar)
      2: B -> A -> C -> D      (upper rectangle perimeter)
      3: T -> D                (left half of cross-bar)
      4: D -> E -> F -> B      (lower rectangle perimeter)

    Junctions:
      at S: K=2 (wires 0,1)
      at T: K=2 (wires 0,3)
      at B: K=3 (wires 1,2,4)
      at D: K=3 (wires 2,3,4)
    """
    half_w = wavelength_design * width_factor / 2.0
    z_mid = wavelength_design * (mid_height_factor - top_height_factor) + z_offset
    z_top = z_offset
    z_bot = -wavelength_design * top_height_factor + z_offset
    eps_feed = _HENTENNA_FEED_GAP

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

    # Uniform N segments per non-feed edge, matching the antenna_designer
    # hentenna convention. Edge lengths differ by ~6× (cross-bar half vs
    # vertical), but length-scaled segmentation would either undersample
    # the cross-bar or wildly oversample the verticals — uniform-per-edge
    # avoids both and matches the reference NEC card layout.
    #
    # Feed-wire parity: pysim's tent basis carries the source on an
    # interior knot (the boundary between two adjacent segments). For
    # that knot to sit exactly on the gap's geometric centre we need an
    # EVEN segment count — n_feed=2 puts the single interior knot at
    # z=0; an odd n_feed offsets the feed by half a segment length.
    # Minimum 2 (n_feed=1 leaves no interior knot to feed on).
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
        "feed_arclength": eps_feed,  # midpoint of the 2*eps_feed feed gap
        "half_width_m": half_w,
        "top_height_m": wavelength_design * top_height_factor,
        "mid_offset_m": z_mid - z_top,
    }


def _solve_hentenna(req: dict) -> dict:
    """Single-band hentenna (narrow rectangular loop with cross-bar feed)."""
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    width_factor = float(req.get("width_factor", 0.1378))
    top_height_factor = float(req.get("top_height_factor", 0.5081))
    mid_height_factor = float(req.get("mid_height_factor", 0.1094))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, height_m, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)

    geom = _hentenna_geometry(
        width_factor,
        top_height_factor,
        mid_height_factor,
        wavelength_design,
        n_per_wire,
        z_offset=z_offset,
    )

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
    sim.wire_radius = wire_radius

    t0_clock = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    knots_per_wire = [
        _polyline_knots(w, npe)
        for w, npe in zip(geom["wires"], geom["n_per_edge_per_wire"])
    ]
    wire_labels = ["feed", "cross_right", "upper", "cross_left", "lower"]
    wire_records = _pack_pysim_wires(sim, coeffs, knots_per_wire, wire_labels)

    # Feed knot index on the feed wire: feed_arclength sits at the midpoint
    # of the single feed-gap edge, so it's the interior knot closest to
    # eps_feed.
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
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": wavelength_design,
        "half_width_m": geom["half_width_m"],
        "top_height_m": geom["top_height_m"],
        "mid_offset_m": geom["mid_offset_m"],
        "solve_ms": solve_ms,
        "ground": ground_on,
        "height_m": z_offset,
        "ground_eps_r": _PEC_GROUND_EPS_R,
        "ground_sigma": _PEC_GROUND_SIGMA,
    }


_BOWTIE_EPS = 0.05  # half-gap (m) at the feed and at the top-of-bowtie pinch


def _bowtie_element_wires(
    slope: float,
    length: float,
    n_per_long_edge: int,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
) -> dict:
    """Build a single bowtie's 4 polylines + per-edge segment counts.

    Replicates antenna_designer/designs/bowtie.py. The antenna lies in the
    y-z plane (x = 0), centred on (y, z) = (0, 0). With

        y = (length/2) / (slope + sqrt(1 + slope^2))
        z = slope * y

    the two triangles' tips are at (±y, 0); their open sides flare to
    (±y, ±z); each side closes to within an `eps` half-gap on the central
    axis at (±eps, ±eps). The feed sits across the bottom gap
    (-eps, -eps) → (eps, -eps).

    Decomposed into 4 pysim polylines (each kink is a continuous bend
    within a polyline so no junction is needed there) connected at 4
    K=2 junctions where the loops meet:

      W0  top arc:        (-y, 0)  -> (-y, z)  -> (-eps,  eps) ->
                          ( eps, eps) -> ( y, z) -> ( y, 0)
      W1  bot-left arc:   (-y, 0)  -> (-y,-z)  -> (-eps, -eps)
      W2  bot-right arc:  ( eps,-eps) -> ( y,-z) -> ( y, 0)
      W3  feed wire:      (-eps,-eps) -> ( eps,-eps)

      J0 at (-y, 0):      W0.start + W1.start
      J1 at ( y, 0):      W0.end   + W2.end
      J2 at (-eps,-eps):  W1.end   + W3.start
      J3 at ( eps,-eps):  W3.end   + W2.start
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

    # Per-edge segment counts: long flare edges get the full count; the
    # short eps-scale edges (top pinch + feed gap) get a few segments so the
    # feed-midpoint basis lands on its own knot. Match antenna_designer's
    # (n_seg0=21, n_seg1=3) ratio when n_per_long_edge=21.
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
    # Junction wire-indices here are LOCAL to this element (0..3); the array
    # wrapper renumbers them to global wire indices.
    junctions = [
        [(0, "start"), (1, "start")],  # J0 at (-y, 0)
        [(0, "end"), (2, "end")],  # J1 at ( y, 0)
        [(1, "end"), (3, "start")],  # J2 at (-eps,-eps)
        [(3, "end"), (2, "start")],  # J3 at ( eps,-eps)
    ]

    return {
        "wires": wires,
        "n_per_edge_per_wire": n_per_edge_per_wire,
        "junctions": junctions,
        "feed_wire_local": 3,
        "feed_arclength": eps_b,  # midpoint of the 2·eps feed gap
        "y_m": y,
        "z_m": z,
        "eps_m": eps_b,
    }


def _bowtie_array_1x2_geometry(
    slope: float,
    length: float,
    n_per_long_edge: int,
    del_y: float,
    phase_lr_deg: float,
    z_offset: float = 0.0,
) -> dict:
    """Build the 1×2 phased-bowtie array geometry.

    Two bowtie elements at y_offset = -del_y and +del_y, sharing the same
    slope/length. Element 0 (left, y < 0) is fed with V = 1+0j; element 1
    (right, y > 0) is fed with V = exp(j·π·phase_lr_deg/180). Mirrors
    `antenna_designer.builder.Array1x2Builder.build_wires`.

    All wires and junctions from the two elements are concatenated; the
    element-1 junction wire-indices are shifted by `n_wires_per_element`
    (=4 for the bowtie). The returned `feeds` list is what
    TriangularPySim(feeds=...) consumes.
    """
    elem_l = _bowtie_element_wires(
        slope, length, n_per_long_edge, y_offset=-del_y, z_offset=z_offset
    )
    elem_r = _bowtie_element_wires(
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


def _bowtie_request_args(req: dict) -> dict:
    """Pull the bowtie-1×2-array params off the request with defaults that
    match antenna_designer's canonical 28.47 MHz `bowtiearray1x2` design.
    """
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
        # 180 = anti-phase, matching antenna_designer's `phase_lr`.
        "phase_lr_deg": float(req.get("phase_lr_deg", 0.0)),
        "wire_radius": float(req.get("wire_radius", 0.0005)),
    }


def _bowtie_feed_knot_index(
    feed_wire_global: int, feed_arclength: float, knots_per_wire: list[np.ndarray]
) -> int:
    """Find the interior knot of `feed_wire_global` whose arc-length from the
    wire's start is closest to `feed_arclength`. Mirrors the
    TriangularPySim._feed_basis_index convention so the frontend marker
    lines up with the actual delta-gap source location.
    """
    feed_knots = knots_per_wire[feed_wire_global]
    arc_at_knot = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(feed_knots, axis=0), axis=1))]
    )
    interior_arc = arc_at_knot[1:-1]
    if len(interior_arc) == 0:
        return 0
    return int(np.argmin(np.abs(interior_arc - feed_arclength))) + 1


def _solve_bowtie(req: dict) -> dict:
    """1×2 phased bowtie array. Two bowtie elements at y = ±del_y, fed with
    V = (1, exp(j·π·phase_lr_deg/180)). Mirrors antenna_designer's
    bowtiearray1x2 design.

    Response carries per-feed driving-point impedances in `feeds[]`; the
    primary feed (element 0) is also exposed on the legacy
    `feed_wire_index` / `feed_knot_index` / `z_in_re` / `z_in_im` keys so
    single-feed-aware frontends keep working without change.
    """
    args = _bowtie_request_args(req)
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    ground_on, _, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    length_m = args["length_factor"] * wavelength_design

    geom = _bowtie_array_1x2_geometry(
        args["slope"],
        length_m,
        args["n_per_wire"],
        args["del_y_m"],
        args["phase_lr_deg"],
        z_offset=z_offset,
    )

    sim = _make_pysim_sim(
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
    wire_records = _pack_pysim_wires(sim, coeffs, knots_per_wire, wire_labels)

    z_arr = np.atleast_1d(z_per_feed)
    feed_entries = []
    for i, (w_global, arc, v_complex) in enumerate(geom["feeds"]):
        kidx = _bowtie_feed_knot_index(w_global, arc, knots_per_wire)
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
        # The bowtie 1×2 array is designed for 100 Ω feedlines per element
        # (matched via a 2:1 transformer to a 50 Ω line, or fed as a 100 Ω
        # parallel pair). Surface that as the Smith-chart / SWR reference
        # impedance instead of the default 50 Ω.
        "z0_ohms": 100.0,
        "solve_ms": solve_ms,
        "ground": ground_on,
        "height_m": z_offset,
        "ground_eps_r": _PEC_GROUND_EPS_R,
        "ground_sigma": _PEC_GROUND_SIGMA,
    }


def _sweep_bowtie(
    req: dict, freqs_mhz: list[float]
) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    """Sweep the 1×2 array. Returns (primary_re, primary_im, feeds_re,
    feeds_im) where `feeds_*` is (n_freqs × n_feeds) per-feed Z and the
    `primary_*` arrays carry feeds[0] for back-compat with the existing
    single-feed sweep stream shape.
    """
    args = _bowtie_request_args(req)
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    ground_on, _, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    length_m = args["length_factor"] * wavelength_design

    geom = _bowtie_array_1x2_geometry(
        args["slope"],
        length_m,
        args["n_per_wire"],
        args["del_y_m"],
        args["phase_lr_deg"],
        z_offset=z_offset,
    )
    sim = _make_pysim_sim(
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


_FANDIPOLE_FEED_GAP = 0.01  # half-gap, matches pynec_backend's eps_feed


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
            np.cos(np.deg2rad(i * step)),
            np.sin(np.deg2rad(i * step)),
        )
        for i in range(k_bands)
    ]


def _fandipole_geometry(req: dict):
    """Build polylines / per-edge segment counts / junctions for the pysim
    fan dipole solver, matching the geometry the PyNEC backend produces.

    Returns a dict with everything needed to construct a TriangularPySim
    and unpack the resulting coefficients into wire records.
    """
    n_per_wire = int(req.get("n_per_wire", 21))
    n_bands = int(req.get("n_bands", 2))
    if not 1 <= n_bands <= 5:
        raise ValueError(f"n_bands must be in [1, 5], got {n_bands}")
    band_lengths_m = list(req.get("band_lengths_m", [10.2551, 5.2691]))[:n_bands]
    if len(band_lengths_m) != n_bands:
        raise ValueError(
            f"band_lengths_m has {len(band_lengths_m)} entries, need {n_bands}"
        )
    band_freqs_mhz = list(req.get("band_freqs_mhz", []))[:n_bands]
    slope = float(req.get("slope", 0.5))
    cone_radius_m = float(req.get("cone_radius_m", 0.12))
    t0_factor = float(req.get("t0_factor", np.sqrt(2.0)))
    _, _, z_offset = _read_ground(req)

    eps_feed = _FANDIPOLE_FEED_GAP
    t0 = cone_radius_m * t0_factor
    Zc = 1.0 / np.sqrt(1.0 + slope * slope)
    Zs = slope * Zc

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
    ls = []
    for i, (q, a) in enumerate(zip(band_lengths_m, A_pos)):
        dsa = float(np.linalg.norm(np.subtract(S, a)))
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

    # Wires:
    #   0:                              feed wire T -> S (2 segments)
    #   1 .. n_bands:                   +y arms S -> A_i -> B_i
    #   n_bands+1 .. 2*n_bands:         -y arms T -> A_neg_i -> B_neg_i
    wires = [np.array([T, S], dtype=float)]
    n_per_edge = [[2]]
    for i in range(n_bands):
        wires.append(np.array([S, A_pos[i], B_pos[i]], dtype=float))
        n_per_edge.append([n_per_wire, n_per_wire])
    for i in range(n_bands):
        wires.append(np.array([T, A_neg[i], B_neg[i]], dtype=float))
        n_per_edge.append([n_per_wire, n_per_wire])

    # Junctions: K=1+n_bands wires meeting at each of S and T.
    j_S = [(0, "end")] + [(1 + i, "start") for i in range(n_bands)]
    j_T = [(0, "start")] + [(1 + n_bands + i, "start") for i in range(n_bands)]

    return {
        "wires": wires,
        "n_per_edge": n_per_edge,
        "junctions": [j_S, j_T],
        "feed_arclength": eps_feed,
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
        "z_offset": z_offset,
        "n_per_wire": n_per_wire,
    }


def _fandipole_pack_wires(g, sim, coeffs):
    """Package the fan-dipole wire records: feed wire (T->S, 2 segments)
    plus n_bands +y arms and n_bands -y arms, each starting at the shared
    junction node. `sim.currents_at_knots(coeffs)` returns one complex
    array per pysim wire; junction-directional bases at S and T flow
    through naturally into the wire-end knots.
    """
    n_bands = g["n_bands"]
    n_per = g["n_per_wire"]
    T, S = g["T"], g["S"]

    def _band_label(i):
        if i < len(g["band_freqs_mhz"]):
            return f"{g['band_freqs_mhz'][i]:.2f} MHz"
        return f"band {i} ({g['band_lengths_m'][i]:.2f} m)"

    knot_arrays = [_polyline_knots(np.array([T, S], dtype=float), [2])]
    labels = ["feed"]
    for side in ("+y", "-y"):
        for i in range(n_bands):
            if side == "+y":
                path = [g["S"], g["A_pos"][i], g["B_pos"][i]]
            else:
                path = [g["T"], g["A_neg"][i], g["B_neg"][i]]
            knot_arrays.append(_polyline_knots(np.array(path), [n_per, n_per]))
            labels.append(f"{_band_label(i)} {side}")
    return _pack_pysim_wires(sim, coeffs, knot_arrays, labels)


def _solve_fandipole(req: dict) -> dict:
    """Fan dipole via pysim's triangular Galerkin with junction support."""
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, z_offset = _read_ground(req)
    g = _fandipole_geometry(req)
    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)

    sim = _make_pysim_sim(
        req,
        wires=g["wires"],
        n_per_edge_per_wire=g["n_per_edge"],
        feed_wire_index=0,
        feed_arclength=g["feed_arclength"],
        wavelength=wavelength_meas,
        nsegs=g["n_per_wire"],
        ground_z=0.0 if ground_on else None,
        junctions=g["junctions"],
    )
    sim.wire_radius = wire_radius

    t0_clock = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    return {
        "geometry": "fan_dipole",
        "wires": _fandipole_pack_wires(g, sim, coeffs),
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


def _sweep_fandipole(
    req: dict, freqs_mhz: list[float]
) -> tuple[list[float], list[float]]:
    """Batched sweep using TriangularPySim.compute_impedance_swept."""
    g = _fandipole_geometry(req)
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, _ = _read_ground(req)
    sim = _make_pysim_sim(
        req,
        wires=g["wires"],
        n_per_edge_per_wire=g["n_per_edge"],
        feed_wire_index=0,
        feed_arclength=g["feed_arclength"],
        wavelength=C_LIGHT / (design_freq_mhz * 1e6),
        nsegs=g["n_per_wire"],
        ground_z=0.0 if ground_on else None,
        junctions=g["junctions"],
    )
    sim.wire_radius = wire_radius
    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def solve(req: dict) -> dict:
    geometry = req.get("geometry", "inverted_v")
    use_pynec = req.get("solver") == "pynec" and pynec_backend.HAVE_PYNEC
    if use_pynec:
        out = pynec_backend.solve(req)
        _compute_directivity_norm(out)
        return out
    if geometry == "yagi":
        out = _solve_yagi(req)
    elif geometry == "moxon":
        out = _solve_moxon(req)
    elif geometry == "hexbeam":
        out = _solve_hexbeam(req)
    elif geometry == "fan_dipole":
        out = _solve_fandipole(req)
    elif geometry == "hentenna":
        out = _solve_hentenna(req)
    elif geometry == "bowtie":
        out = _solve_bowtie(req)
    else:
        out = _solve_inverted_v(req)
    out["solver"] = "pysim"
    _compute_directivity_norm(out)
    return out


def _sweep_inverted_v(
    req: dict, freqs_mhz: list[float]
) -> tuple[list[float], list[float]]:
    """Batched sweep using TriangularPySim.compute_impedance_swept."""
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
        wires=[_inverted_v_polyline(arm_len, angle_deg, z_offset=z_offset)],
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


def _sweep_yagi(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    """Batched sweep using TriangularPySim.compute_impedance_swept."""
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    driver_factor = float(req.get("driver_length_factor", 0.962))
    refl_factor_abs = float(req.get("reflector_length_factor", 1.01))
    spacing_wavelengths = float(req.get("spacing_wavelengths", 0.15))
    wire_radius = float(req.get("wire_radius", 0.0005))
    n_directors = int(req.get("n_directors", 0))
    dir_spacing_wl = float(req.get("director_spacing_wavelengths", 0.2))
    dir_size_factor = float(req.get("director_size_factor", 0.95))
    ground_on, _, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    h_driver = driver_factor * wavelength_design / 4.0
    h_refl = refl_factor_abs * wavelength_design / 4.0
    spacing_m = spacing_wavelengths * wavelength_design
    dir_spacing_m = dir_spacing_wl * wavelength_design
    h_dir = dir_size_factor * h_driver

    wires_polylines = _yagi_polylines(
        h_driver,
        h_refl,
        spacing_m,
        n_directors,
        dir_spacing_m,
        h_dir,
        z_offset=z_offset,
    )
    sim = _make_pysim_sim(
        req,
        wires=wires_polylines,
        n_per_edge_per_wire=[[n_per_wire]] * len(wires_polylines),
        feed_wire_index=0,
        wavelength=wavelength_design,
        halfdriver_factor=driver_factor,
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = wire_radius

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def _sweep_moxon(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    """Batched sweep using TriangularPySim.compute_impedance_swept."""
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.57))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    aspect_ratio = float(req.get("aspect_ratio", 0.3646))
    tipspacer_factor = float(req.get("tipspacer_factor", 0.0773))
    t0_factor = float(req.get("t0_factor", 0.4078))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0

    geom = _moxon_polylines(
        halfdriver,
        aspect_ratio,
        tipspacer_factor,
        t0_factor,
        n_per_wire,
        z_offset=z_offset,
    )
    sim = _make_pysim_sim(
        req,
        wires=[geom["driver"], geom["reflector"]],
        n_per_edge_per_wire=[geom["npe_driver"], geom["npe_reflector"]],
        feed_wire_index=0,
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_design,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = wire_radius

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def _sweep_hexbeam(
    req: dict, freqs_mhz: list[float]
) -> tuple[list[float], list[float]]:
    """Batched sweep using TriangularPySim.compute_impedance_swept."""
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    halfdriver_factor = float(req.get("halfdriver_factor", 1.071))
    tipspacer_factor = float(req.get("tipspacer_factor", 0.1312))
    t0_factor = float(req.get("t0_factor", 0.1243))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0

    geom = _hexbeam_polylines(
        halfdriver,
        tipspacer_factor,
        t0_factor,
        n_per_wire,
        z_offset=z_offset,
    )
    sim = _make_pysim_sim(
        req,
        wires=[geom["driver"], geom["reflector"]],
        n_per_edge_per_wire=[geom["npe_driver"], geom["npe_reflector"]],
        feed_wire_index=0,
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_design,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
    )
    sim.wire_radius = wire_radius

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def _sweep_hentenna(
    req: dict, freqs_mhz: list[float]
) -> tuple[list[float], list[float]]:
    """Batched sweep using TriangularPySim.compute_impedance_swept."""
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    width_factor = float(req.get("width_factor", 0.1378))
    top_height_factor = float(req.get("top_height_factor", 0.5081))
    mid_height_factor = float(req.get("mid_height_factor", 0.1094))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground_on, _, z_offset = _read_ground(req)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)

    geom = _hentenna_geometry(
        width_factor,
        top_height_factor,
        mid_height_factor,
        wavelength_design,
        n_per_wire,
        z_offset=z_offset,
    )
    sim = _make_pysim_sim(
        req,
        wires=geom["wires"],
        n_per_edge_per_wire=geom["n_per_edge_per_wire"],
        feed_wire_index=geom["feed_wire_index"],
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_design,
        nsegs=n_per_wire,
        ground_z=0.0 if ground_on else None,
        junctions=geom["junctions"],
    )
    sim.wire_radius = wire_radius

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


@app.post("/sweep")
async def sweep_endpoint(req: dict, request: Request):
    """Stream sweep points as NDJSON, one (freq, Z) per line.

    Streaming so the UI can show partial results as they're computed, and
    so the server can stop mid-sweep when the client disconnects — without
    this the user's slider drags abort the fetch client-side but the server
    keeps grinding through all 41 expensive PyNEC ground solves, starving
    the live /ws solves of CPU.
    """
    freqs = [float(f) for f in req.get("freqs_mhz", [])]
    geometry = req.get("geometry", "inverted_v")
    use_pynec = req.get("solver") == "pynec" and pynec_backend.HAVE_PYNEC
    solver_name = "pynec" if use_pynec else "pysim"

    async def gen():
        if not freqs:
            yield json.dumps({"done": True, "solver": solver_name}) + "\n"
            return

        if use_pynec:
            # Per-point loop with disconnect check; lets us bail before the
            # next ~100 ms PyNEC ground solve when the user moves a slider.
            for f in freqs:
                if await request.is_disconnected():
                    return
                z = await run_in_threadpool(pynec_backend._sweep_at, req, f)
                yield (
                    json.dumps(
                        {
                            "freq_mhz": f,
                            "z_re": float(z.real),
                            "z_im": float(z.imag),
                            "solver": solver_name,
                        }
                    )
                    + "\n"
                )
        else:
            # pysim's batched sweep is ~10x faster per-call than per-point,
            # but a 5-band fan dipole sweep at n_per_wire=21, 41 freqs takes
            # ~6 s and holds several hundred MB of J tensors — long enough
            # that rapid slider drags would otherwise pile up concurrent
            # computes in the threadpool, exhausting threads or memory and
            # surfacing as a 500 at the Vite proxy.
            #
            # Chunk the sweep so we can check is_disconnected between
            # batches. Per-freq cost has a bowl curve in chunk size:
            # tiny chunks pay per-call overhead, huge chunks thrash memory
            # bandwidth. For the 5-band fan-dipole geometry the sweet spot
            # is chunk_size ≈ 8 (115 ms/freq); for an inverted V it's much
            # larger (single-digit ms/freq, all freqs in one go is fine).
            #
            # Aim each chunk at roughly _CHUNK_TARGET_MS so the cancellation
            # granularity is consistent across geometries. Start with an
            # 8-chunk heuristic, then after each chunk recompute the next
            # size from observed per-freq cost. Converges in ~1 iteration.
            sweep_fn = {
                "yagi": _sweep_yagi,
                "moxon": _sweep_moxon,
                "hexbeam": _sweep_hexbeam,
                "fan_dipole": _sweep_fandipole,
                "hentenna": _sweep_hentenna,
                "bowtie": _sweep_bowtie,
            }.get(geometry, _sweep_inverted_v)
            chunk_size = max(1, len(freqs) // 8)
            start = 0
            while start < len(freqs):
                if await request.is_disconnected():
                    return
                chunk = freqs[start : start + chunk_size]
                t0 = time.perf_counter()
                sweep_result = await run_in_threadpool(sweep_fn, req, chunk)
                # Multi-feed sweeps (bowtie array) return a 4-tuple with
                # per-feed Z appended. Everything else stays on the
                # original 2-tuple shape; the legacy z_re / z_im fields
                # always carry the primary feed for back-compat.
                feeds_re_chunk: list[list[float]] | None = None
                feeds_im_chunk: list[list[float]] | None = None
                if len(sweep_result) == 4:
                    z_re, z_im, feeds_re_chunk, feeds_im_chunk = sweep_result
                else:
                    z_re, z_im = sweep_result
                chunk_ms = (time.perf_counter() - t0) * 1000
                for i, f in enumerate(chunk):
                    record: dict = {
                        "freq_mhz": f,
                        "z_re": z_re[i],
                        "z_im": z_im[i],
                        "solver": solver_name,
                    }
                    if feeds_re_chunk is not None:
                        record["feeds_z_re"] = feeds_re_chunk[i]
                        record["feeds_z_im"] = feeds_im_chunk[i]
                    yield json.dumps(record) + "\n"
                start += len(chunk)
                # Adapt for the next chunk: target _CHUNK_TARGET_MS per
                # batch. Per-freq cost is a weak function of chunk size
                # (bowl curve), so this converges quickly.
                if chunk_ms > 0 and len(chunk) > 0:
                    per_freq_ms = chunk_ms / len(chunk)
                    chunk_size = max(1, round(_CHUNK_TARGET_MS / per_freq_ms))

        yield json.dumps({"done": True, "solver": solver_name}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _solve_z_only(req: dict) -> complex:
    """Run the geometry-specific solver and return only the input impedance.

    Skips the directivity-norm integral that `solve()` would otherwise tack
    on — for the /converge sweep we only need Z(N), and at N ≳ 60 the
    directivity step adds non-negligible cost.
    """
    geometry = req.get("geometry", "inverted_v")
    use_pynec = req.get("solver") == "pynec" and pynec_backend.HAVE_PYNEC
    if use_pynec:
        res = pynec_backend.solve(req)
    elif geometry == "yagi":
        res = _solve_yagi(req)
    elif geometry == "moxon":
        res = _solve_moxon(req)
    elif geometry == "hexbeam":
        res = _solve_hexbeam(req)
    elif geometry == "fan_dipole":
        res = _solve_fandipole(req)
    elif geometry == "hentenna":
        res = _solve_hentenna(req)
    elif geometry == "bowtie":
        res = _solve_bowtie(req)
    else:
        res = _solve_inverted_v(req)
    return complex(res["z_in_re"], res["z_in_im"])


@app.post("/converge")
async def converge_endpoint(req: dict, request: Request):
    """Stream impedance vs segments/wire as NDJSON, one (n, Z) per line.

    The frontend passes `n_values: list[int]`; we re-solve the geometry at
    each N (overriding `n_per_wire`) and yield the result before starting
    the next solve. Streaming so the user sees the trajectory build up
    incrementally — the largest-N solves take noticeably longer (~N³ for
    the dense LU) and the user shouldn't have to wait for the whole sweep
    to see early points.

    Cancels on client disconnect (slider drag interrupts a stale sweep)
    using the same pattern as /sweep.
    """
    n_values = [int(n) for n in req.get("n_values", [])]
    use_pynec = req.get("solver") == "pynec" and pynec_backend.HAVE_PYNEC
    solver_name = "pynec" if use_pynec else "pysim"

    async def gen():
        for n in n_values:
            if await request.is_disconnected():
                return
            req_n = dict(req)
            req_n["n_per_wire"] = n
            try:
                z = await run_in_threadpool(_solve_z_only, req_n)
            except Exception as e:
                # One-off solver failures (e.g. degenerate geometry at very
                # small N) shouldn't abort the whole sweep — note the error
                # for this N and keep going.
                yield (
                    json.dumps(
                        {
                            "n_per_wire": n,
                            "error": str(e),
                            "solver": solver_name,
                        }
                    )
                    + "\n"
                )
                continue
            yield (
                json.dumps(
                    {
                        "n_per_wire": n,
                        "z_re": float(z.real),
                        "z_im": float(z.imag),
                        "solver": solver_name,
                    }
                )
                + "\n"
            )
        yield json.dumps({"done": True, "solver": solver_name}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/pattern")
async def pattern_endpoint(req: dict):
    """NEC's rp_card-computed gain pattern. PyNEC-only."""
    if req.get("solver") != "pynec" or not pynec_backend.HAVE_PYNEC:
        return {"available": False}
    return await run_in_threadpool(pynec_backend.pattern, req)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            req = json.loads(raw)
            result = await run_in_threadpool(solve, req)
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        return
