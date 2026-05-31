"""FastAPI server for the interactive antenna UI.

Supports two geometries:
- inverted_v: a single bent wire (BentTriangularPySim).
- yagi:      driver + reflector, parallel straight wires (TriangularYagiPySim).

Both return a uniform "wire list" response so the frontend draws either
geometry the same way: each wire is a sequence of knots with per-knot complex
currents; the feed lives on one of the wires.

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
#   zgetrf both want available cores. Default to all logical cores; an
#   operator can override via the env if they want to share with other
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

_NPROC = str(os.cpu_count() or 1)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", _NPROC)
os.environ.setdefault("MKL_NUM_THREADS", _NPROC)

# ruff: noqa: E402 — imports below must follow the env-var setup above so
# OpenBLAS picks up the thread count at its own import time.
import json
import time

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pysim.triangular_bent import BentTriangularPySim
from pysim.triangular_bent_multi import BentMultiPySim
from pysim.triangular_yagi import TriangularYagiPySim

from . import pynec_backend


app = FastAPI(title="pysim interactive")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


C_LIGHT = 299_792_458.0  # m/s, matches AbstractPySim's eps*mu derivation to ~1e-9


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


def _wire_record(knots: np.ndarray, coeffs: np.ndarray, label: str) -> dict:
    """Pad interior-knot coefficients with zero endpoints (open-wire BC) and
    package one wire's record for the JSON response.
    """
    n_knots = knots.shape[0]
    full = np.zeros(n_knots, dtype=np.complex128)
    full[1:-1] = coeffs
    return {
        "label": label,
        "knot_positions": knots.tolist(),
        "knot_currents_re": full.real.tolist(),
        "knot_currents_im": full.imag.tolist(),
    }


def _inverted_v_polyline(arm_len: float, angle_deg: float) -> np.ndarray:
    """Inverted-V with apex at the origin and arms drooping in the xz plane.

    angle_deg is each arm's droop from horizontal: 0 = flat dipole, larger =
    more closed V.
    """
    alpha = np.deg2rad(angle_deg)
    cos_a, sin_a = float(np.cos(alpha)), float(np.sin(alpha))
    left = np.array([-arm_len * cos_a, 0.0, -arm_len * sin_a])
    apex = np.array([0.0, 0.0, 0.0])
    right = np.array([arm_len * cos_a, 0.0, -arm_len * sin_a])
    return np.vstack([left, apex, right])


def _solve_inverted_v(req: dict) -> dict:
    angle_deg = float(req.get("angle_deg", 30.0))
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    wire_radius = float(req.get("wire_radius", 0.0005))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    arm_len = halfdriver_factor * wavelength_design / 4.0

    sim = BentTriangularPySim(
        wavelength=wavelength_meas,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
    )
    sim.wire_radius = wire_radius
    polyline = _inverted_v_polyline(arm_len, angle_deg)
    sim.polyline = polyline
    sim.n_per_edge = [n_per_wire, n_per_wire]

    t0 = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0) * 1e3

    knots = np.vstack(
        [
            np.linspace(polyline[0], polyline[1], n_per_wire + 1)[:-1],
            np.linspace(polyline[1], polyline[2], n_per_wire + 1),
        ]
    )
    feed_knot_index = n_per_wire  # apex (midpoint of polyline)

    return {
        "geometry": "inverted_v",
        "wires": [_wire_record(knots, coeffs, "wire")],
        "feed_wire_index": 0,
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": wavelength_design,
        "arm_len_m": arm_len,
        "solve_ms": solve_ms,
    }


def _solve_yagi(req: dict) -> dict:
    """Two-element Yagi (driver + reflector).

    Canonical layout for the UI:
        boom / spacing axis: +x (driver at x=0, reflector at x=-spacing,
                                  directors at x = +i·dir_spacing)
        element direction:   +y (each element runs from -half_len to +half_len
                                  along y)
        beam direction:      +x (away from reflector)
        z = 0 everywhere
    This matches the internal TriangularYagiPySim geometry exactly, so the
    response wires pass through unchanged. Aligns the Yagi convention with
    moxon and hexbeam (also +x beam).
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

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    h_driver = driver_factor * wavelength_design / 4.0
    h_refl = refl_factor_abs * wavelength_design / 4.0
    spacing_m = spacing_wavelengths * wavelength_design
    dir_spacing_m = dir_spacing_wl * wavelength_design
    h_dir = dir_size_factor * h_driver

    # TriangularYagiPySim parameters: factors are relative to the driver's
    # halflength. Director spacing is between consecutive elements.
    refl_factor_rel = h_refl / h_driver
    spacing_factor_rel = spacing_m / h_driver
    dir_spacing_factor_rel = dir_spacing_m / h_driver

    sim = TriangularYagiPySim(
        wavelength=wavelength_meas,
        halfdriver_factor=driver_factor,
        nsegs=n_per_wire,
        reflector_factor=refl_factor_rel,
        spacing_factor=spacing_factor_rel,
        n_directors=n_directors,
        director_spacing_factor=dir_spacing_factor_rel,
        director_size_factor=dir_size_factor,
    )
    sim.wire_radius = wire_radius
    # Decouple geometry from measurement wavelength: the solver computes
    # halfdriver from `wavelength` at construction time, which by default
    # ties the antenna size to measurement freq. Override to fix it to
    # design freq so meas-freq sweeps probe a stationary antenna.
    sim.halfdriver = h_driver

    t0 = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0) * 1e3

    # Canonical layout: elements along y, spacing along x — same as
    # TriangularYagiPySim's internal geometry, so no transpose is needed.
    N = n_per_wire
    nb = N - 1

    def _knots_at(x_pos: float, half_len: float) -> np.ndarray:
        return np.column_stack(
            [
                np.full(N + 1, x_pos),
                np.linspace(-half_len, half_len, N + 1),
                np.zeros(N + 1),
            ]
        )

    wires = [
        _wire_record(_knots_at(0.0, h_driver), coeffs[:nb], "driver"),
        _wire_record(_knots_at(-spacing_m, h_refl), coeffs[nb : 2 * nb], "reflector"),
    ]
    for i in range(n_directors):
        sl = slice((2 + i) * nb, (3 + i) * nb)
        wires.append(
            _wire_record(
                _knots_at((i + 1) * dir_spacing_m, h_dir),
                coeffs[sl],
                f"director {i + 1}" if n_directors > 1 else "director",
            )
        )

    # Feed: TriangularYagiPySim picks the interior knot of the driver closest
    # to L_driver/2 — that's the middle interior knot, at full-list index N//2
    # when interior-list index is (N-2)//2. Reproduce the same logic here.
    interior_arc = np.linspace(0.0, 2 * h_driver, N + 1)[1:-1]
    m_center_interior = int(np.argmin(np.abs(interior_arc - h_driver)))
    feed_knot_index = m_center_interior + 1  # shift past left endpoint

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
    """Moxon (driver + reflector, both bent rectangular) via BentMultiPySim."""
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.57))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    aspect_ratio = float(req.get("aspect_ratio", 0.3646))
    tipspacer_factor = float(req.get("tipspacer_factor", 0.0773))
    t0_factor = float(req.get("t0_factor", 0.4078))
    wire_radius = float(req.get("wire_radius", 0.0005))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0

    geom = _moxon_polylines(
        halfdriver,
        aspect_ratio,
        tipspacer_factor,
        t0_factor,
        n_per_wire,
    )

    sim = BentMultiPySim(
        wires=[geom["driver"], geom["reflector"]],
        n_per_edge_per_wire=[geom["npe_driver"], geom["npe_reflector"]],
        feed_wire_index=0,
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_meas,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
    )
    sim.wire_radius = wire_radius

    t0_clock = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    driver_knots = _polyline_knots(geom["driver"], geom["npe_driver"])
    refl_knots = _polyline_knots(geom["reflector"], geom["npe_reflector"])
    nb_d = sum(geom["npe_driver"]) - 1
    nb_r = sum(geom["npe_reflector"]) - 1

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
        "wires": [
            _wire_record(driver_knots, coeffs[:nb_d], "driver"),
            _wire_record(refl_knots, coeffs[nb_d : nb_d + nb_r], "reflector"),
        ],
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

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0

    geom = _hexbeam_polylines(
        halfdriver,
        tipspacer_factor,
        t0_factor,
        n_per_wire,
    )

    sim = BentMultiPySim(
        wires=[geom["driver"], geom["reflector"]],
        n_per_edge_per_wire=[geom["npe_driver"], geom["npe_reflector"]],
        feed_wire_index=0,
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_meas,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
    )
    sim.wire_radius = wire_radius

    t0_clock = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    driver_knots = _polyline_knots(geom["driver"], geom["npe_driver"])
    refl_knots = _polyline_knots(geom["reflector"], geom["npe_reflector"])
    nb_d = sum(geom["npe_driver"]) - 1
    nb_r = sum(geom["npe_reflector"]) - 1

    arc_at_knot = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(driver_knots, axis=0), axis=1))]
    )
    interior_arc = arc_at_knot[1:-1]
    feed_basis_local = int(np.argmin(np.abs(interior_arc - geom["feed_arclength"])))
    feed_knot_index = feed_basis_local + 1

    return {
        "geometry": "hexbeam",
        "wires": [
            _wire_record(driver_knots, coeffs[:nb_d], "driver"),
            _wire_record(refl_knots, coeffs[nb_d : nb_d + nb_r], "reflector"),
        ],
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
    }


def solve(req: dict) -> dict:
    if req.get("solver") == "pynec" and pynec_backend.HAVE_PYNEC:
        out = pynec_backend.solve(req)
    else:
        geometry = req.get("geometry", "inverted_v")
        if geometry == "yagi":
            out = _solve_yagi(req)
        elif geometry == "moxon":
            out = _solve_moxon(req)
        elif geometry == "hexbeam":
            out = _solve_hexbeam(req)
        else:
            out = _solve_inverted_v(req)
        out["solver"] = "pysim"
    _compute_directivity_norm(out)
    return out


def _sweep_inverted_v(
    req: dict, freqs_mhz: list[float]
) -> tuple[list[float], list[float]]:
    """Batched sweep using BentTriangularPySim.compute_impedance_swept."""
    angle_deg = float(req.get("angle_deg", 30.0))
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    wire_radius = float(req.get("wire_radius", 0.0005))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    arm_len = halfdriver_factor * wavelength_design / 4.0

    # Use design wavelength for the sim construction; geometry is set via
    # polyline override so it's independent of the sim's wavelength field.
    sim = BentTriangularPySim(
        wavelength=wavelength_design,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
    )
    sim.wire_radius = wire_radius
    sim.polyline = _inverted_v_polyline(arm_len, angle_deg)
    sim.n_per_edge = [n_per_wire, n_per_wire]

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def _sweep_yagi(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    """Batched sweep using TriangularYagiPySim.compute_impedance_swept."""
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
    refl_factor_rel = h_refl / h_driver
    spacing_factor_rel = spacing_m / h_driver
    dir_spacing_factor_rel = (dir_spacing_wl * wavelength_design) / h_driver

    sim = TriangularYagiPySim(
        wavelength=wavelength_design,
        halfdriver_factor=driver_factor,
        nsegs=n_per_wire,
        reflector_factor=refl_factor_rel,
        spacing_factor=spacing_factor_rel,
        n_directors=n_directors,
        director_spacing_factor=dir_spacing_factor_rel,
        director_size_factor=dir_size_factor,
    )
    sim.wire_radius = wire_radius
    sim.halfdriver = h_driver

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def _sweep_moxon(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    """Batched sweep using BentMultiPySim.compute_impedance_swept."""
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.57))
    halfdriver_factor = float(req.get("halfdriver_factor", 0.962))
    aspect_ratio = float(req.get("aspect_ratio", 0.3646))
    tipspacer_factor = float(req.get("tipspacer_factor", 0.0773))
    t0_factor = float(req.get("t0_factor", 0.4078))
    wire_radius = float(req.get("wire_radius", 0.0005))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0

    geom = _moxon_polylines(
        halfdriver,
        aspect_ratio,
        tipspacer_factor,
        t0_factor,
        n_per_wire,
    )
    sim = BentMultiPySim(
        wires=[geom["driver"], geom["reflector"]],
        n_per_edge_per_wire=[geom["npe_driver"], geom["npe_reflector"]],
        feed_wire_index=0,
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_design,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
    )
    sim.wire_radius = wire_radius

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


def _sweep_hexbeam(
    req: dict, freqs_mhz: list[float]
) -> tuple[list[float], list[float]]:
    """Batched sweep using BentMultiPySim.compute_impedance_swept."""
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 28.47))
    halfdriver_factor = float(req.get("halfdriver_factor", 1.071))
    tipspacer_factor = float(req.get("tipspacer_factor", 0.1312))
    t0_factor = float(req.get("t0_factor", 0.1243))
    wire_radius = float(req.get("wire_radius", 0.0005))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    halfdriver = halfdriver_factor * wavelength_design / 4.0

    geom = _hexbeam_polylines(
        halfdriver,
        tipspacer_factor,
        t0_factor,
        n_per_wire,
    )
    sim = BentMultiPySim(
        wires=[geom["driver"], geom["reflector"]],
        n_per_edge_per_wire=[geom["npe_driver"], geom["npe_reflector"]],
        feed_wire_index=0,
        feed_arclength=geom["feed_arclength"],
        wavelength=wavelength_design,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_wire,
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
            # pysim is batched (vectorized); compute once, then stream the
            # array. Batched is ~10x faster than per-point here, and pysim is
            # cheap enough that we don't need mid-sweep cancellation.
            geometry = req.get("geometry", "inverted_v")
            if geometry == "yagi":
                z_re, z_im = await run_in_threadpool(_sweep_yagi, req, freqs)
            elif geometry == "moxon":
                z_re, z_im = await run_in_threadpool(_sweep_moxon, req, freqs)
            elif geometry == "hexbeam":
                z_re, z_im = await run_in_threadpool(_sweep_hexbeam, req, freqs)
            else:
                z_re, z_im = await run_in_threadpool(_sweep_inverted_v, req, freqs)
            for i, f in enumerate(freqs):
                yield (
                    json.dumps(
                        {
                            "freq_mhz": f,
                            "z_re": z_re[i],
                            "z_im": z_im[i],
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
