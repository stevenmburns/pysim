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

import json
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from pysim.triangular_bent import BentTriangularPySim
from pysim.triangular_yagi import TriangularYagiPySim


app = FastAPI(title="pysim interactive")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


C_LIGHT = 299_792_458.0  # m/s, matches AbstractPySim's eps*mu derivation to ~1e-9


def _wire_record(
    knots: np.ndarray, coeffs: np.ndarray, label: str
) -> dict:
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
    design_freq_mhz = float(req.get("design_freq_mhz", 13.625))
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

    knots = np.vstack([
        np.linspace(polyline[0], polyline[1], n_per_wire + 1)[:-1],
        np.linspace(polyline[1], polyline[2], n_per_wire + 1),
    ])
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
        wire direction: +x
        spacing axis:   +y (driver at y=0, reflector at y=-spacing)
        beam direction: +y (away from reflector)
        z = 0 everywhere
    The xy plane therefore contains the beam, so the far-field azimuth cut
    actually shows the front-to-back ratio. Internal TriangularYagiPySim
    geometry is transposed to match.
    """
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 13.625))
    meas_freq_mhz = float(req.get("measurement_freq_mhz", design_freq_mhz))
    driver_factor = float(req.get("driver_length_factor", 0.962))
    refl_factor_abs = float(req.get("reflector_length_factor", 1.01))
    # Spacing in wavelengths of the design freq.
    spacing_wavelengths = float(req.get("spacing_wavelengths", 0.15))
    wire_radius = float(req.get("wire_radius", 0.0005))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    wavelength_meas = C_LIGHT / (meas_freq_mhz * 1e6)
    h_driver = driver_factor * wavelength_design / 4.0
    h_refl = refl_factor_abs * wavelength_design / 4.0
    spacing_m = spacing_wavelengths * wavelength_design

    # TriangularYagiPySim parameters: reflector_factor is reflector_half / driver_half;
    # spacing_factor is spacing / driver_half.
    refl_factor_rel = h_refl / h_driver
    spacing_factor_rel = spacing_m / h_driver

    sim = TriangularYagiPySim(
        wavelength=wavelength_meas,
        halfdriver_factor=driver_factor,
        nsegs=n_per_wire,
        reflector_factor=refl_factor_rel,
        spacing_factor=spacing_factor_rel,
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

    # Canonical layout: wires along x, spacing along y, all in the xy plane.
    N = n_per_wire
    driver_knots = np.column_stack([
        np.linspace(-h_driver, h_driver, N + 1),
        np.zeros(N + 1),
        np.zeros(N + 1),
    ])
    refl_knots = np.column_stack([
        np.linspace(-h_refl, h_refl, N + 1),
        np.full(N + 1, -spacing_m),
        np.zeros(N + 1),
    ])

    nb = N - 1
    driver_coeffs = coeffs[:nb]
    refl_coeffs = coeffs[nb:]

    # Feed: TriangularYagiPySim picks the interior knot of the driver closest
    # to L_driver/2 — that's the middle interior knot, at full-list index N//2
    # when interior-list index is (N-2)//2. Reproduce the same logic here.
    interior_arc = np.linspace(0.0, 2 * h_driver, N + 1)[1:-1]
    m_center_interior = int(np.argmin(np.abs(interior_arc - h_driver)))
    feed_knot_index = m_center_interior + 1  # shift past left endpoint

    return {
        "geometry": "yagi",
        "wires": [
            _wire_record(driver_knots, driver_coeffs, "driver"),
            _wire_record(refl_knots, refl_coeffs, "reflector"),
        ],
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
        "solve_ms": solve_ms,
    }


def solve(req: dict) -> dict:
    geometry = req.get("geometry", "inverted_v")
    if geometry == "yagi":
        return _solve_yagi(req)
    return _solve_inverted_v(req)


def _sweep_inverted_v(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    """Batched sweep using BentTriangularPySim.compute_impedance_swept."""
    angle_deg = float(req.get("angle_deg", 30.0))
    n_per_wire = int(req.get("n_per_wire", 30))
    design_freq_mhz = float(req.get("design_freq_mhz", 13.625))
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
    design_freq_mhz = float(req.get("design_freq_mhz", 13.625))
    driver_factor = float(req.get("driver_length_factor", 0.962))
    refl_factor_abs = float(req.get("reflector_length_factor", 1.01))
    spacing_wavelengths = float(req.get("spacing_wavelengths", 0.15))
    wire_radius = float(req.get("wire_radius", 0.0005))

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    h_driver = driver_factor * wavelength_design / 4.0
    h_refl = refl_factor_abs * wavelength_design / 4.0
    spacing_m = spacing_wavelengths * wavelength_design
    refl_factor_rel = h_refl / h_driver
    spacing_factor_rel = spacing_m / h_driver

    sim = TriangularYagiPySim(
        wavelength=wavelength_design,
        halfdriver_factor=driver_factor,
        nsegs=n_per_wire,
        reflector_factor=refl_factor_rel,
        spacing_factor=spacing_factor_rel,
    )
    sim.wire_radius = wire_radius
    sim.halfdriver = h_driver

    k_array = np.array([2 * np.pi * f * 1e6 / C_LIGHT for f in freqs_mhz])
    z_array = sim.compute_impedance_swept(k_array)
    return z_array.real.tolist(), z_array.imag.tolist()


@app.post("/sweep")
async def sweep_endpoint(req: dict):
    """Run a measurement-freq sweep across freqs_mhz for a fixed antenna."""
    freqs = [float(f) for f in req.get("freqs_mhz", [])]
    if not freqs:
        return {"freqs_mhz": [], "z_re": [], "z_im": []}
    geometry = req.get("geometry", "inverted_v")
    if geometry == "yagi":
        z_re, z_im = _sweep_yagi(req, freqs)
    else:
        z_re, z_im = _sweep_inverted_v(req, freqs)
    return {"freqs_mhz": freqs, "z_re": z_re, "z_im": z_im}


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
            result = solve(req)
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        return
