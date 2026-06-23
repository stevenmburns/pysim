"""Test that the PyNEC web backend agrees with the momwire backend.

Skipped when PyNEC isn't installed. Exercises the dispatch path through
`web.server.solve` and `web.server.sweep_endpoint`'s helpers as a side
benefit, since both paths call into the per-backend modules.
"""

import pytest

PyNEC = pytest.importorskip("PyNEC")  # noqa: F841

from web import pynec_backend  # noqa: E402
from web.examples import REGISTRY as _EXAMPLES  # noqa: E402

_solve_inverted_v = _EXAMPLES["inverted_v"].momwire_solve
_sweep_inverted_v = _EXAMPLES["inverted_v"].momwire_sweep
_solve_yagi = _EXAMPLES["yagi"].momwire_solve
_sweep_yagi = _EXAMPLES["yagi"].momwire_sweep
_solve_bowtie = _EXAMPLES["bowtie"].momwire_solve
_sweep_bowtie = _EXAMPLES["bowtie"].momwire_sweep
pynec_backend.solve_inverted_v = _EXAMPLES["inverted_v"].pynec_solve
pynec_backend.solve_yagi = _EXAMPLES["yagi"].pynec_solve
pynec_backend.solve_fandipole = _EXAMPLES["fan_dipole"].pynec_solve
pynec_backend.solve_bowtie = _EXAMPLES["bowtie"].pynec_solve


# The two backends use different basis functions (NEC2 pulse basis vs
# momwire's triangular Galerkin) and slightly different feed models, so
# they don't agree bit-for-bit. Near resonance, |Z| is small (~60 Ω) and
# the delta is ~1 Ω; off-resonance, |Z| can reach ~200 Ω and the delta
# scales roughly with it. Use a 3% relative tolerance with a 0.5 Ω floor,
# which comfortably covers both regimes while catching geometry-construction
# bugs and outright solver regressions.
def _close(
    z_a: complex, z_b: complex, rel: float = 0.03, abs_floor: float = 0.5
) -> bool:
    return abs(z_a - z_b) < rel * abs(z_a) + abs_floor


def _z_complex(res):
    return complex(res["z_in_re"], res["z_in_im"])


def test_inverted_v_agrees_at_n30():
    req = {
        "geometry": "inverted_v",
        "n_per_wire": 30,
        "design_freq_mhz": 14.3,
        "measurement_freq_mhz": 14.3,
        "halfdriver_factor": 0.962,
        "angle_deg": 30.0,
        "wire_radius": 0.0005,
    }
    z_momwire = _z_complex(_solve_inverted_v(req))
    z_pynec = _z_complex(pynec_backend.solve_inverted_v(req))
    assert _close(z_momwire, z_pynec), (
        f"V N=30: momwire={z_momwire}, pynec={z_pynec}, |delta|={abs(z_momwire - z_pynec):.3f}"
    )


def test_yagi_agrees_at_n30():
    req = {
        "geometry": "yagi",
        "n_per_wire": 30,
        "design_freq_mhz": 14.3,
        "measurement_freq_mhz": 14.3,
        "driver_length_factor": 0.962,
        "reflector_length_factor": 1.01,
        "spacing_wavelengths": 0.15,
        "wire_radius": 0.0005,
    }
    z_momwire = _z_complex(_solve_yagi(req))
    z_pynec = _z_complex(pynec_backend.solve_yagi(req))
    assert _close(z_momwire, z_pynec), (
        f"Yagi N=30: momwire={z_momwire}, pynec={z_pynec}, |delta|={abs(z_momwire - z_pynec):.3f}"
    )


def test_sweep_inverted_v_agrees():
    """Three-point sweep around resonance: both backends should track each
    other across the band, not just at the center frequency."""
    req = {
        "geometry": "inverted_v",
        "n_per_wire": 30,
        "design_freq_mhz": 14.3,
        "halfdriver_factor": 0.962,
        "angle_deg": 30.0,
        "wire_radius": 0.0005,
    }
    freqs = [13.0, 14.3, 15.5]
    z_re_p, z_im_p = _sweep_inverted_v(req, freqs)
    z_re_n, z_im_n = pynec_backend.sweep(req, freqs)
    for f, rp, ip, rn, ni in zip(freqs, z_re_p, z_im_p, z_re_n, z_im_n):
        zp, zn = complex(rp, ip), complex(rn, ni)
        assert _close(zp, zn), (
            f"V sweep @ {f} MHz: momwire={zp}, pynec={zn}, |delta|={abs(zp - zn):.3f}"
        )


def test_sweep_yagi_agrees():
    req = {
        "geometry": "yagi",
        "n_per_wire": 30,
        "design_freq_mhz": 14.3,
        "driver_length_factor": 0.962,
        "reflector_length_factor": 1.01,
        "spacing_wavelengths": 0.15,
        "wire_radius": 0.0005,
    }
    freqs = [13.0, 14.3, 15.5]
    z_re_p, z_im_p = _sweep_yagi(req, freqs)
    z_re_n, z_im_n = pynec_backend.sweep(req, freqs)
    for f, rp, ip, rn, ni in zip(freqs, z_re_p, z_im_p, z_re_n, z_im_n):
        zp, zn = complex(rp, ip), complex(rn, ni)
        assert _close(zp, zn), (
            f"Yagi sweep @ {f} MHz: momwire={zp}, pynec={zn}, |delta|={abs(zp - zn):.3f}"
        )


def test_fandipole_two_band_smoke():
    """Two-band fan dipole (20m + 10m), 2 wires per arm, cone arrangement
    from the antenna_designer reference. Both bands should land within
    ~30 Ω of 50+j0 — the design is nominally co-resonant on 14.3 and 28.47.
    """
    req = {
        "geometry": "fan_dipole",
        "n_per_wire": 21,
        "n_bands": 2,
        "design_freq_mhz": 14.3,
        "band_lengths_m": [10.2551, 5.2691],
        "band_freqs_mhz": [14.3, 28.47],
        "slope": 0.5,
        "cone_radius_m": 0.12,
        "wire_radius": 0.0005,
    }

    req_20 = {**req, "measurement_freq_mhz": 14.3}
    z_20 = _z_complex(pynec_backend.solve_fandipole(req_20))
    assert 30.0 < z_20.real < 80.0, f"20m R out of range: {z_20}"
    assert -30.0 < z_20.imag < 30.0, f"20m X out of range: {z_20}"

    req_10 = {**req, "measurement_freq_mhz": 28.47}
    z_10 = _z_complex(pynec_backend.solve_fandipole(req_10))
    assert 30.0 < z_10.real < 100.0, f"10m R out of range: {z_10}"
    assert -30.0 < z_10.imag < 60.0, f"10m X out of range: {z_10}"

    # Wire records: 1 feed + 2 arms * n_bands = 5 wires; feed has 3 knots
    # (T, midpoint, S) so feed_knot_index=1 lands on the midpoint.
    res = pynec_backend.solve_fandipole(req_20)
    assert len(res["wires"]) == 1 + 2 * req["n_bands"]
    assert res["feed_wire_index"] == 0
    assert res["feed_knot_index"] == 1
    assert len(res["wires"][0]["knot_positions"]) == 3


def test_response_shape_matches():
    """The frontend reads exact field names; make sure both backends produce
    the same keys with the same types so a backend swap can't silently
    break the UI."""
    req = {
        "geometry": "inverted_v",
        "n_per_wire": 30,
        "design_freq_mhz": 14.3,
        "halfdriver_factor": 0.962,
        "angle_deg": 30.0,
        "wire_radius": 0.0005,
    }
    p = _solve_inverted_v(req)
    n = pynec_backend.solve_inverted_v(req)
    # PyNEC backend adds a "solver" field; momwire's _solve_inverted_v doesn't
    # add it (the dispatch wrapper does), so drop it from the comparison.
    for k in (
        "wires",
        "feed_wire_index",
        "feed_knot_index",
        "z_in_re",
        "z_in_im",
        "design_freq_mhz",
        "measurement_freq_mhz",
        "lambda_design_m",
        "arm_len_m",
    ):
        assert k in p, f"momwire response missing {k}"
        assert k in n, f"pynec response missing {k}"
    # Wire structure: same wire count, same knot count per wire.
    assert len(p["wires"]) == len(n["wires"])
    for wp, wn in zip(p["wires"], n["wires"]):
        assert len(wp["knot_positions"]) == len(wn["knot_positions"])
        assert len(wp["knot_currents_re"]) == len(wn["knot_currents_re"])
        assert len(wp["knot_currents_im"]) == len(wn["knot_currents_im"])


# ---------------------------------------------------------------------------
# Bowtie 1×2 phased array — multi-feed parity between the two backends.
# ---------------------------------------------------------------------------


def _bowtie_parity_req(**over) -> dict:
    req = {
        "geometry": "bowtie",
        "n_per_wire": 21,
        "design_freq_mhz": 28.47,
        "measurement_freq_mhz": 28.47,
        "wire_radius": 0.0005,
        "slope": 0.5376,
        "length_factor": 0.515,
        "del_y_m": 4.0,
        "phase_lr_deg": 0.0,
    }
    req.update(over)
    return req


def _feeds_z(res):
    return [complex(f["z_re"], f["z_im"]) for f in res["feeds"]]


# The bowtie array discretization differs more than the dipole / Yagi
# cases — momwire's tent basis lives on a 4-polyline mesh with K=2
# junctions while NEC uses 10 wire cards per element with implicit
# coordinate-match continuity. At antenna_designer's resonant-ish
# defaults the per-feed delta is ~3 Ω on a ~180 Ω port, so a 5%
# relative tolerance + 2 Ω floor catches geometry / EX-card bugs
# without flagging genuine basis-vs-basis disagreement.
def _close_bowtie(z_a: complex, z_b: complex) -> bool:
    return abs(z_a - z_b) < 0.05 * abs(z_a) + 2.0


def test_bowtie_in_phase_per_feed_agrees():
    req = _bowtie_parity_req(phase_lr_deg=0.0)
    p = _solve_bowtie(req)
    n = pynec_backend.solve_bowtie(req)
    z_p = _feeds_z(p)
    z_n = _feeds_z(n)
    assert len(z_p) == len(z_n) == 2
    for i, (zp, zn) in enumerate(zip(z_p, z_n)):
        assert _close_bowtie(zp, zn), (
            f"feed {i} in-phase: momwire={zp}, pynec={zn}, |delta|={abs(zp - zn):.3f}"
        )


def test_bowtie_90deg_phase_per_feed_agrees():
    """The interesting case: V₁ ≠ V₂ in phase, so mutual coupling makes
    Z_0 ≠ Z_1. Both backends must reproduce the same asymmetric drive
    impedances — catches EX-card ordering / voltage-sign bugs that a
    symmetric in-phase test would hide."""
    req = _bowtie_parity_req(phase_lr_deg=90.0)
    p = _solve_bowtie(req)
    n = pynec_backend.solve_bowtie(req)
    z_p = _feeds_z(p)
    z_n = _feeds_z(n)
    for i, (zp, zn) in enumerate(zip(z_p, z_n)):
        assert _close_bowtie(zp, zn), (
            f"feed {i} 90°: momwire={zp}, pynec={zn}, |delta|={abs(zp - zn):.3f}"
        )
    # Also assert asymmetry actually showed up — otherwise both sides
    # could be failing silently in lockstep.
    assert abs(z_p[0] - z_p[1]) > 5.0
    assert abs(z_n[0] - z_n[1]) > 5.0


def test_bowtie_response_shape_matches():
    """Cross-backend response-shape contract: every key the frontend
    indexes on the bowtie response must be present on both backends, and
    the geometry-level structure (wire count, feed count, z0_ohms) must
    agree exactly."""
    req = _bowtie_parity_req()
    p = _solve_bowtie(req)
    n = pynec_backend.solve_bowtie(req)
    for k in (
        "wires",
        "feeds",
        "feed_wire_index",
        "feed_knot_index",
        "z_in_re",
        "z_in_im",
        "z0_ohms",
        "phase_lr_deg",
        "del_y_m",
    ):
        assert k in p, f"momwire response missing {k}"
        assert k in n, f"pynec response missing {k}"
    assert p["z0_ohms"] == n["z0_ohms"] == 100.0
    assert len(p["wires"]) == len(n["wires"]) == 8
    assert len(p["feeds"]) == len(n["feeds"]) == 2
    # Same feed wire indices on both — guarantees the frontend's marker
    # rendering finds the same wire on either backend.
    for fp, fn in zip(p["feeds"], n["feeds"]):
        assert fp["wire_index"] == fn["wire_index"]


def test_bowtie_sweep_per_feed_agrees():
    """Sweep parity across the band: at every frequency the per-feed Z
    must agree between backends. Catches sweep-loop bugs (wrong NEC
    context reuse, missing FR card reset) that single-frequency tests
    miss."""
    req = _bowtie_parity_req()
    freqs = [28.0, 28.47, 29.0]
    _, _, feeds_re_p, feeds_im_p = _sweep_bowtie(req, freqs)
    # PyNEC has no batched sweep API; loop the multifeed point helper.
    feeds_re_n, feeds_im_n = [], []
    for f in freqs:
        _, fz = pynec_backend._sweep_at_multifeed(req, f)
        feeds_re_n.append([z.real for z in fz])
        feeds_im_n.append([z.imag for z in fz])
    for fi, (rp, ip, rn, ni) in enumerate(
        zip(feeds_re_p, feeds_im_p, feeds_re_n, feeds_im_n)
    ):
        for j in range(2):
            zp = complex(rp[j], ip[j])
            zn = complex(rn[j], ni[j])
            assert _close_bowtie(zp, zn), (
                f"freq {freqs[fi]} feed {j}: momwire={zp}, pynec={zn}, "
                f"|delta|={abs(zp - zn):.3f}"
            )
