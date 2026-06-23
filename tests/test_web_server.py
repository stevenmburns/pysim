"""Integration tests for the web/server.py momwire solver paths.

The solver-level momwire API has its own unit coverage in tests/test_momwire.py;
this file pins the *response schema* and array-assembly logic the frontend
depends on. Catches regressions in the bowtie 1×2 geometry build, the
multi-feed response shape, the multi-feed sweep tuple shape, and the
100 Ω Z₀ surfacing — all easy to break with a careless refactor of
web/server.py and hard to notice without driving the UI by hand.
"""

import numpy as np
import pytest

# web.server is the FastAPI app — skip the whole module when fastapi
# isn't installed. The plain `test` CI job ships a minimal scientific
# stack without the web dependencies; the `test-pynec` jobs install
# fastapi and exercise this file.
pytest.importorskip("fastapi")

from web.examples import REGISTRY as _EXAMPLES  # noqa: E402
from web.server import solve  # noqa: E402

_solve_bowtie = _EXAMPLES["bowtie"].momwire_solve
_sweep_bowtie = _EXAMPLES["bowtie"].momwire_sweep


def _bowtie_req(**over) -> dict:
    """Minimal bowtie request body matching the live frontend payload."""
    req = {
        "geometry": "bowtie",
        "solver": "momwire",
        "momwire_model": "triangular",
        "n_per_wire": 15,
        "design_freq_mhz": 28.47,
        "measurement_freq_mhz": 28.47,
        "wire_radius": 0.0005,
        "ground": False,
        "ground_fast": False,
        "height_m": 0.0,
        "slope": 0.5376,
        "length_factor": 0.515,
        "del_y_m": 4.0,
        "phase_lr_deg": 0.0,
    }
    req.update(over)
    return req


def test_bowtie_response_shape():
    res = _solve_bowtie(_bowtie_req())
    assert res["geometry"] == "bowtie"
    # 4 polylines per element × 2 elements
    assert len(res["wires"]) == 8
    # 1×2 array → 2 feeds
    assert len(res["feeds"]) == 2
    for f in res["feeds"]:
        assert set(f.keys()) == {
            "wire_index",
            "knot_index",
            "z_re",
            "z_im",
            "v_re",
            "v_im",
        }
    # Legacy single-feed fields still populated for back-compat with the
    # existing single-feed render path.
    primary = res["feeds"][0]
    assert res["feed_wire_index"] == primary["wire_index"]
    assert res["feed_knot_index"] == primary["knot_index"]
    assert res["z_in_re"] == primary["z_re"]
    assert res["z_in_im"] == primary["z_im"]


def test_bowtie_response_z0_is_100_ohms():
    """The bowtie is designed for 100 Ω feedlines per element — the
    frontend uses this for SWR and Smith chart reference."""
    res = _solve_bowtie(_bowtie_req())
    assert res["z0_ohms"] == 100.0


def test_bowtie_in_phase_equal_by_symmetry():
    """V₁ = V₂ = 1 + 0j → the two elements see identical drive impedance
    by mirror symmetry. Mutual coupling shifts Z away from the isolated
    bowtie value, but Z_0 must equal Z_1 to numerical precision."""
    res = _solve_bowtie(_bowtie_req(phase_lr_deg=0.0))
    z0 = complex(res["feeds"][0]["z_re"], res["feeds"][0]["z_im"])
    z1 = complex(res["feeds"][1]["z_re"], res["feeds"][1]["z_im"])
    assert abs(z0 - z1) / abs(z0) < 1e-6
    # Sanity-bound the magnitude so a wire-assembly bug that gives a
    # tiny or huge Z lights up.
    assert 50.0 < abs(z0) < 500.0


def test_bowtie_anti_phase_changes_drive_point():
    """The (Z_self + Z_mut) in-phase mode and the (Z_self − Z_mut)
    anti-phase mode must differ — otherwise the phase knob is wired up
    but not actually shifting NEC's RHS."""
    z_in = complex(
        _solve_bowtie(_bowtie_req(phase_lr_deg=0.0))["z_in_re"],
        _solve_bowtie(_bowtie_req(phase_lr_deg=0.0))["z_in_im"],
    )
    z_anti = complex(
        _solve_bowtie(_bowtie_req(phase_lr_deg=180.0))["z_in_re"],
        _solve_bowtie(_bowtie_req(phase_lr_deg=180.0))["z_in_im"],
    )
    assert abs(z_in - z_anti) > 5.0


def test_bowtie_voltage_phasor_matches_request():
    """The reported v_re/v_im on each feed entry should reflect exactly
    the (1, exp(j·π·phase_lr/180)) drive — this is the contract the
    frontend's per-feed phase labels rely on."""
    res = _solve_bowtie(_bowtie_req(phase_lr_deg=90.0))
    v0 = complex(res["feeds"][0]["v_re"], res["feeds"][0]["v_im"])
    v1 = complex(res["feeds"][1]["v_re"], res["feeds"][1]["v_im"])
    assert abs(v0 - (1.0 + 0.0j)) < 1e-9
    assert abs(v1 - np.exp(1j * np.pi / 2)) < 1e-9


def test_bowtie_sweep_returns_4_tuple_with_per_feed_data():
    """The /sweep dispatcher routes by tuple length: single-feed returns
    2-tuple, multi-feed returns 4-tuple with feeds_re/feeds_im appended."""
    freqs = [28.0, 28.47, 29.0]
    result = _sweep_bowtie(_bowtie_req(), freqs)
    assert len(result) == 4
    primary_re, primary_im, feeds_re, feeds_im = result
    assert len(primary_re) == 3
    assert len(primary_im) == 3
    # feeds_re is (n_freq × n_feeds).
    assert len(feeds_re) == 3
    assert len(feeds_im) == 3
    for row_re, row_im in zip(feeds_re, feeds_im):
        assert len(row_re) == 2
        assert len(row_im) == 2
    # Primary mirrors feeds[0] at every frequency.
    for i in range(3):
        assert primary_re[i] == feeds_re[i][0]
        assert primary_im[i] == feeds_im[i][0]


def test_solve_dispatch_routes_bowtie_geometry():
    """End-to-end through the top-level dispatcher (the path /solve
    actually hits). Pins the geometry name match and confirms
    directivity_norm tacks on without crashing on multi-feed coeffs."""
    res = solve(_bowtie_req())
    assert res["geometry"] == "bowtie"
    assert res["solver"] == "momwire"
    assert "directivity_norm" in res
    assert res["directivity_norm"] > 0
    assert len(res["feeds"]) == 2


def test_bowtie_single_feed_geometries_still_have_no_feeds_key():
    """Regression guard on back-compat — multi-feed response field must
    not leak into single-feed geometries."""
    req = {
        "geometry": "inverted_v",
        "solver": "momwire",
        "momwire_model": "triangular",
        "n_per_wire": 20,
        "design_freq_mhz": 14.3,
        "measurement_freq_mhz": 14.3,
        "wire_radius": 0.0005,
        "ground": False,
        "ground_fast": False,
        "height_m": 0.0,
        "angle_deg": 30.0,
        "halfdriver_factor": 0.962,
    }
    res = solve(req)
    assert res["geometry"] == "inverted_v"
    assert "feeds" not in res
    assert "z0_ohms" not in res
