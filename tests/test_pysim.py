import pytest
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import time

from pysim import PySim
from pysim.yagi import YagiPySim
from pysim.triangular import TriangularPySim

from pysim._util import save_or_show
from pysim._accelerators import dist_outer_product

from matplotlib import pyplot as plt
import numpy as np
import scipy

import skrf

fn = None
# fn = '/dev/null'


def test_extension():
    nsegs = 20
    pts = np.array([[0, 0, z] for z in range(nsegs + 1)]) / (2 * nsegs)

    result = dist_outer_product(pts, pts)
    expected = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.testing.assert_allclose(result, expected)


@pytest.mark.slow
@pytest.mark.plot
def test_impedance_nsegs():
    xs = [21, 41, 61, 81, 101, 201, 401, 801]
    xs = np.array(xs)
    z0 = 50

    fig, ax0 = plt.subplots()
    skrf.plotting.smith(draw_labels=True, chart_type="z")

    # plt.plot(xs, np.abs(zs), marker='s')
    # plt.plot(xs, np.imag(zs), marker='s')

    for ntrap, color in [
        (0, "tab:green"),
        (2, "tab:blue"),
        (8, "tab:red"),
        (16, "tab:purple"),
    ]:
        zs = []
        for nsegs in xs:
            z, _ = PySim(nsegs=nsegs).compute_impedance(ntrap=ntrap)
            print(f"nsegs={nsegs}, z={z}")
            zs.append(z)

        zs = np.array(zs)

        normalized_zs = zs / z0
        reflection_coefficients = (normalized_zs - 1) / (normalized_zs + 1)
        skrf.plotting.plot_smith(
            reflection_coefficients,
            color=color,
            draw_labels=True,
            chart_type="z",
            marker="s",
            linestyle="None",
        )

    save_or_show(plt, fn)


@pytest.mark.slow
@pytest.mark.plot
def test_spline_fit():
    nsegs = 1001
    nsample = 100
    assert (nsegs - 1) % nsample == 0

    halfdriver_factors = np.linspace(0.9, 1, 6)

    fig, ax0 = plt.subplots()
    ax1 = ax0.twinx()

    for halfdriver_factor in halfdriver_factors:
        _, i = PySim(
            nsegs=nsegs, halfdriver_factor=halfdriver_factor
        ).compute_impedance(ntrap=16)

        xs = np.linspace(0, nsegs - 1, nsegs)

        i_sample = i[::nsample]
        xs_sample = xs[::nsample]

        interp = scipy.interpolate.CubicSpline(xs_sample, i_sample)

        color = "tab:blue"
        ax0.plot(xs, np.abs(i), color=color)
        color = "tab:purple"
        ax0.plot(xs, np.abs(interp(xs)), color=color)

        color = "tab:green"
        ax1.plot(xs, np.angle(i) * 180 / np.pi, color=color)
        color = "tab:olive"
        ax1.plot(xs, np.angle(interp(xs)) * 180 / np.pi, color=color)
    save_or_show(plt, fn)


@pytest.mark.plot
def test_svd_currents_nsmallest():

    nsegs = 101

    _, (i, i_svd_all) = PySim(nsegs=nsegs, run_svd=True).compute_impedance(ntrap=0)

    color = "tab:blue"
    plt.plot(np.abs(i), color=color)

    color = "tab:red"
    plt.plot(np.abs(i_svd_all), color=color)

    for nsmallest, color in [(1, "tab:green"), (2, "tab:purple")]:
        _, (_, i_svd) = PySim(
            nsegs=nsegs, nsmallest=nsmallest, run_svd=True
        ).compute_impedance(ntrap=0)
        print(f"nsmallest={nsmallest}, |i_svd - i_svd_all|={np.linalg.norm(i_svd - i_svd_all)}")
        plt.plot(np.abs(i_svd), color=color)

    save_or_show(plt, fn)


@pytest.mark.plot
def test_new_currents():

    fig, ax = plt.subplots(2, 2)

    for nsegs, color in [
        (21, "tab:green"),
        (41, "tab:purple"),
        (81, "tab:blue"),
        (121, "tab:orange"),
    ]:
        xs = np.linspace(0, 1, nsegs)

        _, i = PySim(nsegs=nsegs).compute_impedance(ntrap=2)

        ax[0][0].plot(xs, np.abs(i), color=color, label=f"{nsegs}")
        ax[0][1].plot(xs, np.angle(i) * 180 / np.pi, color=color, label=f"{nsegs}")

        _, i = PySim(nsegs=nsegs).compute_impedance(ntrap=2)

        ax[1][0].plot(xs, np.abs(i), color=color, label=f"{nsegs}")
        ax[1][1].plot(xs, np.angle(i) * 180 / np.pi, color=color, label=f"{nsegs}")

    ax[0][0].legend()
    ax[0][1].legend()
    ax[1][0].legend()
    ax[1][1].legend()
    save_or_show(plt, fn)


def test_iterative_improvement():
    PySim(nsegs=401, run_iterative_improvement=True).compute_impedance(ntrap=0)


@pytest.mark.slow
@pytest.mark.plot
def test_sweep_halfdriver():

    nsegs = 401
    z0 = 50

    fig, ax0 = plt.subplots()
    skrf.plotting.smith(draw_labels=True, chart_type="z")

    xs = np.linspace(0.9, 1, 21)

    for ntrap, color in ((0, "tab:green"), (4, "tab:blue"), (16, "tab:purple")):
        t = time.time()
        zs = []
        for x in xs:
            z, _ = PySim(halfdriver_factor=x, nsegs=nsegs).compute_impedance(ntrap=4)
            zs.append(z)
        print("augmented ntrap=4", time.time() - t)
        zs = np.array(zs)

        normalized_zs = zs / z0
        reflection_coefficients = (normalized_zs - 1) / (normalized_zs + 1)
        skrf.plotting.plot_smith(
            reflection_coefficients,
            color=color,
            draw_labels=True,
            chart_type="z",
            marker="s",
            linestyle="None",
        )

    save_or_show(plt, fn)


nsegs = 801
nrepeat = 1
ntrap = 8


@pytest.mark.parametrize(
    "engine,ntrap",
    [("python", 0), ("python", ntrap), ("accelerated", ntrap)],
)
def test_param(engine, ntrap):
    ps = PySim(nsegs=nsegs)

    t = time.time()
    for i in range(nrepeat):
        z, i = ps.compute_impedance(ntrap=ntrap, engine=engine)

    print(f"engine {engine}: {time.time() - t:.4f}s")


@pytest.mark.parametrize("nsegs", [21, 41, 101])
@pytest.mark.parametrize("ntrap", [0, 4, 8])
def test_python_vs_accelerated(nsegs, ntrap):
    ps = PySim(nsegs=nsegs)
    z_py, i_py = ps.compute_impedance(ntrap=ntrap, engine="python")
    z_acc, i_acc = ps.compute_impedance(ntrap=ntrap, engine="accelerated")

    np.testing.assert_allclose(z_acc, z_py, rtol=1e-10)
    np.testing.assert_allclose(i_acc, i_py, rtol=1e-10)


@pytest.mark.parametrize("nsegs", [21, 41, 101])
@pytest.mark.parametrize("ntrap", [0, 4, 8])
def test_yagi_smoke(nsegs, ntrap):
    z, i = YagiPySim(nsegs=nsegs).compute_impedance(ntrap=ntrap)
    assert i.shape == (2 * nsegs,)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(i).all()


@pytest.mark.parametrize("nsegs", [20, 40, 80])
def test_triangular_smoke(nsegs):
    z, c = TriangularPySim(nsegs=nsegs).compute_impedance()
    assert c.shape == (nsegs - 1,)
    assert np.isfinite(z.real) and np.isfinite(z.imag)
    assert np.isfinite(c).all()
    # NEC reference for the default dipole geometry: 69.64 - j18.21.
    # Triangular basis converges quickly; even N=20 is within ~2 Ohm on the
    # real part and ~2 Ohm on the imag.
    assert abs(z.real - 69.64) < 3.0
    assert abs(z.imag - (-18.21)) < 6.0
