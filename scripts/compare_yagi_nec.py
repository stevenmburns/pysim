"""Compare YagiPySim against NEC2 for a matched 2-element Yagi in free space.

Run with antenna_designer's venv (which has PyNEC), pointing PYTHONPATH at this
repo's src so pysim is importable:

    PYTHONPATH=/home/smburns/antennas/pysim/src \
        /home/smburns/antennas/antenna_designer/.venv/bin/python \
        scripts/compare_yagi_nec.py
"""
import PyNEC as nec

from pysim.yagi import YagiPySim
from pysim import PySim as NewPySim


def _run_nec(c, n_seg, freq_mhz):
    c.geometry_complete(0)
    c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)  # free space

    feed_seg = (n_seg + 1) // 2
    c.ex_card(0, 1, feed_seg, 0, 1.0, 0.0, 0, 0, 0, 0)

    c.fr_card(0, 1, freq_mhz, 0)
    c.xq_card(0)

    sc = c.get_structure_currents(0)
    currents = sc.get_current()
    tags = sc.get_current_segment_tag()

    driver_indices = [i for i, t in enumerate(tags) if t == 1]
    feed_idx = driver_indices[feed_seg - 1]
    return 1.0 / currents[feed_idx]


def nec_free_space_dipole(*, freq_mhz, halfdriver, wire_radius, n_seg):
    c = nec.nec_context()
    geo = c.get_geometry()

    geo.wire(
        1, n_seg,
        0.0, -halfdriver, 0.0,
        0.0, +halfdriver, 0.0,
        wire_radius, 1.0, 1.0,
    )
    return _run_nec(c, n_seg, freq_mhz)


def nec_free_space_yagi(
    *, freq_mhz, halfdriver, reflector_factor, spacing, wire_radius, n_seg
):
    c = nec.nec_context()
    geo = c.get_geometry()

    geo.wire(
        1, n_seg,
        0.0, -halfdriver, 0.0,
        0.0, +halfdriver, 0.0,
        wire_radius, 1.0, 1.0,
    )

    refl_h = halfdriver * reflector_factor
    geo.wire(
        2, n_seg,
        -spacing, -refl_h, 0.0,
        -spacing, +refl_h, 0.0,
        wire_radius, 1.0, 1.0,
    )
    return _run_nec(c, n_seg, freq_mhz)


def main():
    ps = YagiPySim()
    wavelength = ps.wavelength
    freq_mhz = 299.792458 / wavelength
    halfdriver = ps.halfdriver
    spacing = halfdriver
    refl_factor = 1.05
    wire_radius = ps.wire_radius

    print(f"Geometry: wavelength={wavelength:.3f} m  freq={freq_mhz:.4f} MHz")
    print(
        f"          halfdriver={halfdriver:.4f} m  "
        f"spacing={spacing:.4f} m ({spacing/wavelength:.3f} lambda)  "
        f"reflector_factor={refl_factor}"
    )
    print(f"          wire_radius={wire_radius:.4f} m")
    print()

    print("=== Single dipole (driver only) ===")
    print("NewPySim (Python integral):")
    for nsegs in [21, 41, 101]:
        for ntrap in [0, 8]:
            z, _ = NewPySim(nsegs=nsegs).compute_impedance(ntrap=ntrap)
            print(
                f"  nsegs={nsegs:3d} ntrap={ntrap:2d}: "
                f"Z = {z.real:8.3f} + j{z.imag:8.3f}"
            )
        print()

    print("NEC2 free space (dipole only):")
    for n_seg in [21, 41, 101]:
        z = nec_free_space_dipole(
            freq_mhz=freq_mhz, halfdriver=halfdriver,
            wire_radius=wire_radius, n_seg=n_seg,
        )
        print(f"  n_seg={n_seg:3d}: Z = {z.real:8.3f} + j{z.imag:8.3f}")

    print()
    print("=== Two-element Yagi (driver + reflector) ===")
    print("YagiPySim (Python integral):")
    for nsegs in [21, 41, 101]:
        for ntrap in [0, 8]:
            z, _ = YagiPySim(nsegs=nsegs).compute_impedance(ntrap=ntrap)
            print(
                f"  nsegs={nsegs:3d} ntrap={ntrap:2d}: "
                f"Z = {z.real:8.3f} + j{z.imag:8.3f}"
            )
        print()

    print("NEC2 free space:")
    for n_seg in [21, 41, 101]:
        z = nec_free_space_yagi(
            freq_mhz=freq_mhz, halfdriver=halfdriver, reflector_factor=refl_factor,
            spacing=spacing, wire_radius=wire_radius, n_seg=n_seg,
        )
        print(f"  n_seg={n_seg:3d}: Z = {z.real:8.3f} + j{z.imag:8.3f}")


if __name__ == "__main__":
    main()
