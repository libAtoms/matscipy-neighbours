"""Cross-checks for the Lennard-Jones Langevin examples.

The array example (``examples/lj_langevin/lj_langevin.py``) is imported as a
module and its force routine is compared against an O(N^2) brute-force
reference for both systems: the droplet in vacuum (no periodicity) and the
bulk liquid in a periodic box (minimum-image reference; the box is larger than
twice the cutoff). If the C++ example binary has been built
(``BUILD_EXAMPLES=ON``) its initial potential energy is checked against the
array example too, which exercises the shift-array path of the fused kernel.
"""

import importlib.util
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EXAMPLE = _ROOT / "examples" / "lj_langevin" / "lj_langevin.py"

CUTOFF = 2.5
DENSITY = 0.8442
KT = 0.7


@pytest.fixture(scope="module")
def lj():
    spec = importlib.util.spec_from_file_location("lj_langevin", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def brute_force(positions, cutoff, L=None):
    """Reference LJ forces/energy; ``L`` is the periodic box edge (minimum
    image) or None for a non-periodic system."""
    n = positions.shape[0]
    D = positions[None, :, :] - positions[:, None, :]       # D[i, j] = r_j - r_i
    if L is not None:
        D -= L * np.round(D / L)
    r2 = (D * D).sum(axis=-1)
    np.fill_diagonal(r2, np.inf)
    mask = r2 < cutoff * cutoff
    inv_r2 = np.where(mask, 1.0 / r2, 0.0)
    inv_r6 = inv_r2 ** 3
    energy = 0.5 * (4.0 * inv_r6 * (inv_r6 - 1.0)).sum()
    coef = np.where(mask, -24.0 * inv_r2 * inv_r6 * (2.0 * inv_r6 - 1.0), 0.0)
    forces = (coef[..., None] * D).sum(axis=1)
    # Virial sum_{i<j} r_ij . F_ij with r_ij = r_i - r_j = -D and F_ij = coef D.
    virial = -0.5 * (coef * r2)[mask].sum()
    return forces, energy, virial, int(mask.sum())


def _system(lj, system, n):
    positions, origin, cell, pbc = lj.make_system(
        np, system, n, 0, 1.6, DENSITY, CUTOFF)
    return positions, origin, cell, pbc


@pytest.mark.parametrize("system", ["droplet", "liquid"])
def test_example_forces_match_brute_force(lj, system):
    from matscipy_neighbours import neighbour_list
    n = 256
    positions, origin, cell, pbc = _system(lj, system, n)
    L = cell[0, 0] if pbc else None
    if L is not None:
        assert L > 2 * CUTOFF, "minimum-image reference needs L > 2 rc"
    build = lj.make_pairs_builder("matscipy", neighbour_list, np, CUTOFF,
                                  origin, cell, pbc)
    forces, energy, virial, npairs = lj.lj_forces_energy(np, build, positions)
    ref_forces, ref_energy, ref_virial, ref_pairs = brute_force(positions, CUTOFF, L)
    assert npairs == ref_pairs
    assert energy == pytest.approx(ref_energy, rel=1e-10)
    assert virial == pytest.approx(ref_virial, rel=1e-10)
    np.testing.assert_allclose(forces, ref_forces, rtol=1e-9, atol=1e-9)
    # Newton's third law holds pairwise, so the net force vanishes.
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)


def test_liquid_lattice_is_homogeneous_and_inside_box(lj):
    for n in (100, 256, 500, 4000):
        positions, L = lj.fcc_liquid_n(np, n, DENSITY)
        assert positions.shape == (n, 3)
        assert L == pytest.approx((n / DENSITY) ** (1.0 / 3.0))
        assert (positions >= 0).all() and (positions < L).all()
        # Nearest-neighbour distance stays above the LJ core.
        D = positions[None] - positions[:, None]
        D -= L * np.round(D / L)
        r = np.sqrt((D * D).sum(-1))
        np.fill_diagonal(r, np.inf)
        assert r.min() > 1.0


def _cpp_binary():
    for p in _ROOT.rglob("lj_langevin_cpu"):
        if p.is_file() and ".git" not in p.parts and "CMakeFiles" not in p.parts:
            return p
    return None


@pytest.mark.parametrize("system", ["droplet", "liquid"])
def test_cpp_example_matches_array_example(lj, system):
    exe = _cpp_binary()
    if exe is None:
        pytest.skip("lj_langevin_cpu not built (BUILD_EXAMPLES=ON)")
    from matscipy_neighbours import neighbour_list
    n = 500
    positions, origin, cell, pbc = _system(lj, system, n)
    build = lj.make_pairs_builder("matscipy", neighbour_list, np, CUTOFF,
                                  origin, cell, pbc)
    _, energy, _, npairs = lj.lj_forces_energy(np, build, positions)

    out = subprocess.run([str(exe), "--system", system, "--atoms", str(n),
                          "--steps", "0", "--out", "/dev/null"],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r"pairs~(\d+)\s+E_pot=([-\d.]+)", out)
    assert m, out
    assert int(m.group(1)) == npairs
    assert float(m.group(2)) == pytest.approx(energy, rel=1e-8)


def test_langevin_holds_temperature_and_liquid_is_stable(lj):
    """The Allen-Tildesley integrator must thermostat the liquid at the target
    temperature (the previous single-kick scheme heated it until it exploded)
    and the default state point must stay a bound liquid at positive
    pressure."""
    from matscipy_neighbours import neighbour_list
    np.random.seed(1)
    n, steps, sample_from = 256, 5000, 3000   # the lattice melts in ~3000 steps
    positions, origin, cell, pbc = _system(lj, "liquid", n)
    volume = float(np.linalg.det(cell))
    build = lj.make_pairs_builder("matscipy", neighbour_list, np, CUTOFF,
                                  origin, cell, pbc)
    velocities = np.zeros_like(positions)
    lc = lj.langevin_constants(0.005, 1.0, KT)
    forces, energy, virial, _ = lj.lj_forces_energy(np, build, positions)
    T_samples, P_samples, U_samples = [], [], []
    for step in range(steps):
        lj.langevin_drift(np, positions, velocities, forces, lc)
        forces, energy, virial, _ = lj.lj_forces_energy(np, build, positions)
        lj.langevin_kick(velocities, forces, lc)
        if step >= sample_from:
            T = lj.kinetic_temperature(np, velocities)
            T_samples.append(T)
            P_samples.append(lj.pressure(n, T, virial, volume))
            U_samples.append(energy / n)
    assert np.mean(T_samples) == pytest.approx(KT, abs=0.05)
    assert -6.5 < np.mean(U_samples) < -4.5          # a dense LJ liquid
    assert np.mean(P_samples) > 0.0                  # not under tension
    # Newton's third law still holds and nothing has left the neighbourhood.
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)
