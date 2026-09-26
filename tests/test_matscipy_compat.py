"""Compatibility tests: the public ``matscipy_neighbours`` API against ASE.

These cover graphite coordination, FCC bulk, element-pair dict cutoffs, atoms
outside the box, supercell consistency, and the minimum-image convention,
exercising the wrapper that provides matscipy API compatibility.
"""

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase.build import bulk, molecule  # noqa: E402
import ase.lattice.hexagonal  # noqa: E402

from matscipy_neighbours import (  # noqa: E402
    coordination,
    first_neighbours,
    mic,
    neighbour_list,
)


def test_fcc_bulk_coordination_and_distance():
    # FCC aluminium: 12 nearest neighbours at a/sqrt(2).
    a = bulk("Al", "fcc", a=4.05, cubic=False)
    i, j, d = neighbour_list("ijd", a, 3.1)
    assert (np.bincount(i) == [12]).all()
    np.testing.assert_allclose(d, 4.05 / np.sqrt(2), atol=1e-6)


def test_graphite_coordination_is_three():
    for n in range(1, 4):
        a = ase.lattice.hexagonal.Graphite(
            "C", latticeconstant=(2.5, 10.0), size=[n, n, 1])
        assert (coordination(a, 1.85) == 3).all()


def test_shift_reconstructs_distance_vector():
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    a.rattle(0.1, seed=12)
    i, j, D, S = neighbour_list("ijDS", a, 3.0)
    np.testing.assert_allclose(
        D, a.positions[j] - a.positions[i] + S.dot(a.cell), atol=1e-10)


def test_distance_vector_matches_mic():
    # Supercell so the cutoff stays below half the box: the minimum image
    # convention (and hence mic) is only well defined when cutoff < L/2.
    a = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    a.rattle(0.1, seed=3)
    i, j, D, d = neighbour_list("ijDd", a, 3.0)
    direct = mic(a.positions[j] - a.positions[i], np.asarray(a.cell))
    np.testing.assert_allclose(D, direct, atol=1e-10)
    np.testing.assert_allclose(d, np.linalg.norm(D, axis=1), atol=1e-12)


def test_pair_list_is_symmetric():
    a = bulk("Si", "diamond", a=5.43, cubic=True)
    i, j = neighbour_list("ij", a, 2.6)
    assert (np.bincount(i) == np.bincount(j)).all()


def test_element_pair_dict_cutoffs():
    # Formic acid HCOOH, generously padded with vacuum.
    a = molecule("HCOOH")
    a.center(vacuum=5.0)

    # Plain global cutoff: full coordination.
    assert (np.bincount(neighbour_list("i", a, 1.85)) == [2, 3, 1, 1, 1]).all()

    # Only C-H bonds shorter than 1.2.
    assert (np.bincount(neighbour_list("i", a, {(1, 6): 1.2})) ==
            [0, 1, 0, 0, 1]).all()

    # Only C-O bonds shorter than 1.4 (symbols and numbers mix freely).
    assert (np.bincount(neighbour_list("i", a, {("C", "O"): 1.4})) ==
            [1, 2, 1]).all()

    # Two different element-pair cutoffs at once.
    assert (np.bincount(neighbour_list("i", a, {("H", "C"): 1.2, (6, 8): 1.4}))
            == [1, 3, 1, 0, 1]).all()


def test_per_atom_radii_cutoff():
    a = molecule("HCOOH")
    a.center(vacuum=5.0)
    # Per-atom radii: pair is a neighbour when the spheres overlap.
    radii = np.full(len(a), 0.9)
    i_pa = neighbour_list("i", a, radii)
    i_global = neighbour_list("i", a, 1.8)  # equivalent global cutoff
    assert (np.bincount(i_pa, minlength=len(a)) ==
            np.bincount(i_global, minlength=len(a))).all()


def test_atoms_outside_box_are_wrapped():
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    ref_i = neighbour_list("i", a, 3.0)
    moved = a.copy()
    moved.positions[0] += moved.cell[0] + 2 * moved.cell[1]
    out_i = neighbour_list("i", moved, 3.0)
    assert (np.bincount(ref_i) == np.bincount(out_i)).all()


def test_supercell_coordination_consistency():
    # Per-atom coordination is invariant under building a supercell.
    a = bulk("Si", "diamond", a=5.43, cubic=True)
    c1 = np.bincount(neighbour_list("i", a, 2.6))
    rep = a.repeat((2, 1, 2))
    c2 = np.bincount(neighbour_list("i", rep, 2.6))
    assert set(np.unique(c1)) == set(np.unique(c2))
    assert (c2 == c1[0]).all()


def test_shrink_wrapped_positions_only():
    rng = np.random.default_rng(0)
    r = rng.uniform(0, 8, size=(30, 3))
    i, j, D, d = neighbour_list("ijDd", positions=r, cutoff=1.5)
    # Non-periodic shrink-wrapped: distance vectors are just r[j] - r[i].
    np.testing.assert_allclose(D, r[j] - r[i], atol=1e-12)
    np.testing.assert_allclose(d, np.linalg.norm(D, axis=1), atol=1e-12)
    assert (np.bincount(i) == np.bincount(j)).all()


def test_shift_is_zero_for_non_periodic_atom_outside_cell():
    # Regression test for matscipy PR #315 / issue #313: shifts must be zero
    # for a non-periodic system, even when an atom lies outside the (shrink-
    # wrapped or supplied) cell -- no cell boundary can be crossed. Otherwise
    # the contract D == r[j] - r[i] + S @ cell is silently violated.
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 1.2, 1.3]])
    for cell in (None, np.eye(3), np.diag([1.1, 1.2, 1.3])):
        i, j, S, D = neighbour_list(
            "ijSD", cutoff=5.0, positions=positions, cell=cell,
            pbc=[False, False, False])
        assert len(i) > 0  # the pair is within the cutoff
        np.testing.assert_array_equal(S, np.zeros_like(S))
        # D must be the direct difference for a non-periodic system.
        np.testing.assert_allclose(D, positions[j] - positions[i], atol=1e-12)


def test_first_neighbours_reference_values():
    np.testing.assert_array_equal(
        first_neighbours(5, [1, 1, 1, 1, 3, 3, 3]), [-1, 0, 4, 4, 7, 7])
    np.testing.assert_array_equal(
        first_neighbours(6, [0, 1, 2, 3, 4, 5]), [0, 1, 2, 3, 4, 5, 6])


def test_missing_cutoff_raises():
    a = bulk("Al", "fcc", a=4.05)
    with pytest.raises(ValueError):
        neighbour_list("i", a)


# ---------------------------------------------------------------------------
# mic: mixed periodicity
# ---------------------------------------------------------------------------

def _brute_force_mic(dr, cell, pbc):
    """Minimum image over explicit shifts along the periodic directions."""
    import itertools
    ranges = [range(-8, 9) if p else [0] for p in pbc]
    out = np.empty_like(dr)
    for p, v in enumerate(dr):
        images = [v + np.array(s) @ cell for s in itertools.product(*ranges)]
        out[p] = min(images, key=lambda w: w @ w)
    return out


def test_mic_orthorhombic_mixed_pbc_matches_brute_force():
    rng = np.random.default_rng(0)
    cell = np.diag([10.0, 9.0, 11.0])
    pbc = np.array([True, False, True])
    dr = rng.uniform(-30, 30, (300, 3))
    np.testing.assert_allclose(mic(dr, cell, pbc),
                               _brute_force_mic(dr, cell, pbc), atol=1e-12)


def test_mic_triclinic_never_shifts_along_non_periodic_direction():
    # Regression: masking the rows of inv(cell) (Cartesian components) instead
    # of its columns (lattice directions) shifted vectors along the
    # non-periodic direction of a sheared cell.
    rng = np.random.default_rng(1)
    cell = np.array([[10.0, 0, 0], [4.0, 9.0, 0], [1.0, 2.0, 11.0]])
    pbc = np.array([True, False, True])
    dr = rng.uniform(-30, 30, (500, 3))
    wrapped = mic(dr, cell, pbc)
    shifts = np.round((wrapped - dr) @ np.linalg.inv(cell))
    assert (shifts[:, 1] == 0).all()
    # ... and the shifts along the periodic directions are integers.
    np.testing.assert_allclose((wrapped - dr) @ np.linalg.inv(cell), shifts,
                               atol=1e-9)


def test_mic_triclinic_fully_periodic_reconstructs_neighbour_list():
    a = bulk("Cu", "fcc", a=3.6, cubic=False).repeat((3, 3, 3))
    a.rattle(0.05, seed=4)
    i, j, D = neighbour_list("ijD", a, 2.7)
    np.testing.assert_allclose(mic(a.positions[j] - a.positions[i], a.cell),
                               D, atol=1e-10)


# ---------------------------------------------------------------------------
# Missing / incomplete cells
# ---------------------------------------------------------------------------

def test_shrink_wrapped_cell_with_pbc_raises():
    # A shrink-wrapped box made atoms at opposite faces periodic images.
    pos = np.random.default_rng(2).uniform(0, 10, (30, 3))
    with pytest.raises(ValueError):
        neighbour_list("d", positions=pos, pbc=True, cutoff=3.0)


def test_molecule_without_cell():
    # ASE molecules carry an all-zero cell; the planar water molecule used to
    # give a singular shrink-wrapped cell.
    water = molecule("H2O")
    i, j, d = neighbour_list("ijd", water, 1.2)
    assert sorted(zip(i.tolist(), j.tolist())) == [(0, 1), (0, 2), (1, 0), (2, 0)]
    np.testing.assert_allclose(d, 0.969, atol=1e-3)
    assert (coordination(water, 1.2) == [2, 1, 1]).all()
    # Element-pair cutoffs on a molecule.
    i, j = neighbour_list("ij", water, {("H", "O"): 1.2})
    assert len(i) == 4
    i, j = neighbour_list("ij", water, {("H", "H"): 1.2})
    assert len(i) == 0


@pytest.mark.parametrize("positions", [
    np.zeros((1, 3)),                                    # single atom
    np.c_[np.arange(5.0), np.zeros(5), np.zeros(5)],     # a line
    np.c_[np.arange(4.0), np.arange(4.0) ** 2, np.zeros(4)],  # planar
])
def test_degenerate_configurations_without_cell(positions):
    i, j, d = neighbour_list("ijd", positions=positions, cutoff=1.5)
    assert (np.bincount(i, minlength=len(positions)) ==
            np.bincount(j, minlength=len(positions))).all()
    assert (d < 1.5).all()


def test_slab_with_zero_lattice_vector():
    from ase.build import fcc111
    slab = fcc111("Cu", (3, 3, 3), vacuum=None)   # zero c vector
    slab.pbc = [True, True, False]
    counts = np.bincount(neighbour_list("i", slab, 2.7))
    # Nearest neighbours only: 9 at the two surfaces, 12 in the middle layer.
    assert (counts[:9] == 9).all()
    assert (counts[9:18] == 12).all()
    assert (counts[18:] == 9).all()
    slab.pbc = True
    with pytest.raises(ValueError):
        neighbour_list("i", slab, 2.7)


def test_mic_strongly_sheared_cell_matches_brute_force():
    # Regression (review): rounding fractional coordinates is not a minimum
    # image for a skewed cell; the periodic basis must be reduced first.
    cell = np.array([[1.0, 0, 0], [0.9, 1.0, 0], [0, 0, 1.0]])
    dr = np.array([[0.931, 0.49, 0.0]])
    np.testing.assert_allclose(np.linalg.norm(mic(dr, cell), axis=1),
                               [0.494834], atol=1e-6)

    # Random strongly sheared cells. Fully periodic: against ASE's exact
    # find_mic (which reduces the basis too). Mixed periodicity: against a
    # brute force with wide lattice offsets in the original basis (ASE's
    # find_mic is not minimal for partially periodic skewed cells; with at
    # most two periodic directions the wide search stays cheap).
    import itertools
    from ase.geometry import find_mic
    rng = np.random.default_rng(7)
    for _ in range(20):
        cell = np.diag(rng.uniform(1, 3, 3)) + rng.uniform(-1.5, 1.5, (3, 3))
        if abs(np.linalg.det(cell)) < 0.3:
            continue
        pbc = (rng.integers(0, 2, 3).astype(bool) if rng.random() < 0.5
               else np.ones(3, bool))
        dr = rng.uniform(-6, 6, (100, 3))
        got = mic(dr, cell, pbc)
        if pbc.all():
            _, ref_len = find_mic(dr, cell, pbc)
        else:
            ranges = [range(-25, 26) if p else [0] for p in pbc]
            ref_len = np.array([
                min(np.linalg.norm(v + np.array(s) @ cell)
                    for s in itertools.product(*ranges)) for v in dr])
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), ref_len,
                                   atol=1e-9)
        # The result differs from the input by periodic lattice vectors only.
        shifts = (got - dr) @ np.linalg.inv(cell)
        np.testing.assert_allclose(shifts, np.round(shifts), atol=1e-9)
        assert np.allclose(shifts[:, ~pbc], 0)
