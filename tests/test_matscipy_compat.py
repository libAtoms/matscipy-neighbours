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


def test_first_neighbours_reference_values():
    np.testing.assert_array_equal(
        first_neighbours(5, [1, 1, 1, 1, 3, 3, 3]), [-1, 0, 4, 4, 7, 7])
    np.testing.assert_array_equal(
        first_neighbours(6, [0, 1, 2, 3, 4, 5]), [0, 1, 2, 3, 4, 5, 6])


def test_missing_cutoff_raises():
    a = bulk("Al", "fcc", a=4.05)
    with pytest.raises(ValueError):
        neighbour_list("i", a)
