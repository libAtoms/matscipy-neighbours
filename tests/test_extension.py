"""Tests for the low-level ``_matscipy_neighbours`` C extension.

These call the C functions directly (no ase / matscipy Python wrapper needed):

    neighbour_list(quantities, cell_origin, cell, inv_cell, pbc, r, cutoffs,
                   [types])
    first_neighbours(n, i)
    get_jump_indicies(sorted_array)
    triplet_list(first_i, [abs_distances, cutoff])

The neighbour list itself is validated against an independent brute-force
reference implemented in pure Python/NumPy below.
"""

import itertools

import numpy as np
import pytest

import _matscipy_neighbours as nl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_args(cell, positions, pbc):
    """Build the argument arrays expected by the C ``neighbour_list``.

    ``cell`` is a 3x3 matrix whose *rows* are the lattice vectors. The C code
    computes fractional coordinates as ``inv_cell . r``, so the matrix it needs
    is ``inv(cell.T)``.
    """
    cell = np.ascontiguousarray(cell, dtype=np.float64)
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    cell_origin = np.zeros(3, dtype=np.float64)
    inv_cell = np.ascontiguousarray(np.linalg.inv(cell.T), dtype=np.float64)
    pbc = np.ascontiguousarray(np.broadcast_to(pbc, (3,)), dtype=bool)
    return cell_origin, cell, inv_cell, pbc, positions


def neighbour_list(quantities, cell, positions, cutoff, pbc=True, types=None):
    cell_origin, cell, inv_cell, pbc, positions = make_args(cell, positions, pbc)
    args = [quantities, cell_origin, cell, inv_cell, pbc, positions, cutoff]
    if types is not None:
        args.append(np.ascontiguousarray(types, dtype=np.int32))
    return nl.neighbour_list(*args)


def brute_force(cell, positions, pbc, cutoff):
    """O(N^2) reference neighbour list with explicit periodic images.

    Returns a sorted list of (i, j) pairs (with multiplicity, one per image)
    with distance strictly less than ``cutoff``.
    """
    cell = np.asarray(cell, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    pbc = np.broadcast_to(pbc, (3,))
    n = len(pos)

    # Wrap atoms into the cell along periodic axes (the C code does the
    # equivalent via bin_wrap), so a small image range suffices regardless of
    # how far outside the box an atom starts.
    frac = pos @ np.linalg.inv(cell)
    for k in range(3):
        if pbc[k]:
            frac[:, k] -= np.floor(frac[:, k])
    pos = frac @ cell

    volume = abs(np.linalg.det(cell))
    img_range = []
    for k in range(3):
        if pbc[k]:
            a, b = cell[(k + 1) % 3], cell[(k + 2) % 3]
            face_dist = volume / np.linalg.norm(np.cross(a, b))
            img_range.append(int(np.ceil(cutoff / face_dist)))
        else:
            img_range.append(0)

    shifts = list(itertools.product(
        *[range(-r, r + 1) for r in img_range]))

    pairs = []
    for i in range(n):
        for j in range(n):
            for s in shifts:
                if i == j and s == (0, 0, 0):
                    continue
                d = pos[j] + np.array(s) @ cell - pos[i]
                if np.linalg.norm(d) < cutoff:
                    pairs.append((i, j))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# neighbour_list: small cell / periodic images (mirrors upstream test_small_cell)
# ---------------------------------------------------------------------------

CUBIC = np.eye(3)


def test_single_atom_unit_cube_counts():
    pos = [[0.5, 0.5, 0.5]]
    i, j, D, S = neighbour_list("ijDS", CUBIC, pos, 1.1, pbc=True)
    assert np.bincount(i)[0] == 6           # 6 faces at distance 1
    # For a unit cube and a self-pair, D == shift @ cell == shift.
    assert (D == S).all()

    i = neighbour_list("i", CUBIC, pos, 1.5, pbc=True)
    assert np.bincount(i)[0] == 18          # +12 edge images at sqrt(2)


def test_single_atom_mixed_pbc():
    pos = [[0.5, 0.5, 0.5]]
    assert len(neighbour_list("i", CUBIC, pos, 1.1, pbc=False)) == 0
    assert np.bincount(neighbour_list(
        "i", CUBIC, pos, 1.1, pbc=[True, False, False]))[0] == 2
    assert np.bincount(neighbour_list(
        "i", CUBIC, pos, 1.1, pbc=[True, False, True]))[0] == 4


# ---------------------------------------------------------------------------
# neighbour_list validated against the brute-force reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pbc", [
    True, False, (True, False, True), (False, True, False)])
def test_brute_force_cubic(pbc):
    rng = np.random.default_rng(1234)
    cell = 5.0 * np.eye(3)
    pos = rng.uniform(0, 5.0, size=(40, 3))
    i, j = neighbour_list("ij", cell, pos, 1.7, pbc=pbc)
    got = sorted(zip(i.tolist(), j.tolist()))
    assert got == brute_force(cell, pos, pbc, 1.7)


@pytest.mark.parametrize("pbc", [True, False, (True, True, False)])
def test_brute_force_triclinic(pbc):
    rng = np.random.default_rng(7)
    cell = np.array([[4.0, 0.0, 0.0],
                     [1.3, 3.7, 0.0],
                     [0.8, 0.6, 4.2]])
    # Scatter atoms, some deliberately outside the cell.
    frac = rng.uniform(-0.3, 1.3, size=(30, 3))
    pos = frac @ cell
    i, j = neighbour_list("ij", cell, pos, 1.6, pbc=pbc)
    got = sorted(zip(i.tolist(), j.tolist()))
    assert got == brute_force(cell, pos, pbc, 1.6)


@pytest.mark.parametrize("pbc", [True, False, (True, False, True)])
def test_sparse_backend_vacuum(pbc):
    """A small cluster in a big box has a huge cell grid -> hashed (compact)
    backend. Results must still match brute force."""
    rng = np.random.default_rng(3)
    cell = 120.0 * np.eye(3)          # ~1.7e6 cells >> 8*natoms -> sparse path
    pos = rng.uniform(0, 6.0, size=(250, 3))
    i, j = neighbour_list("ij", cell, pos, 1.5, pbc=pbc)
    got = sorted(zip(i.tolist(), j.tolist()))
    assert got == brute_force(cell, pos, pbc, 1.5)


def test_atoms_outside_box_match_brute_force():
    """Atoms displaced far outside the cell must still be handled (wrap/trunc)."""
    rng = np.random.default_rng(99)
    cell = 6.0 * np.eye(3)
    pos = rng.uniform(0, 6.0, size=(25, 3))
    pos[3] += cell[0]          # shove a few outside
    pos[7] -= 2 * cell[1]
    pos[11] += 3 * cell[2]
    for pbc in (True, False, (True, False, True)):
        i, j = neighbour_list("ij", cell, pos, 1.5, pbc=pbc)
        got = sorted(zip(i.tolist(), j.tolist()))
        assert got == brute_force(cell, pos, pbc, 1.5)


# ---------------------------------------------------------------------------
# neighbour_list: returned quantities are self-consistent
# ---------------------------------------------------------------------------

def test_distance_vector_and_shift_invariant():
    rng = np.random.default_rng(2024)
    cell = np.array([[5.0, 0.0, 0.0],
                     [0.5, 4.5, 0.0],
                     [0.2, 0.3, 5.5]])
    pos = rng.uniform(-1, 6, size=(35, 3))
    i, j, D, d, S = neighbour_list("ijDdS", cell, pos, 1.9, pbc=True)

    # D == r[j] - r[i] + S @ cell
    expect = pos[j] - pos[i] + S @ cell
    np.testing.assert_allclose(D, expect, atol=1e-10)
    # abs distance == |D|
    np.testing.assert_allclose(d, np.linalg.norm(D, axis=1), atol=1e-12)
    # symmetry: every i appears as a j the same number of times
    assert (np.bincount(i) == np.bincount(j)).all()


def test_no_self_pairs():
    rng = np.random.default_rng(5)
    cell = 4.0 * np.eye(3)
    pos = rng.uniform(0, 4.0, size=(20, 3))
    # Non-periodic: an atom can never be its own neighbour.
    i, j = neighbour_list("ij", cell, pos, 1.5, pbc=False)
    assert not np.any(i == j)


# ---------------------------------------------------------------------------
# neighbour_list: per-atom and per-type cutoffs
# ---------------------------------------------------------------------------

def test_per_atom_cutoffs():
    cell = 10.0 * np.eye(3)
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.5, 0, 0]])
    # pair cutoff is the sum of the two per-atom radii
    radii = np.array([0.4, 0.4, 1.2])      # 0-1: 0.8 (no), 1-2: 1.6 (yes)
    i, j = neighbour_list("ij", cell, pos, radii, pbc=False)
    got = sorted(zip(i.tolist(), j.tolist()))
    assert got == [(1, 2), (2, 1)]


def test_per_type_cutoffs():
    cell = 10.0 * np.eye(3)
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    types = np.array([0, 1, 0], dtype=np.int32)
    # 2x2 cutoff matrix: only the (0,1)/(1,0) interaction within 1.5 counts
    cutoffs = np.array([[0.5, 1.5],
                        [1.5, 0.5]])
    i, j = neighbour_list("ij", cell, pos, cutoffs, pbc=False, types=types)
    got = sorted(zip(i.tolist(), j.tolist()))
    # 0-1 (d=1) yes; 1-2 (d=1) yes; 0-2 (d=2, same type 0, cutoff 0.5) no
    assert got == [(0, 1), (1, 0), (1, 2), (2, 1)]


# ---------------------------------------------------------------------------
# neighbour_list: error handling and return shape
# ---------------------------------------------------------------------------

def test_unsupported_quantity_raises():
    pos = [[0.5, 0.5, 0.5]]
    with pytest.raises(ValueError):
        neighbour_list("x", CUBIC, pos, 1.1, pbc=True)


def test_wrong_number_of_cutoffs_raises():
    cell = 10.0 * np.eye(3)
    pos = np.zeros((10, 3))
    with pytest.raises(TypeError):
        neighbour_list("ij", cell, pos, np.ones(9), pbc=False)


def test_single_quantity_returns_bare_array():
    pos = [[0.5, 0.5, 0.5]]
    out = neighbour_list("i", CUBIC, pos, 1.1, pbc=True)
    assert isinstance(out, np.ndarray)        # not a tuple


def test_zero_cell_volume_raises():
    bad_cell = np.array([[1.0, 0, 0], [2.0, 0, 0], [0, 0, 1.0]])
    pos = [[0.0, 0, 0]]
    with pytest.raises(RuntimeError):
        # inv(cell.T) is singular -> build inv ourselves to reach the C check
        cell_origin = np.zeros(3)
        inv_cell = np.zeros((3, 3))
        nl.neighbour_list("i", cell_origin, bad_cell, inv_cell,
                          np.array([True, True, True]),
                          np.ascontiguousarray(pos, dtype=float), 1.0)


# ---------------------------------------------------------------------------
# first_neighbours (seed/row-start array)
# ---------------------------------------------------------------------------

def _fn(n, i):
    return nl.first_neighbours(n, np.array(i, dtype=np.int32))


def test_first_neighbours_reference_values():
    # values taken from the upstream matscipy test suite
    np.testing.assert_array_equal(_fn(5, [1, 1, 1, 1, 3, 3, 3]),
                                  [-1, 0, 4, 4, 7, 7])
    np.testing.assert_array_equal(_fn(6, [0, 1, 2, 3, 4, 5]),
                                  [0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(_fn(8, [0, 1, 2, 3, 5, 6]),
                                  [0, 1, 2, 3, 4, 4, 5, 6, 6])
    np.testing.assert_array_equal(_fn(8, [0, 1, 2, 3, 3, 5, 6]),
                                  [0, 1, 2, 3, 5, 5, 6, 7, 7])


def test_first_neighbours_empty():
    # Regression test: an empty neighbour list must not read out of bounds.
    out = _fn(5, [])
    np.testing.assert_array_equal(out, [0, 0, 0, 0, 0, 0])


# ---------------------------------------------------------------------------
# get_jump_indicies
# ---------------------------------------------------------------------------

def test_get_jump_indicies():
    out = nl.get_jump_indicies(np.array(
        [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        dtype=np.int32))
    np.testing.assert_array_equal(out, [0, 3, 8, 11, 15, 19])

    out = nl.get_jump_indicies(np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(out, [0, 1])


# ---------------------------------------------------------------------------
# triplet_list
# ---------------------------------------------------------------------------

def test_triplet_list_no_cutoff():
    first_i = np.array([0, 2, 6, 10], dtype=np.int32)
    ij, ik = nl.triplet_list(first_i)
    ij_comp = [0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5,
               5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
    ik_comp = [1, 0, 3, 4, 5, 2, 4, 5, 2, 3, 5, 2, 3,
               4, 7, 8, 9, 6, 8, 9, 6, 7, 9, 6, 7, 8]
    np.testing.assert_array_equal(ij, ij_comp)
    np.testing.assert_array_equal(ik, ik_comp)


def test_triplet_list_with_cutoff():
    first_i = np.array([0, 2, 6, 10], dtype=np.int32)
    absdist = np.array([2.2] * 4 + [3.0] * 2 + [2.0] * 4, dtype=np.float64)
    ij, ik = nl.triplet_list(first_i, absdist, 2.6)
    ij_comp = [0, 1, 2, 3, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
    ik_comp = [1, 0, 3, 2, 7, 8, 9, 6, 8, 9, 6, 7, 9, 6, 7, 8]
    np.testing.assert_array_equal(ij, ij_comp)
    np.testing.assert_array_equal(ik, ik_comp)


def test_triplet_list_cutoff_requires_distances():
    first_i = np.array([0, 2, 6, 10], dtype=np.int32)
    with pytest.raises(TypeError):
        nl.triplet_list(first_i, np.ones(10))   # cutoff missing
