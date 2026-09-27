"""Input validation: every malformed input raises a Python exception.

Each case here used to read or write out of bounds, abort the interpreter
(uncaught C++ exception), hang, or silently return wrong pairs. The tests call
both the public wrapper and the raw extension so that both layers are covered.
"""

import numpy as np
import pytest

import _matscipy_neighbours as ext
from matscipy_neighbours import (
    first_neighbours,
    get_jump_indicies,
    neighbour_list,
    neighbour_matrix,
    triplet_list,
)

CELL = 10.0 * np.eye(3)
INV = np.linalg.inv(CELL.T)
PBC = np.ones(3, dtype=bool)
POS = np.random.default_rng(0).uniform(0, 10, (50, 3))
TYPES = np.ones(50, dtype=np.int64)


def raw(quantities="i", origin=np.zeros(3), cell=CELL, inv=INV, pbc=PBC,
        pos=POS, cutoff=3.0, types=TYPES):
    """Call the extension directly, bypassing the wrapper's normalisation."""
    return ext.neighbour_list(quantities, origin, cell, inv, pbc, pos, cutoff,
                              types)


# ---------------------------------------------------------------------------
# Array shapes (extension level)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(pos=POS[:, :2].copy()),
    dict(pos=np.hstack([POS, POS[:, :1]])),
    dict(cell=np.eye(2)),
    dict(cell=CELL[:2]),
    dict(inv=INV[:, :2].copy()),
    dict(pbc=np.array([True, True])),
    dict(origin=np.zeros(1)),
    dict(types=np.ones(7, dtype=np.int64)),
])
def test_wrong_array_shapes_raise(kwargs):
    with pytest.raises(TypeError):
        raw(**kwargs)


def test_wrapper_rejects_wrong_position_shape():
    with pytest.raises(TypeError):
        neighbour_list("i", positions=POS[:, :2].copy(), cell=CELL, pbc=True,
                       cutoff=3.0)


# ---------------------------------------------------------------------------
# Cutoffs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cutoff", [0.0, -1.0, np.nan, np.inf])
def test_invalid_scalar_cutoff_raises(cutoff):
    with pytest.raises(ValueError):
        neighbour_list("i", positions=POS, cell=CELL, pbc=True, cutoff=cutoff)


@pytest.mark.parametrize("cutoff", [3, np.int64(3), np.float32(3.0),
                                    np.array(3.0)])
def test_scalar_cutoff_of_any_numeric_type(cutoff):
    ref = neighbour_list("i", positions=POS, cell=CELL, pbc=True, cutoff=3.0)
    got = neighbour_list("i", positions=POS, cell=CELL, pbc=True, cutoff=cutoff)
    np.testing.assert_array_equal(got, ref)


def test_string_cutoff_raises():
    with pytest.raises(TypeError):
        raw(cutoff="3")


@pytest.mark.parametrize("array_namespace", [None, np])
def test_per_atom_cutoff_wrong_length_raises(array_namespace):
    with pytest.raises(TypeError):
        neighbour_list("i", positions=POS, cell=CELL, pbc=True,
                       cutoff=np.ones(5), array_namespace=array_namespace)


def test_per_atom_cutoff_negative_or_nan_raises():
    with pytest.raises(ValueError):
        neighbour_list("i", positions=POS, cell=CELL, pbc=True,
                       cutoff=-np.ones(50))
    bad = np.ones(50)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        neighbour_list("i", positions=POS, cell=CELL, pbc=True, cutoff=bad)


@pytest.mark.parametrize("array_namespace", [None, np])
def test_per_type_cutoff_not_square_raises(array_namespace):
    with pytest.raises((TypeError, ValueError)):
        neighbour_list("i", positions=POS, cell=CELL, pbc=True,
                       cutoff=np.ones((3, 2)), numbers=np.zeros(50, int),
                       array_namespace=array_namespace)


def test_type_outside_per_type_matrix_raises():
    # Used to fall back silently to the *maximum* cutoff for such atoms.
    matrix = np.array([[1.0, 1.0], [1.0, 3.0]])
    for numbers in (np.full(50, 7), np.full(50, -1)):
        with pytest.raises(ValueError):
            neighbour_list("i", positions=POS, cell=CELL, pbc=True,
                           cutoff=matrix, numbers=numbers)
    # Bypassing the wrapper's check, the core rejects it too.
    with pytest.raises(ValueError):
        raw(cutoff=matrix, types=np.full(50, 7, dtype=np.int64))


def test_per_type_matrix_without_types_uses_wrapper_default():
    # The wrapper passes numbers=1 for every atom; a 2x2 matrix covers that.
    matrix = np.array([[1.0, 1.0], [1.0, 3.0]])
    ref = neighbour_list("i", positions=POS, cell=CELL, pbc=True, cutoff=3.0)
    got = neighbour_list("i", positions=POS, cell=CELL, pbc=True,
                         cutoff=matrix)
    np.testing.assert_array_equal(got, ref)


def test_unknown_element_symbol_in_dict_raises():
    with pytest.raises(ValueError):
        neighbour_list("i", positions=POS, cell=CELL, pbc=True,
                       cutoff={("Xx", "Yy"): 3.0})


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_positions_raise(value):
    # A NaN position used to pass the distance test and yield NaN pairs.
    pos = POS.copy()
    pos[20, 1] = value
    with pytest.raises(ValueError):
        neighbour_list("i", positions=pos, cell=CELL, pbc=True, cutoff=3.0)


def test_astronomically_far_atoms_terminate_quickly():
    # The cell index of an atom 1e30 cells away is clamped, not undefined, and
    # wrapping it is O(1). The far atoms (all at the same x) may only pair
    # with each other, never with the atoms in the box.
    pos = POS.copy()
    pos[:5, 0] = 1e30
    i, j = neighbour_list("ij", positions=pos, cell=CELL, pbc=True, cutoff=3.0)
    far_i, far_j = i < 5, j < 5
    assert (far_i == far_j).all()


def test_empty_positions():
    i, j = neighbour_list("ij", positions=np.zeros((0, 3)), cell=CELL,
                          pbc=True, cutoff=3.0)
    assert i.shape == (0,) and j.shape == (0,)
    i = neighbour_list("i", positions=np.zeros((0, 3)), cutoff=3.0)
    assert i.shape == (0,)


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def test_periodic_without_cell_raises():
    with pytest.raises(ValueError):
        neighbour_list("i", positions=POS, pbc=True, cutoff=3.0)


def test_zero_lattice_vector_in_periodic_direction_raises():
    cell = CELL.copy()
    cell[2] = 0
    with pytest.raises(ValueError):
        neighbour_list("i", positions=POS, cell=cell, pbc=True, cutoff=3.0)


def test_degenerate_cell_raises_runtime_error():
    bad = np.array([[1.0, 0, 0], [2.0, 0, 0], [0, 0, 1.0]])
    with pytest.raises(RuntimeError):
        raw(cell=bad, inv=np.eye(3), pos=POS[:3], types=TYPES[:3])


def test_huge_cell_with_small_cutoff():
    # 2.2e6 cells per direction would overflow a 64-bit cell count; the
    # resolution is clamped instead.
    pos = np.random.default_rng(1).uniform(0, 10, (10, 3))
    i = neighbour_list("i", positions=pos, cell=2.2e6 * np.eye(3), pbc=False,
                       cutoff=1.0)
    assert i.shape[0] >= 0


# ---------------------------------------------------------------------------
# first_neighbours / get_jump_indicies / triplet_list
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, i", [
    (3, [0, 1, 7]),      # index beyond n (was a heap overflow)
    (3, [-2, 1]),        # negative index (was heap corruption)
    (3, [2, 1]),         # unsorted
    (-5, [0, 1]),        # negative n (was an interpreter abort)
])
def test_first_neighbours_rejects_bad_input(n, i):
    with pytest.raises(ValueError):
        first_neighbours(n, np.array(i))


def test_first_neighbours_accepts_valid_input():
    seed = first_neighbours(5, np.array([1, 1, 1, 1, 3, 3, 3]))
    np.testing.assert_array_equal(seed, [-1, 0, 4, 4, 7, 7])
    assert seed.dtype == np.int64


@pytest.mark.parametrize("sorted_array", [[0, 0, 2], [1, 1, 2], [0, 1, 0]])
def test_get_jump_indicies_rejects_bad_input(sorted_array):
    with pytest.raises(ValueError):
        get_jump_indicies(np.array(sorted_array))


def test_triplet_list_rejects_bad_row_starts():
    with pytest.raises(ValueError):
        triplet_list([0, 10], np.ones(3), 1.0)      # beyond the distances
    with pytest.raises(ValueError):
        triplet_list([5, 0])                         # decreasing
    with pytest.raises(ValueError):
        triplet_list([0, 2, 4], cutoff=1.0)          # cutoff without distances
    with pytest.raises(ValueError):
        triplet_list([-1, 3])                        # -1 not followed by 0


def test_triplet_list_accepts_leading_minus_one():
    # first_neighbours() emits -1 for atoms before the first pair.
    ij, ik = triplet_list([-1, -1, 0, 2], np.ones(2), 2.0)
    np.testing.assert_array_equal(ij, [0, 1])
    np.testing.assert_array_equal(ik, [1, 0])


def test_triplet_list_accepts_integer_cutoff():
    ij, _ = triplet_list([0, 2, 4], np.ones(4), 2)
    assert len(ij) == 4


# ---------------------------------------------------------------------------
# neighbour_matrix
# ---------------------------------------------------------------------------

def test_neighbour_matrix_rejects_negative_capacity():
    # Used to throw std::length_error out of the extension and abort.
    with pytest.raises(ValueError):
        neighbour_matrix(positions=POS, cell=CELL, pbc=True, cutoff=3.0,
                         max_neighbours=-1)


def test_neighbour_matrix_zero_capacity_overflows_cleanly():
    with pytest.raises(ValueError):
        neighbour_matrix(positions=POS, cell=CELL, pbc=True, cutoff=3.0,
                         max_neighbours=0)


# ---------------------------------------------------------------------------
# DLPack binding specifics
# ---------------------------------------------------------------------------

def test_device_positions_with_cpu_backend_raise():
    # Positions that claim to be on a device cannot be used by the CPU backend.
    with pytest.raises(TypeError):
        ext.neighbour_list_dlpack("i", np.zeros(3), CELL, INV, PBC,
                                  np.zeros((0, 3)), 3.0, None, 0, POS, -1)


def test_empty_quantity_string_returns_empty_tuple():
    assert neighbour_list("", positions=POS, cell=CELL, pbc=True,
                          cutoff=3.0) == ()
