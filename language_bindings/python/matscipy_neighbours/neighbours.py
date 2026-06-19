"""Public neighbour-list API, compatible with ``matscipy.neighbours``.

Independent, MIT-licensed reimplementation on top of the ``_matscipy_neighbours``
C-extension. The C-extension signature (``neighbour_list(quantities,
cell_origin, cell, inv_cell, pbc, positions, cutoff, numbers)``) is the same one
matscipy's ``_matscipy`` backend exposes, so the two are interchangeable.
"""

import numpy as np

# The compiled extension. When this package and the extension are installed
# side by side the relative import wins; the fallback covers the in-tree test
# layout where the .so sits at the top of the build directory.
try:
    from . import _matscipy_neighbours as _ext
except ImportError:  # pragma: no cover - exercised only outside an install
    import _matscipy_neighbours as _ext

try:
    from ase.data import atomic_numbers as _atomic_numbers
except ImportError:  # pragma: no cover - ase is optional
    _atomic_numbers = {}

__all__ = [
    "neighbour_list",
    "first_neighbours",
    "get_jump_indicies",
    "triplet_list",
    "mic",
    "coordination",
]

# These are pure C-extension functions; re-export them unchanged.
first_neighbours = _ext.first_neighbours
get_jump_indicies = _ext.get_jump_indicies


def mic(dr, cell, pbc=None):
    """Apply the minimum image convention to an array of distance vectors.

    Parameters
    ----------
    dr : array_like
        Distance vectors, shape ``(n, 3)``.
    cell : array_like
        ``3x3`` cell matrix (rows are the lattice vectors).
    pbc : array_like, optional
        Per-direction periodicity. Defaults to periodic in all directions.

    Returns
    -------
    numpy.ndarray
        ``dr`` wrapped into the minimum image.
    """
    dr = np.asarray(dr, dtype=float)
    cell = np.asarray(cell, dtype=float)
    rec = np.linalg.inv(cell)
    if pbc is not None:
        rec = rec * np.asarray(pbc, dtype=int).reshape(3, 1)
    offset = np.round(dr @ rec)
    return dr - offset @ cell


class _DLPackArray:
    """Adapt a raw DLPack capsule from the C-extension to the ``from_dlpack``
    protocol. ``numpy``/``cupy`` ``from_dlpack`` call these two methods, then
    take ownership of the capsule (zero-copy)."""

    def __init__(self, capsule, device):
        self._capsule = capsule
        self._device = device  # (DLDeviceType, device_id)

    def __dlpack__(self, *args, **kwargs):
        return self._capsule

    def __dlpack_device__(self):
        return self._device


def _is_on_device(x):
    """True if ``x`` is a GPU array (e.g. cupy) — detected without importing
    cupy, via the CUDA Array Interface."""
    return hasattr(x, "__cuda_array_interface__") and not isinstance(x, np.ndarray)


def _neighbour_list_device(quantities, positions, cutoff, *, cell, pbc,
                           numbers, cell_origin):
    """GPU path: positions stay on the device, results come back as cupy arrays
    via DLPack (no host round-trip)."""
    if not getattr(_ext, "_has_gpu", 0):
        raise RuntimeError(
            "Device (GPU) positions were given, but the _matscipy_neighbours "
            "extension was built without a GPU backend (-DENABLE_CUDA=ON).")
    import cupy as cp

    positions = cp.ascontiguousarray(positions, dtype=cp.float64)
    nat = int(positions.shape[0])
    device_ptr = int(positions.data.ptr)
    device_id = int(positions.device.id)

    if cell is None:
        rmin = cp.asnumpy(positions.min(axis=0))
        rmax = cp.asnumpy(positions.max(axis=0))
        cell_origin = rmin if cell_origin is None else cell_origin
        cell = np.diag(rmax - rmin)
    if cell_origin is None:
        cell_origin = np.zeros(3)
    if pbc is None:
        pbc = np.zeros(3, dtype=bool)
    if numbers is None:
        numbers = np.ones(nat, dtype=np.int32)

    # All metadata is tiny and lives on the host.
    cell = np.ascontiguousarray(np.asarray(cell, dtype=float))
    cell_origin = np.ascontiguousarray(np.asarray(cell_origin, dtype=float))
    pbc = np.ascontiguousarray(np.broadcast_to(pbc, (3,)), dtype=bool)
    numbers = np.ascontiguousarray(np.asarray(numbers), dtype=np.int32)
    resolved_cutoff, numbers = _resolve_cutoff(cutoff, numbers)
    inv_cell = np.ascontiguousarray(np.linalg.inv(cell.T))

    # The extension reads positions from the device pointer; the host positions
    # argument is an unused placeholder (nat is passed explicitly).
    placeholder = np.empty((0, 3), dtype=float)
    capsules = _ext.neighbour_list_dlpack(
        quantities, cell_origin, cell, inv_cell, pbc, placeholder,
        resolved_cutoff, numbers, 1, device_ptr, nat)

    dev = (2, device_id)  # kDLCUDA
    arrays = tuple(cp.from_dlpack(_DLPackArray(c, dev)) for c in capsules)
    return arrays[0] if len(quantities) == 1 else arrays


def _resolve_cutoff(cutoff, numbers):
    """Turn a scalar / per-atom array / element-pair dict into a value the
    C-extension understands, plus the per-atom type array to pass along."""
    if isinstance(cutoff, dict):
        maxnum = int(np.max(numbers))
        matrix = np.zeros((maxnum + 1, maxnum + 1), dtype=float)
        for (el1, el2), c in cutoff.items():
            el1 = _atomic_numbers.get(el1, el1)
            el2 = _atomic_numbers.get(el2, el2)
            if el1 <= maxnum and el2 <= maxnum:
                matrix[el1, el2] = c
                matrix[el2, el1] = c
        return matrix, numbers
    # Scalar or per-atom array: types are unused by the extension.
    return cutoff, numbers


def neighbour_list(quantities, atoms=None, cutoff=None, *, positions=None,
                   cell=None, pbc=None, numbers=None, cell_origin=None):
    """Compute a neighbour list for an atomic configuration.

    Accepts either an ASE ``Atoms`` object or explicit ``positions``/``cell``/
    ``pbc``. Mirrors :func:`matscipy.neighbours.neighbour_list`.

    Parameters
    ----------
    quantities : str
        Any of ``i`` (first index), ``j`` (second index), ``d`` (distance),
        ``D`` (distance vector), ``S`` (cell shift). Outputs are returned in
        the order given; a single character returns a bare array.
    atoms : ase.Atoms, optional
        Atomic configuration. Mutually exclusive with the explicit arguments.
    cutoff : float, dict or array_like
        Global cutoff, ``{(el1, el2): cutoff}`` per element-pair dict (element
        numbers or symbols), or a per-atom radius array (pair cutoff = sum).
    positions, cell, pbc, numbers, cell_origin : array_like, optional
        Explicit configuration, used when ``atoms`` is not given.

    Returns
    -------
    tuple or numpy.ndarray
        One array per requested quantity. The shift ``S`` satisfies
        ``D == positions[j] - positions[i] + S @ cell``.
    """
    if cutoff is None:
        raise ValueError("Please provide a value for the cutoff radius.")

    # GPU path: if positions live on the device (e.g. a cupy array), run the
    # GPU backend and return device arrays without a host round-trip.
    if positions is not None and _is_on_device(positions):
        if atoms is not None:
            raise ValueError("Cannot combine an ASE Atoms object with device "
                             "positions.")
        return _neighbour_list_device(quantities, positions, cutoff, cell=cell,
                                      pbc=pbc, numbers=numbers,
                                      cell_origin=cell_origin)

    if atoms is not None:
        if positions is not None or cell is not None or pbc is not None or \
                numbers is not None or cell_origin is not None:
            raise ValueError("Cannot combine an ASE Atoms object with explicit "
                             "positions/cell/pbc/numbers/cell_origin.")
        positions = atoms.positions
        cell = np.asarray(atoms.cell)
        pbc = atoms.pbc
        numbers = atoms.numbers.astype(np.int32)
        cell_origin = np.zeros(3)
    else:
        if positions is None:
            raise ValueError("Provide either an ASE Atoms object or a "
                             "positions array.")
        positions = np.asarray(positions, dtype=float)
        if cell is None:
            # Shrink-wrapped cell around the atoms.
            rmin = positions.min(axis=0)
            rmax = positions.max(axis=0)
            cell_origin = rmin if cell_origin is None else cell_origin
            cell = np.diag(rmax - rmin)
        if cell_origin is None:
            cell_origin = np.zeros(3)
        if pbc is None:
            pbc = np.zeros(3, dtype=bool)
        if numbers is None:
            numbers = np.ones(len(positions), dtype=np.int32)

    cell = np.ascontiguousarray(cell, dtype=float)
    positions = np.ascontiguousarray(positions, dtype=float)
    cell_origin = np.ascontiguousarray(cell_origin, dtype=float)
    pbc = np.ascontiguousarray(np.broadcast_to(pbc, (3,)), dtype=bool)
    numbers = np.ascontiguousarray(numbers, dtype=np.int32)

    resolved_cutoff, numbers = _resolve_cutoff(cutoff, numbers)
    inv_cell = np.ascontiguousarray(np.linalg.inv(cell.T))

    return _ext.neighbour_list(quantities, cell_origin, cell, inv_cell, pbc,
                               positions, resolved_cutoff, numbers)


def triplet_list(first_neighbours, abs_dr_p=None, cutoff=None):
    """Compute a triplet list from a first-neighbour (row-start) array.

    Mirrors :func:`matscipy.neighbours.triplet_list` (without the optional
    ``jk_t`` output).
    """
    first_neighbours = np.ascontiguousarray(first_neighbours, dtype=np.int32)
    if abs_dr_p is not None and cutoff is not None:
        abs_dr_p = np.ascontiguousarray(abs_dr_p, dtype=float)
        return _ext.triplet_list(first_neighbours, abs_dr_p, float(cutoff))
    return _ext.triplet_list(first_neighbours)


def coordination(atoms, cutoff):
    """Number of neighbours of each atom within ``cutoff``."""
    i = neighbour_list("i", atoms, cutoff)
    return np.bincount(i, minlength=len(atoms))
