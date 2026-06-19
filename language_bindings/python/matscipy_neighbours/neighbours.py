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
    "DLPackTensor",
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


class DLPackTensor:
    """A zero-copy DLPack tensor produced by the neighbour list. It implements
    the ``__dlpack__`` / ``__dlpack_device__`` protocol, so any consumer
    (``numpy``/``cupy``/``torch``/``jax``) can adopt it with ``from_dlpack``.
    Request these directly with ``array_namespace="dlpack"``."""

    __slots__ = ("_capsule", "_device")

    def __init__(self, capsule, device):
        self._capsule = capsule
        self._device = device  # (DLDeviceType, device_id)

    def __dlpack__(self, *args, **kwargs):
        return self._capsule

    def __dlpack_device__(self):
        return self._device


# Backwards-compatible internal alias.
_DLPackArray = DLPackTensor

_DLPACK_CPU = 1
_DLPACK_CUDA = 2
_DLPACK_ROCM = 10


def _dlpack_device(x):
    """``(device_type, device_id)`` of an array via the DLPack protocol, or
    None if ``x`` does not expose it."""
    fn = getattr(x, "__dlpack_device__", None)
    if fn is None:
        return None
    try:
        dt, did = fn()
        return (int(dt), int(did))
    except Exception:  # pragma: no cover - defensive
        return None


def _is_on_device(x):
    """True if ``x`` lives on a GPU (its DLPack device is not the CPU)."""
    dev = _dlpack_device(x)
    return dev is not None and dev[0] != _DLPACK_CPU


def _resolve_device(device, on_device_input):
    """Map the ``device=`` argument + input location to (use_gpu, device_id).

    device: None (auto), "cpu", "cuda"/"gpu", an int id, or ("cuda", id)."""
    if device is None:
        return (on_device_input, -1)
    if isinstance(device, str):
        d = device.lower()
        if d == "cpu":
            return (False, -1)
        if d in ("cuda", "gpu", "rocm", "hip"):
            return (True, -1)
        raise ValueError(f"Unknown device {device!r}.")
    if isinstance(device, int):
        return (True, device)
    if isinstance(device, (tuple, list)) and len(device) == 2:
        return (True, int(device[1]))
    raise ValueError(f"Unknown device {device!r}.")


def _consume(wrappers, array_namespace, use_gpu):
    """Turn DLPack tensors into the requested array type. ``None`` => default
    (cupy on device, numpy on host); ``"dlpack"`` => the tensors themselves; a
    module => its ``from_dlpack``."""
    if array_namespace == "dlpack":
        return wrappers
    if array_namespace is None:
        if use_gpu:
            import cupy as array_namespace
        else:
            array_namespace = np
    return [array_namespace.from_dlpack(w) for w in wrappers]


def _host_metadata(cell, pbc, numbers, cell_origin, nat, *, positions=None):
    """Normalise the (small, host-resident) geometry/type arrays. ``positions``
    is used only to shrink-wrap a cell when none is given (host arrays only)."""
    if cell is None:
        r = np.asarray(positions, dtype=float)
        rmin, rmax = r.min(axis=0), r.max(axis=0)
        cell_origin = rmin if cell_origin is None else cell_origin
        cell = np.diag(rmax - rmin)
    if cell_origin is None:
        cell_origin = np.zeros(3)
    if pbc is None:
        pbc = np.zeros(3, dtype=bool)
    if numbers is None:
        numbers = np.ones(nat, dtype=np.int32)
    cell = np.ascontiguousarray(np.asarray(cell, dtype=float))
    cell_origin = np.ascontiguousarray(np.asarray(cell_origin, dtype=float))
    pbc = np.ascontiguousarray(np.broadcast_to(pbc, (3,)), dtype=bool)
    numbers = np.ascontiguousarray(np.asarray(numbers), dtype=np.int32)
    inv_cell = np.ascontiguousarray(np.linalg.inv(cell.T))
    return cell_origin, cell, inv_cell, pbc, numbers


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


def _gather(atoms, positions, cell, pbc, numbers, cell_origin):
    """Normalise the ASE-Atoms-or-explicit inputs to a 5-tuple, leaving a device
    positions array untouched (so it can stay on the GPU)."""
    if atoms is not None:
        if any(x is not None
               for x in (positions, cell, pbc, numbers, cell_origin)):
            raise ValueError("Cannot combine an ASE Atoms object with explicit "
                             "positions/cell/pbc/numbers/cell_origin.")
        return (atoms.positions, np.asarray(atoms.cell), atoms.pbc,
                atoms.numbers.astype(np.int32), np.zeros(3))
    if positions is None:
        raise ValueError("Provide either an ASE Atoms object or a positions "
                         "array.")
    return positions, cell, pbc, numbers, cell_origin


def neighbour_list(quantities, atoms=None, cutoff=None, *, positions=None,
                   cell=None, pbc=None, numbers=None, cell_origin=None,
                   device=None, array_namespace=None):
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
        Global cutoff, ``{(el1, el2): cutoff}`` per element-pair dict, or a
        per-atom radius array (pair cutoff = sum).
    positions, cell, pbc, numbers, cell_origin : array_like, optional
        Explicit configuration. ``positions`` may be a device array (cupy /
        torch / jax / any ``__dlpack__`` producer); then the GPU backend runs
        and results stay on the device. ``cell`` is required for device input.
    device : optional
        ``None`` (auto: follow the input), ``"cpu"``, ``"cuda"``/``"gpu"``, an
        integer device id, or ``("cuda", id)`` — to force the backend/device.
    array_namespace : optional
        Output type. ``None`` (default): cupy on the GPU, numpy on the host.
        A module with ``from_dlpack`` (numpy/cupy/torch/jax.numpy): return that.
        ``"dlpack"``: return :class:`DLPackTensor` capsules for the caller to
        consume with its own ``from_dlpack``.

    Returns
    -------
    tuple or array
        One array per requested quantity; a single character returns a bare
        array. The shift ``S`` satisfies ``D == r[j] - r[i] + S @ cell``.
    """
    if cutoff is None:
        raise ValueError("Please provide a value for the cutoff radius.")
    positions, cell, pbc, numbers, cell_origin = _gather(
        atoms, positions, cell, pbc, numbers, cell_origin)

    on_device = _is_on_device(positions)
    use_gpu, device_id = _resolve_device(device, on_device)
    if on_device and not use_gpu:
        raise ValueError("device='cpu' with device-resident positions is not "
                         "supported; move the array to the host first.")

    # Fast host path: numpy in, numpy out, no DLPack round-trip.
    if not use_gpu and array_namespace is None:
        positions = np.ascontiguousarray(positions, dtype=float)
        co, ce, inv, pb, nums = _host_metadata(cell, pbc, numbers, cell_origin,
                                               len(positions),
                                               positions=positions)
        rc, nums = _resolve_cutoff(cutoff, nums)
        return _ext.neighbour_list(quantities, co, ce, inv, pb, positions, rc,
                                   nums)

    # DLPack path: device backend, and/or a non-default output framework.
    if use_gpu and not getattr(_ext, "_has_gpu", 0):
        raise RuntimeError("GPU requested but the extension was built without a "
                           "GPU backend (-DENABLE_CUDA=ON).")
    if use_gpu and on_device and cell is None:
        raise ValueError("cell must be given for device-resident positions.")
    nat = int(positions.shape[0])
    if on_device:
        py_in, host_pos = positions, np.empty((0, 3), dtype=float)
    else:
        py_in, host_pos = None, np.ascontiguousarray(positions, dtype=float)
        if use_gpu and device_id < 0:
            device_id = 0  # default device for a host->GPU upload
    co, ce, inv, pb, nums = _host_metadata(
        cell, pbc, numbers, cell_origin, nat,
        positions=None if on_device else host_pos)
    rc, nums = _resolve_cutoff(cutoff, nums)

    capsules = _ext.neighbour_list_dlpack(quantities, co, ce, inv, pb, host_pos,
                                          rc, nums, 1 if use_gpu else 0, py_in,
                                          device_id)
    if use_gpu:
        dtype = getattr(_ext, "_device_type", _DLPACK_CUDA)
        out_id = _dlpack_device(positions)[1] if on_device else device_id
        out_dev = (dtype, out_id)
    else:
        out_dev = (_DLPACK_CPU, 0)
    wrappers = [DLPackTensor(c, out_dev) for c in capsules]
    arrays = _consume(wrappers, array_namespace, use_gpu)
    return arrays[0] if len(quantities) == 1 else tuple(arrays)


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


def coordination(atoms=None, cutoff=None, *, positions=None, cell=None,
                 pbc=None, numbers=None, cell_origin=None,
                 device=None, array_namespace=None):
    """Number of neighbours of each atom within ``cutoff``.

    With device ``positions`` (or ``device=`` forcing the GPU) this runs the
    count-only kernel — never materialising the pair list — and returns a device
    array; otherwise it counts the host neighbour list. ``device`` and
    ``array_namespace`` behave as in :func:`neighbour_list`."""
    positions, cell, pbc, numbers, cell_origin = _gather(
        atoms, positions, cell, pbc, numbers, cell_origin)
    on_device = _is_on_device(positions)
    use_gpu, device_id = _resolve_device(device, on_device)
    if on_device and not use_gpu:
        raise ValueError("device='cpu' with device-resident positions is not "
                         "supported; move the array to the host first.")

    if use_gpu:
        if not getattr(_ext, "_has_gpu", 0):
            raise RuntimeError("GPU requested but the extension was built "
                               "without a GPU backend (-DENABLE_CUDA=ON).")
        if on_device and cell is None:
            raise ValueError("cell must be given for device-resident positions.")
        nat = int(positions.shape[0])
        if on_device:
            py_in, host_pos = positions, np.empty((0, 3), dtype=float)
        else:
            py_in, host_pos = None, np.ascontiguousarray(positions, dtype=float)
            if device_id < 0:
                device_id = 0
        co, ce, inv, pb, nums = _host_metadata(
            cell, pbc, numbers, cell_origin, nat,
            positions=None if on_device else host_pos)
        rc, nums = _resolve_cutoff(cutoff, nums)
        capsule = _ext.coordination_dlpack(co, ce, inv, pb, host_pos, rc, nums,
                                           py_in, device_id)
        dtype = getattr(_ext, "_device_type", _DLPACK_CUDA)
        out_id = _dlpack_device(positions)[1] if on_device else device_id
        return _consume([DLPackTensor(capsule, (dtype, out_id))],
                        array_namespace, True)[0]

    # Host: count the pair list.
    i = neighbour_list("i", cutoff=cutoff, positions=positions, cell=cell,
                       pbc=pbc, numbers=numbers, cell_origin=cell_origin)
    counts = np.bincount(i, minlength=int(np.asarray(positions).shape[0]))
    if array_namespace in (None,):
        return counts
    if array_namespace == "dlpack":
        return DLPackTensor(counts.__dlpack__(), counts.__dlpack_device__())
    return array_namespace.from_dlpack(counts)
