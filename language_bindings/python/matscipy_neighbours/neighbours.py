"""Public neighbour-list API, compatible with ``matscipy.neighbours``.

Built on the ``_matscipy_neighbours`` C-extension. The C-extension signature
(``neighbour_list(quantities, cell_origin, cell, inv_cell, pbc, positions,
cutoff, numbers)``) matches the one matscipy's ``_matscipy`` backend exposes, so
the two are interchangeable.
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

# A pure C-extension function; re-exported unchanged.
get_jump_indicies = _ext.get_jump_indicies


def _gauss_reduce(b1, b2):
    """Lagrange/Gauss reduction of a 2D lattice basis (shortest two vectors)."""
    b1, b2 = (b1, b2) if b1 @ b1 <= b2 @ b2 else (b2, b1)
    while True:
        b2 = b2 - np.round((b2 @ b1) / (b1 @ b1)) * b1
        if b2 @ b2 >= b1 @ b1:
            return b1, b2
        b1, b2 = b2, b1


def _closest_in_plane(v, b1, b2):
    """Lattice vector of the Gauss-reduced basis (b1, b2) closest to v.

    Rounding the (Gram) coordinates and checking the neighbouring cells is
    exact for a reduced 2D basis.
    """
    gram = np.array([[b1 @ b1, b1 @ b2], [b1 @ b2, b2 @ b2]])
    c = np.round(np.linalg.solve(gram, [v @ b1, v @ b2]))
    best, best_d = None, np.inf
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            w = (c[0] + di) * b1 + (c[1] + dj) * b2
            d = (v - w) @ (v - w)
            if d < best_d:
                best, best_d = w, d
    return best


def _reduce_basis(vectors):
    """Minkowski-reduce up to three lattice vectors (greedy algorithm, exact
    in dimension <= 3): the returned basis spans the same lattice and its
    vectors are the successive shortest ones."""
    b = [np.asarray(v, dtype=float) for v in vectors]
    if len(b) == 1:
        return np.array(b)
    if len(b) == 2:
        return np.array(_gauss_reduce(*b))
    for _ in range(100):
        b.sort(key=lambda v: v @ v)
        b1, b2 = _gauss_reduce(b[0], b[1])
        b3 = b[2] - _closest_in_plane(b[2], b1, b2)
        if b3 @ b3 >= b2 @ b2:
            return np.array([b1, b2, b3])
        b = [b1, b2, b3]
    raise RuntimeError("Lattice reduction did not converge.")  # pragma: no cover


def mic(dr, cell, pbc=None):
    """Apply the minimum image convention to an array of distance vectors.

    Each vector is replaced by the shortest vector that differs from it by an
    integer combination of the *periodic* lattice vectors. The periodic
    sublattice is Minkowski-reduced first, so the result is the true minimum
    image for any (arbitrarily skewed) cell, not just for orthogonal ones.

    Parameters
    ----------
    dr : array_like
        Distance vectors, shape ``(n, 3)``.
    cell : array_like
        ``3x3`` cell matrix (rows are the lattice vectors).
    pbc : array_like, optional
        Periodicity of each *lattice direction* (row of ``cell``). Vectors are
        only ever shifted by multiples of the periodic lattice vectors.
        Defaults to periodic in all directions.

    Returns
    -------
    numpy.ndarray
        ``dr`` wrapped into the minimum image.
    """
    dr = np.array(dr, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if pbc is None:
        pbc = np.ones(3, dtype=bool)
    pbc = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,))
    if not pbc.any():
        return dr
    basis = _reduce_basis(cell[pbc])        # (k, 3), k periodic directions
    k = len(basis)
    # Coordinates of dr in the reduced basis (least squares: components
    # perpendicular to the periodic sublattice cannot be changed).
    gram = basis @ basis.T
    s = np.round(np.linalg.solve(gram, basis @ dr.T).T)
    # For a Minkowski-reduced basis the closest lattice point lies within one
    # step of the rounded coordinates in every direction.
    best = dr - s @ basis
    best_d = np.einsum("ij,ij->i", best, best)
    for offset in np.ndindex(*(3,) * k):
        off = np.asarray(offset) - 1
        if not off.any():
            continue
        cand = dr - (s + off) @ basis
        cand_d = np.einsum("ij,ij->i", cand, cand)
        better = cand_d < best_d
        best[better] = cand[better]
        best_d[better] = cand_d[better]
    return best


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


def _namespace_of(x):
    """The array module ``x`` belongs to (cupy, jax.numpy or torch), for
    returning results in the caller's framework; cupy if unrecognised."""
    root = type(x).__module__.split(".")[0]
    if root in ("jax", "jaxlib"):
        import jax.numpy as xp
    elif root == "torch":
        import torch as xp
    else:
        import cupy as xp
    return xp


def first_neighbours(n, i, *, array_namespace=None):
    """Row-start ("seed") array of a pair list sorted by its first index.

    Pairs ``seed[k]:seed[k+1]`` belong to atom ``k``; ``seed[n]`` is the number
    of pairs. Atoms before the first pair get ``-1`` (as in matscipy), and an
    atom without neighbours further on starts where the next one does, so
    ``maximum(seed, 0)`` gives offsets with an empty segment for every atom
    that has no neighbours.

    Parameters
    ----------
    n : int
        Number of atoms.
    i : array_like
        Sorted first-atom indices of the pairs, e.g. ``neighbour_list("i",
        ...)``. A device array (cupy, jax, torch; int64) is processed on its
        GPU and the result stays there.
    array_namespace : module or "dlpack", optional
        For device input: framework of the result (default: that of ``i``),
        or ``"dlpack"`` for a :class:`DLPackTensor`.

    Returns
    -------
    array
        ``seed`` of length ``n + 1``: numpy for host input, a device array
        for device input.
    """
    if not _is_on_device(i):
        return _ext.first_neighbours(n, i)
    if not getattr(_ext, "_has_gpu", 0):
        raise RuntimeError("Device input requires a GPU build (-DENABLE_CUDA=ON "
                           "or -DENABLE_HIP=ON).")
    capsule = _ext.first_neighbours_dlpack(n, i)
    tensor = DLPackTensor(capsule, _dlpack_device(i))
    if array_namespace is None:
        array_namespace = _namespace_of(i)
    return _consume([tensor], array_namespace, True)[0]


def _shrink_wrapped_cell(positions):
    """Orthorhombic cell spanning ``positions`` (non-periodic use only).

    Degenerate extents (planar or linear molecules, a single atom) are padded
    to the largest extent, or to 1 if every extent is zero: the cell only bins
    atoms here, so with no periodic direction the padding cannot change the
    result, but a zero-volume cell could not be binned at all.
    """
    r = np.asarray(positions, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise TypeError(f"positions must have shape (n, 3), got {r.shape}.")
    if len(r) == 0:
        return np.zeros(3), np.eye(3)
    rmin, rmax = r.min(axis=0), r.max(axis=0)
    extent = rmax - rmin
    pad = extent.max() if extent.max() > 0 else 1.0
    extent = np.where(extent > 1e-9 * pad, extent, pad)
    return rmin, np.diag(extent)


def _complete_cell(cell, cell_origin, pbc, positions):
    """Replace zero lattice vectors (ASE's "no cell in this direction") by
    vectors orthogonal to the given ones that span the atoms, moving the
    origin so the atoms sit inside. Only allowed for non-periodic directions:
    the completed direction merely bins atoms, so its length cannot change
    the result."""
    zero = ~cell.any(axis=1)
    if not zero.any():
        return cell_origin, cell
    if (zero & pbc).any():
        raise ValueError("Lattice vector(s) "
                         f"{np.flatnonzero(zero & pbc).tolist()} are zero but "
                         "those directions are periodic.")
    if positions is None:
        raise ValueError("The cell has zero lattice vectors; a complete cell "
                         "is required for device-resident positions.")
    r = np.asarray(positions, dtype=float)
    if zero.all():
        return _shrink_wrapped_cell(r)
    cell = cell.copy()
    cell_origin = np.array(cell_origin, dtype=float)
    given = cell[~zero]
    if len(given) == 2:
        normals = [np.cross(given[0], given[1])]
    else:
        a = given[0]
        e = np.eye(3)[np.argmin(np.abs(a))]      # least-aligned axis
        n1 = np.cross(a, e)
        normals = [n1, np.cross(a, n1)]
    scale = max(np.linalg.norm(given, axis=1).max(), 1.0)
    for k, n in zip(np.flatnonzero(zero), normals):
        n = n / np.linalg.norm(n)
        proj = r @ n
        lo, hi = (proj.min(), proj.max()) if len(proj) else (0.0, 0.0)
        length = hi - lo if hi - lo > 1e-9 * scale else scale
        cell[k] = n * length
        cell_origin += n * (lo - cell_origin @ n)
    return cell_origin, cell


def _host_metadata(cell, pbc, numbers, cell_origin, nat, *, positions=None):
    """Normalise the (small, host-resident) geometry/type arrays. ``positions``
    is used only to shrink-wrap a cell when none is given, or to complete zero
    lattice vectors (host arrays only)."""
    if pbc is None:
        pbc = np.zeros(3, dtype=bool)
    pbc = np.ascontiguousarray(np.broadcast_to(pbc, (3,)), dtype=bool)
    if cell is None:
        if pbc.any():
            raise ValueError("A cell is required when any direction is "
                             "periodic; pass cell= or set pbc=False.")
        rmin, cell = _shrink_wrapped_cell(positions)
        cell_origin = rmin if cell_origin is None else cell_origin
    if cell_origin is None:
        cell_origin = np.zeros(3)
    if numbers is None:
        numbers = np.ones(nat, dtype=np.int64)
    cell = np.ascontiguousarray(np.asarray(cell, dtype=float))
    cell_origin = np.ascontiguousarray(np.asarray(cell_origin, dtype=float))
    numbers = np.ascontiguousarray(np.asarray(numbers), dtype=np.int64)
    if cell.shape != (3, 3):
        raise TypeError(f"cell must have shape (3, 3), got {cell.shape}.")
    cell_origin, cell = _complete_cell(cell, cell_origin, pbc, positions)
    inv_cell = np.ascontiguousarray(np.linalg.inv(cell.T))
    return cell_origin, cell, inv_cell, pbc, numbers


def _atomic_number(el):
    """Atomic number of an element given as a symbol or a number."""
    if isinstance(el, str):
        try:
            return _atomic_numbers[el]
        except KeyError:
            raise ValueError(
                f"Unknown element symbol {el!r} in the cutoff dictionary"
                + ("" if _atomic_numbers else " (ase is needed to resolve "
                   "element symbols)")) from None
    return int(el)


def _resolve_cutoff(cutoff, numbers):
    """Turn a scalar / per-atom array / element-pair dict / per-type matrix
    into a value the C-extension understands, plus the per-atom type array to
    pass along."""
    if isinstance(cutoff, dict):
        maxnum = int(np.max(numbers)) if numbers.size else 0
        matrix = np.zeros((maxnum + 1, maxnum + 1), dtype=float)
        for (el1, el2), c in cutoff.items():
            el1 = _atomic_number(el1)
            el2 = _atomic_number(el2)
            if el1 <= maxnum and el2 <= maxnum:
                matrix[el1, el2] = c
                matrix[el2, el1] = c
        cutoff = matrix
    elif np.ndim(cutoff) == 0:
        # Any scalar (int, float32, 0-d array): the extension wants a float.
        return float(cutoff), numbers
    else:
        cutoff = np.ascontiguousarray(cutoff, dtype=float)
    if cutoff.ndim == 2:
        if cutoff.shape[0] != cutoff.shape[1]:
            raise ValueError("Per-type cutoff matrix must be square, got "
                             f"shape {cutoff.shape}.")
        if numbers.size and (numbers.min() < 0
                             or numbers.max() >= cutoff.shape[0]):
            raise ValueError(
                f"numbers must index the {cutoff.shape[0]}x{cutoff.shape[0]} "
                f"per-type cutoff matrix, i.e. lie in [0, {cutoff.shape[0]}); "
                f"got values in [{numbers.min()}, {numbers.max()}].")
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
                atoms.numbers.astype(np.int64), np.zeros(3))
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


# Per-slot quantities of neighbour_matrix, as the core's QUANTITY_* flags.
_MATRIX_QUANTITIES = {"D": 1 << 2, "S": 1 << 4}


def neighbour_matrix(atoms=None, cutoff=None, max_neighbours=None, *,
                     positions=None, cell=None, pbc=None, numbers=None,
                     cell_origin=None, device=None, array_namespace=None,
                     quantities="D"):
    """Dense fixed-capacity neighbour list: ``(idx, *per_slot, count)``.

    Each atom's neighbours occupy a row of an ``n x max_neighbours`` matrix, so
    the output shape is static (it depends only on ``n`` and ``max_neighbours``,
    not on the number of pairs). This suits frameworks that compile for fixed
    shapes (e.g. JAX), where forces are a masked sum over the neighbour axis with
    no scatter.

    ``quantities`` selects the per-slot arrays returned between ``idx`` and
    ``count``, in the order given: ``"D"`` the distance vectors
    ``D == r[j] - r[i] + S @ cell`` (float, shape ``(n, max_neighbours, 3)``),
    ``"S"`` the cell shifts (int64, same shape). The default ``"D"`` returns
    ``(idx, dist, count)``; ``""`` returns ``(idx, count)``, the cheapest form
    for a consumer that recomputes the distances from the positions (with
    ``"S"`` for the periodic images).

    ``idx`` has shape ``(n, max_neighbours)`` (int64 neighbour indices) and
    ``count`` shape ``(n,)`` (true neighbour count). Only the first ``count``
    slots of a row are defined: the unused slots are not cleared, so mask with
    ``arange(max_neighbours) < count[:, None]``. ``device`` and
    ``array_namespace`` behave as in :func:`neighbour_list`.

    Raises ``ValueError`` if any atom has more than ``max_neighbours`` neighbours
    (the capacity is too small); retry with a larger ``max_neighbours``. Raises
    ``MemoryError`` if the host or GPU runs out of memory.
    """
    bad = [c for c in quantities if c not in _MATRIX_QUANTITIES]
    if bad or len(set(quantities)) != len(quantities):
        raise ValueError(f"quantities={quantities!r}: each of 'D' (distance "
                         "vector) and 'S' (cell shift) may appear at most once.")
    flags = sum(_MATRIX_QUANTITIES[c] for c in quantities)
    if cutoff is None:
        raise ValueError("Please provide a value for the cutoff radius.")
    if max_neighbours is None:
        raise ValueError("Please provide max_neighbours (the row capacity).")
    positions, cell, pbc, numbers, cell_origin = _gather(
        atoms, positions, cell, pbc, numbers, cell_origin)

    on_device = _is_on_device(positions)
    use_gpu, device_id = _resolve_device(device, on_device)
    if on_device and not use_gpu:
        raise ValueError("device='cpu' with device-resident positions is not "
                         "supported; move the array to the host first.")
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
            device_id = 0
    co, ce, inv, pb, nums = _host_metadata(
        cell, pbc, numbers, cell_origin, nat,
        positions=None if on_device else host_pos)
    rc, nums = _resolve_cutoff(cutoff, nums)

    idx_cap, dist_cap, shift_cap, count_cap, overflow = \
        _ext.neighbour_matrix_dlpack(co, ce, inv, pb, host_pos, rc,
                                     int(max_neighbours), nums,
                                     1 if use_gpu else 0, py_in, device_id,
                                     flags)
    if overflow:
        raise ValueError(
            f"max_neighbours={max_neighbours} is too small: some atom has more "
            "neighbours than that. Retry with a larger max_neighbours.")

    if use_gpu:
        dtype = getattr(_ext, "_device_type", _DLPACK_CUDA)
        out_id = _dlpack_device(positions)[1] if on_device else device_id
        out_dev = (dtype, out_id)
    else:
        out_dev = (_DLPACK_CPU, 0)
    per_slot = {"D": dist_cap, "S": shift_cap}
    caps = [idx_cap] + [per_slot[c] for c in quantities] + [count_cap]
    wrappers = [DLPackTensor(c, out_dev) for c in caps]
    return tuple(_consume(wrappers, array_namespace, use_gpu))


def empty_gpu_cache():
    """Return GPU memory cached by matscipy-neighbours to the driver.

    The GPU backend keeps freed buffers in its own memory pool and reuses them
    in later calls, which avoids the cost of allocating afresh every time. The
    cached memory is not available to other libraries (CuPy, PyTorch, JAX, ...)
    until it is returned; call this before handing the memory to them. Buffers
    still referenced by returned arrays are not affected. No-op without a GPU
    backend."""
    _ext.empty_gpu_cache()


def triplet_list(first_neighbours, abs_dr_p=None, cutoff=None):
    """Compute a triplet list from a first-neighbour (row-start) array.

    Mirrors :func:`matscipy.neighbours.triplet_list` (without the optional
    ``jk_t`` output).
    """
    first_neighbours = np.ascontiguousarray(first_neighbours, dtype=np.int64)
    if (abs_dr_p is None) != (cutoff is None):
        raise ValueError("abs_dr_p and cutoff must be given together.")
    if abs_dr_p is not None:
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
