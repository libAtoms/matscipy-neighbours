"""EXPERIMENTAL: the neighbour matrix as a JAX primitive (XLA FFI), callable
inside `jit`, `lax.cond` and `lax.fori_loop`, on the CPU or a CUDA GPU, without
leaving the device.

    idx, shift, count, overflow = neighbour_matrix(positions, cell, cutoff=6.0,
                                                   pbc=(True, True, True), max_neighbours=64)
    D = positions[idx] - positions[:, None] + shift @ cell   # where slot < count

positions [n, 3] and cell [3, 3] (rows: lattice vectors) are float32 or float64
arrays (the search itself runs in float64); cutoff, pbc and max_neighbours are
static.  Per-type cutoffs: `types` [n] (int) and `cutoffs` [m, m] (symmetric,
not squared); `cutoff` is then their largest unless given.  Returns int32 idx
[n, K], shift [n, K, 3] (cell shifts S: periodic images of any order, so the
cell may be smaller than the cutoff), count [n] (the true count, which may
exceed K) and overflow [] (1: some row was clipped; build again with a larger
K).  Slots past min(count, K) are unspecified: select with the mask, do not
multiply by it.

`neighbour_matrix_batched` builds B structures (padded to N atoms) in one call
-- a brute-force search over each structure's atoms and periodic images, for the
many small cells of a batched evaluation -- and `jax.vmap` of `neighbour_matrix`
or `neighbour_matrix_planned` calls it (one build for the batch, not B builds).

`plan` + `neighbour_matrix_planned` (CUDA only) is the device-only build: the
cell grid and workspace are fixed once on the host, and the build runs on XLA's
stream with no host synchronisation, so it can sit inside a CUDA graph (e.g. a
`lax.cond` in a jitted MD loop) at no host round trip.

Needs the handler library `_matscipy_neighbours_xla` (built when CMake is given
JAX_FFI_INCLUDE_DIR); the CUDA handlers exist only in a CUDA build.
"""
import ctypes
import functools
import os
import pathlib

import numpy as np

try:                                                        # jax is optional for the package, not here
    from jax.custom_batching import custom_vmap
except ImportError:  # pragma: no cover
    custom_vmap = None

_TARGET = "matscipy_neighbour_matrix"
_PLANNED = "matscipy_neighbour_matrix_planned"
_BATCHED = "matscipy_neighbour_matrix_batched"
_LIB = {}


@functools.cache
def _register():
    """Load the handler library and register its targets: the platforms it serves."""
    import jax
    here = pathlib.Path(__file__).parent
    lib_path = os.environ.get("MATSCIPY_NEIGHBOURS_XLA_LIB") or next(
        (str(p) for p in here.glob("_matscipy_neighbours_xla*.so")), None)
    if lib_path is None:
        raise ImportError("matscipy-neighbours was built without the XLA FFI library "
                          "(_matscipy_neighbours_xla); rebuild with -DJAX_FFI_INCLUDE_DIR=...")
    lib = ctypes.cdll.LoadLibrary(lib_path)
    platforms = ["cpu"]
    jax.ffi.register_ffi_target(_TARGET, jax.ffi.pycapsule(lib.MatscipyNeighbourMatrixCpu), platform="cpu")
    jax.ffi.register_ffi_target(_BATCHED, jax.ffi.pycapsule(lib.MatscipyNeighbourMatrixBatchedCpu),
                                platform="cpu")
    if hasattr(lib, "MatscipyNeighbourMatrixCuda"):
        jax.ffi.register_ffi_target(_TARGET, jax.ffi.pycapsule(lib.MatscipyNeighbourMatrixCuda),
                                    platform="CUDA")
        platforms.append("cuda")
    if hasattr(lib, "MatscipyNeighbourMatrixBatchedCuda"):
        jax.ffi.register_ffi_target(_BATCHED, jax.ffi.pycapsule(lib.MatscipyNeighbourMatrixBatchedCuda),
                                    platform="CUDA")
    if hasattr(lib, "MatscipyNeighbourMatrixPlannedCuda"):
        jax.ffi.register_ffi_target(_PLANNED, jax.ffi.pycapsule(lib.MatscipyNeighbourMatrixPlannedCuda),
                                    platform="CUDA")
        lib.MatscipyPlanNeighbourMatrix.argtypes = [
            ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.c_double,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)]
        lib.MatscipyLastError.restype = ctypes.c_char_p
        _LIB["planned"] = lib
    return tuple(platforms)


def _pbc(pbc):
    pbc = tuple(bool(b) for b in (pbc if hasattr(pbc, "__len__") else (pbc,) * 3))
    return pbc, sum(1 << k for k, b in enumerate(pbc) if b)


def _inputs(positions, cell, cutoff, types, cutoffs):
    """(positions, cell, types, cutoffs, cutoff) as the handlers take them: one float
    dtype, int32 types, and empty per-type inputs for a global cutoff."""
    import jax.numpy as jnp
    positions = jnp.asarray(positions)
    dt = positions.dtype if positions.dtype in (jnp.float32, jnp.float64) else jnp.float64
    positions, cell = positions.astype(dt), jnp.asarray(cell, dt)
    if (types is None) != (cutoffs is None):
        raise ValueError("per-type cutoffs need both `types` and `cutoffs`")
    if types is None:
        if cutoff is None:
            raise ValueError("give `cutoff`, or `types` and `cutoffs`")
        return positions, cell, jnp.zeros((0,), jnp.int32), jnp.zeros((0, 0), dt), float(cutoff)
    cutoffs_np = np.asarray(cutoffs, float) if not hasattr(cutoffs, "aval") else None
    if cutoff is None:
        if cutoffs_np is None:
            raise ValueError("traced `cutoffs`: give `cutoff` (the largest), which sizes the grid")
        cutoff = float(cutoffs_np.max())
    return positions, cell, jnp.asarray(types, jnp.int32), jnp.asarray(cutoffs, dt), float(cutoff)


def neighbour_matrix(positions, cell, cutoff, pbc, max_neighbours, types=None, cutoffs=None):
    import jax
    import jax.numpy as jnp
    _register()
    n, K = positions.shape[0], int(max_neighbours)
    positions, cell, types, cutoffs, cutoff = _inputs(positions, cell, cutoff, types, cutoffs)
    pbc, mask = _pbc(pbc)
    i32 = jnp.int32
    out = (jax.ShapeDtypeStruct((n, K), i32), jax.ShapeDtypeStruct((n, K, 3), i32),
           jax.ShapeDtypeStruct((n,), i32), jax.ShapeDtypeStruct((), i32))

    @custom_vmap
    def call(positions, cell, types, cutoffs):
        return jax.ffi.ffi_call(_TARGET, out, vmap_method="sequential")(
            positions, cell, types, cutoffs, cutoff=cutoff, pbc=mask)

    @call.def_vmap
    def _batched_rule(axis_size, in_batched, positions, cell, types, cutoffs):
        idx, shift, count, status = _batched_from_vmap(axis_size, in_batched, positions, cell, types, cutoffs,
                                                       cutoff, pbc, K)
        return (idx, shift, count, status[:, 0]), (True,) * 4

    return call(positions, cell, types, cutoffs)


# ------------------------------------------------------------------ planned (EXPERIMENTAL, CUDA)
def plan(n_atoms, cell, cutoff, pbc, max_neighbours, ncutoffs=0, dtype=np.float64):
    """Host-side plan of the device-only build: (n1, n2, n3, hashed, workspace_bytes).
    The cell grid (and the dense or hashed cell lookup) is fixed from this
    (reference) `cell` [3, 3] and `cutoff` (the largest, with per-type cutoffs);
    `ncutoffs` is the per-type matrix's side (0: global cutoff) and `dtype` the
    positions' (float32 or float64).  Plan again when the cell changes a lot: the
    build stays correct but scans more or fewer cells than needed."""
    _register()
    if "planned" not in _LIB:
        raise ImportError("this matscipy-neighbours XLA library has no planned (CUDA) build")
    lib = _LIB["planned"]
    _, mask = _pbc(pbc)
    c = np.ascontiguousarray(cell, np.float64)
    out = (ctypes.c_int64 * 5)()
    single = int(np.dtype(dtype) == np.float32)
    rc = lib.MatscipyPlanNeighbourMatrix(int(n_atoms), c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                         mask, float(cutoff), int(max_neighbours), int(ncutoffs), single, out)
    if rc:
        raise ValueError(lib.MatscipyLastError().decode())
    return tuple(int(v) for v in out)


def neighbour_matrix_planned(positions, cell, cutoff, pbc, max_neighbours, plan, types=None, cutoffs=None):
    """`neighbour_matrix` (CUDA only) built entirely on XLA's stream: no host
    synchronisation, allocation or read-back, and declared CUDA-graph compatible,
    so under jit a rebuild inside `lax.cond` needs no host round trip.  `plan`
    from `plan(...)` with the same n_atoms, cutoff, pbc, max_neighbours, number of
    types and dtype.  Returns idx, shift, count and status [2] = (overflow,
    invalid): non-finite positions, a degenerate cell, a type out of range or a
    per-type cutoff above `cutoff` set invalid (counts are then 0), where the
    regular build raises."""
    import jax
    import jax.numpy as jnp
    _register()
    n, K = positions.shape[0], int(max_neighbours)
    positions, cell, types, cutoffs, cutoff = _inputs(positions, cell, cutoff, types, cutoffs)
    n1, n2, n3, hashed, ws = plan
    _, mask = _pbc(pbc)
    i32 = jnp.int32
    out = (jax.ShapeDtypeStruct((n, K), i32), jax.ShapeDtypeStruct((n, K, 3), i32),
           jax.ShapeDtypeStruct((n,), i32), jax.ShapeDtypeStruct((2,), i32),
           jax.ShapeDtypeStruct((ws,), jnp.uint8))
    pbc = _pbc(pbc)[0]

    @custom_vmap
    def call(positions, cell, types, cutoffs):
        idx, shift, count, status, _ = jax.ffi.ffi_call(_PLANNED, out, vmap_method="sequential")(
            positions, cell, types, cutoffs, cutoff=cutoff, pbc=mask, n1=np.int64(n1), n2=np.int64(n2),
            n3=np.int64(n3), hashed=np.int64(hashed))
        return idx, shift, count, status

    @call.def_vmap
    def _batched_rule(axis_size, in_batched, positions, cell, types, cutoffs):
        return (_batched_from_vmap(axis_size, in_batched, positions, cell, types, cutoffs, cutoff, pbc, K),
                (True,) * 4)

    return call(positions, cell, types, cutoffs)


# ------------------------------------------------------------------ batched (EXPERIMENTAL)
def neighbour_matrix_batched(positions, cell, cutoff, pbc, max_neighbours, n_atoms=None, types=None,
                             cutoffs=None):
    """B structures padded to N atoms in one build: positions [B, N, 3] and cell [B, 3, 3] (or one [3, 3]
    for all; rows: lattice vectors), float32 or float64; n_atoms [B] live atoms each (the first rows;
    default N); pbc a 3-tuple for all or a [B, 3] boolean array; per-type `types` [B, N] (or [N] for all)
    with `cutoffs` [m, m] as `neighbour_matrix`.  Returns int32 idx [B, N, K], shift [B, N, K, 3]
    (positions[b, j] - positions[b, i] + shift @ cell[b] is the pair vector), count [B, N] (0 on padding
    rows) and status [B, 2] per structure: (some row clipped at K, invalid input -- a non-finite position,
    a degenerate cell, a type out of range, a per-type cutoff above `cutoff`, or n_atoms outside [0, N];
    that structure's counts are then 0).  A brute-force search, O(n^2 S) per structure (S periodic
    images): the algorithm for many small cells; use `neighbour_matrix` for large ones.  Positions may lie
    outside the cell; the cell may be smaller than the cutoff.  Row order differs from the cell-list
    builds'.  `jax.vmap` of `neighbour_matrix` / `neighbour_matrix_planned` calls this."""
    import jax
    import jax.numpy as jnp
    _register()
    positions = jnp.asarray(positions)
    B, N = positions.shape[0], positions.shape[1]
    cell = jnp.asarray(cell)
    if cell.ndim == 2:
        cell = jnp.broadcast_to(cell, (B, 3, 3))
    n_atoms = jnp.full((B,), N, jnp.int32) if n_atoms is None else jnp.asarray(n_atoms, jnp.int32)
    if types is not None and jnp.ndim(types) == 1:
        types = jnp.broadcast_to(jnp.asarray(types, jnp.int32), (B, N))
    positions, cell, types, cutoffs, cutoff = _inputs(positions, cell, cutoff, types, cutoffs)
    if types.size == 0:
        types = jnp.zeros((0,), jnp.int32)
    if isinstance(pbc, (tuple, list)) or np.ndim(pbc) == 1 or np.ndim(pbc) == 0:
        masks = jnp.full((B,), _pbc(pbc)[1], jnp.int32)
    else:                                                   # [B, 3] booleans, per structure
        masks = (jnp.asarray(pbc).astype(jnp.int32) * jnp.asarray([1, 2, 4], jnp.int32)).sum(-1, dtype=jnp.int32)
    K = int(max_neighbours)
    i32 = jnp.int32
    out = (jax.ShapeDtypeStruct((B, N, K), i32), jax.ShapeDtypeStruct((B, N, K, 3), i32),
           jax.ShapeDtypeStruct((B, N), i32), jax.ShapeDtypeStruct((B, 2), i32))
    call = jax.ffi.ffi_call(_BATCHED, out, vmap_method="sequential")
    return call(positions, cell, n_atoms, masks, types, cutoffs, cutoff=cutoff)


def _batched_from_vmap(axis_size, in_batched, positions, cell, types, cutoffs, cutoff, pbc, K):
    """The vmap rule of the single-structure builds: their (unbatched) inputs broadcast to the batch, one
    batched build.  A batched per-type cutoff matrix is refused: the batched build takes one."""
    import jax.numpy as jnp
    pb, cb, tb, mb = in_batched
    if mb:
        raise NotImplementedError("matscipy-neighbours: vmap over the per-type cutoff matrix is not supported")
    if not pb:
        positions = jnp.broadcast_to(positions, (axis_size,) + positions.shape)
    if not cb:
        cell = jnp.broadcast_to(cell, (axis_size, 3, 3))
    if types.size and not tb:
        types = jnp.broadcast_to(types, (axis_size,) + types.shape)
    per_type = types.size > 0
    return neighbour_matrix_batched(positions, cell, cutoff, pbc, K, types=types if per_type else None,
                                    cutoffs=cutoffs if per_type else None)
