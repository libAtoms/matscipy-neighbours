"""Phase 4 — DLPack interop.

Validates the zero-copy DLPack export: the host (numpy) path against the
existing copy-based extension call, and the device (cupy) path against the host
oracle. cupy tests skip automatically when cupy or a GPU is unavailable.
"""

import numpy as np
import pytest

import _matscipy_neighbours as _ext
from matscipy_neighbours.neighbours import _DLPackArray
import matscipy_neighbours.neighbours as nl


def _random_config(N=2000, L=12.0, seed=1):
    rng = np.random.default_rng(seed)
    positions = np.ascontiguousarray(rng.uniform(0.0, L, size=(N, 3)))
    cell = np.ascontiguousarray(np.diag([L, L, L]).astype(float))
    cell_origin = np.zeros(3)
    pbc = np.ones(3, dtype=bool)
    inv_cell = np.ascontiguousarray(np.linalg.inv(cell.T))
    return positions, cell, inv_cell, cell_origin, pbc


def _canonical(i, j, S):
    """Sorted (i, j, shift) rows — order-independent comparison key."""
    rows = np.column_stack([i, j, S]).astype(np.int64)
    return rows[np.lexsort(rows.T[::-1])]


def test_dlpack_host_matches_copy_path():
    """neighbour_list_dlpack (CPU backend) -> numpy.from_dlpack must equal the
    high-level (copy-based) neighbour_list."""
    positions, cell, inv_cell, origin, pbc = _random_config()
    cutoff = 1.0
    quant = "ijD"

    i0, j0, D0 = nl.neighbour_list(quant, positions=positions, cell=cell,
                                   pbc=pbc, cutoff=cutoff)

    caps = _ext.neighbour_list_dlpack(quant, origin, cell, inv_cell, pbc,
                                      positions, cutoff, None, 0, 0, -1)
    i1, j1, D1 = (np.from_dlpack(_DLPackArray(c, (1, 0))) for c in caps)

    assert i1.shape == i0.shape and D1.shape == D0.shape
    assert D1.shape[1] == 3
    assert np.array_equal(np.sort(i1), np.sort(i0))
    assert np.array_equal(_canonical(i0, j0, np.zeros((len(i0), 3))),
                          _canonical(i1, j1, np.zeros((len(i1), 3))))
    # Distance vectors agree once pairs are matched by sorting on (i, j).
    o0 = np.lexsort((j0, i0))
    o1 = np.lexsort((j1, i1))
    assert np.allclose(D0[o0], D1[o1], atol=1e-12)


def test_dlpack_host_frees_without_consumer():
    """A capsule that is never consumed must free its buffer via its own
    destructor (no leak / crash)."""
    positions, cell, inv_cell, origin, pbc = _random_config(N=500)
    caps = _ext.neighbour_list_dlpack("i", origin, cell, inv_cell, pbc,
                                      positions, 1.0, None, 0, 0, -1)
    del caps  # capsule destructor must run the DLManagedTensor deleter


cupy = pytest.importorskip("cupy")


def _gpu_available():
    """A CUDA device AND an extension built with the GPU backend."""
    if not getattr(_ext, "_has_gpu", 0):
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="no CUDA device, or extension built without the GPU backend")


@requires_gpu
def test_dlpack_cupy_roundtrip_matches_cpu():
    """cupy positions in -> cupy arrays out, matching the CPU oracle."""
    positions, cell, inv_cell, origin, pbc = _random_config(N=3000, seed=5)
    cutoff = 1.0

    i_cpu, j_cpu, S_cpu = nl.neighbour_list(
        "ijS", positions=positions, cell=cell, pbc=pbc, cutoff=cutoff)

    pos_d = cupy.asarray(positions)
    out = nl.neighbour_list("ijS", positions=pos_d, cell=cell, pbc=pbc,
                            cutoff=cutoff)
    i_g, j_g, S_g = out
    # Results must be cupy arrays (stayed on the device).
    assert isinstance(i_g, cupy.ndarray)
    assert isinstance(S_g, cupy.ndarray)

    i_g, j_g, S_g = (cupy.asnumpy(a) for a in (i_g, j_g, S_g))
    assert len(i_g) == len(i_cpu)
    assert np.array_equal(_canonical(i_cpu, j_cpu, S_cpu),
                          _canonical(i_g, j_g, S_g))


@requires_gpu
def test_dlpack_cupy_multi_gpu():
    """The result must land on the same device as the input cupy array."""
    ndev = cupy.cuda.runtime.getDeviceCount()
    positions, cell, inv_cell, origin, pbc = _random_config(N=2000, seed=4)
    i_cpu = nl.neighbour_list("i", positions=positions, cell=cell, pbc=pbc,
                              cutoff=1.0)
    for d in range(ndev):
        with cupy.cuda.Device(d):
            pos_d = cupy.asarray(positions)
            i, j = nl.neighbour_list("ij", positions=pos_d, cell=cell, pbc=pbc,
                                     cutoff=1.0)
        assert int(i.device.id) == d
        assert len(i) == len(i_cpu)


@requires_gpu
def test_coordination_cupy_matches_cpu():
    """GPU coordination (count-only) -> cupy, matching the host bincount."""
    positions, cell, inv_cell, origin, pbc = _random_config(N=3000, seed=8)
    cutoff = 1.3
    c_cpu = nl.coordination(positions=positions, cell=cell, pbc=pbc,
                            cutoff=cutoff)
    pos_d = cupy.asarray(positions)
    c_gpu = nl.coordination(positions=pos_d, cell=cell, pbc=pbc, cutoff=cutoff)
    assert isinstance(c_gpu, cupy.ndarray)
    assert np.array_equal(cupy.asnumpy(c_gpu), c_cpu)


@requires_gpu
def test_dlpack_cupy_distance_matches_cpu():
    positions, cell, inv_cell, origin, pbc = _random_config(N=2500, seed=9)
    cutoff = 1.5

    d_cpu = nl.neighbour_list("d", positions=positions, cell=cell, pbc=pbc,
                              cutoff=cutoff)
    pos_d = cupy.asarray(positions)
    d_g = cupy.asnumpy(nl.neighbour_list("d", positions=pos_d, cell=cell,
                                         pbc=pbc, cutoff=cutoff))
    assert np.allclose(np.sort(d_cpu), np.sort(d_g), atol=1e-12)
