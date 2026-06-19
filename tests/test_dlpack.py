"""DLPack interop and the framework-agnostic API.

Covers the host path (numpy fast path, DLPack capsules, other frameworks),
the device path (cupy in -> cupy out via DLPack import/export), the device
override, and GPU coordination. GPU tests skip when cupy or a GPU is absent;
the JAX test skips when jax is absent.
"""

import numpy as np
import pytest

import _matscipy_neighbours as _ext
import matscipy_neighbours.neighbours as nl
from matscipy_neighbours.neighbours import DLPackTensor

# cupy is optional: importing it must NOT skip the whole module, or the host and
# JAX tests below (which need no GPU) would never run in a cupy-less CI.
try:
    import cupy
except ImportError:  # pragma: no cover - exercised in CPU-only CI
    cupy = None


def _random_config(N=2000, L=12.0, seed=1):
    rng = np.random.default_rng(seed)
    positions = np.ascontiguousarray(rng.uniform(0.0, L, size=(N, 3)))
    cell = np.ascontiguousarray(np.diag([L, L, L]).astype(float))
    pbc = np.ones(3, dtype=bool)
    return positions, cell, pbc


def _canonical(i, j, S):
    """Sorted (i, j, shift) rows — order-independent comparison key."""
    rows = np.column_stack([np.asarray(i), np.asarray(j), np.asarray(S)])
    rows = rows.astype(np.int64)
    return rows[np.lexsort(rows.T[::-1])]


# --------------------------------------------------------------- host paths

def test_host_default_numpy():
    pos, cell, pbc = _random_config()
    i, j, D = nl.neighbour_list("ijD", positions=pos, cell=cell, pbc=pbc,
                                cutoff=1.0)
    assert isinstance(i, np.ndarray) and D.shape[1] == 3
    # coordination consistency
    assert int(i.shape[0]) == int(np.bincount(i).sum())


def test_host_array_namespace_numpy_matches_fast_path():
    """array_namespace=numpy routes through DLPack; must equal the fast path."""
    pos, cell, pbc = _random_config(seed=2)
    i0, j0, S0 = nl.neighbour_list("ijS", positions=pos, cell=cell, pbc=pbc,
                                   cutoff=1.0)
    i1, j1, S1 = nl.neighbour_list("ijS", positions=pos, cell=cell, pbc=pbc,
                                   cutoff=1.0, array_namespace=np)
    assert all(isinstance(a, np.ndarray) for a in (i1, j1, S1))
    assert np.array_equal(_canonical(i0, j0, S0), _canonical(i1, j1, S1))


def test_host_dlpack_capsules():
    """array_namespace='dlpack' returns DLPackTensor objects any framework can
    consume; numpy.from_dlpack here."""
    pos, cell, pbc = _random_config(seed=3)
    i0, j0 = nl.neighbour_list("ij", positions=pos, cell=cell, pbc=pbc,
                               cutoff=1.0)
    caps = nl.neighbour_list("ij", positions=pos, cell=cell, pbc=pbc, cutoff=1.0,
                             array_namespace="dlpack")
    assert all(isinstance(c, DLPackTensor) for c in caps)
    i1, j1 = (np.from_dlpack(c) for c in caps)
    z = np.zeros((len(i0), 3))
    assert np.array_equal(_canonical(i0, j0, z), _canonical(i1, j1, z))


def test_host_dlpack_capsule_frees_without_consumer():
    """A capsule that is never consumed frees its buffer via its destructor."""
    pos, cell, pbc = _random_config(N=500, seed=4)
    caps = nl.neighbour_list("i", positions=pos, cell=cell, pbc=pbc, cutoff=1.0,
                             array_namespace="dlpack")
    del caps  # must not leak / crash


def test_host_jax_namespace():
    """JAX (host) consumes the same capsules — proves output isn't cupy-pinned."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    pos, cell, pbc = _random_config(seed=6)
    i0 = nl.neighbour_list("i", positions=pos, cell=cell, pbc=pbc, cutoff=1.0)
    i1 = nl.neighbour_list("i", positions=pos, cell=cell, pbc=pbc, cutoff=1.0,
                           array_namespace=jnp)
    assert isinstance(i1, jax.Array)             # really a JAX array
    assert np.array_equal(np.sort(np.asarray(i1)), np.sort(i0))


# --------------------------------------------------------------- device paths

def _gpu_available():
    if cupy is None or not getattr(_ext, "_has_gpu", 0):
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="no CUDA device, or extension built without the GPU backend")


@requires_gpu
def test_cupy_in_cupy_out_matches_cpu():
    """cupy positions in (DLPack import) -> cupy arrays out, matching CPU."""
    pos, cell, pbc = _random_config(N=3000, seed=5)
    i_cpu, j_cpu, S_cpu = nl.neighbour_list("ijS", positions=pos, cell=cell,
                                            pbc=pbc, cutoff=1.0)
    pos_d = cupy.asarray(pos)
    i_g, j_g, S_g = nl.neighbour_list("ijS", positions=pos_d, cell=cell, pbc=pbc,
                                      cutoff=1.0)
    assert isinstance(i_g, cupy.ndarray) and isinstance(S_g, cupy.ndarray)
    assert np.array_equal(_canonical(i_cpu, j_cpu, S_cpu),
                          _canonical(cupy.asnumpy(i_g), cupy.asnumpy(j_g),
                                     cupy.asnumpy(S_g)))


@requires_gpu
def test_device_override_host_in_gpu_out():
    """numpy positions + device='cuda' forces the GPU and returns cupy."""
    pos, cell, pbc = _random_config(N=2500, seed=7)
    i_cpu, j_cpu = nl.neighbour_list("ij", positions=pos, cell=cell, pbc=pbc,
                                     cutoff=1.0)
    i_g, j_g = nl.neighbour_list("ij", positions=pos, cell=cell, pbc=pbc,
                                 cutoff=1.0, device="cuda")
    assert isinstance(i_g, cupy.ndarray)
    z = np.zeros((len(i_cpu), 3))
    assert np.array_equal(_canonical(i_cpu, j_cpu, z),
                          _canonical(cupy.asnumpy(i_g), cupy.asnumpy(j_g), z))


@requires_gpu
def test_device_dlpack_capsules():
    """array_namespace='dlpack' on the device returns DLPackTensors for cupy."""
    pos, cell, pbc = _random_config(N=1500, seed=8)
    pos_d = cupy.asarray(pos)
    caps = nl.neighbour_list("ij", positions=pos_d, cell=cell, pbc=pbc,
                             cutoff=1.0, array_namespace="dlpack")
    assert all(isinstance(c, DLPackTensor) for c in caps)
    i, j = (cupy.from_dlpack(c) for c in caps)
    assert isinstance(i, cupy.ndarray)


@requires_gpu
def test_multi_gpu_follows_input_device():
    ndev = cupy.cuda.runtime.getDeviceCount()
    pos, cell, pbc = _random_config(N=2000, seed=9)
    i_cpu = nl.neighbour_list("i", positions=pos, cell=cell, pbc=pbc, cutoff=1.0)
    for d in range(ndev):
        with cupy.cuda.Device(d):
            pos_d = cupy.asarray(pos)
            i, j = nl.neighbour_list("ij", positions=pos_d, cell=cell, pbc=pbc,
                                     cutoff=1.0)
        assert int(i.device.id) == d
        assert len(i) == len(i_cpu)


@requires_gpu
def test_coordination_cupy_and_override():
    pos, cell, pbc = _random_config(N=3000, seed=10)
    c_cpu = nl.coordination(positions=pos, cell=cell, pbc=pbc, cutoff=1.3)
    # device input
    c_dev = nl.coordination(positions=cupy.asarray(pos), cell=cell, pbc=pbc,
                            cutoff=1.3)
    assert isinstance(c_dev, cupy.ndarray)
    assert np.array_equal(cupy.asnumpy(c_dev), c_cpu)
    # host input forced onto the GPU
    c_ovr = nl.coordination(positions=pos, cell=cell, pbc=pbc, cutoff=1.3,
                            device="cuda")
    assert np.array_equal(cupy.asnumpy(c_ovr), c_cpu)
