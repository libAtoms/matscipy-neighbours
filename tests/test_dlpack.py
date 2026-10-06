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


# --------------------------------------------------- dense fixed-capacity matrix

def _per_atom_sets(i, j, n):
    from collections import defaultdict
    s = defaultdict(set)
    for a, b in zip(np.asarray(i), np.asarray(j)):
        s[int(a)].add(int(b))
    return s


def test_matrix_host_matches_pairs():
    pos, cell, pbc = _random_config(seed=11)
    n = len(pos)
    i, j = nl.neighbour_list("ij", positions=pos, cell=cell, pbc=pbc, cutoff=1.0)
    idx, dist, count = nl.neighbour_matrix(positions=pos, cell=cell, pbc=pbc,
                                           cutoff=1.0, max_neighbours=64)
    idx, dist, count = np.asarray(idx), np.asarray(dist), np.asarray(count)
    assert idx.shape == (n, 64) and dist.shape == (n, 64, 3)
    assert np.array_equal(count, np.bincount(i, minlength=n))
    ref = _per_atom_sets(i, j, n)
    for a in range(n):
        assert {int(x) for x in idx[a, :count[a]]} == ref[a]


def test_matrix_overflow_raises():
    pos, cell, pbc = _random_config(seed=12)
    with pytest.raises(ValueError):
        nl.neighbour_matrix(positions=pos, cell=cell, pbc=pbc, cutoff=1.5,
                            max_neighbours=2)


def _matrix_pairs(idx, shift, count):
    """Sorted (i, j, shift) rows of a neighbour matrix, over the valid slots."""
    idx, count = np.asarray(idx), np.asarray(count)
    shift = None if shift is None else np.asarray(shift)
    rows = []
    for a in range(len(count)):
        for s in range(int(count[a])):
            sh = (0, 0, 0) if shift is None else tuple(shift[a, s])
            rows.append((a, int(idx[a, s])) + tuple(int(x) for x in sh))
    return sorted(rows)


@pytest.mark.parametrize("quantities", ["", "D", "S", "DS", "SD"])
def test_matrix_quantities_host(quantities):
    pos, cell, pbc = _random_config(N=500, L=6.0, seed=21)
    n, K = len(pos), 48
    out = nl.neighbour_matrix(positions=pos, cell=cell, pbc=pbc, cutoff=1.2,
                              max_neighbours=K, quantities=quantities)
    assert len(out) == 2 + len(quantities)
    idx, count = np.asarray(out[0]), np.asarray(out[-1])
    per_slot = dict(zip(quantities, (np.asarray(a) for a in out[1:-1])))
    assert idx.shape == (n, K) and idx.dtype == np.int64
    for a in per_slot.values():
        assert a.shape == (n, K, 3)
    i, j, S = nl.neighbour_list("ijS", positions=pos, cell=cell, pbc=pbc,
                                cutoff=1.2)
    assert np.array_equal(count, np.bincount(i, minlength=n))
    ref = _canonical(i, j, S if "S" in per_slot else np.zeros_like(S))
    got = np.array(_matrix_pairs(idx, per_slot.get("S"), count))
    assert np.array_equal(got, ref)
    if "D" in per_slot and "S" in per_slot:  # D == r[j] - r[i] + S @ cell
        mask = np.arange(K)[None, :] < count[:, None]
        D = pos[idx] - pos[:, None, :] + per_slot["S"] @ cell
        assert np.allclose(per_slot["D"][mask], D[mask])


@pytest.mark.parametrize("quantities", ["X", "DD", "d"])
def test_matrix_quantities_invalid(quantities):
    pos, cell, pbc = _random_config(N=50, seed=22)
    with pytest.raises(ValueError):
        nl.neighbour_matrix(positions=pos, cell=cell, pbc=pbc, cutoff=1.2,
                            max_neighbours=8, quantities=quantities)


def test_empty_gpu_cache_is_callable():
    nl.empty_gpu_cache()   # a no-op without the GPU backend


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
def test_device_wrong_dtype_raises_not_crash():
    """A device array of the wrong dtype must raise cleanly (the DLPack importer
    must not double-free the producer's capsule)."""
    pos = cupy.asarray(np.random.rand(400, 3).astype(np.float32))
    with pytest.raises(TypeError):
        nl.neighbour_list("ij", positions=pos, cell=np.diag([12.0, 12, 12]),
                          cell_origin=np.zeros(3), pbc=False, cutoff=2.5)


@requires_gpu
def test_matrix_cupy_matches_cpu():
    pos, cell, pbc = _random_config(seed=13)
    n = len(pos)
    idx_c, _, cnt_c = nl.neighbour_matrix(positions=pos, cell=cell, pbc=pbc,
                                          cutoff=1.0, max_neighbours=64)
    idx_g, dist_g, cnt_g = nl.neighbour_matrix(positions=cupy.asarray(pos),
                                               cell=cell, pbc=pbc, cutoff=1.0,
                                               max_neighbours=64)
    assert isinstance(idx_g, cupy.ndarray) and isinstance(dist_g, cupy.ndarray)
    cnt_c = np.asarray(cnt_c)
    assert np.array_equal(cupy.asnumpy(cnt_g), cnt_c)
    ig, ic = cupy.asnumpy(idx_g), np.asarray(idx_c)
    for a in range(n):
        assert set(ig[a, :cnt_c[a]]) == set(ic[a, :cnt_c[a]])


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


@requires_gpu
@pytest.mark.parametrize("quantities", ["", "S", "DS"])
def test_matrix_quantities_cupy_match_host(quantities):
    pos, cell, pbc = _random_config(N=500, L=6.0, seed=23)
    kw = dict(cell=cell, pbc=pbc, cutoff=1.2, max_neighbours=48,
              quantities=quantities)
    host = nl.neighbour_matrix(positions=pos, **kw)
    dev = nl.neighbour_matrix(positions=cupy.asarray(pos), **kw)
    assert all(isinstance(a, cupy.ndarray) for a in dev)
    h_slot = dict(zip(quantities, host[1:-1]))
    d_slot = dict(zip(quantities, (cupy.asnumpy(a) for a in dev[1:-1])))
    assert np.array_equal(cupy.asnumpy(dev[-1]), np.asarray(host[-1]))
    assert (_matrix_pairs(cupy.asnumpy(dev[0]), d_slot.get("S"),
                          cupy.asnumpy(dev[-1]))
            == _matrix_pairs(host[0], h_slot.get("S"), host[-1]))


@requires_gpu
def test_gpu_out_of_memory_raises_memoryerror():
    """Running out of GPU memory raises MemoryError instead of aborting the
    process, and the next sensible call works."""
    pos, cell, pbc = _random_config(N=1000, seed=24)
    with pytest.raises(MemoryError):
        nl.neighbour_matrix(positions=cupy.asarray(pos), cell=cell, pbc=pbc,
                            cutoff=1.2, max_neighbours=(1 << 31) - 1)
    nl.empty_gpu_cache()
    idx, dist, count = nl.neighbour_matrix(positions=cupy.asarray(pos),
                                           cell=cell, pbc=pbc, cutoff=1.2,
                                           max_neighbours=64)
    assert isinstance(idx, cupy.ndarray) and int(count.max()) <= 64


# The host reference cases of test_first_neighbours_reference_values, plus
# leading, inner and trailing atoms without pairs and the empty list.
_FIRST_NEIGHBOURS_CASES = [
    (5, [1, 1, 1, 1, 3, 3, 3]),
    (6, [0, 1, 2, 3, 4, 5]),
    (8, [0, 1, 2, 3, 4, 5, 6]),
    (8, [0, 1, 2, 3, 3, 5, 6]),
    (10, [3, 3, 7, 7, 7]),
    (4, []),
]


@requires_gpu
@pytest.mark.parametrize("n, i", _FIRST_NEIGHBOURS_CASES)
def test_first_neighbours_cupy_matches_host(n, i):
    host = nl.first_neighbours(n, np.array(i, dtype=np.int64))
    dev = nl.first_neighbours(n, cupy.asarray(i, dtype=cupy.int64))
    assert isinstance(dev, cupy.ndarray)
    assert np.array_equal(cupy.asnumpy(dev), host)


@requires_gpu
def test_first_neighbours_cupy_on_neighbour_list():
    """Device seeds of a GPU pair list equal the host seeds of the same list,
    and give each atom's pairs."""
    pos, cell, pbc = _random_config(N=3000, seed=25)
    i = nl.neighbour_list("i", positions=cupy.asarray(pos), cell=cell, pbc=pbc,
                          cutoff=1.5)
    seed = nl.first_neighbours(len(pos), i)
    assert np.array_equal(cupy.asnumpy(seed),
                          nl.first_neighbours(len(pos), cupy.asnumpy(i)))
    counts = cupy.diff(cupy.maximum(seed, 0))
    assert cupy.array_equal(counts, cupy.bincount(i, minlength=len(pos)))


@requires_gpu
@pytest.mark.parametrize("n, i", [(3, [0, 2, 1]), (3, [0, 3]), (3, [-1, 0]),
                                  (-1, [0])])
def test_first_neighbours_cupy_rejects_bad_input(n, i):
    with pytest.raises(ValueError):
        nl.first_neighbours(n, cupy.asarray(i, dtype=cupy.int64))


@requires_gpu
def test_first_neighbours_cupy_wrong_dtype_raises():
    with pytest.raises(TypeError):
        nl.first_neighbours(3, cupy.asarray([0, 1, 2], dtype=cupy.int32))


@requires_gpu
def test_first_neighbours_jax_device_in_jax_out():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("no JAX GPU device")
    i = jax.device_put(jax.numpy.asarray([1, 1, 3, 3, 3], dtype="int64"), gpu)
    seed = nl.first_neighbours(5, i)
    assert type(seed).__module__.split(".")[0] in ("jax", "jaxlib")
    assert np.array_equal(np.asarray(seed), [-1, 0, 2, 2, 5, 5])
