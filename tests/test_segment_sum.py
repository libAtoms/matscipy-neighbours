"""segment_sum and the matscipy-compatible mabincount.

Host tests always run; the device tests skip without cupy or a GPU build.
"""

import numpy as np
import pytest

import _matscipy_neighbours as _ext
import matscipy_neighbours as msn

try:
    import cupy
except ImportError:  # pragma: no cover - exercised in CPU-only CI
    cupy = None


def _gpu_available():
    if cupy is None or not getattr(_ext, "_has_gpu", 0):
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not _gpu_available(),
                                  reason="no GPU, or extension built without "
                                         "a GPU backend")

# Atoms 0-1 before the first pair, 4 and 6 without pairs, 8-9 after the last.
I = np.array([2, 2, 2, 3, 5, 5, 7, 7, 7, 7])
N = 10


def _reference(i, w, n):
    out = np.zeros((n,) + w.shape[1:], dtype=w.dtype)
    np.add.at(out, i, w)
    return out


@pytest.mark.parametrize("shape", [(), (3,), (3, 3)])
def test_segment_sum_host_matches_add_at(shape):
    w = np.random.default_rng(1).standard_normal((len(I),) + shape)
    seed = msn.first_neighbours(N, I)
    out, total = msn.segment_sum(w, seed, total=True)
    ref = _reference(I, w, N)
    assert out.shape == ref.shape and total.shape == shape
    assert np.allclose(out, ref, rtol=0, atol=1e-14)
    assert np.allclose(total, ref.sum(axis=0), rtol=0, atol=1e-13)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_segment_sum_host_dtypes(dtype):
    w = np.arange(len(I) * 2, dtype=dtype).reshape(-1, 2)
    out = msn.segment_sum(w, msn.first_neighbours(N, I))
    assert out.dtype == dtype
    assert np.array_equal(out, _reference(I, w, N))


def test_segment_sum_empty_list():
    out, total = msn.segment_sum(np.zeros((0, 3)), np.zeros(5, dtype=np.int64),
                                 total=True)
    assert out.shape == (4, 3) and not out.any() and not total.any()


@pytest.mark.parametrize("seed", [[0, 3, 2], [0, 2, 11]])
def test_segment_sum_rejects_bad_seed(seed):
    with pytest.raises(ValueError):
        msn.segment_sum(np.ones((10, 3)), np.array(seed))


def test_mabincount_matches_matscipy_semantics():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((len(I), 3))
    assert np.allclose(msn.mabincount(I, w, N), _reference(I, w, N))
    # Binning along another axis.
    wt = rng.standard_normal((4, len(I), 2))
    want = np.moveaxis(_reference(I, np.moveaxis(wt, 1, 0), N), 0, 1)
    assert np.allclose(msn.mabincount(I, wt, N, axis=1), want)


@pytest.mark.parametrize("x", [[0, 2, 1], [0, 1, 3]])
def test_mabincount_requires_sorted_indices_in_range(x):
    with pytest.raises(ValueError):
        msn.mabincount(np.array(x), np.ones(3), 3)


@requires_gpu
@pytest.mark.parametrize("shape", [(), (3,), (3, 3)])
def test_segment_sum_cupy_matches_host(shape):
    w = np.random.default_rng(3).standard_normal((len(I),) + shape)
    seed = msn.first_neighbours(N, I)
    out, total = msn.segment_sum(cupy.asarray(w), cupy.asarray(seed), total=True)
    assert isinstance(out, cupy.ndarray) and isinstance(total, cupy.ndarray)
    ref = _reference(I, w, N)
    assert np.allclose(cupy.asnumpy(out), ref, rtol=0, atol=1e-14)
    assert np.allclose(cupy.asnumpy(total), ref.sum(axis=0), rtol=0, atol=1e-13)


@requires_gpu
def test_segment_sum_cupy_on_neighbour_list():
    """Forces from a GPU pair list: segment_sum over device seeds equals a
    host bincount, and the total equals the sum over all pairs."""
    rng = np.random.default_rng(4)
    L = 12.0
    pos = rng.uniform(0, L, size=(3000, 3))
    i, D = msn.neighbour_list("iD", positions=cupy.asarray(pos),
                              cell=np.diag([L, L, L]), pbc=True, cutoff=1.5)
    seed = msn.first_neighbours(len(pos), i)
    out, total = msn.segment_sum(D, seed, total=True)
    ref = _reference(cupy.asnumpy(i), cupy.asnumpy(D), len(pos))
    assert np.allclose(cupy.asnumpy(out), ref, rtol=0, atol=1e-12)
    assert np.allclose(cupy.asnumpy(total), cupy.asnumpy(D).sum(axis=0),
                       rtol=0, atol=1e-9)


@requires_gpu
def test_segment_sum_seed_on_other_device_raises():
    with pytest.raises(ValueError):
        msn.segment_sum(cupy.ones((10, 3)), msn.first_neighbours(N, I))


@requires_gpu
def test_mabincount_cupy():
    w = np.random.default_rng(5).standard_normal((len(I), 3))
    out = msn.mabincount(cupy.asarray(I), cupy.asarray(w), N)
    assert isinstance(out, cupy.ndarray)
    assert np.allclose(cupy.asnumpy(out), _reference(I, w, N))
