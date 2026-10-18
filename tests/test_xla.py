"""The XLA FFI neighbour matrix (matscipy_neighbours.xla) against the pair list.

Every build -- the regular CPU and CUDA handlers and the planned (device-only)
CUDA build -- must give exactly neighbour_list's (i, j, S) set: float64 and
float32 inputs, a global cutoff and per-type cutoffs, cells smaller than the
cutoff, sheared, slab and sparse (hashed lookup) geometries, and cells changed
after planning. Skips without jax or without the XLA library; the CUDA tests
skip without a CUDA device.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from matscipy_neighbours import neighbour_list  # noqa: E402

xla = pytest.importorskip("matscipy_neighbours.xla")
try:
    PLATFORMS = xla._register()
except ImportError as e:  # pragma: no cover - a build without JAX_FFI_INCLUDE_DIR
    pytest.skip(str(e), allow_module_level=True)

CUDA = "cuda" in PLATFORMS and any(d.platform == "gpu" for d in jax.devices())
PLANNED = CUDA and "planned" in xla._LIB
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _diamond(reps, a=5.43):
    """Si diamond: 8 atoms per cubic cell, `reps` cells per direction."""
    basis = np.array([[0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0],
                      [1, 1, 1], [1, 3, 3], [3, 1, 3], [3, 3, 1]]) * a / 4
    r = np.array([b + a * np.array([i, j, k]) for i in range(reps[0])
                  for j in range(reps[1]) for k in range(reps[2]) for b in basis], float)
    return r, np.diag(np.array(reps, float) * a)


def _geometries():
    rng = np.random.default_rng(0)
    out = {}
    r, c = _diamond((2, 2, 2))                      # 10.9 A: cell below 2 cutoffs
    out["small"] = (r, c, (True, True, True))
    r, c = _diamond((4, 4, 4))
    c = c + np.array([[0, 0, 0], [3.0, 0, 0], [1.0, 2.0, 0]])   # sheared
    r = r @ np.linalg.solve(np.diag([4 * 5.43] * 3), c)
    out["sheared"] = (r, c, (True, True, True))
    r, c = _diamond((3, 3, 2))
    c[2, 2] += 12.0                                 # vacuum along z, not periodic
    out["slab"] = (r + [0, 0, 6.0], c, (True, True, False))
    rs = rng.uniform(0, 1000.0, (400, 3))           # sparse: 166^3 cells > 2^20, hashed lookup
    rs[:200] = rs[200:] + rng.normal(0, 2.0, (200, 3))   # pairs, so there are neighbours
    out["sparse"] = (rs, np.diag([1000.0] * 3), (False, False, False))
    for k in out:
        r, c, p = out[k]
        out[k] = (r + rng.normal(0, 0.05, r.shape), c, p)
    return out


GEOMETRIES = _geometries()
TYPE_CUTOFFS = np.array([[5.0, 6.0], [6.0, 4.5]])


def _types(n):
    return (np.arange(n) % 3 == 0).astype(np.int32)   # two types, uneven


def _reference(r, c, pbc, cutoff, types=None):
    """neighbour_list's (i, j, S) rows, sorted."""
    if types is None:
        i, j, S = neighbour_list("ijS", positions=r, cell=c, pbc=pbc, cutoff=cutoff)
    else:
        cut = {(a + 1, b + 1): TYPE_CUTOFFS[a, b] for a in range(2) for b in range(2)}
        i, j, S = neighbour_list("ijS", positions=r, cell=c, pbc=pbc, cutoff=cut, numbers=types + 1)
    return sorted(zip(i.tolist(), j.tolist(), map(tuple, S.tolist())))


def _rows(idx, shift, count):
    idx, shift, count = map(np.asarray, (idx, shift, count))
    K = idx.shape[1]
    return sorted((i, int(idx[i, k]), tuple(shift[i, k].tolist()))
                  for i in range(len(count)) for k in range(min(int(count[i]), K)))


def _cases():
    for name in GEOMETRIES:
        for per_type in (False, True):
            for dtype in ("float64", "float32"):
                yield name, per_type, dtype


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name, per_type, dtype", list(_cases()))
def test_regular_build_matches_the_pair_list(device, name, per_type, dtype):
    r, c, pbc = GEOMETRIES[name]
    types = _types(len(r)) if per_type else None
    with jax.default_device(jax.devices("gpu" if device == "cuda" else "cpu")[0]):
        R = jnp.asarray(r, dtype)
        kw = dict(types=types, cutoffs=TYPE_CUTOFFS) if per_type else {}
        f = jax.jit(lambda R, C: xla.neighbour_matrix(R, C, None if per_type else 6.0, pbc, 96, **kw))
        idx, shift, count, ov = f(R, jnp.asarray(c, dtype))
    assert int(ov) == 0
    # float32 inputs: compare against the list of the float32-rounded positions
    r_ref, c_ref = np.asarray(R, float), np.asarray(jnp.asarray(c, dtype), float)
    assert _rows(idx, shift, count) == _reference(r_ref, c_ref, pbc, 6.0, types)


@pytest.mark.skipif(not PLANNED, reason="no CUDA device, or no planned build")
@pytest.mark.parametrize("name, per_type, dtype", list(_cases()))
@pytest.mark.parametrize("deform", ["none", "x0.95", "x1.1", "shear"])
def test_planned_build_matches_the_pair_list(name, per_type, dtype, deform):
    """The grid is planned on the undeformed cell; the build reads the deformed one."""
    r, c, pbc = GEOMETRIES[name]
    M = {"none": np.eye(3), "x0.95": 0.95 * np.eye(3), "x1.1": 1.1 * np.eye(3),
         "shear": np.array([[1, 0.06, 0], [0, 1, 0.04], [0, 0, 1.0]])}[deform]
    r2, c2 = r @ np.linalg.solve(c, c @ M), c @ M
    types = _types(len(r)) if per_type else None
    K = 128
    cutoff = 6.0
    P = xla.plan(len(r), c, cutoff, pbc, K, ncutoffs=2 if per_type else 0, dtype=dtype)
    assert bool(P[3]) == (name == "sparse")          # the hashed lookup where the grid is sparse
    kw = dict(types=types, cutoffs=TYPE_CUTOFFS) if per_type else {}
    R, C = jnp.asarray(r2, dtype), jnp.asarray(c2, dtype)
    f = jax.jit(lambda R, C: xla.neighbour_matrix_planned(R, C, cutoff, pbc, K, P, **kw))
    idx, shift, count, status = f(R, C)
    assert np.asarray(status).tolist() == [0, 0]
    assert _rows(idx, shift, count) == _reference(np.asarray(R, float), np.asarray(C, float), pbc, cutoff, types)


@pytest.mark.skipif(not PLANNED, reason="no CUDA device, or no planned build")
def test_planned_build_flags_on_the_device():
    r, c, pbc = GEOMETRIES["small"]
    P = xla.plan(len(r), c, 6.0, pbc, 8)
    f = jax.jit(lambda R, C: xla.neighbour_matrix_planned(R, C, 6.0, pbc, 8, P)[3])
    assert np.asarray(f(jnp.asarray(r), jnp.asarray(c))).tolist() == [1, 0]          # K too small
    P = xla.plan(len(r), c, 6.0, pbc, 96)
    g = jax.jit(lambda R, C: xla.neighbour_matrix_planned(R, C, 6.0, pbc, 96, P)[2:])
    count, status = g(jnp.asarray(r).at[3, 1].set(jnp.nan), jnp.asarray(c))
    assert np.asarray(status).tolist() == [0, 1] and int(np.asarray(count).sum()) == 0   # non-finite
    count, status = g(jnp.asarray(r), jnp.zeros((3, 3)))
    assert np.asarray(status)[1] == 1                                                   # degenerate cell
    P = xla.plan(len(r), c, 5.0, pbc, 96, ncutoffs=2)
    h = jax.jit(lambda R, C: xla.neighbour_matrix_planned(R, C, 5.0, pbc, 96, P, types=_types(len(r)),
                                                          cutoffs=TYPE_CUTOFFS)[3])
    assert np.asarray(h(jnp.asarray(r), jnp.asarray(c)))[1] == 1                         # cutoff above the grid's


@pytest.mark.skipif(not PLANNED, reason="no CUDA device, or no planned build")
def test_planned_build_inside_a_conditional_in_a_loop():
    """Rebuilt every fifth step inside lax.cond in a fori_loop (one CUDA graph per step):
    the last rebuild is the pair list of that step's positions."""
    r, c, pbc = GEOMETRIES["sheared"]
    P = xla.plan(len(r), c, 6.0, pbc, 96)
    R0, C = jnp.asarray(r), jnp.asarray(c)
    key = jax.random.PRNGKey(0)
    build = lambda R: xla.neighbour_matrix_planned(R, C, 6.0, pbc, 96, P)[:3]

    @jax.jit
    def loop(carry):
        def body(i, c):
            Ri = R0 + 0.2 * jax.random.normal(jax.random.fold_in(key, i), R0.shape)
            return jax.lax.cond(i % 5 == 4, lambda: build(Ri), lambda: c)
        return jax.lax.fori_loop(0, 20, body, carry)

    out = loop(build(R0))
    R_last = np.asarray(R0 + 0.2 * jax.random.normal(jax.random.fold_in(key, 19), R0.shape))
    assert _rows(*out) == _reference(R_last, c, pbc, 6.0)
