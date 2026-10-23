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


# ------------------------------------------------------------------ batched (EXPERIMENTAL)
def _batch_structures():
    """Small cells of mixed size, shape and periodicity: the batched build's use case."""
    rng = np.random.default_rng(1)
    a = 5.43
    prim = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    out = [(np.array([[0, 0, 0], [a / 4] * 3]), prim, (True, True, True))]          # 2 atoms, cell << cutoff
    r, c = _diamond((1, 1, 1))
    out.append((r, c, (True, True, True)))                                          # 8 atoms
    r, c = _diamond((2, 1, 1))
    c = c + np.array([[0, 0, 0], [1.5, 0, 0], [0.7, 1.1, 0]])                       # sheared, 16 atoms
    r = r @ np.linalg.solve(np.diag([2 * a, a, a]), c)
    out.append((r, c, (True, True, True)))
    r, c = _diamond((2, 2, 1))
    c[2, 2] += 10.0                                                                 # slab, 32 atoms
    out.append((r + [0, 0, 5.0], c, (True, True, False)))
    r, c = _diamond((2, 2, 2))
    out.append((r + 3 * c[0] - 2 * c[2], c, (True, True, True)))                    # unwrapped positions
    return [(r + rng.normal(0, 0.05, r.shape), c, p) for r, c, p in out]


BATCH = _batch_structures()


def _pad(structs, dtype):
    N = max(len(r) for r, _, _ in structs)
    R = np.zeros((len(structs), N, 3))
    for b, (r, _, _) in enumerate(structs):
        R[b, :len(r)] = r
    C = np.stack([c for _, c, _ in structs])
    P = np.array([p for _, _, p in structs])
    n = np.array([len(r) for r, _, _ in structs], np.int32)
    return jnp.asarray(R, dtype), jnp.asarray(C, dtype), P, n


def _batched_rows(idx, shift, count, b):
    return _rows(np.asarray(idx)[b], np.asarray(shift)[b], np.asarray(count)[b])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("per_type", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_batched_build_matches_the_pair_list_per_structure(device, per_type, dtype):
    """One build for a padded batch of mixed small cells (2-atom primitive Si below the cutoff, sheared,
    a slab, unwrapped positions; mixed periodicity): each structure's rows are exactly neighbour_list's."""
    with jax.default_device(jax.devices("gpu" if device == "cuda" else "cpu")[0]):
        R, C, P, n = _pad(BATCH, dtype)
        types = None
        kw = {}
        if per_type:
            types = np.stack([np.pad(_types(len(r)), (0, R.shape[1] - len(r))) for r, _, _ in BATCH])
            kw = dict(types=types, cutoffs=TYPE_CUTOFFS)
        f = jax.jit(lambda R, C: xla.neighbour_matrix_batched(R, C, None if per_type else 6.0, P, 160,
                                                              n_atoms=n, **kw))
        idx, shift, count, status = f(R, C)
    assert np.asarray(status).tolist() == [[0, 0]] * len(BATCH)
    for b, (r, c, pbc) in enumerate(BATCH):
        nb = len(r)
        assert np.all(np.asarray(count)[b, nb:] == 0)                  # padding rows have no neighbours
        r_ref, c_ref = np.asarray(R[b, :nb], float), np.asarray(C[b], float)
        t = types[b, :nb] if per_type else None
        assert _batched_rows(idx, shift, count, b) == _reference(r_ref, c_ref, pbc, 6.0, t), b


@pytest.mark.parametrize("device", DEVICES)
def test_batched_build_flags_overflow_and_invalid_per_structure(device):
    """Status is per structure: a clipped row flags only its structure; a NaN position, a degenerate cell
    or a bad n_atoms invalidates only its own (counts 0), the others are still built."""
    with jax.default_device(jax.devices("gpu" if device == "cuda" else "cpu")[0]):
        R, C, P, n = _pad(BATCH[:3], "float64")
        R = R.at[1, 0, 0].set(jnp.nan)                                 # structure 1: a NaN position
        C = C.at[2].set(jnp.zeros((3, 3)))                             # structure 2: a degenerate cell
        idx, shift, count, status = xla.neighbour_matrix_batched(R, C, 6.0, P, 160, n_atoms=n)
        assert np.asarray(status).tolist() == [[0, 0], [0, 1], [0, 1]]
        assert np.all(np.asarray(count)[1:] == 0)
        _, _, count, status = xla.neighbour_matrix_batched(R[:1], C[:1], 6.0, P[:1], 4, n_atoms=n[:1])
        assert np.asarray(status).tolist() == [[1, 0]] and int(np.asarray(count).max()) > 4
    if device == "cuda":                                               # the host build refuses it instead
        with jax.default_device(jax.devices("gpu")[0]):
            _, _, _, status = xla.neighbour_matrix_batched(R[:1], C[:1], 6.0, P[:1], 160,
                                                           n_atoms=np.array([99], np.int32))
        assert np.asarray(status).tolist() == [[0, 1]]


@pytest.mark.parametrize("device", DEVICES)
def test_vmap_of_neighbour_matrix_is_one_batched_build(device):
    """jax.vmap of neighbour_matrix (and, on CUDA, of the planned build) routes to the batched handler --
    one custom call for the batch, not B -- with each structure's rows exactly as built alone."""
    structs = [s for s in BATCH if s[2] == (True, True, True) and len(s[0]) == 8] * 3
    with jax.default_device(jax.devices("gpu" if device == "cuda" else "cpu")[0]):
        R = jnp.asarray(np.stack([r + 0.01 * k for k, (r, _, _) in enumerate(structs)]))
        C = jnp.asarray(np.stack([c for _, c, _ in structs]))
        f = jax.jit(jax.vmap(lambda r, c: xla.neighbour_matrix(r, c, 6.0, (True,) * 3, 160)))
        hlo = f.lower(R, C).as_text()
        assert xla._BATCHED in hlo and hlo.count("custom_call_target=\"" + xla._TARGET + "\"") == 0
        idx, shift, count, ov = f(R, C)
        assert np.asarray(ov).tolist() == [0] * len(structs)
        for b in range(len(structs)):
            alone = xla.neighbour_matrix(R[b], C[b], 6.0, (True,) * 3, 160)
            assert _batched_rows(idx, shift, count, b) == _rows(*alone[:3])
        if device == "cuda" and PLANNED:
            P = xla.plan(8, np.asarray(C[0]), 6.0, (True,) * 3, 160)
            g = jax.jit(jax.vmap(lambda r, c: xla.neighbour_matrix_planned(r, c, 6.0, (True,) * 3, 160, P)))
            idx2, shift2, count2, status2 = g(R, C)
            assert np.asarray(status2).tolist() == [[0, 0]] * len(structs)
            for b in range(len(structs)):
                assert _batched_rows(idx2, shift2, count2, b) == _batched_rows(idx, shift, count, b)
