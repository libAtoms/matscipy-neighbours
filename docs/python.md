# Python API

```python
from matscipy_neighbours import (
    neighbour_list, coordination, first_neighbours, segment_sum, mabincount,
    triplet_list, get_jump_indicies, mic, DLPackTensor,
)
```

## `neighbour_list`

```python
neighbour_list(quantities, atoms=None, cutoff=None, *,
               positions=None, cell=None, pbc=None, numbers=None,
               cell_origin=None, device=None, array_namespace=None)
```

Compute a neighbour list and return one array per requested quantity.

**Quantities.** `quantities` is a string built from:

| char | meaning |
|------|---------|
| `i`  | first atom index |
| `j`  | second atom index |
| `d`  | distance |
| `D`  | distance vector, shape `(npairs, 3)` |
| `S`  | cell shift, shape `(npairs, 3)` |

Arrays come back in the order requested; a single character returns a bare
array. Index arrays (`i`, `j`, `S`) are `int64`, distances are `float64`. The
shift satisfies `D == r[j] - r[i] + S @ cell`, and pairs are sorted by `i`.

**Configuration.** Pass either an ASE `Atoms` object as `atoms`, or explicit
`positions` (plus `cell`, `pbc`, `numbers`, `cell_origin`). If `cell` is omitted
for host input, or has zero lattice vectors (an ASE molecule), a box around the
atoms is used for the missing directions; those directions must be
non-periodic, and a full cell is required for device input. Invalid input
(wrong shapes, a non-positive or non-finite cutoff, non-finite positions, types
outside a per-type cutoff matrix) raises `TypeError` or `ValueError`.

**Cutoff.** `cutoff` is a scalar, a per-atom radius array (the pair cutoff is the
sum of the two radii), or a dict `{(el1, el2): cutoff}` of element-pair cutoffs.

```python
i, j, D = neighbour_list("ijD", positions=r, cell=cell, pbc=True, cutoff=2.5)
S = neighbour_list("S", atoms=atoms, cutoff=5.0)
i, j = neighbour_list("ij", positions=r, cell=cell, pbc=True,
                      cutoff={("Si", "Si"): 2.7, ("Si", "O"): 1.8},
                      numbers=numbers)
```

### Device selection

The backend follows the input array by default: a host (NumPy) array runs on the
CPU; a device array (CuPy, or any DLPack producer on a GPU) runs on the GPU with
results staying on the device. Override with `device`:

- `None` — auto (follow the input).
- `"cpu"` — force the CPU backend.
- `"cuda"` / `"gpu"`, an integer id, or `("cuda", id)` — force the GPU backend
  on a chosen device.

```python
import cupy as cp
i, j, D = neighbour_list("ijD", positions=cp.asarray(r), cell=cell,
                         pbc=True, cutoff=2.5)            # GPU in, GPU out
i, j = neighbour_list("ij", positions=r, cell=cell, pbc=True,
                      cutoff=2.5, device="cuda")          # host in, GPU out
```

`cell` is required for device-resident input.

### Output framework

`array_namespace` selects the array type of the result:

- `None` (default) — CuPy on the device, NumPy on the host.
- a module with `from_dlpack` (`numpy`, `cupy`, `torch`, `jax.numpy`) — that
  framework's arrays.
- `"dlpack"` — [`DLPackTensor`](#dlpacktensor) capsules the caller consumes with
  its own `from_dlpack`.

```python
import jax.numpy as jnp
i = neighbour_list("i", positions=r, cell=cell, pbc=True, cutoff=2.5,
                   array_namespace=jnp)          # JAX array out
```

## `coordination`

```python
coordination(atoms=None, cutoff=None, *, positions=None, cell=None,
             pbc=None, numbers=None, cell_origin=None,
             device=None, array_namespace=None)
```

Number of neighbours of each atom within `cutoff`. On the GPU this runs a
count-only kernel that never materialises the pair list. `device` and
`array_namespace` behave as for `neighbour_list`.

## `neighbour_matrix`

```python
neighbour_matrix(atoms=None, cutoff=None, max_neighbours=None, *,
                 positions=None, cell=None, pbc=None, numbers=None,
                 cell_origin=None, device=None, array_namespace=None,
                 quantities="D")
```

Dense, **fixed-capacity** neighbour list — each atom's neighbours fill a row of
an `n × max_neighbours` matrix, so the output shape is *static* (it depends only
on `n` and `max_neighbours`, not on the number of pairs). This is the form to use
with frameworks that compile for fixed shapes (e.g. JAX `jit`), where forces are
a masked sum over the neighbour axis with no scatter.

Returns `(idx, *per_slot, count)`:

- `idx` — shape `(n, max_neighbours)`, int64 neighbour indices;
- the per-slot arrays selected by `quantities`, in the order given: `"D"` the
  distance vectors `D` (float64), `"S"` the cell shifts (int64), each of shape
  `(n, max_neighbours, 3)`. The default `"D"` returns `(idx, dist, count)`;
  `""` returns `(idx, count)`, the cheapest form when the consumer recomputes
  the distances from the positions (add `"S"` for a periodic cell);
- `count` — shape `(n,)`, true neighbour count; mask with
  `arange(max_neighbours) < count[:, None]`.

!!! warning "Unused slots are not cleared"
    Only the first `count` slots of a row are defined; the rest may hold any
    bit pattern, including NaN distances and out-of-range indices. Select with
    the mask rather than multiplying by it (`0 * NaN` is NaN), and do not gather
    with the indices of unused slots.

`device` and `array_namespace` behave as for `neighbour_list`. Raises
`ValueError` if any atom has more than `max_neighbours` neighbours (capacity too
small) — retry with a larger value.

```python
import jax.numpy as jnp
idx, dist, count = neighbour_matrix(positions=r_jax, cell=cell, pbc=True,
                                    cutoff=2.5, max_neighbours=96,
                                    array_namespace=jnp)
mask = jnp.arange(idx.shape[1])[None, :] < count[:, None]
dist = jnp.where(mask[..., None], dist, 0.0)              # select, don't multiply
r2 = jnp.where(mask, (dist * dist).sum(-1), 1.0)          # masked, jit-friendly
```

## JAX inside `jit`: `matscipy_neighbours.xla` (experimental)

`neighbour_matrix` above is called from Python, between jitted steps. The
`matscipy_neighbours.xla` module registers it as an XLA FFI call instead, so a
JAX program can build its list *inside* `jit` — under `lax.cond`, in a
`lax.fori_loop` — on the CPU or a CUDA GPU. It needs the handler library, built
when CMake is given JAX's FFI headers:

```bash
pip install . -C cmake.define.JAX_FFI_INCLUDE_DIR="$(python -c 'import jax.ffi; print(jax.ffi.include_dir())')"
#   add -C cmake.define.ENABLE_CUDA=ON for the CUDA handlers
```

```python
from matscipy_neighbours import xla

idx, shift, count, overflow = xla.neighbour_matrix(positions, cell, cutoff=6.0,
                                                   pbc=(True, True, True), max_neighbours=96)
mask = jnp.arange(96)[None, :] < count[:, None]
D = positions[idx] - positions[:, None] + shift @ cell     # differentiable in positions
```

Positions and cell (rows: lattice vectors) are float32 or float64; the search
runs in float64. The outputs are int32: `shift` is the cell shift of each slot,
so cells smaller than the cutoff work, and `D` can be formed from the positions
being differentiated. `overflow` (a device scalar) is 1 when a row was clipped.
Per-type cutoffs: pass `types` (int, `[n]`) and `cutoffs` (`[m, m]`, symmetric,
not squared).

On CUDA the regular build waits for and reads back from the device several
times per call (sizes, flags, the cell). `xla.plan` + `xla.neighbour_matrix_planned`
remove that: the cell grid, the dense or hashed cell lookup and a workspace are
fixed once on the host from a reference cell, and each build is enqueued on
XLA's stream with no host synchronisation, allocation or read-back. It is
declared CUDA-graph compatible, so a rebuild inside `lax.cond` in a jitted MD
loop stays on the device (no host round trip per step). The cell may change
between builds (NPT); plan again if it changes a lot. Its flags are a device
array `status = (overflow, invalid)`: invalid input (non-finite positions, a
degenerate cell, a type out of range) sets `invalid` where the regular build
raises.

```python
plan = xla.plan(len(positions), cell, 6.0, (True, True, True), 96)
idx, shift, count, status = xla.neighbour_matrix_planned(positions, cell, 6.0,
                                                         (True, True, True), 96, plan)
```

### Many small structures at once

Each build above has a fixed cost per call (a cell grid, a scan, launches),
which dominates for unit cells and small supercells. Evaluating many of them is
then limited by the list, not the model. `xla.neighbour_matrix_batched` builds
the lists of a padded batch of independent structures in one call: one CUDA
launch (or one OpenMP loop) over every row of the batch.

```python
# positions [B, N, 3] (padded), cells [B, 3, 3], n_atoms [B] live atoms each
idx, shift, count, status = xla.neighbour_matrix_batched(positions, cells, 6.0,
                                                         (True, True, True), 96, n_atoms=n_atoms)
# idx [B, N, K], shift [B, N, K, 3], count [B, N] (0 on padding rows), status [B, 2]
```

The search is brute force over the images within the cutoff, O(n^2) per
structure: right for small cells, wrong for large ones (use the cell-list
builds). Positions need not lie in the cell, and a cell may be smaller than the
cutoff. `pbc` is a 3-tuple for all or a `[B, 3]` boolean array; `types` is `[B, N]`.
`status` is per structure, `(overflow, invalid)`: one bad structure (non-finite
positions, a degenerate cell) gets no neighbours and leaves the others built.

`jax.vmap` of `neighbour_matrix` or `neighbour_matrix_planned` calls the batched
build, so a vmapped single-structure program gets it without changes. On an
RTX 4000 Ada a batch of 1024 64-atom Si cells (cutoff 6 Å) builds in 4 ms
(3.9 µs a cell), and 1024 2-atom primitive cells in 0.09 ms.

## GPU memory

The GPU backend allocates from its own caching memory pool: freed buffers stay
reserved and are reused by the next call instead of being allocated afresh,
which matters for calls made every step. The cached memory is not available to
other libraries (CuPy, PyTorch, JAX, …); return it with `empty_gpu_cache()`
before handing the device to them. Running out of memory — on the host or the
GPU — raises `MemoryError`; on the GPU the cache is emptied and the
allocation retried once first.

## Other functions

- `first_neighbours(n, i)` — row-start (CSR) offsets for an `i`-sorted list:
  the pairs of atom `k` are `seed[k]:seed[k+1]`. Atoms before the first pair get
  `-1` (as in matscipy), so `maximum(seed, 0)` gives offsets with an empty
  segment for every atom without neighbours. A device `i` (CuPy, JAX, PyTorch;
  int64) is processed on its GPU and the result stays there, by default in the
  framework of `i`.
- `triplet_list(first_neighbours, abs_dr_p=None, cutoff=None)` — triplets from a
  first-neighbour array.
- `get_jump_indicies(sorted_array)` — jump indices of an ordered array.
- `mic(dr, cell, pbc=None)` — minimum-image-convention wrap of distance vectors.
- `empty_gpu_cache()` — return the GPU memory cached by the library's
  allocator to the driver (see [GPU memory](#gpu-memory); no-op without a GPU
  backend).

## Per-atom sums: `segment_sum` and `mabincount`

```python
segment_sum(values, seed, *, total=False, array_namespace=None)
mabincount(x, weights, minlength, axis=0)
```

Summing per-pair quantities per atom (forces, per-atom energies or virials)
is a bincount over the pairs' first index, usually written as a scatter: a
weighted `bincount`, `cupyx.scatter_add` or `jax.ops.segment_sum`. The pair
list is sorted by its first index, so the pairs of an atom form one
contiguous segment, and `segment_sum` sums each segment without atomics. On
an AMD MI300A this takes 0.55 ms for 5.4×10⁷ force vectors, against 4.6 ms for
`cupyx.scatter_add` and about 440 ms for CuPy's weighted `bincount`, which is
dominated by its input validation (two full reductions, slow with CuPy on
ROCm). The summation order is fixed, so results are reproducible bit for
bit, unlike with atomics.

- `segment_sum(values, seed)` sums the rows of `values` (shape
  `(npairs, ...)`, float32, float64, int32 or int64, C-contiguous) over the
  segments given by `seed` from `first_neighbours`, and returns an array of
  shape `(n, ...)`. It runs on the host (OpenMP) for NumPy input and on the
  GPU for device input (CuPy, JAX, PyTorch via DLPack; `seed` on the same
  device), returning the result in the framework of `values`.
- `total=True` also returns the sum over all atoms, of shape
  `values.shape[1:]`, computed in the same call. On the GPU this saves a
  separate full reduction, which can be slow too (CuPy on ROCm, see the
  [benchmark notes on performance portability](benchmark.md#performance-portability-nvidia-and-amd)).
- `mabincount(x, weights, minlength, axis=0)` has the signature of
  `matscipy.numpy_tricks.mabincount`, but requires `x` to be **sorted**, with
  every entry in `[0, minlength)` (`ValueError` otherwise). It builds the
  segments with `first_neighbours` and sums them with `segment_sum`.

Forces, energy and virial of a pair potential in one call, as in the array
example:

```python
i, D = neighbour_list("iD", positions=positions, cell=cell, pbc=True,
                      cutoff=cutoff)
r2 = (D * D).sum(axis=1)
... # per-pair force prefactor `coef` and energy `e_pair`
pair = xp.empty((len(i), 5))
pair[:, :3] = coef[:, None] * D       # force on i
pair[:, 3] = e_pair
pair[:, 4] = coef * r2                # virial term
per_atom, total = segment_sum(pair, first_neighbours(n, i), total=True)
forces = per_atom[:, :3]
energy, virial = 0.5 * float(total[3]), -0.5 * float(total[4])
```

Inside `jax.jit` these functions cannot be called (they exchange data with
DLPack eagerly); there, use the neighbour matrix, where each atom sums its own
row.

## `DLPackTensor`

A zero-copy DLPack tensor returned when `array_namespace="dlpack"`. It
implements `__dlpack__` / `__dlpack_device__`, so any DLPack consumer can adopt
it:

```python
caps = neighbour_list("ijD", positions=r, cell=cell, pbc=True, cutoff=2.5,
                      array_namespace="dlpack")
import numpy as np
i = np.from_dlpack(caps[0])
```
