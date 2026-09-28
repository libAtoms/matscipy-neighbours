# Python API

```python
from matscipy_neighbours import (
    neighbour_list, coordination, first_neighbours, triplet_list,
    get_jump_indicies, mic, DLPackTensor,
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

## GPU memory

The GPU backend allocates from its own caching memory pool: freed buffers stay
reserved and are reused by the next call instead of being allocated afresh,
which matters for calls made every step. The cached memory is not available to
other libraries (CuPy, PyTorch, JAX, …); return it with `empty_gpu_cache()`
before handing the device to them. Running out of memory — on the host or the
GPU — raises `MemoryError`; on the GPU the cache is emptied and the
allocation retried once first.

## Other functions

- `first_neighbours(n, i)` — row-start (CSR) offsets for an `i`-sorted list.
- `triplet_list(first_neighbours, abs_dr_p=None, cutoff=None)` — triplets from a
  first-neighbour array.
- `get_jump_indicies(sorted_array)` — jump indices of an ordered array.
- `mic(dr, cell, pbc=None)` — minimum-image-convention wrap of distance vectors.
- `empty_gpu_cache()` — return the GPU memory cached by the library's
  allocator to the driver (see [GPU memory](#gpu-memory); no-op without a GPU
  backend).

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
