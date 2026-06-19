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
array. The shift satisfies `D == r[j] - r[i] + S @ cell`, and pairs are sorted
by `i`.

**Configuration.** Pass either an ASE `Atoms` object as `atoms`, or explicit
`positions` (plus `cell`, `pbc`, `numbers`, `cell_origin`). If `cell` is omitted
for host input, a shrink-wrapped box around the atoms is used.

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

## Other functions

- `first_neighbours(n, i)` — row-start (CSR) offsets for an `i`-sorted list.
- `triplet_list(first_neighbours, abs_dr_p=None, cutoff=None)` — triplets from a
  first-neighbour array.
- `get_jump_indicies(sorted_array)` — jump indices of an ordered array.
- `mic(dr, cell, pbc=None)` — minimum-image-convention wrap of distance vectors.

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
