# matscipy-neighbours

Fast neighbour lists for particle simulations. The library has a Python-free
C++ core, an optional single-source CUDA/HIP GPU backend, and zero-copy
NumPy/CuPy interop through DLPack.

## Highlights

- **Compatible API.** `neighbour_list("ijdDS", …)` returns one array per
  requested quantity — first index `i`, second index `j`, distance `d`,
  distance vector `D`, and cell shift `S`. The contract is
  `D == r[j] - r[i] + S @ cell`, and the output is sorted by `i`.
- **Parallel CPU core.** A sorted cell list (counting-sort CSR) with an
  open-addressing hashed backend for sparse/vacuum systems, parallelised with
  OpenMP over a two-pass count → fill that allocates exactly.
- **GPU backend.** The same algorithm compiled for CUDA or HIP. Results can stay
  on the device and are handed to CuPy/PyTorch/JAX zero-copy via DLPack.
- **General geometry.** Triclinic cells, per-direction periodicity, and scalar,
  per-atom, or per-type cutoffs.

## A first example

```python
import numpy as np
from matscipy_neighbours import neighbour_list

positions = np.random.uniform(0, 10, (1000, 3))
cell = np.diag([10.0, 10.0, 10.0])
i, j, D = neighbour_list("ijD", positions=positions, cell=cell, pbc=True,
                         cutoff=2.5)
```

## Where to go next

- [Installation](installation.md) — build the CPU core and the optional GPU
  backends.
- [Python API](python.md) — the high-level `neighbour_list` / `coordination`
  interface, device selection, and DLPack output.
- [C++ API](cpp.md) — the Python-free core, the `CellGrid`, and the GPU entry
  points.
- [Algorithm & references](algorithm.md) — how the cell list works and the
  literature it draws on.
