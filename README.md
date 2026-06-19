# matscipy-neighbours

Fast neighbour lists for particle simulations, with a Python-free C++ core, an
optional CUDA/HIP GPU backend, and zero-copy NumPy/CuPy interop via DLPack.

- **Drop-in API** compatible with `matscipy.neighbours`: request any of the
  quantities `"ijdDS"` and get one array back per character.
- **Parallel CPU core** (OpenMP) built from a sorted cell list with a hashed
  compact backend for sparse/vacuum systems.
- **GPU backend** (single-source CUDA/HIP) that keeps results on the device and
  hands them to CuPy/PyTorch/JAX zero-copy through DLPack.
- **General geometry**: triclinic cells, per-direction periodicity, and scalar,
  per-atom, or per-type cutoffs.

## Quick start (Python)

```python
import numpy as np
from matscipy_neighbours import neighbour_list

positions = np.random.uniform(0, 10, (1000, 3))
cell = np.diag([10.0, 10.0, 10.0])
i, j, D = neighbour_list("ijD", positions=positions, cell=cell, pbc=True, cutoff=2.5)
# D[p] == positions[j[p]] - positions[i[p]] + S @ cell, output sorted by i
```

With a CuPy array in, the GPU backend runs and CuPy arrays come back, with no
host round-trip:

```python
import cupy as cp
i, j, D = neighbour_list("ijD", positions=cp.asarray(positions), cell=cell,
                         pbc=True, cutoff=2.5)
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Enable a GPU backend (one at a time):

```bash
cmake -S . -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80   # NVIDIA
cmake -S . -B build -DENABLE_HIP=ON                                  # AMD
```

## Documentation

Full documentation (installation, Python and C++ APIs, the algorithm and its
references) is at **https://libatoms.github.io/matscipy-neighbours/** and in the
[`docs/`](docs/) folder.

## License

MIT — see [LICENSE.md](LICENSE.md).
