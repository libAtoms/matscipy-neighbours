# matscipy-neighbours

Fast neighbour lists for particle simulations, with a Python-free C++ core, an
optional CUDA/HIP GPU backend, and zero-copy NumPy/CuPy interop via DLPack.

- **Simple API**: `neighbour_list("ijdDS", …)` returns one array per requested
  quantity (indices `i`, `j`, distance `d`, distance vector `D`, cell shift
  `S`), sorted by `i`.
- **Parallel CPU core** (OpenMP) built from a sorted cell list with a hashed
  compact backend for sparse/vacuum systems.
- **GPU backend** (single-source CUDA/HIP) that keeps results on the device and
  hands them to CuPy/PyTorch/JAX zero-copy through DLPack.
- **General geometry**: triclinic cells, per-direction periodicity, and scalar,
  per-atom, or per-type cutoffs.

This interface is compatible with
[`matscipy.neighbours`](https://github.com/libAtoms/matscipy).

## Installation

```bash
pip install matscipy-neighbours
```

> **Note:** The binary wheels on PyPI are **CPU-only** and have **no GPU
> support**. For the CUDA or HIP backend, build from source against your GPU
> toolkit:
>
> ```bash
> # NVIDIA (needs nvcc on PATH; set the compute capability of your GPU)
> pip install --no-binary matscipy-neighbours matscipy-neighbours \
>     -C cmake.define.ENABLE_CUDA=ON -C cmake.define.CMAKE_CUDA_ARCHITECTURES=80
> # AMD (needs hipcc/ROCm; set the architecture of your GPU)
> pip install --no-binary matscipy-neighbours matscipy-neighbours \
>     -C cmake.define.ENABLE_HIP=ON -C cmake.define.CMAKE_HIP_ARCHITECTURES=gfx90a
> ```
>
> Check with `python -c "import matscipy_neighbours._matscipy_neighbours as m; print(m._has_gpu)"`
> (prints `1` for a GPU build). See the
> [installation docs](https://libatoms.github.io/matscipy-neighbours/installation/)
> for details.

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

## Build from a checkout (C++ core and tests)

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
