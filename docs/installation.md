# Installation

The project builds with CMake (≥ 3.18). It produces a static C++ core
(`neighbours`) and a Python extension (`_matscipy_neighbours`).

## Requirements

- A C++17 compiler.
- CMake ≥ 3.18.
- Python with development headers and NumPy (for the extension).
- Optional: OpenMP (CPU parallelism), a CUDA or HIP toolkit (GPU backend),
  CuPy (GPU use from Python), pytest/ASE/JAX (tests).

## CPU build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

OpenMP is detected automatically; without it the core builds single-threaded.

To use the package from the build tree, put the built extension and the
pure-Python wrapper on `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/build:$PWD/language_bindings/python
python -c "import matscipy_neighbours; print('ok')"
```

## GPU build

Enable exactly one GPU backend. The same kernel sources compile under `nvcc`
(CUDA) or `hipcc` (HIP).

### CUDA (NVIDIA)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --parallel
```

Set `CMAKE_CUDA_ARCHITECTURES` to your device's compute capability (e.g. `52`,
`70`, `80`). `CuPy` matching the CUDA version is needed to drive the GPU path
from Python.

### HIP (AMD)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build --parallel
```

!!! note
    A build targets a single backend (CPU-only, CUDA, or HIP). The CPU path is
    always present; the GPU backend is additive and opt-in. Whether the loaded
    extension has GPU support is reported by `_matscipy_neighbours._has_gpu`.

## Running the tests

```bash
ctest --test-dir build --output-on-failure     # C++ (GoogleTest) + Python (pytest)
```

GPU tests skip automatically when no device is present; the JAX interop test
skips when JAX is not installed.
