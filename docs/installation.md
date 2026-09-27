# Installation

## From PyPI

```bash
pip install matscipy-neighbours
```

Binary wheels are provided for Linux (x86_64, aarch64), macOS (arm64) and
Windows (x86_64) for CPython 3.10–3.14. On other platforms pip builds from the
source distribution, which needs a C++17 compiler.

!!! warning "The wheels have no GPU support"
    The wheels on PyPI are built for the CPU only. To use the CUDA or HIP
    backend, install from source as described below.

## From PyPI with GPU support

Force pip to build from the source distribution and pass the GPU backend
options to CMake with `-C cmake.define.<OPTION>=<value>`. Only one backend can
be enabled at a time.

**NVIDIA (CUDA).** Needs the CUDA toolkit (`nvcc` on `PATH`, or `CUDACXX`
pointing at it). Set `CMAKE_CUDA_ARCHITECTURES` to the compute capability of
your GPU (e.g. `70` for V100, `80` for A100, `90` for H100):

```bash
pip install --no-binary matscipy-neighbours matscipy-neighbours \
    -C cmake.define.ENABLE_CUDA=ON \
    -C cmake.define.CMAKE_CUDA_ARCHITECTURES=80
pip install cupy-cuda12x   # CuPy matching your CUDA version
```

**AMD (HIP).** Needs ROCm (`hipcc`). Set `CMAKE_HIP_ARCHITECTURES` to your GPU
(e.g. `gfx90a` for MI200, `gfx942` for MI300):

```bash
pip install --no-binary matscipy-neighbours matscipy-neighbours \
    -C cmake.define.ENABLE_HIP=ON \
    -C cmake.define.CMAKE_HIP_ARCHITECTURES=gfx90a
```

The same options work for a checkout (`pip install . -C ...`) or for the
latest development version
(`pip install "git+https://github.com/libAtoms/matscipy-neighbours" -C ...`).
If a CPU-only wheel is already installed, add `--force-reinstall`, otherwise
pip keeps the existing installation. Check that the GPU backend was built:

```bash
python -c "import matscipy_neighbours._matscipy_neighbours as m; print(m._has_gpu)"
```

This prints `1` for a GPU build and `0` for a CPU-only build.

## Building with CMake

The project builds with CMake (≥ 3.18). It produces a static C++ core
(`neighbours`) and a Python extension (`_matscipy_neighbours`).

### Requirements

- A C++17 compiler.
- CMake ≥ 3.18.
- Python with development headers and NumPy (for the extension).
- Optional: OpenMP (CPU parallelism), a CUDA or HIP toolkit (GPU backend),
  CuPy (GPU use from Python), pytest/ASE/JAX (tests).

### CPU build

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

### GPU build

Enable exactly one GPU backend. The same kernel sources compile under `nvcc`
(CUDA) or `hipcc` (HIP).

#### CUDA (NVIDIA)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --parallel
```

Set `CMAKE_CUDA_ARCHITECTURES` to your device's compute capability (e.g. `52`,
`70`, `80`). `CuPy` matching the CUDA version is needed to drive the GPU path
from Python.

#### HIP (AMD)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build --parallel
```

!!! note
    A build targets a single backend (CPU-only, CUDA, or HIP). The CPU path is
    always present; the GPU backend is additive and opt-in. Whether the loaded
    extension has GPU support is reported by `_matscipy_neighbours._has_gpu`.

### Running the tests

```bash
ctest --test-dir build --output-on-failure     # C++ (GoogleTest) + Python (pytest)
```

GPU tests skip automatically when no device is present; the JAX interop test
skips when JAX is not installed.
