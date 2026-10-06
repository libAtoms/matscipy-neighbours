# Changelog

All notable changes to matscipy-neighbours are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] - Unreleased

### Core

- GPU buffers come from a per-device caching memory pool (stream-ordered
  allocation with an unlimited release threshold) instead of a fresh
  `cudaMalloc`/`cudaFree` per buffer and call; the device primitives' scratch
  buffers use it too. A GPU pair list of 10⁶ atoms builds in about half the
  time. `empty_gpu_cache()` (C++ and Python) returns the cached memory to the
  driver.
- The GPU `neighbour_matrix` is filled in a single pass of the cell-list
  search, one thread per atom writing its own row, instead of building the
  pair list and scattering it; it no longer clears the unused slots. About
  4.7× faster at 10⁶ atoms, and faster than the pair list.
- `neighbour_matrix` takes `quantities` (Python, default `"D"`) or a trailing
  `quantities` flag argument (C++, default `QUANTITY_DISTVEC`) selecting the
  per-slot extras: distance vectors `D`, cell shifts `S`, both or neither.
  Indices stay int64.
- **Behaviour change:** the unused slots of a neighbour matrix (beyond `count`)
  are unspecified instead of 0. Consumers must select with the mask rather
  than multiply by it; the JAX example and the documented pattern do.
- Running out of memory no longer aborts the process: allocation failures
  (GPU, and host `malloc`) throw `std::bad_alloc`, which the public entry
  points return as the new `NL_OUT_OF_MEMORY` code and Python raises as
  `MemoryError`. On the GPU the cache is emptied and the allocation retried
  once first.
- On HIP the GPU buffers come from a per-device cache of `hipMalloc` blocks
  (free lists by size, requests rounded up to one of eight sizes per power of
  two) instead of the stream-ordered pool: ROCm 6.4's pools did not reliably
  reuse freed blocks, so every call mapped its outputs afresh. A GPU pair list
  of 10⁶ atoms on an MI300A builds in about 20 ms instead of 160 ms. CUDA is
  unchanged.
- `tests/test_neighbour_list_gpu.cc` builds and runs with the HIP backend, and
  the device tests of `tests/test_memory_space.cc` run on HIP too, plus a test
  that the HIP block cache reuses freed blocks.
- `first_neighbours` accepts a device index array (CuPy, JAX, PyTorch via
  DLPack) and computes the row-start array on its GPU, returning it there in
  the input's framework (`array_namespace` overrides); host input is
  unchanged. The C++ core gains `first_neighbours_gpu_device`.

### Examples and benchmark

- The Lennard-Jones Langevin examples (NumPy/CuPy, JAX, Warp, C++ CPU and
  CUDA) gain a second system, `--system liquid`: a bulk liquid at a chosen
  reduced density in a fully periodic box, next to the existing droplet in
  vacuum. The fused Warp and C++ kernels consume the cell-shift array `S` of
  the list for the periodic case; the array and JAX kernels are unchanged
  because the shifts are folded into the distance vectors.
- The array example sums the pair forces per atom with a segment sum over the
  `i`-sorted pairs instead of a weighted `bincount` per component, whose
  float64 atomic scatter is very slow on ROCm (about 440 ms against 1.5 ms at
  10⁶ atoms on an MI300A): `first_neighbours` gives the segment starts, and a
  small CuPy kernel (one thread per atom) sums each segment on the GPU,
  `add.reduceat` on the CPU. CuPy's `add.reduceat` was not used on the GPU: it
  computes a cumulative sum of all pairs, a temporary the size of the pair
  forces, and loses accuracy to it. Results files record
  the method (`array_force_sum`); the GPU comparison labels the array curves
  with it.
- The benchmark page gains a section on performance portability between
  NVIDIA and AMD GPUs: caching allocators, float64 atomics, CuPy's backend
  defaults, the MI300A's unified memory, and installing the ROCm stack.
- `benchmark.py` compares GPUs: `--add-machine RESULTS.json` stores a run as
  an additional machine in an existing results file (keyed by `--machine`,
  default the detected GPU) without touching its main results, and
  `--kernels`, `--lists` and `--devices` restrict a run to a subset. A new
  figure (`docs/benchmark_gpus.png`) shows the array and JAX kernels on the
  matscipy-neighbours GPU list for every machine in the file. AMD GPUs are
  detected through `rocm-smi`.
- `benchmark.py` sweeps both systems and writes one figure per system
  (`docs/benchmark_liquid.png`, `docs/benchmark_droplet.png`; panels ordered
  array / JAX / Warp / C++) plus a kernel-comparison figure
  (`docs/benchmark_kernels.png`) on the matscipy-neighbours list alone; the
  generated page leads with a note on Verlet shells versus per-step rebuilds.
  `--systems` restricts the run, and older results files still replot.
- The Langevin integrator in all example implementations applied the whole
  force kick with the old forces; it now follows the Allen-Tildesley scheme
  (half kick with the old forces, list and forces at the new positions, half
  kick with the new forces), which reduces to velocity Verlet at zero
  friction. The old scheme ran the droplet at 0.9 instead of 0.7 and heated
  the periodic liquid until it exploded. The examples now write the kinetic
  temperature to the trajectory.
- The liquid's default density is 0.8442, the Verlet (1967) state point at
  kT 0.7 (positive pressure, no cavitation at constant volume); the
  benchmark driver takes `--density` and records it.
- The NumPy/CuPy example computes the virial from the pair arrays and reports
  the pressure of the periodic liquid.
- The C++ examples print the potential energy, and the Warp example no longer
  halves the energy twice.
- `tests/test_examples.py` checks the array example's forces, energy and
  virial against a brute-force reference for both systems, checks that the
  integrator holds the target temperature with the liquid stable at positive
  pressure, and, when built, compares the C++ example against the array
  example.
- NVIDIA ALCHEMI (`nvalchemiops`) is a fourth neighbour-list backend
  (`--neighbours alchemi`, GPU only): through its PyTorch interface in the
  array and Warp examples, and through its JAX interface (jit-compiled) in the
  JAX example, which gains `--neighbours {matscipy,alchemi}`. The Warp example
  gains `--format matrix`, which consumes the fixed-capacity neighbour matrix
  (matscipy-neighbours' `neighbour_matrix` or ALCHEMI's native output) with one
  thread per atom instead of the pair list. The benchmark adds the ALCHEMI
  curves and, in the Warp panels, the matrix-format GPU curves.
- The JAX benchmark runs no longer let JAX reserve 75% of GPU memory up front;
  the neighbour list allocates outside that pool, so the JAX GPU curves ran
  out of memory at a fraction of the card's capacity.
- The CUDA example printed and logged `E_pot=0`: its `CUDA_CHECK` macro
  declared a local `e` that shadowed the caller's `e` in
  `CUDA_CHECK(cudaMemcpy(&e, ...))`. Forces and timings were unaffected. The
  example test now also checks the GPU binary when it is built, and finds
  binaries of an out-of-tree build through `$MATSCIPY_EXAMPLES_BUILD`.
- The GPU force kernels of the Warp and CUDA examples added every pair's
  energy to one global value with an atomic, which serialised the launch:
  95 of the 110 ms of a Warp step at 10⁶ atoms went to the energy sum. The
  energy is now reduced within each thread block (Warp tiles; warp shuffles
  and shared memory in CUDA) and added with one atomic per block. The CUDA
  kernel also sizes its grid in 64-bit arithmetic.
- The array, JAX and C++ examples wrote a trajectory frame at step 0 even
  with `--write-every` larger than the step count, inside the timed loop, so
  their benchmark timings included one full XYZ frame (with only 5 steps
  per point at large sizes, several times the actual step time).
  `--write-every 0` now disables the trajectory, and the benchmark uses it.
  The JAX warm-up also runs a full discarded iteration, so that no
  compilation falls into the timed loop.
- `benchmark.py` stops a configuration before a run predicted to exceed
  `--max-run-seconds` (default 60 s), caps CPU sizes with `--max-atoms-cpu`,
  runs the multi-threaded CPU curves on the cores the process may use (not
  all cores of the machine), saves the results after every configuration and
  continues an interrupted run with `--resume`.

### Packaging

- The package version is taken from the git tag (setuptools-scm) instead of
  being hard-coded; `matscipy_neighbours.__version__` is available.
- CPU-only binary wheels for Linux (x86_64, aarch64), macOS (arm64) and Windows
  (x86_64), CPython 3.10–3.14, built and published to PyPI on each tag.

## [1.0.0] - 2026-09-26

Initial release. The neighbour-list code of
[matscipy](https://github.com/libAtoms/matscipy) as a standalone package with a
Python-free C++ core, an optional GPU backend and zero-copy array interop.

### Core (C++)

- Cell-list neighbour search for arbitrary triclinic cells with per-direction
  periodicity; atoms may lie outside the cell, and cutoffs may exceed the cell
  size (multiple periodic images of a pair are returned).
- Global, per-atom (pair cutoff = sum of radii) and per-type (matrix) cutoffs.
- Per-pair outputs selectable via flags: first and second index, distance
  vector, absolute distance and cell shift, with the contract
  `D == r[j] - r[i] + S @ cell` and output sorted by the first index.
- Dense (CSR) and hashed compact cell lists, chosen automatically so sparse
  or vacuum-dominated systems do not pay for empty cells; linear or Morton
  (Z-curve) cell ordering.
- Two-pass (count, then fill) construction with exactly sized output buffers,
  parallelised per atom with OpenMP.
- Fixed-capacity neighbour matrix (`n x max_neighbours`) for static-shape
  consumers such as JAX, with an overflow flag.
- Row-start (`first_neighbours`, `get_jump_indicies`) and triplet-list helpers.
- 64-bit indices throughout, so atom, cell and pair counts cannot overflow.
- Input validation with error codes and messages: non-positive or non-finite
  cutoffs, negative radii, types outside the cutoff matrix, non-finite
  positions, degenerate cells and malformed row-start arrays are rejected
  rather than computed on.

### GPU backend (CUDA / HIP)

- Single-source kernels sharing the cell-index, hashing and traversal code with
  the CPU path; binning via atomic histograms, CUB/hipCUB scans and radix sort.
- Pair lists, per-atom neighbour counts (coordination) and the dense neighbour
  matrix computed entirely on the device, with results left in device memory.
- Positions already on the device are consumed in place through DLPack; no
  host round-trip.

### Python package

- `neighbour_list` with the matscipy-compatible signature and quantity string
  (`"ijdDS"`), accepting ASE `Atoms` objects or explicit positions, cell, pbc
  and numbers; cutoffs as a scalar, per-atom array, per-type matrix or
  element-pair dictionary.
- `neighbour_matrix`, `coordination`, `first_neighbours`, `get_jump_indicies`,
  `triplet_list` and `mic` (minimum-image convention, with per-lattice-direction
  periodicity).
- Zero-copy results through DLPack: NumPy by default on the host, CuPy on a
  GPU, or any `from_dlpack` consumer (PyTorch, JAX) via `array_namespace`, or
  the raw `DLPackTensor` objects.
- Missing cells are shrink-wrapped and zero lattice vectors (ASE molecules and
  slabs) are completed for non-periodic directions; periodic directions
  require a cell.
- Errors surface as `TypeError`, `ValueError`, `MemoryError` or
  `RuntimeError`; the extension never aborts the interpreter.
- An ASE v4 neighbour-list plugin (`matscipy-neighbours` backend).

### Build, tests and documentation

- CMake build of the static core, the CPython extension and the C++ tests
  (GoogleTest), with `-DENABLE_CUDA=ON` / `-DENABLE_HIP=ON` for the GPU
  backend; pip-installable through scikit-build-core.
- C++ unit tests, a pytest suite validated against brute-force references
  (including a randomised fuzz over triclinic cells, mixed periodicity, multiple
  images and per-atom radii), input-validation tests, and DLPack/GPU tests that
  skip without a device.
- Documentation site (installation, Python and C++ APIs, algorithm and
  references, benchmarks) and a Lennard-Jones Langevin example in C++, NumPy,
  JAX and Warp.

[1.1.0]: https://github.com/libAtoms/matscipy-neighbours/releases/tag/1.1.0
[1.0.0]: https://github.com/libAtoms/matscipy-neighbours/releases/tag/1.0.0
