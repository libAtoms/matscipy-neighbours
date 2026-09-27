# Changelog

All notable changes to matscipy-neighbours are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Examples and benchmark

- The Lennard-Jones Langevin examples (NumPy/CuPy, JAX, Warp, C++ CPU and
  CUDA) gain a second system, `--system liquid`: a bulk liquid at a chosen
  reduced density in a fully periodic box, next to the existing droplet in
  vacuum. The fused Warp and C++ kernels consume the cell-shift array `S` of
  the list for the periodic case; the array and JAX kernels are unchanged
  because the shifts are folded into the distance vectors.
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

[Unreleased]: https://github.com/libAtoms/matscipy-neighbours/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/libAtoms/matscipy-neighbours/releases/tag/v1.0.0
