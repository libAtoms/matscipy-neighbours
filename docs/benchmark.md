# Benchmark

Per-step wall time of the [Lennard-Jones Langevin](examples.md) example for two
systems — a **bulk liquid in a periodic box** and a **droplet in vacuum**
(non-periodic) — across system sizes, for the full cross-product of **device**
(CPU / GPU), **neighbour list** and **kernels** (Warp / array (NumPy/CuPy) /
JAX / C++). Lower is better. The neighbour-list backends are:

- **matscipy-neighbours** — this library (`matscipy_neighbours`), CPU + GPU;
- **matscipy 1.2.0** — the classic [`matscipy`](https://pypi.org/project/matscipy/)
  package's `neighbour_list`, the CPU reference this library descends from;
- **vesin** — [`vesin`](https://github.com/luthaf/vesin), CPU + GPU;
- **ALCHEMI** — NVIDIA ALCHEMI's
  [`nvalchemiops`](https://github.com/NVIDIA/nvalchemi-toolkit-ops) cell list
  (`method="cell_list"`), GPU only, through its PyTorch interface (Warp and
  array kernels) and its JAX interface (JAX kernel).

!!! warning "What is being measured"
    Every step rebuilds the neighbour list from scratch, and the list is the
    dominant cost of a step here. That is deliberate: these runs are set up to
    expose the neighbour-list performance as much as possible. In a real
    application one would build the list with a Verlet shell (a skin added to
    the cutoff) and reuse it for many steps until an atom has moved half the
    skin, so the total time spent constructing neighbour lists plays a much
    smaller role in the overall cost of a simulation than it does below.

!!! info "Test machine"
    - **CPU:** AMD EPYC 9654 96-Core Processor (23 usable cores)
    - **GPU:** NVIDIA H200

!!! warning "CPU threading"
    On the CPU the **matscipy-neighbours** list is benchmarked **both
    single-threaded** (`OMP_NUM_THREADS=1`, the `(1t)` curves) **and
    multi-threaded** (all 23 usable cores, the `(mt)` curves). The
    C++ force loop is OpenMP-parallel and follows the same setting; the Warp and
    array kernels and the JAX backend use their own threading. The classic
    **matscipy 1.2.0** and **vesin** CPU lists are single-threaded. The GPU
    curves are unaffected.

!!! note "Pair list or neighbour matrix"
    A neighbour list comes in two layouts. The **pair list** is three flat
    arrays `i`, `j`, `S` with one entry per pair; every backend returns it, and
    the array, Warp and C++ kernels run one thread (or array element) per
    pair. The **neighbour matrix** gives each atom a row of fixed capacity
    (here 96 slots) plus a neighbour count; its shapes are static, so the JAX
    kernel uses it, and it is ALCHEMI's native layout (its pair list is
    compacted from the matrix). The Warp panels therefore show the GPU curves
    of matscipy-neighbours (`neighbour_matrix`) and ALCHEMI twice: on the pair
    list, and on the matrix (`(matrix)`, one thread per atom summing its own
    row).

!!! note "Missing curves"
    vesin and matscipy 1.2.0 only feed the Warp and array kernels, and ALCHEMI
    the Warp, array and JAX kernels: JAX needs a fixed-capacity neighbour
    matrix, and the C++ example uses the in-tree C++ core. matscipy 1.2.0 is
    CPU-only, and ALCHEMI is benchmarked on the GPU only. A curve that ends
    before the largest size either ran **out of GPU memory** at the next size
    (e.g. the JAX dense neighbour matrix, or the per-pair arrays of the array
    kernels) or was stopped because its next run was predicted to take longer
    than 60 s.

Run configuration: reduced LJ units, cutoff 2.5, dt 0.005, friction 1.0,
temperature 0.7; sizes from 100 to 30,000,000 atoms.
CPU runs stop at 1,000,000 atoms; the GPU runs cover the full range.
Up to 40 steps per point (fewer for the largest systems; JAX and
Warp are compiled once during an untimed warm-up; JAX runs without its
default up-front reservation of GPU memory, which would leave too little for
the neighbour list). Both systems start from an
FCC lattice; the droplet uses lattice constant 1.6, the liquid fills its box at
the stated density.

## Periodic: bulk liquid

A homogeneous liquid at reduced number density 0.8442
in a **fully periodic** cubic box: every grid cell is occupied (dense grid), and
pairs across the boundary carry a non-zero cell shift. At temperature 0.7 the
default density is the Verlet (1967) state point (also the LAMMPS LJ
benchmark's), a liquid at positive pressure well away from coexistence. This is the geometry a
bulk molecular-dynamics or structure-analysis workload sees, and it removes the
vacuum that favours cell lists in the droplet case, so it is the fairer
comparison between the list implementations.

![Liquid: time vs. number of atoms](benchmark_liquid.png)

How to read it (on the test machine):

- On the GPU the list build dominates the step for the Warp and C++ kernels,
  which avoid per-pair arrays; the array and JAX kernels add a roughly constant
  factor on top. The **list** choice therefore drives the scaling.
- On the pair list, ALCHEMI is ahead of matscipy-neighbours up to a few million
  atoms (by 20–40%), matscipy-neighbours is ahead at 10⁷ atoms (by about 20%),
  and only matscipy-neighbours reaches the largest size. ALCHEMI's native
  **matrix** format is the fastest GPU list here at every size from 10⁵ atoms
  on; matscipy-neighbours' `neighbour_matrix` is slower than its own pair
  list, because it also writes the distance vectors (and int64 indices).
- In the **JAX** panel ALCHEMI's jit-compiled cell list is faster than
  matscipy-neighbours' `neighbour_matrix` but runs out of GPU memory above
  3×10⁶ atoms, where matscipy-neighbours reaches 10⁷.
- vesin's GPU path is more than an order of magnitude slower than both at 10⁶
  atoms and grows super-linearly.
- On the CPU, single-threaded matscipy-neighbours `(1t)` is faster than both
  the classic matscipy 1.2.0 and the vesin CPU lists at every size measured.
- Per atom the liquid has more neighbours within the cutoff than the droplet,
  whose surface atoms have fewer, so absolute times are somewhat higher.
- The periodic shifts are folded into the distance vectors by the list (array
  and JAX kernels) or applied from the shift array in the fused kernels (Warp
  and C++), so no kernel wraps positions itself; atoms are free to drift out
  of the box.

## Non-periodic: droplet in vacuum

A self-bound liquid droplet in a **non-periodic**, generously padded box: most
of the cell grid is empty, so this exercises the sparse (hashed) grid and a
list without periodic images.

![Droplet: time vs. number of atoms](benchmark_droplet.png)

How to read it (on the test machine):

- matscipy-neighbours' list scales linearly on the GPU up to the largest size
  and, on the pair list, is the fastest GPU list from about 10⁶ atoms on.
- ALCHEMI is fast up to a few million atoms, fastest of all in its native
  matrix format, but its list build grows super-linearly beyond that on this
  sparse geometry: from 3×10⁶ to 10⁷ atoms its build time rises about twentyfold
  for 3.3× the atoms, and its curves end well above matscipy-neighbours'.
- vesin's GPU path grows super-linearly and falls far behind for these large,
  low-density droplets.
- On the CPU, single-threaded matscipy-neighbours `(1t)` is faster than the
  classic matscipy 1.2.0 and vesin CPU lists from about 10⁵ atoms on; the
  multi-threaded `(mt)` curves pull away as the system grows (at small sizes
  the thread start-up cost dominates).

## Kernel comparison: the cost of the Lennard-Jones implementation

The figure below keeps the neighbour list fixed — **matscipy-neighbours** on
the GPU, and on the CPU **single-threaded** — and varies only the kernels that
consume it: array (NumPy/CuPy), JAX, Warp and C++/CUDA. Within one device the
array, Warp and C++ curves share the same pair-list build, so the vertical
spread between them is the cost of the potential, not of the list; JAX
consumes `neighbour_matrix` instead, whose build is slower on the GPU (see the
matrix curves in the Warp panels). This section therefore does
**not** primarily benchmark the neighbour list but the implementation of the
Lennard-Jones potential on top of it:

- the **array** path materialises the per-pair distance vectors returned by the
  list and scatters the forces with a `bincount` per component;
- **JAX** `jit`-compiles a dense masked sum over the fixed-capacity
  `neighbour_matrix` (no scatter, but padded rows);
- **Warp** and **C++/CUDA** run one fused pass over the `ij` pairs (plus the
  shift `S` in the periodic box) that recomputes each distance and never
  materialises per-pair arrays; on the GPU the potential energy is reduced
  within each thread block and added with one atomic per block.

![Kernel comparison](benchmark_kernels.png)

The C++ force loop is OpenMP-parallel and honours the single-thread setting;
the NumPy, Warp-CPU and JAX-CPU kernels use their own threading and are not
pinned to one core, so on the CPU the comparison is indicative rather than
strict.

This page is generated by `examples/lj_langevin/benchmark.py`. Regenerate it on
your own hardware with:

```bash
python examples/lj_langevin/benchmark.py --build build --doc-out docs/benchmark.md
```

The raw timings are written to `--results-out` (JSON); pass that file to
`--replot` to redraw the plots and this page without re-running the benchmark
(the kernel-comparison figure is drawn from the same results). `--systems
liquid` or `--systems droplet` restricts the run to one system. For
the C++ curves, build with `-DBUILD_EXAMPLES=ON` (and `-DENABLE_CUDA=ON` for
the GPU binary); the others need `pip install jax warp-lang vesin muTimer
matscipy==1.2.0 matplotlib nvalchemi-toolkit-ops` and a CUDA build of PyTorch
(for ALCHEMI) in the interpreter that runs this driver.
