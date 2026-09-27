# Benchmark

Per-step wall time of the [Lennard-Jones Langevin](examples.md) example for two
systems — a **bulk liquid in a periodic box** and a **droplet in vacuum**
(non-periodic) — across system sizes, for the full cross-product of **device**
(CPU / GPU), **neighbour list** and **kernels** (Warp / array (NumPy/CuPy) /
JAX / C++). Lower is better. The neighbour-list backends are:

- **matscipy-neighbours** — this library (`matscipy_neighbours`), CPU + GPU;
- **matscipy 1.2.0** — the classic [`matscipy`](https://pypi.org/project/matscipy/)
  package's `neighbour_list`, the CPU reference this library descends from;
- **vesin** — [`vesin`](https://github.com/luthaf/vesin), CPU + GPU.

!!! warning "What is being measured"
    Every step rebuilds the neighbour list from scratch, and the list is the
    dominant cost of a step here. That is deliberate: these runs are set up to
    expose the neighbour-list performance as much as possible. In a real
    application one would build the list with a Verlet shell (a skin added to
    the cutoff) and reuse it for many steps until an atom has moved half the
    skin, so the total time spent constructing neighbour lists plays a much
    smaller role in the overall cost of a simulation than it does below.

!!! info "Test machine"
    - **CPU:** AMD EPYC 9655 96-Core Processor (23 usable cores)
    - **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition

!!! warning "CPU threading"
    On the CPU the **matscipy-neighbours** list is benchmarked **both
    single-threaded** (`OMP_NUM_THREADS=1`, the `(1t)` curves) **and
    multi-threaded** (all 23 usable cores, the `(mt)` curves). The
    C++ force loop is OpenMP-parallel and follows the same setting; the Warp and
    array kernels and the JAX backend use their own threading. The classic
    **matscipy 1.2.0** and **vesin** CPU lists are single-threaded. The GPU
    curves are unaffected.

!!! note "Missing curves"
    vesin and matscipy 1.2.0 only feed the Warp and array kernels: JAX uses the
    dense `neighbour_matrix` and the C++ example uses the in-tree C++ core, so
    those panels show matscipy-neighbours only. matscipy 1.2.0 is CPU-only, so it
    has no GPU curve. A curve that ends before the largest size either ran
    **out of GPU memory** at the next size (e.g. the JAX dense neighbour
    matrix, or the per-pair arrays of the array kernels) or was stopped because
    its next run was predicted to take longer than
    60 s.

Run configuration: reduced LJ units, cutoff 2.5, dt 0.005, friction 1.0,
temperature 0.7; sizes from 100 to 30,000,000 atoms.
CPU runs stop at 1,000,000 atoms; the GPU runs cover the full range.
Up to 40 steps per point (fewer for the largest systems; JAX and
Warp are compiled once during an untimed warm-up). Both systems start from an
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

!!! note "Not yet measured on the test machine"
    The liquid figure (`benchmark_liquid.png`) is produced by the same driver
    run that produces the droplet figure below. Re-run the command at the end
    of this page on the test machine to add it; the driver writes all three
    figures and regenerates this page.

How to read it:

- The same list/kernel combinations as above; compare curve by curve with the
  droplet figure. Per atom the liquid has more neighbours within the cutoff
  than the droplet's surface-heavy clusters, so the absolute times are higher
  and the per-pair cost is the better like-for-like number.
- The periodic shifts are folded into the distance vectors by the list (array
  and JAX kernels) or applied from the shift array in the fused kernels (Warp
  and C++), so no kernel wraps positions itself; atoms are free to drift out
  of the box.

## Non-periodic: droplet in vacuum

A self-bound liquid droplet in a **non-periodic**, generously padded box: most
of the cell grid is empty, so this exercises the sparse (hashed) grid and a
list without periodic images.

![Droplet: time vs. number of atoms](benchmark_droplet.png)

!!! note "Panel order"
    This figure predates the current panel layout (top row array / JAX,
    bottom row Warp / C++) and the integrator fix; it is replaced by the
    next run of the driver on the test machine. The timings are unaffected
    by the integrator, since each point covers a few dozen steps from a
    lattice.

How to read it:

- The neighbour-list build dominates the step, so the **list** choice drives the
  scaling: matscipy-neighbours' cell list stays close to linear on both devices
  (on the GPU up to the largest size), the classic matscipy 1.2.0 list is a
  single-threaded CPU reference, and vesin's GPU path grows super-linearly and
  falls far behind for these large, low-density droplets.
- The **kernel** choice mostly shifts the curve: the fused C++/CUDA and Warp
  kernels avoid materialising per-pair arrays, the array (NumPy/CuPy) path is the
  simplest, and JAX `jit`-compiles a dense masked sum.
- On the CPU, the matscipy-neighbours `(mt)` curves pull away from `(1t)` as the
  system grows (at small sizes the thread start-up cost dominates); and from
  about 10⁵ atoms on, even single-threaded matscipy-neighbours `(1t)` is faster
  than the classic matscipy 1.2.0 and vesin CPU lists.

## Kernel comparison: the cost of the Lennard-Jones implementation

The figure below keeps the neighbour list fixed — **matscipy-neighbours** on
the GPU, and on the CPU **single-threaded** — and varies only the kernels that
consume it: array (NumPy/CuPy), JAX, Warp and C++/CUDA. Within one device the
list build time is the same for all four curves, so the vertical spread between
them is the cost of the potential, not of the list. This section therefore does
**not** primarily benchmark the neighbour list but the implementation of the
Lennard-Jones potential on top of it:

- the **array** path materialises the per-pair distance vectors returned by the
  list and scatters the forces with a `bincount` per component;
- **JAX** `jit`-compiles a dense masked sum over the fixed-capacity
  `neighbour_matrix` (no scatter, but padded rows);
- **Warp** and **C++/CUDA** run one fused pass over the `ij` pairs (plus the
  shift `S` in the periodic box) that recomputes each distance and never
  materialises per-pair arrays.

!!! note "Not yet measured on the test machine"
    The kernel-comparison figure (`benchmark_kernels.png`) is drawn from the
    same results as the two figures above and appears here after the next run
    of the driver on the test machine.

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
matscipy==1.2.0 matplotlib` in the interpreter that runs this driver.
