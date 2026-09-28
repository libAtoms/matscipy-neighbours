# Examples

## Lennard-Jones Langevin dynamics

`examples/lj_langevin/` runs Lennard-Jones dynamics with a Langevin thermostat
(the Allen–Tildesley integrator: drift and half kick with the old forces,
neighbour list and forces at the new positions, second half kick; reduced
units), writing an XYZ trajectory. Every implementation offers two systems,
selected with `--system`:

- **`droplet`** (default) — a self-bound liquid droplet in vacuum: a
  non-periodic, generously padded box in which most grid cells are empty. This
  exercises the sparse (hashed) cell grid.
- **`liquid`** — a homogeneous bulk liquid at reduced number density
  `--density` (default 0.8442) in a fully periodic cubic box, started from an
  FCC lattice that fills the box. With the default temperature 0.7 this is the
  Verlet (1967) state point that the LAMMPS LJ benchmark also uses: a liquid
  at positive pressure, well inside the liquid region, so a constant-volume
  run does not cavitate. This exercises the dense cell grid and periodic
  images: pairs across the boundary carry a non-zero cell shift `S`, and the
  neighbour list returns their distance vector as `D = r[j] - r[i] + S @ cell`.

The examples come in several implementations that share the physics but make
different points — and demonstrate that the **library stays
neighbour-list-only**: all Lennard-Jones code lives in the examples.

- **Python NumPy/CuPy (`lj_langevin.py`)** — *prototyping*. The neighbour list
  returns the distance vectors `D`, so the LJ force is a few array operations and
  the per-atom forces are scattered with a `bincount`-per-component accumulation.
  The same code runs on CPU (NumPy) or GPU (CuPy) via `--device`, and the same
  code handles both systems, because the periodic shifts are already folded
  into `D`. The virial, and from it the pressure of the periodic liquid, is one
  more reduction over the same pair arrays; the example reports the
  temperature and pressure in the trajectory and their averages over the
  second half of the run. Started from the lattice, the liquid needs about
  3000 steps to melt and equilibrate; at the default state point it then sits
  at a pressure of about 0.8.
- **Python JAX (`lj_langevin_jax.py`)** — uses the dense fixed-capacity
  `neighbour_matrix` (`array_namespace=jax.numpy`) so shapes are static and the
  per-step force + Langevin update `jit`-compile once; forces are a masked sum
  over the neighbour axis (no scatter). See the note below.
- **C++ (`lj_langevin_cpu.cc` / `lj_langevin_gpu.cu`)** — *performance*. The
  neighbour list supplies only the `ij` connectivity (plus the shift `S` for the
  periodic liquid); a single fused pass recomputes distances and accumulates
  the LJ force, never materialising per-pair arrays.
- **Python Warp (`lj_langevin_warp.py`)** — *interop*. The LJ force/energy and
  the Langevin integrator are [NVIDIA Warp](https://github.com/NVIDIA/warp)
  kernels (compiled once, launched every step), but the neighbour list is built
  by this library — or, for comparison, by
  [`vesin`](https://github.com/luthaf/vesin), the classic `matscipy` 1.2.0
  package, or NVIDIA ALCHEMI's
  [`nvalchemiops`](https://github.com/NVIDIA/nvalchemi-toolkit-ops) (GPU only)
  (`--neighbours {matscipy,matscipy-classic,vesin,alchemi}`). With
  `--format matrix` the kernels consume the fixed-capacity neighbour matrix
  instead of the pair list (one thread per atom summing its own row; matscipy
  and ALCHEMI only).
  Positions live in one device buffer that Warp wraps zero-copy through DLPack
  and the list builder reads directly; the fused kernel recomputes each pair's
  distance (applying the shift `S` in the periodic case) and atomically
  accumulates the force on `i`. Timing is broken down per phase with
  [`muTimer`](https://pypi.org/project/muTimer/) (build list / LJ force /
  integrate), with the neighbour-list build reported separately.

The array and Warp examples also take
`--neighbours {matscipy,matscipy-classic,vesin,alchemi}`, and the JAX example
`--neighbours {matscipy,alchemi}`, so the [Benchmark](benchmark.md) compares
this library's list against the classic `matscipy` 1.2.0 package,
[`vesin`](https://github.com/luthaf/vesin) and NVIDIA ALCHEMI feeding the same
kernels. Every implementation prints the initial potential energy
`E_pot`, and all of them — for both systems and every list backend — agree to
printed precision on the same initial configuration
(`tests/test_examples.py` checks the array and C++ paths against a brute-force
reference).

### Running

```bash
# Python (extension on PYTHONPATH)
python examples/lj_langevin/lj_langevin.py     --device cpu --steps 2000 --out traj.xyz
python examples/lj_langevin/lj_langevin.py     --device gpu --steps 2000 --out traj.xyz
python examples/lj_langevin/lj_langevin.py     --system liquid --atoms 4000 --steps 2000 --out liquid.xyz
python examples/lj_langevin/lj_langevin_jax.py --device cpu --steps 2000 --out traj.xyz

# Warp kernels + this library's neighbour list (or --neighbours vesin / alchemi)
python examples/lj_langevin/lj_langevin_warp.py --device gpu --neighbours matscipy --atoms 2000
python examples/lj_langevin/lj_langevin_warp.py --device gpu --neighbours alchemi --format matrix --atoms 2000
python examples/lj_langevin/lj_langevin_warp.py --device gpu --system liquid --atoms 2000

# C++ (BUILD_EXAMPLES=ON; the GPU binary needs ENABLE_CUDA=ON)
./build/examples/lj_langevin/lj_langevin_cpu --steps 2000 --out traj.xyz
./build/examples/lj_langevin/lj_langevin_gpu --system liquid --atoms 4000 --steps 2000 --out liquid.xyz
```

Common flags: `--system`, `--atoms` (exact atom count), `--ncells` (used when
`--atoms` is 0: droplet radius, or liquid box edge in FCC cells), `--lattice`
(droplet), `--density` (liquid), `--dt`, `--gamma`, `--kT`, `--cutoff`,
`--steps`, `--write-every`, `--out`.

The Warp example additionally needs `warp-lang`, `muTimer`, and (for the vesin
comparison) `vesin`: `pip install warp-lang muTimer vesin`; `--neighbours
alchemi` needs `nvalchemi-toolkit-ops` and a CUDA build of PyTorch. On WSL, put
`libcuda.so` on `LD_LIBRARY_PATH` (e.g. `/usr/lib/wsl/lib`) for the vesin GPU
path.

### Scaling benchmark

The [Benchmark](benchmark.md) page sweeps logarithmically spaced system sizes
(100, 1000, … up to the GPU memory) for **both systems** over the full
cross-product of **device** (CPU/GPU), **neighbour list** and **kernels**
(array / JAX / Warp / C++), with one time-vs-atoms plot per system faceted by
kernel (periodic liquid first, then the droplet), plus a third figure that
holds the list fixed and compares only the four kernels — a benchmark of the
Lennard-Jones implementations rather than of the neighbour list. The neighbour-list backends are **matscipy-neighbours** (this library),
**matscipy 1.2.0** (the classic `matscipy` package, the CPU reference this
library descends from), **vesin** and NVIDIA **ALCHEMI** (GPU only). It is
generated by
`examples/lj_langevin/benchmark.py` (re-run it to refresh the numbers for your
own hardware; `--systems droplet` or `--systems liquid` restricts it to one
system). The broad picture:

- The **neighbour-list build dominates** the step, so the list choice drives the
  scaling: matscipy-neighbours' cell list stays close to linear on both devices,
  matscipy 1.2.0 is a single-threaded CPU reference, while vesin's GPU path falls
  behind for the large, low-density droplets.
- On the GPU, NVIDIA **ALCHEMI** is the closest competitor. On the periodic
  liquid its neighbour matrix is 10–25% faster than matscipy-neighbours' from
  10⁶ to 3×10⁶ atoms; below and above that range matscipy-neighbours is
  ahead, and only its pair list reaches 3×10⁷ atoms. On the sparse droplet
  ALCHEMI's list build grows super-linearly above 3×10⁶ atoms, and
  matscipy-neighbours' pair list and matrix are faster than ALCHEMI's at every
  size.
- The **kernel** choice mostly shifts the curve — the fused C++/CUDA and Warp
  kernels avoid materialising per-pair arrays; the array (NumPy/CuPy) path is the
  simplest; JAX `jit`-compiles a dense masked sum.
- On the CPU the matscipy list is benchmarked **single-threaded** (`(1t)`) and
  **multi-threaded** (`(mt)`); the gap widens with size, and multithreading
  actually *hurts* for tiny systems (thread-spawn overhead).
- The **periodic liquid** has no vacuum, so it is the fairer comparison between
  the list implementations; the droplet shows what the sparse grid buys.

vesin and matscipy 1.2.0 only feed the Warp and array kernels, and ALCHEMI the
Warp, array and JAX kernels (JAX needs a fixed-capacity neighbour matrix, C++
uses the in-tree core); matscipy 1.2.0 is CPU-only and ALCHEMI is run on the
GPU only. Those cells are left empty.

### JAX and the dense neighbour list

JAX compiles for static array shapes, so the *variable-size* pair list is a poor
fit: it makes XLA recompile every step (a naive eager loop is ~1.3 s/step,
compile-bound, on both CPU and GPU). The dense fixed-capacity `neighbour_matrix`
(`n x K` rows) gives static shapes so the per-step update `jit`-compiles once —
which is what makes the JAX rows above fast. Pick `--max-neighbours` large enough
for the densest atom (the call raises if it is too small). JAX defaults to
float32; the example enables `jax_enable_x64` (the list works in float64).
