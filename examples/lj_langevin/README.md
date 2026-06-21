# Lennard-Jones Langevin droplet

A self-bound Lennard-Jones liquid droplet (non-periodic) thermostatted with a
Langevin integrator, in reduced LJ units (ε = σ = m = kB = 1). The integrator is
the Allen–Tildesley scheme. Every implementation writes an XYZ trajectory.

Three implementations make the same physics but illustrate different points. The
library itself stays neighbour-list-only — all Lennard-Jones code lives in the
examples.

- **`lj_langevin.py`** — *prototyping (NumPy/CuPy)*. The neighbour list returns
  the distance vectors `D`, so the LJ force is a few array operations; per-atom
  forces are accumulated with a `bincount`-per-component scatter. The same code
  runs on the CPU (NumPy) or the GPU (CuPy) via `--device`.
- **`lj_langevin_jax.py`** — *JAX*. Uses the dense fixed-capacity
  `neighbour_matrix` (`array_namespace=jax.numpy`), so the shapes are static and
  the per-step force + Langevin update `jit`-compile once; forces are a masked
  sum over the neighbour axis (no scatter). The list is exchanged zero-copy
  through DLPack.
- **`lj_langevin_cpu.cc` / `lj_langevin_gpu.cu`** — *performance*. The neighbour
  list supplies only the `ij` connectivity; a single fused pass recomputes the
  distances and accumulates the LJ force, never materialising per-pair arrays.
- **`lj_langevin_warp.py`** — *interop*. The LJ force/energy and the Langevin
  integrator are [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels, but the
  neighbour list is built by this library — or, with `--neighbours vesin`, by
  [vesin](https://github.com/luthaf/vesin), feeding the *same* kernels.
  Positions live in one device buffer shared zero-copy with Warp through DLPack.
  Per-phase timing (build list / force / integrate) is reported with
  [muTimer](https://pypi.org/project/muTimer/), with the neighbour-list build
  listed separately. Needs `pip install warp-lang muTimer vesin`.

## Running

Python (needs the built extension on `PYTHONPATH`):

```bash
export PYTHONPATH=$PWD/build:$PWD/language_bindings/python
python examples/lj_langevin/lj_langevin.py     --device cpu --steps 2000 --out traj.xyz
python examples/lj_langevin/lj_langevin.py     --device gpu --steps 2000 --out traj.xyz
python examples/lj_langevin/lj_langevin_jax.py --device cpu --steps 2000 --out traj.xyz

# Warp kernels + this library's neighbour list (or vesin)
python examples/lj_langevin/lj_langevin_warp.py --device gpu --neighbours matscipy --atoms 2000
python examples/lj_langevin/lj_langevin_warp.py --device gpu --neighbours vesin    --atoms 2000
```

C++ (built when `BUILD_EXAMPLES=ON`; the GPU binary needs `ENABLE_CUDA=ON`):

```bash
./build/examples/lj_langevin/lj_langevin_cpu --steps 2000 --out traj.xyz
./build/examples/lj_langevin/lj_langevin_gpu --steps 2000 --out traj.xyz   # CUDA
```

Common flags: `--ncells`, `--lattice`, `--dt`, `--gamma`, `--kT`, `--cutoff`,
`--steps`, `--write-every`, `--out`.

## Scaling benchmark

`benchmark.py` sweeps logarithmically spaced droplet sizes (100, 1000, 10000, …
up to GPU memory) over the full cross-product of **device** (CPU/GPU),
**neighbour list** and **kernels** (Warp / array / JAX / C++), reports a combined
`ms/step` table, and writes a time-vs-atoms plot faceted by kernel:

```bash
python examples/lj_langevin/benchmark.py --build build --doc-out docs/benchmark.md
```

The three neighbour-list backends are **matscipy-neighbours** (this library,
CPU+GPU), **matscipy 1.2.0** (the classic `matscipy` package, the CPU reference
this library descends from), and **vesin** (CPU+GPU). On the CPU the
matscipy-neighbours list is run **single-threaded** (`OMP_NUM_THREADS=1`, the
`(1t)` rows) and **multi-threaded** (all cores, the `(mt)` rows); matscipy 1.2.0
and vesin are single-threaded. vesin and matscipy 1.2.0 only feed the Warp and
array kernels (JAX uses the dense `neighbour_matrix`, C++ uses the in-tree core),
and matscipy 1.2.0 is CPU-only, so those cells are empty. See the
[Benchmark](../../docs/benchmark.md) page for the generated table, plot and
discussion.

The benchmark rows need `pip install jax warp-lang vesin muTimer matscipy==1.2.0`.

## JAX and the dense neighbour list

JAX compiles for static array shapes, so the *variable-size* pair list is a poor
fit — it makes XLA recompile every step (a naive eager JAX loop is ~1.3 s/step,
compile-bound, on both CPU and GPU). The fix is the dense fixed-capacity
`neighbour_matrix`: `n x K` rows give static shapes, so the per-step force +
Langevin update `jit`-compile once. That is what makes the JAX rows above fast.
Choose `--max-neighbours` large enough for the densest atom; if it is too small
the call raises and you retry with a larger value. JAX defaults to float32, so
the example enables `jax_enable_x64` (the neighbour list works in float64).
