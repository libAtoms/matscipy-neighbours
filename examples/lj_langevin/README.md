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

`benchmark.py` runs every backend over a range of droplet sizes and reports the
per-step wall time:

```bash
python examples/lj_langevin/benchmark.py --build build --sizes 3 4 5 6 8 \
       --jax-python /path/to/python-with-jax
```

The CPU rows are **multi-threaded** (the matscipy neighbour list is
OpenMP-parallel, and the C++/NumPy/JAX backends use all cores); set
`OMP_NUM_THREADS=1` for a single-threaded comparison.

`benchmark_warp.py` is a separate scaling benchmark for the Warp example: it
sweeps logarithmically spaced sizes (100, 1000, 10000, … up to GPU memory),
compares this library's neighbour list against vesin on CPU and GPU, lists the
neighbour-list build time separately, and writes a time-vs-atoms plot:

```bash
python examples/lj_langevin/benchmark_warp.py --doc-out docs/benchmark_warp.md
```

Per-step wall time (**ms/step**) on a 48-core CPU and a GTX TITAN X (Maxwell):

| Implementation | 456 atoms | 1088 atoms | 2112 atoms | 3604 atoms | 8628 atoms |
|----------------|----------:|-----------:|-----------:|-----------:|-----------:|
| NumPy (CPU)    | 8.9       | 16.8       | 23.7       | 36.7       | 88.7       |
| CuPy (GPU)     | 4.5       | 5.6        | 10.8       | 12.8       | 22.9       |
| JAX (CPU)      | 12.5      | 22.4       | 17.9       | 26.1       | 41.1       |
| JAX (GPU)      | 8.7       | 9.1        | 8.9        | 11.3       | 11.9       |
| C++ (CPU, 48t) | 12.7      | 16.8       | 13.6       | 15.1       | 21.3       |
| C++ + CUDA     | 1.4       | 1.7        | 2.5        | 2.8        | 5.5        |

How to read it:

- **NumPy** is compute-bound and scales roughly linearly with atom count.
- **CuPy** is launch-overhead-bound for small droplets and pulls ahead as the
  system grows — ~3.9× faster than NumPy at 8628 atoms.
- **JAX** (using the dense fixed-capacity `neighbour_matrix` + `jit`) is
  competitive: JAX-CPU tracks NumPy, and **JAX-GPU is nearly flat (~9–12 ms/step)
  and beats CuPy at the largest size** (its `jit`-fused masked sum avoids CuPy's
  many per-step kernel launches).
- **C++ CPU** is *slower than NumPy on tiny droplets* (48-thread overhead) and
  only wins in the mid-range.
- **C++ + CUDA** is fastest everywhere — the fused kernel avoids materialising
  per-pair arrays — and stays ~2× ahead of JAX-GPU and ~16× ahead of NumPy at the
  largest size.

The takeaway: prototype in Python (CuPy or JAX both give a real GPU speed-up; on
the GPU they are within a small factor of each other), but a performant run wants
the fused C++/CUDA kernel.

## JAX and the dense neighbour list

JAX compiles for static array shapes, so the *variable-size* pair list is a poor
fit — it makes XLA recompile every step (a naive eager JAX loop is ~1.3 s/step,
compile-bound, on both CPU and GPU). The fix is the dense fixed-capacity
`neighbour_matrix`: `n x K` rows give static shapes, so the per-step force +
Langevin update `jit`-compile once. That is what makes the JAX rows above fast.
Choose `--max-neighbours` large enough for the densest atom; if it is too small
the call raises and you retry with a larger value. JAX defaults to float32, so
the example enables `jax_enable_x64` (the neighbour list works in float64).
