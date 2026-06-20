# Examples

## Lennard-Jones Langevin droplet

`examples/lj_langevin/` simulates a self-bound Lennard-Jones liquid droplet
(non-periodic) with a Langevin thermostat (Allen–Tildesley integrator, reduced
units), writing an XYZ trajectory. It comes in three implementations that share
the physics but make different points — and demonstrate that the **library stays
neighbour-list-only**: all Lennard-Jones code lives in the examples.

- **Python NumPy/CuPy (`lj_langevin.py`)** — *prototyping*. The neighbour list
  returns the distance vectors `D`, so the LJ force is a few array operations and
  the per-atom forces are scattered with a `bincount`-per-component accumulation.
  The same code runs on CPU (NumPy) or GPU (CuPy) via `--device`.
- **Python JAX (`lj_langevin_jax.py`)** — uses the dense fixed-capacity
  `neighbour_matrix` (`array_namespace=jax.numpy`) so shapes are static and the
  per-step force + Langevin update `jit`-compile once; forces are a masked sum
  over the neighbour axis (no scatter). See the note below.
- **C++ (`lj_langevin_cpu.cc` / `lj_langevin_gpu.cu`)** — *performance*. The
  neighbour list supplies only the `ij` connectivity; a single fused pass
  recomputes distances and accumulates the LJ force, never materialising per-pair
  arrays.

### Running

```bash
# Python (extension on PYTHONPATH)
python examples/lj_langevin/lj_langevin.py     --device cpu --steps 2000 --out traj.xyz
python examples/lj_langevin/lj_langevin.py     --device gpu --steps 2000 --out traj.xyz
python examples/lj_langevin/lj_langevin_jax.py --device cpu --steps 2000 --out traj.xyz

# C++ (BUILD_EXAMPLES=ON; the GPU binary needs ENABLE_CUDA=ON)
./build/examples/lj_langevin/lj_langevin_cpu --steps 2000 --out traj.xyz
./build/examples/lj_langevin/lj_langevin_gpu --steps 2000 --out traj.xyz
```

`benchmark.py` runs all backends across droplet sizes.

### Scaling benchmark

Per-step wall time (**ms/step**) on a 48-core CPU and a GTX TITAN X:

| Implementation | 456 atoms | 1088 atoms | 2112 atoms | 3604 atoms | 8628 atoms |
|----------------|----------:|-----------:|-----------:|-----------:|-----------:|
| NumPy (CPU)    | 8.9       | 16.8       | 23.7       | 36.7       | 88.7       |
| CuPy (GPU)     | 4.5       | 5.6        | 10.8       | 12.8       | 22.9       |
| JAX (CPU)      | 12.5      | 22.4       | 17.9       | 26.1       | 41.1       |
| JAX (GPU)      | 8.7       | 9.1        | 8.9        | 11.3       | 11.9       |
| C++ (CPU, 48t) | 12.7      | 16.8       | 13.6       | 15.1       | 21.3       |
| C++ + CUDA     | 1.4       | 1.7        | 2.5        | 2.8        | 5.5        |

- **NumPy** is compute-bound and scales ~linearly with atom count.
- **CuPy** is launch-overhead-bound for small droplets and pulls ahead as the
  system grows (~3.9× over NumPy at 8628 atoms).
- **JAX** (dense `neighbour_matrix` + `jit`) is competitive: JAX-CPU tracks NumPy,
  and **JAX-GPU is nearly flat (~9–12 ms/step) and beats CuPy at the largest
  size** (its `jit`-fused masked sum avoids CuPy's per-step kernel launches).
- **C++ CPU** is *slower than NumPy on tiny droplets* (48-thread overhead) and
  only wins in the mid-range.
- **C++ + CUDA** is fastest everywhere (fused kernel, no per-pair materialisation)
  — ~2× ahead of JAX-GPU and ~16× ahead of NumPy at the largest size.

### JAX and the dense neighbour list

JAX compiles for static array shapes, so the *variable-size* pair list is a poor
fit: it makes XLA recompile every step (a naive eager loop is ~1.3 s/step,
compile-bound, on both CPU and GPU). The dense fixed-capacity `neighbour_matrix`
(`n x K` rows) gives static shapes so the per-step update `jit`-compiles once —
which is what makes the JAX rows above fast. Pick `--max-neighbours` large enough
for the densest atom (the call raises if it is too small). JAX defaults to
float32; the example enables `jax_enable_x64` (the list works in float64).
