#!/usr/bin/env python3
"""Lennard-Jones Langevin dynamics (droplet or periodic liquid) — JAX.

Uses the dense fixed-capacity neighbour list (`neighbour_matrix`): each atom's
neighbours occupy a row of an ``n x K`` matrix, so the shapes are static and the
per-step force + Langevin update can be `jit`-compiled once (no per-step
recompilation). Forces are a masked sum over the neighbour axis — no scatter. The
neighbour list is exchanged with JAX zero-copy through DLPack
(`array_namespace=jax.numpy`). Select CPU or GPU with ``--device`` (the GPU path
needs a CUDA build of JAX). ``--system liquid`` runs a bulk liquid in a fully
periodic box instead of the droplet in vacuum; the periodic shifts are folded
into the distance vectors of the dense list, so the kernel is unchanged.
``--neighbours alchemi`` builds the same fixed-capacity matrix with NVIDIA
ALCHEMI's JAX cell list (`nvalchemiops.jax`, GPU only) instead, asking it for
the distance vectors (``return_vectors``) so that the kernel is shared.
Reduced LJ units (epsilon = sigma = mass = kB = 1); output is an XYZ trajectory.
"""

import argparse
import functools
import time

import numpy as np


def fcc_droplet(ncells, lattice):
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    span = range(-ncells, ncells + 1)
    r = np.array([(np.array([ix, iy, iz]) + b) * lattice
                  for ix in span for iy in span for iz in span for b in basis])
    r -= r.mean(axis=0)
    radius = ncells * lattice
    return np.ascontiguousarray(r[(r * r).sum(axis=1) <= radius * radius])


def fcc_droplet_n(target_n, lattice):
    """An FCC cluster of *exactly* ``target_n`` atoms (the sites closest to the
    centre), so the benchmark can hit round atom counts for every backend."""
    import math
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    ncells = int(math.ceil((3.0 * target_n / (16.0 * math.pi)) ** (1.0 / 3.0))) + 2
    span = range(-ncells, ncells + 1)
    r = np.array([(np.array([ix, iy, iz]) + b) * lattice
                  for ix in span for iy in span for iz in span for b in basis])
    r -= r.mean(axis=0)
    order = np.argsort((r * r).sum(axis=1))
    return np.ascontiguousarray(r[order[:target_n]], dtype=float)


def fcc_liquid_n(target_n, density):
    """Exactly ``target_n`` atoms on FCC sites filling a periodic cubic box at
    the given number density; returns ``(positions, L)``. Same deterministic
    site selection as the other implementations (see lj_langevin.py)."""
    import math
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    L = (target_n / density) ** (1.0 / 3.0)
    ncells = int(math.ceil((target_n / 4.0) ** (1.0 / 3.0)))
    a = L / ncells
    span = range(ncells)
    r = np.array([(np.array([ix, iy, iz]) + b) * a
                  for ix in span for iy in span for iz in span for b in basis])
    keep = (np.arange(target_n) * r.shape[0]) // target_n
    return np.ascontiguousarray(r[keep], dtype=float), L


def langevin_constants(dt, gamma, kT, mass=1.0):
    D = kT / (mass * gamma)
    c0 = np.exp(-gamma * dt)
    c1 = (1.0 - c0) / (gamma * dt)
    c2 = (1.0 - c1) / (gamma * dt)
    sr = np.sqrt(dt * D * (2.0 - (3.0 - 4.0 * c0 + c0 * c0) / (gamma * dt)))
    sv = np.sqrt(gamma * D * (1.0 - c0 * c0))
    crv = D * (1.0 - c0) ** 2 / (sr * sv)
    return dict(c0=c0, c1=c1, c2=c2, sr=sr, sv=sv, crv=crv, dt=dt, mass=mass)


def make_step(jnp, jrandom, lc):
    """A jit-compiled Langevin step. The dense `(idx, dist, count)` have static
    shapes, so this compiles once and reruns without recompilation.

    The Allen-Tildesley scheme needs the forces at the new positions for the
    second half kick, i.e. a neighbour-list rebuild in the middle of the step.
    To keep one force evaluation per step, the step is shifted by half a kick:
    it receives the list at the *current* positions, first finishes the
    previous step's half kick with these (new) forces, then drifts the
    positions and applies the half kick with the same (now old) forces.
    ``finish`` is 0.0 on the very first step (nothing to finish) and 1.0
    afterwards; passing it as a traced scalar avoids a recompilation."""

    @jax.jit
    def step(positions, velocities, dist, count, key, finish):
        K = dist.shape[1]
        mask = jnp.arange(K)[None, :] < count[:, None]      # valid neighbours
        # Unused slots are not cleared (they may hold anything, NaN included),
        # so select, don't multiply: 0 * NaN would poison the force sum.
        dist = jnp.where(mask[..., None], dist, 0.0)
        r2 = (dist * dist).sum(axis=-1)
        safe = jnp.where(mask, r2, 1.0)                      # avoid 1/0 in pads
        inv_r2 = 1.0 / safe
        inv_r6 = inv_r2 ** 3
        coef = jnp.where(mask, -24.0 * inv_r2 * inv_r6 * (2.0 * inv_r6 - 1.0), 0.0)
        forces = (coef[..., None] * dist).sum(axis=1)        # (n, 3), no scatter
        energy = 0.5 * jnp.where(mask, 4.0 * inv_r6 * (inv_r6 - 1.0), 0.0).sum()
        a = forces / lc["mass"]

        # Second half kick of the previous step (forces at the new positions).
        velocities = velocities + finish * lc["c2"] * lc["dt"] * a
        temperature = (velocities * velocities).sum() / (3.0 * positions.shape[0])

        # Drift plus the first half kick of this step (forces at the old positions).
        k1, k2 = jrandom.split(key)
        g1 = jrandom.normal(k1, positions.shape)
        g2 = jrandom.normal(k2, positions.shape)
        gr = lc["sr"] * g1
        gv = lc["sv"] * (lc["crv"] * g1 + (1.0 - lc["crv"] ** 2) ** 0.5 * g2)
        positions = positions + lc["c1"] * lc["dt"] * velocities + \
            lc["c2"] * lc["dt"] ** 2 * a + gr
        velocities = lc["c0"] * velocities + \
            (lc["c1"] - lc["c2"]) * lc["dt"] * a + gv
        return positions, velocities, energy, temperature

    return step


def write_xyz(handle, positions_host, comment):
    n = positions_host.shape[0]
    handle.write(f"{n}\n{comment}\n")
    for p in positions_host:
        handle.write(f"Ar {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def alchemi_neighbours(jnp, positions, cutoff, origin, cell, pbc, K):
    """``neighbours(p) -> (idx, dist, count)`` from NVIDIA ALCHEMI's JAX cell
    list, in the layout of `neighbour_matrix`. The cell-grid sizing is
    host-side and not jit-compatible, so it is estimated once here; the build
    itself is then `jit`-compiled."""
    from nvalchemiops.jax.neighbors.cell_list import (cell_list,
                                                      estimate_cell_list_sizes)
    origin = jnp.asarray(origin)
    cell3 = jnp.asarray(cell)[None]
    pbc3 = jnp.full((1, 3), pbc)
    max_total_cells, _, radius = estimate_cell_list_sizes(
        positions - origin, cell3, cutoff, pbc3)
    max_total_cells = int(max_total_cells)
    radius = np.asarray(radius)          # concrete, so ALCHEMI may pick a strategy

    @jax.jit
    def neighbours(p):
        # ALCHEMI has no cell origin: shift into the cell (translation invariant).
        idx, count, _, dist = cell_list(
            p - origin, cutoff, cell=cell3, pbc=pbc3, max_neighbors=K,
            max_total_cells=max_total_cells, neighbor_search_radius=radius,
            return_vectors=True)
        return idx, dist, count
    return neighbours


def check_capacity(count, K):
    """ALCHEMI's matrix output truncates overflowing rows without raising (the
    count stays exact), so check explicitly; `neighbour_matrix` raises itself."""
    most = int(count.max())
    if most > K:
        raise SystemExit(f"--max-neighbours={K} is too small (an atom has "
                         f"{most} neighbours)")


# `jax` is imported in main() (after enabling x64); referenced by make_step's
# decorator at call time.
jax = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--system", choices=["droplet", "liquid"], default="droplet",
                    help="droplet in vacuum (non-periodic) or bulk liquid in a "
                         "periodic box")
    ap.add_argument("--ncells", type=int, default=6,
                    help="FCC cells each way (droplet: radius; liquid: box "
                         "edge, 4*ncells^3 atoms) unless --atoms is given")
    ap.add_argument("--atoms", type=int, default=0,
                    help="exact atom count (overrides --ncells if > 0)")
    ap.add_argument("--lattice", type=float, default=1.6,
                    help="droplet FCC lattice constant")
    ap.add_argument("--density", type=float, default=0.8442,
                    help="liquid number density (reduced units); the default "
                         "with kT=0.7 is the Verlet (1967) liquid state point")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kT", type=float, default=0.7)
    ap.add_argument("--cutoff", type=float, default=2.5)
    ap.add_argument("--max-neighbours", type=int, default=96)
    ap.add_argument("--neighbours", choices=["matscipy", "alchemi"],
                    default="matscipy",
                    help="builder of the neighbour matrix (matscipy = this "
                         "library; alchemi = NVIDIA ALCHEMI, GPU only)")
    ap.add_argument("--out", default="traj_jax.xyz")
    ap.add_argument("--write-every", type=int, default=50,
                    help="write a trajectory frame every N steps (0: none)")
    args = ap.parse_args()

    global jax
    import jax as _jax
    jax = _jax
    jax.config.update("jax_enable_x64", True)   # the list works in float64
    import jax.numpy as jnp
    import jax.random as jrandom
    from matscipy_neighbours import neighbour_matrix

    devices = jax.devices(args.device)
    if not devices:
        raise SystemExit(f"no JAX {args.device} device available")
    dev = devices[0]

    cutoff, K = args.cutoff, args.max_neighbours
    if args.system == "liquid":
        n_target = args.atoms if args.atoms > 0 else 4 * args.ncells ** 3
        pos_np, L = fcc_liquid_n(n_target, args.density)
        lo, pbc = 0.0, True
    elif args.atoms > 0:
        pos_np = fcc_droplet_n(args.atoms, args.lattice)
        half = float(abs(pos_np).max()) + cutoff + 5.0
        lo, L, pbc = -half, 2.0 * half, False
    else:
        pos_np = fcc_droplet(args.ncells, args.lattice)
        half = args.ncells * args.lattice + cutoff + 5.0
        lo, L, pbc = -half, 2.0 * half, False
    n = pos_np.shape[0]
    positions = jax.device_put(jnp.asarray(pos_np), dev)
    velocities = jnp.zeros_like(positions)
    lc = langevin_constants(args.dt, args.gamma, args.kT)
    key = jrandom.PRNGKey(12345)
    step = make_step(jnp, jrandom, lc)

    origin = np.ascontiguousarray(np.full(3, lo))
    cell = np.ascontiguousarray(np.diag([L, L, L]).astype(float))

    if args.neighbours == "alchemi":
        if args.device != "gpu":
            raise SystemExit("alchemi is benchmarked on the GPU only; use "
                             "--device gpu")
        neighbours = alchemi_neighbours(jnp, positions, cutoff, origin, cell,
                                        pbc, K)
    else:
        def neighbours(p):
            return neighbour_matrix(positions=p, cell=cell, cell_origin=origin,
                                    pbc=pbc, cutoff=cutoff, max_neighbours=K,
                                    array_namespace=jnp)

    _, dist, count = neighbours(positions)
    check_capacity(count, K)
    print(f"device={args.device}  neighbours={args.neighbours}  "
          f"system={args.system}  atoms={n}  K={K}  backend={dev.platform}")

    # Warm up: one full, discarded iteration of the loop below (key split, step,
    # list rebuild at the moved positions), so that every one-time compilation
    # happens before timing.
    _, _sub = jrandom.split(key)
    _wp, _wv, _, _ = step(positions, velocities, dist, count, _sub, 1.0)
    jax.block_until_ready(neighbours(_wp))

    out = open(args.out, "w")
    jax.block_until_ready(positions)
    t0 = time.perf_counter()
    energy = 0.0
    for s in range(args.steps):
        key, sub = jrandom.split(key)
        positions, velocities, energy, temperature = step(
            positions, velocities, dist, count, sub, 0.0 if s == 0 else 1.0)
        _, dist, count = neighbours(positions)
        if args.write_every > 0 and s % args.write_every == 0:
            write_xyz(out, np.asarray(positions),
                      f"step={s} E_pot={float(energy):.4f} T={float(temperature):.4f}")
    jax.block_until_ready(positions)
    elapsed = time.perf_counter() - t0
    out.close()
    check_capacity(count, K)

    per_step = elapsed / args.steps
    print(f"steps={args.steps}  total={elapsed:.3f}s  {per_step * 1e3:.3f} ms/step")


if __name__ == "__main__":
    main()
