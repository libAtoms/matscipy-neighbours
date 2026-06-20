#!/usr/bin/env python3
"""Lennard-Jones liquid droplet with a Langevin thermostat — JAX.

Uses the dense fixed-capacity neighbour list (`neighbour_matrix`): each atom's
neighbours occupy a row of an ``n x K`` matrix, so the shapes are static and the
per-step force + Langevin update can be `jit`-compiled once (no per-step
recompilation). Forces are a masked sum over the neighbour axis — no scatter. The
neighbour list is exchanged with JAX zero-copy through DLPack
(`array_namespace=jax.numpy`). Select CPU or GPU with ``--device`` (the GPU path
needs a CUDA build of JAX). Reduced LJ units (epsilon = sigma = mass = kB = 1);
output is an XYZ trajectory.
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
    shapes, so this compiles once and reruns without recompilation."""

    @jax.jit
    def step(positions, velocities, dist, count, key):
        K = dist.shape[1]
        mask = jnp.arange(K)[None, :] < count[:, None]      # valid neighbours
        r2 = (dist * dist).sum(axis=-1)
        safe = jnp.where(mask, r2, 1.0)                      # avoid 1/0 in pads
        inv_r2 = 1.0 / safe
        inv_r6 = inv_r2 ** 3
        coef = jnp.where(mask, -24.0 * inv_r2 * inv_r6 * (2.0 * inv_r6 - 1.0), 0.0)
        forces = (coef[..., None] * dist).sum(axis=1)        # (n, 3), no scatter
        energy = 0.5 * jnp.where(mask, 4.0 * inv_r6 * (inv_r6 - 1.0), 0.0).sum()

        k1, k2 = jrandom.split(key)
        g1 = jrandom.normal(k1, positions.shape)
        g2 = jrandom.normal(k2, positions.shape)
        gr = lc["sr"] * g1
        gv = lc["sv"] * (lc["crv"] * g1 + (1.0 - lc["crv"] ** 2) ** 0.5 * g2)
        fm = forces / lc["mass"]
        positions = positions + lc["c1"] * lc["dt"] * velocities + \
            lc["c2"] * lc["dt"] ** 2 * fm + gr
        velocities = velocities + (lc["c0"] - 1.0) * velocities + \
            lc["c1"] * lc["dt"] * fm + gv
        return positions, velocities, energy

    return step


def write_xyz(handle, positions_host, comment):
    n = positions_host.shape[0]
    handle.write(f"{n}\n{comment}\n")
    for p in positions_host:
        handle.write(f"Ar {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


# `jax` is imported in main() (after enabling x64); referenced by make_step's
# decorator at call time.
jax = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--ncells", type=int, default=6)
    ap.add_argument("--lattice", type=float, default=1.6)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kT", type=float, default=0.7)
    ap.add_argument("--cutoff", type=float, default=2.5)
    ap.add_argument("--max-neighbours", type=int, default=96)
    ap.add_argument("--out", default="traj_jax.xyz")
    ap.add_argument("--write-every", type=int, default=50)
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

    pos_np = fcc_droplet(args.ncells, args.lattice)
    n = pos_np.shape[0]
    positions = jax.device_put(jnp.asarray(pos_np), dev)
    velocities = jnp.zeros_like(positions)
    lc = langevin_constants(args.dt, args.gamma, args.kT)
    cutoff, K = args.cutoff, args.max_neighbours
    key = jrandom.PRNGKey(12345)
    step = make_step(jnp, jrandom, lc)

    half = args.ncells * args.lattice + cutoff + 5.0
    L = 2.0 * half
    origin = np.ascontiguousarray(np.full(3, -half))
    cell = np.ascontiguousarray(np.diag([L, L, L]).astype(float))

    def neighbours(p):
        return neighbour_matrix(positions=p, cell=cell, cell_origin=origin,
                                pbc=False, cutoff=cutoff, max_neighbours=K,
                                array_namespace=jnp)

    _, dist, count = neighbours(positions)
    print(f"device={args.device}  atoms={n}  K={K}  backend={dev.platform}")

    # Warm up: trigger the one-time jit compilation before timing.
    _wp, _wv, _ = step(positions, velocities, dist, count, key)
    jax.block_until_ready(_wp)

    out = open(args.out, "w")
    jax.block_until_ready(positions)
    t0 = time.perf_counter()
    energy = 0.0
    for s in range(args.steps):
        key, sub = jrandom.split(key)
        positions, velocities, energy = step(positions, velocities, dist, count,
                                             sub)
        _, dist, count = neighbours(positions)
        if s % args.write_every == 0:
            write_xyz(out, np.asarray(positions),
                      f"step={s} E_pot={float(energy):.4f}")
    jax.block_until_ready(positions)
    elapsed = time.perf_counter() - t0
    out.close()

    per_step = elapsed / args.steps
    print(f"steps={args.steps}  total={elapsed:.3f}s  {per_step * 1e3:.3f} ms/step")


if __name__ == "__main__":
    main()
