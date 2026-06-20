#!/usr/bin/env python3
"""Lennard-Jones liquid droplet with a Langevin thermostat.

Prototyping-style implementation: the neighbour list returns the distance
vectors ``D``, and the Lennard-Jones potential is then a handful of array
operations. The same code runs on the CPU (NumPy) or the GPU (CuPy) — select
with ``--device``. Output is an XYZ trajectory.

The Langevin integrator is the Allen-Tildesley scheme (Baron, "Continuous
stochastic variables", Algorithm: Langevin dynamics). Reduced LJ units
(epsilon = sigma = mass = kB = 1).
"""

import argparse
import time


def get_backend(device):
    """Return the array module (NumPy on the CPU, CuPy on the GPU)."""
    if device == "gpu":
        import cupy as xp
        return xp, True
    import numpy as xp
    return xp, False


def fcc_droplet(xp, ncells, lattice):
    """A roughly spherical FCC cluster, centred at the origin."""
    basis = xp.asarray([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    r = []
    span = range(-ncells, ncells + 1)
    for ix in span:
        for iy in span:
            for iz in span:
                for b in basis:
                    r.append((xp.asarray([ix, iy, iz]) + b) * lattice)
    r = xp.asarray(r, dtype=float)
    r -= r.mean(axis=0)
    radius = ncells * lattice
    return xp.ascontiguousarray(r[(r * r).sum(axis=1) <= radius * radius])


def mabincount(xp, idx, weights, n):
    """Sum ``weights`` (shape (npairs, d)) over pairs grouped by ``idx``,
    giving (n, d). ``bincount`` weights are 1-D, so accumulate per component —
    the operation matscipy performs with its ``mabincount`` helper."""
    idx = idx.astype(xp.int64)
    out = xp.empty((n, weights.shape[1]), dtype=weights.dtype)
    for k in range(weights.shape[1]):
        out[:, k] = xp.bincount(idx, weights=weights[:, k], minlength=n)
    return out


def fixed_box(ncells, lattice, cutoff):
    """A cubic box centred at the origin enclosing the (self-bound) droplet for
    the whole run, so the neighbour grid need not be recomputed each step."""
    import numpy as np
    half = ncells * lattice + cutoff + 5.0
    L = 2.0 * half
    origin = np.ascontiguousarray(np.full(3, -half))
    cell = np.ascontiguousarray(np.diag([L, L, L]).astype(float))
    return origin, cell


def lj_forces_energy(xp, neighbour_list, positions, cutoff, origin, cell):
    """Lennard-Jones forces and potential energy from the neighbour list.

    The list returns directed pairs sorted by ``i`` with the distance vector
    ``D == r[j] - r[i]``; the force on ``i`` from each pair is accumulated per
    atom. This is the whole potential."""
    n = positions.shape[0]
    i, j, D = neighbour_list("ijD", positions=positions, cell=cell,
                             cell_origin=origin, pbc=False, cutoff=cutoff)

    r2 = (D * D).sum(axis=1)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    energy = 0.5 * float((4.0 * inv_r6 * (inv_r6 - 1.0)).sum())
    coef = -24.0 * inv_r2 * inv_r6 * (2.0 * inv_r6 - 1.0)   # force prefactor
    fpair = coef[:, None] * D                                # force on i
    forces = mabincount(xp, i, fpair, n)
    return forces, energy, int(i.shape[0])


def langevin_constants(dt, gamma, kT, mass=1.0):
    """Precompute the Allen-Tildesley Langevin coefficients."""
    import numpy as np
    D = kT / (mass * gamma)
    c0 = np.exp(-gamma * dt)
    c1 = (1.0 - c0) / (gamma * dt)
    c2 = (1.0 - c1) / (gamma * dt)
    sr = np.sqrt(dt * D * (2.0 - (3.0 - 4.0 * c0 + c0 * c0) / (gamma * dt)))
    sv = np.sqrt(gamma * D * (1.0 - c0 * c0))
    crv = D * (1.0 - c0) ** 2 / (sr * sv)
    return dict(c0=c0, c1=c1, c2=c2, sr=sr, sv=sv, crv=crv, dt=dt, mass=mass)


def langevin_step(xp, positions, velocities, forces, lc):
    """One Langevin update (positions and velocities), in place."""
    g1 = xp.random.standard_normal(positions.shape)
    g2 = xp.random.standard_normal(positions.shape)
    gr = lc["sr"] * g1
    gv = lc["sv"] * (lc["crv"] * g1 + (1.0 - lc["crv"] ** 2) ** 0.5 * g2)
    fm = forces / lc["mass"]
    positions += lc["c1"] * lc["dt"] * velocities + \
        lc["c2"] * lc["dt"] ** 2 * fm + gr
    velocities += (lc["c0"] - 1.0) * velocities + lc["c1"] * lc["dt"] * fm + gv


def write_xyz(handle, positions_host, comment):
    n = positions_host.shape[0]
    handle.write(f"{n}\n{comment}\n")
    for p in positions_host:
        handle.write(f"Ar {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--ncells", type=int, default=6, help="FCC cells each way")
    ap.add_argument("--lattice", type=float, default=1.6)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kT", type=float, default=0.7)
    ap.add_argument("--cutoff", type=float, default=2.5)
    ap.add_argument("--out", default="traj.xyz")
    ap.add_argument("--write-every", type=int, default=50)
    args = ap.parse_args()

    xp, on_gpu = get_backend(args.device)
    import numpy as np
    from matscipy_neighbours import neighbour_list

    positions = fcc_droplet(xp, args.ncells, args.lattice)
    n = positions.shape[0]
    velocities = xp.zeros_like(positions)
    lc = langevin_constants(args.dt, args.gamma, args.kT)

    to_host = (lambda a: xp.asnumpy(a)) if on_gpu else (lambda a: np.asarray(a))
    origin, cell = fixed_box(args.ncells, args.lattice, args.cutoff)

    forces, energy, npairs = lj_forces_energy(xp, neighbour_list, positions,
                                              args.cutoff, origin, cell)
    print(f"device={args.device}  atoms={n}  pairs~{npairs}")

    out = open(args.out, "w")
    t0 = time.perf_counter()
    for step in range(args.steps):
        langevin_step(xp, positions, velocities, forces, lc)
        forces, energy, npairs = lj_forces_energy(xp, neighbour_list, positions,
                                                  args.cutoff, origin, cell)
        if step % args.write_every == 0:
            write_xyz(out, to_host(positions), f"step={step} E_pot={energy:.4f}")
    if on_gpu:
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    out.close()

    per_step = elapsed / args.steps
    print(f"steps={args.steps}  total={elapsed:.3f}s  "
          f"{per_step * 1e3:.3f} ms/step  {per_step * 1e9 / npairs:.1f} ns/pair")


if __name__ == "__main__":
    main()
