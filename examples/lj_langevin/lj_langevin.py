#!/usr/bin/env python3
"""Lennard-Jones Langevin dynamics: a droplet or a periodic liquid.

Prototyping-style implementation: the neighbour list returns the distance
vectors ``D``, and the Lennard-Jones potential is then a handful of array
operations. The same code runs on the CPU (NumPy) or the GPU (CuPy) — select
with ``--device``. Output is an XYZ trajectory.

Two systems (``--system``):

- ``droplet`` — a self-bound liquid droplet in vacuum (non-periodic, padded
  box; exercises the sparse/hashed cell grid);
- ``liquid`` — a homogeneous bulk liquid at reduced density ``--density`` in a
  fully periodic cubic box (dense cell grid; pairs across the boundary carry a
  non-zero cell shift, which is already folded into ``D``). The default state
  point, density 0.8442 at kT 0.7, is the Verlet (1967) liquid that the LAMMPS
  LJ benchmark also uses; it sits safely inside the liquid region at positive
  pressure.

The Langevin integrator is the Allen-Tildesley scheme (Allen & Tildesley,
Computer Simulation of Liquids, Sec. 9.3, after van Gunsteren & Berendsen
1982): positions and a half kick with the old forces, then the neighbour list
and forces at the new positions, then the second half kick. In the
zero-friction limit it reduces to velocity Verlet. The virial pressure is
computed from the same pair arrays as the forces and reported for the periodic
liquid. Reduced LJ units (epsilon = sigma = mass = kB = 1).
"""

import argparse
import math
import time

import numpy as np

FCC_BASIS = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


def get_backend(device):
    """Return the array module (NumPy on the CPU, CuPy on the GPU)."""
    if device == "gpu":
        import cupy as xp
        return xp, True
    return np, False


def fcc_droplet(xp, ncells, lattice):
    """A roughly spherical FCC cluster, centred at the origin."""
    span = range(-ncells, ncells + 1)
    r = np.array([(np.array([ix, iy, iz]) + b) * lattice
                  for ix in span for iy in span for iz in span for b in FCC_BASIS])
    r -= r.mean(axis=0)
    radius = ncells * lattice
    return xp.asarray(np.ascontiguousarray(r[(r * r).sum(axis=1) <= radius * radius]))


def fcc_droplet_n(xp, target_n, lattice):
    """An FCC cluster of *exactly* ``target_n`` atoms (the sites closest to the
    centre), so the benchmark can hit round atom counts for every backend."""
    ncells = int(math.ceil((3.0 * target_n / (16.0 * math.pi)) ** (1.0 / 3.0))) + 2
    span = range(-ncells, ncells + 1)
    r = np.array([(np.array([ix, iy, iz]) + b) * lattice
                  for ix in span for iy in span for iz in span for b in FCC_BASIS])
    r -= r.mean(axis=0)
    order = np.argsort((r * r).sum(axis=1))
    r = np.ascontiguousarray(r[order[:target_n]], dtype=float)
    return xp.asarray(r)


def fcc_liquid_n(xp, target_n, density):
    """Exactly ``target_n`` atoms on FCC sites filling a periodic cubic box at
    the given number density. Returns ``(positions, L)``.

    The box edge is ``L = (n / density)^(1/3)``; the smallest FCC lattice with
    at least ``n`` sites is scaled to fill it, and ``n`` sites are kept at
    evenly spaced lattice indices (``floor(k * nsites / n)``), so a partially
    filled lattice has its vacancies spread homogeneously. The same
    deterministic rule is used by every implementation (C++ included), so all
    backends start from an identical configuration."""
    L = (target_n / density) ** (1.0 / 3.0)
    ncells = int(math.ceil((target_n / 4.0) ** (1.0 / 3.0)))
    a = L / ncells
    span = range(ncells)
    r = np.array([(np.array([ix, iy, iz]) + b) * a
                  for ix in span for iy in span for iz in span for b in FCC_BASIS])
    nsites = r.shape[0]
    keep = (np.arange(target_n) * nsites) // target_n
    r = np.ascontiguousarray(r[keep], dtype=float)
    return xp.asarray(r), L


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
    half = ncells * lattice + cutoff + 5.0
    return _box(-half, 2.0 * half)


def fixed_box_pos(positions, cutoff):
    """As fixed_box, but sized from the actual droplet extent (used with the
    exact-count droplet, where there is no single ncells)."""
    half = float(abs(positions).max()) + cutoff + 5.0
    return _box(-half, 2.0 * half)


def periodic_box(L):
    """The periodic cubic box [0, L)^3 of the bulk liquid."""
    return _box(0.0, L)


def _box(lo, L):
    origin = np.ascontiguousarray(np.full(3, lo))
    cell = np.ascontiguousarray(np.diag([L, L, L]).astype(float))
    return origin, cell


def make_system(xp, system, atoms, ncells, lattice, density, cutoff):
    """Initial positions plus the neighbour-list geometry ``(origin, cell,
    pbc)`` for the chosen system."""
    if system == "liquid":
        n = atoms if atoms > 0 else 4 * ncells ** 3
        positions, L = fcc_liquid_n(xp, n, density)
        origin, cell = periodic_box(L)
        return positions, origin, cell, True
    if atoms > 0:
        positions = fcc_droplet_n(xp, atoms, lattice)
        origin, cell = fixed_box_pos(positions, cutoff)
    else:
        positions = fcc_droplet(xp, ncells, lattice)
        origin, cell = fixed_box(ncells, lattice, cutoff)
    return positions, origin, cell, False


def make_pairs_builder(kind, neighbour_list, xp, cutoff, origin, cell, pbc,
                       max_neighbours=96):
    """Return ``build(positions) -> (i, j, D)`` for the chosen neighbour-list
    backend. ``D == r[j] - r[i] + S @ cell`` (all backends share this
    convention; ``S`` is the cell shift of the periodic image)."""
    if kind == "matscipy":
        def build(positions):
            return neighbour_list("ijD", positions=positions, cell=cell,
                                  cell_origin=origin, pbc=pbc, cutoff=cutoff)
        return build
    if kind == "vesin":
        import vesin
        box = xp.asarray(cell)
        nl = vesin.NeighborList(cutoff=cutoff, full_list=True, sorted=True)

        def build(positions):
            return nl.compute(points=positions, box=box, periodic=pbc,
                              quantities="ijD")
        return build
    if kind == "matscipy-classic":
        # The classic matscipy package (pinned to 1.2.0). CPU/host only: its C
        # extension wants positions inside the cell along non-periodic
        # directions, so shift by -origin (the list is translation invariant).
        from matscipy.neighbours import neighbour_list as ms_nl
        pbc3 = [pbc, pbc, pbc]

        def build(positions):
            return ms_nl("ijD", positions=positions - origin, cell=cell,
                         pbc=pbc3, cutoff=cutoff)
        return build
    if kind == "alchemi":
        # NVIDIA ALCHEMI (nvalchemiops) through its PyTorch entry point, GPU
        # only. It has no cell origin (shift the positions into the cell; the
        # list is translation invariant) and returns indices and cell shifts,
        # not distance vectors, so D is assembled here.
        import torch
        from nvalchemiops.torch.neighbors import neighbor_list as alchemi_nl
        cell_t = torch.as_tensor(cell, device="cuda")
        pbc_t = torch.tensor([pbc] * 3, device="cuda")
        origin_d = xp.asarray(origin)
        cell_d = xp.asarray(cell)

        def build(positions):
            nl, _, S = alchemi_nl(torch.from_dlpack(positions - origin_d),
                                  cutoff, cell=cell_t, pbc=pbc_t,
                                  method="cell_list",
                                  max_neighbors=max_neighbours,
                                  return_neighbor_list=True)
            i, j = xp.from_dlpack(nl[0]), xp.from_dlpack(nl[1])
            D = positions[j] - positions[i]
            if pbc:
                D += xp.from_dlpack(S).astype(D.dtype) @ cell_d
            return i, j, D
        return build
    raise SystemExit(f"unknown neighbour backend: {kind}")


def lj_forces_energy(xp, build_ijD, positions):
    """Lennard-Jones forces, potential energy and virial from the neighbour
    list.

    The builder returns directed pairs sorted by ``i`` with the distance vector
    ``D == r[j] - r[i] + S @ cell`` (periodic images already applied); the
    force on ``i`` from each pair is accumulated per atom. This is the whole
    potential. The virial ``W = sum_{i<j} r_ij . F_ij`` (with ``r_ij = -D``)
    is one more reduction over the same pair arrays; the pressure is
    ``P = (N kT + W / 3) / V``."""
    n = positions.shape[0]
    i, j, D = build_ijD(positions)

    r2 = (D * D).sum(axis=1)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    energy = 0.5 * float((4.0 * inv_r6 * (inv_r6 - 1.0)).sum())
    coef = -24.0 * inv_r2 * inv_r6 * (2.0 * inv_r6 - 1.0)   # force prefactor
    fpair = coef[:, None] * D                                # force on i
    forces = mabincount(xp, i, fpair, n)
    virial = -0.5 * float((coef * r2).sum())                 # directed pairs: 1/2
    return forces, energy, virial, int(i.shape[0])


def kinetic_temperature(xp, velocities):
    """Instantaneous kinetic temperature, ``2 K / (3 N)`` in reduced units."""
    n = velocities.shape[0]
    return float((velocities * velocities).sum()) / (3.0 * n)


def pressure(n, kT, virial, volume):
    """Virial pressure ``(N kT + W / 3) / V``."""
    return (n * kT + virial / 3.0) / volume


def langevin_constants(dt, gamma, kT, mass=1.0):
    """Precompute the Allen-Tildesley Langevin coefficients."""
    D = kT / (mass * gamma)
    c0 = np.exp(-gamma * dt)
    c1 = (1.0 - c0) / (gamma * dt)
    c2 = (1.0 - c1) / (gamma * dt)
    sr = np.sqrt(dt * D * (2.0 - (3.0 - 4.0 * c0 + c0 * c0) / (gamma * dt)))
    sv = np.sqrt(gamma * D * (1.0 - c0 * c0))
    crv = D * (1.0 - c0) ** 2 / (sr * sv)
    return dict(c0=c0, c1=c1, c2=c2, sr=sr, sv=sv, crv=crv, dt=dt, mass=mass)


def langevin_drift(xp, positions, velocities, forces, lc):
    """First half of a Langevin step, in place: move the positions and apply
    the friction, the random kicks and the half kick with the *old* forces.
    Rebuild the neighbour list and the forces at the new positions, then call
    :func:`langevin_kick`."""
    g1 = xp.random.standard_normal(positions.shape)
    g2 = xp.random.standard_normal(positions.shape)
    gr = lc["sr"] * g1
    gv = lc["sv"] * (lc["crv"] * g1 + (1.0 - lc["crv"] ** 2) ** 0.5 * g2)
    a = forces / lc["mass"]
    positions += lc["c1"] * lc["dt"] * velocities + \
        lc["c2"] * lc["dt"] ** 2 * a + gr
    velocities *= lc["c0"]
    velocities += (lc["c1"] - lc["c2"]) * lc["dt"] * a + gv


def langevin_kick(velocities, forces, lc):
    """Second half of a Langevin step: the half kick with the *new* forces."""
    velocities += lc["c2"] * lc["dt"] * forces / lc["mass"]


def write_xyz(handle, positions_host, comment):
    n = positions_host.shape[0]
    handle.write(f"{n}\n{comment}\n")
    for p in positions_host:
        handle.write(f"Ar {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


def add_system_arguments(ap):
    """The system-selection flags shared by the Python examples."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--neighbours",
                    choices=["matscipy", "matscipy-classic", "vesin", "alchemi"],
                    default="matscipy",
                    help="neighbour-list builder (matscipy = this library; "
                         "matscipy-classic = the matscipy 1.2.0 package, CPU "
                         "only; alchemi = NVIDIA ALCHEMI, GPU only)")
    ap.add_argument("--max-neighbours", type=int, default=96,
                    help="per-atom capacity of ALCHEMI's internal neighbour "
                         "matrix (alchemi only)")
    add_system_arguments(ap)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kT", type=float, default=0.7)
    ap.add_argument("--cutoff", type=float, default=2.5)
    ap.add_argument("--out", default="traj.xyz")
    ap.add_argument("--write-every", type=int, default=50,
                    help="write a trajectory frame every N steps (0: none)")
    args = ap.parse_args()

    if args.neighbours == "matscipy-classic" and args.device == "gpu":
        raise SystemExit("matscipy-classic (the matscipy 1.2.0 package) is "
                         "CPU only; use --device cpu")
    if args.neighbours == "alchemi" and args.device == "cpu":
        raise SystemExit("alchemi is benchmarked on the GPU only; use "
                         "--device gpu")
    xp, on_gpu = get_backend(args.device)
    from matscipy_neighbours import neighbour_list

    positions, origin, cell, pbc = make_system(
        xp, args.system, args.atoms, args.ncells, args.lattice, args.density,
        args.cutoff)
    n = positions.shape[0]
    velocities = xp.zeros_like(positions)
    lc = langevin_constants(args.dt, args.gamma, args.kT)

    to_host = (lambda a: xp.asnumpy(a)) if on_gpu else (lambda a: np.asarray(a))
    build_ijD = make_pairs_builder(args.neighbours, neighbour_list, xp,
                                   args.cutoff, origin, cell, pbc,
                                   args.max_neighbours)

    volume = float(np.linalg.det(cell))
    forces, energy, virial, npairs = lj_forces_energy(xp, build_ijD, positions)
    print(f"device={args.device}  neighbours={args.neighbours}  "
          f"system={args.system}  atoms={n}  pairs~{npairs}  E_pot={energy:.6f}")

    out = open(args.out, "w")
    T_sum = P_sum = 0.0
    t0 = time.perf_counter()
    for step in range(args.steps):
        langevin_drift(xp, positions, velocities, forces, lc)
        forces, energy, virial, npairs = lj_forces_energy(xp, build_ijD, positions)
        langevin_kick(velocities, forces, lc)
        if args.write_every > 0 and step % args.write_every == 0:
            T = kinetic_temperature(xp, velocities)
            comment = f"step={step} E_pot={energy:.4f} T={T:.4f}"
            if pbc:   # the pressure of the padded droplet box is not meaningful
                comment += f" P={pressure(n, T, virial, volume):.4f}"
            write_xyz(out, to_host(positions), comment)
        if step >= args.steps // 2:
            T = kinetic_temperature(xp, velocities)
            T_sum += T
            P_sum += pressure(n, T, virial, volume)
    if on_gpu:
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    out.close()

    per_step = elapsed / max(args.steps, 1)
    nsamples = args.steps - args.steps // 2
    if nsamples > 0:
        summary = f"second half: <T>={T_sum / nsamples:.4f}"
        if pbc:
            summary += f"  <P>={P_sum / nsamples:.4f}"
        print(summary)
    print(f"steps={args.steps}  total={elapsed:.3f}s  "
          f"{per_step * 1e3:.3f} ms/step  {per_step * 1e9 / npairs:.1f} ns/pair")


if __name__ == "__main__":
    main()
