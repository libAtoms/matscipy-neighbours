#!/usr/bin/env python3
"""Lennard-Jones Langevin dynamics (droplet or periodic liquid) — NVIDIA Warp.

The Lennard-Jones force/energy and the Langevin integrator are written as
`warp` kernels (compiled once, then launched every step), but the neighbour
list is built by *this* library — `matscipy_neighbours` — or, for comparison,
by `vesin` (https://github.com/luthaf/vesin) or by NVIDIA ALCHEMI
(`nvalchemiops`, https://github.com/NVIDIA/nvalchemi-toolkit-ops; GPU only,
through its PyTorch interface). Select the builder with
``--neighbours {matscipy,vesin,alchemi}``.

The point is interop: positions live in a single device buffer (CuPy on the
GPU, NumPy on the CPU); Warp wraps it zero-copy through DLPack, and the
neighbour-list builder reads the very same buffer. The builder returns the
directed pair list ``(i, j)`` (full list, sorted by ``i``); the fused Warp
kernel recomputes the distance for each pair and atomically accumulates the
force on ``i`` — never materialising per-pair arrays. The potential energy is
reduced within each thread block and added with one atomic per block. For the periodic liquid
(``--system liquid``) the builder also returns the cell shift ``S`` of each
pair, and the kernel evaluates the periodic image D = r[j] - r[i] + S @ cell.

``--format matrix`` swaps the pair list for the fixed-capacity *neighbour
matrix* (one row of ``--max-neighbours`` slots per atom plus a per-atom count),
consumed by a kernel with one thread per atom that sums its own row — no
atomics on the forces. Each library supplies the matrix in its native form:
matscipy-neighbours (`neighbour_matrix`) with the distance vectors ``D``,
ALCHEMI with neighbour indices (and cell shifts for a periodic box), from
which the kernel recomputes ``D``.

Timing is broken down per phase with `muTimer` (build neighbour list / LJ
forces / Langevin integrate), and the neighbour-list build time is reported
separately. Each timed phase synchronises the device so the breakdown is
attributable; the headline ``ms/step`` is their sum.

Reduced LJ units (epsilon = sigma = mass = kB = 1); optional XYZ trajectory.
Warp's RNG draws single-precision normals for the thermostat noise; positions,
velocities and forces are double precision.
"""

import argparse
import math
from typing import Any

import numpy as np
import warp as wp
from muTimer import Timer

vec3d = wp.vec3d


# --------------------------------------------------------------------------- #
# Warp kernels
# --------------------------------------------------------------------------- #
# Threads per block of the force kernels. The launch is rounded up to whole
# blocks (see `launch_forces`); threads past the end contribute nothing.
BLOCK_DIM = 256


# Energy reduction: each thread holds the energy of its pair (or row), zero if
# it has none. `wp.tile(e)` gathers these values across the thread block,
# `wp.tile_sum` reduces them cooperatively (warp shuffles, then shared memory),
# and `wp.tile_atomic_add` adds the block's total to the global energy: one
# atomic per block instead of one per pair, which would serialise every thread
# of the launch on a single address. All threads of a block must reach the
# tile operations, so the kernels guard their work with `if` rather than
# returning early.


@wp.kernel
def lj_forces(pos: wp.array(dtype=vec3d),
              ii: wp.array(dtype=wp.int32),
              jj: wp.array(dtype=wp.int32),
              npairs: int,
              cutoff_sq: wp.float64,
              forces: wp.array(dtype=vec3d),
              energy: wp.array(dtype=wp.float64)):
    """One thread per directed pair. ``forces`` and ``energy`` must be zeroed
    beforehand; the full (both-directions) list means the force is accumulated
    on ``i`` only and the energy carries the 1/2 double-counting factor."""
    p = wp.tid()
    e = wp.float64(0.0)
    if p < npairs:
        i = ii[p]
        dr = pos[jj[p]] - pos[i]
        r2 = wp.dot(dr, dr)
        if r2 < cutoff_sq:
            inv_r2 = wp.float64(1.0) / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            coef = wp.float64(-24.0) * inv_r2 * inv_r6 * (wp.float64(2.0) * inv_r6 - wp.float64(1.0))
            wp.atomic_add(forces, i, coef * dr)
            e = wp.float64(2.0) * inv_r6 * (inv_r6 - wp.float64(1.0))
    wp.tile_atomic_add(energy, wp.tile_sum(wp.tile(e)))


@wp.kernel
def lj_forces_pbc(pos: wp.array(dtype=vec3d),
                  ii: wp.array(dtype=wp.int32),
                  jj: wp.array(dtype=wp.int32),
                  shift: wp.array(dtype=wp.vec3i),
                  npairs: int,
                  cell_t: wp.mat33d,
                  cutoff_sq: wp.float64,
                  forces: wp.array(dtype=vec3d),
                  energy: wp.array(dtype=wp.float64)):
    """As `lj_forces`, for a periodic box: the pair's cell shift selects the
    periodic image, D = r[j] - r[i] + S @ cell (`cell_t` is the transposed
    cell, rows = lattice vectors, so that S @ cell == cell_t * S)."""
    p = wp.tid()
    e = wp.float64(0.0)
    if p < npairs:
        i = ii[p]
        s = shift[p]
        sd = vec3d(wp.float64(s[0]), wp.float64(s[1]), wp.float64(s[2]))
        dr = pos[jj[p]] - pos[i] + cell_t * sd
        r2 = wp.dot(dr, dr)
        if r2 < cutoff_sq:
            inv_r2 = wp.float64(1.0) / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            coef = wp.float64(-24.0) * inv_r2 * inv_r6 * (wp.float64(2.0) * inv_r6 - wp.float64(1.0))
            wp.atomic_add(forces, i, coef * dr)
            e = wp.float64(2.0) * inv_r6 * (inv_r6 - wp.float64(1.0))
    wp.tile_atomic_add(energy, wp.tile_sum(wp.tile(e)))


@wp.kernel
def lj_forces_matrix(pos: wp.array(dtype=vec3d),
                     nbr: wp.array2d(dtype=Any),
                     count: wp.array(dtype=Any),
                     n: int,
                     cutoff_sq: wp.float64,
                     forces: wp.array(dtype=vec3d),
                     energy: wp.array(dtype=wp.float64)):
    """Neighbour-matrix form: one thread per atom sums the pairs in its row
    (the first ``count[a]`` slots), so the force is written, not accumulated
    atomically. ``energy`` must be zeroed beforehand."""
    a = wp.tid()
    e = wp.float64(0.0)
    if a < n:
        f = vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        for k in range(int(count[a])):
            dr = pos[int(nbr[a, k])] - pos[a]
            r2 = wp.dot(dr, dr)
            if r2 < cutoff_sq:
                inv_r2 = wp.float64(1.0) / r2
                inv_r6 = inv_r2 * inv_r2 * inv_r2
                f += wp.float64(-24.0) * inv_r2 * inv_r6 * (wp.float64(2.0) * inv_r6 - wp.float64(1.0)) * dr
                e += wp.float64(2.0) * inv_r6 * (inv_r6 - wp.float64(1.0))
        forces[a] = f
    wp.tile_atomic_add(energy, wp.tile_sum(wp.tile(e)))


@wp.kernel
def lj_forces_matrix_pbc(pos: wp.array(dtype=vec3d),
                         nbr: wp.array2d(dtype=Any),
                         count: wp.array(dtype=Any),
                         shift: wp.array2d(dtype=wp.vec3i),
                         n: int,
                         cell_t: wp.mat33d,
                         cutoff_sq: wp.float64,
                         forces: wp.array(dtype=vec3d),
                         energy: wp.array(dtype=wp.float64)):
    """As `lj_forces_matrix`, with the per-slot cell shift of a periodic box."""
    a = wp.tid()
    e = wp.float64(0.0)
    if a < n:
        f = vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        for k in range(int(count[a])):
            s = shift[a, k]
            sd = vec3d(wp.float64(s[0]), wp.float64(s[1]), wp.float64(s[2]))
            dr = pos[int(nbr[a, k])] - pos[a] + cell_t * sd
            r2 = wp.dot(dr, dr)
            if r2 < cutoff_sq:
                inv_r2 = wp.float64(1.0) / r2
                inv_r6 = inv_r2 * inv_r2 * inv_r2
                f += wp.float64(-24.0) * inv_r2 * inv_r6 * (wp.float64(2.0) * inv_r6 - wp.float64(1.0)) * dr
                e += wp.float64(2.0) * inv_r6 * (inv_r6 - wp.float64(1.0))
        forces[a] = f
    wp.tile_atomic_add(energy, wp.tile_sum(wp.tile(e)))


@wp.kernel
def lj_forces_matrix_D(dist: wp.array2d(dtype=vec3d),
                       count: wp.array(dtype=Any),
                       n: int,
                       cutoff_sq: wp.float64,
                       forces: wp.array(dtype=vec3d),
                       energy: wp.array(dtype=wp.float64)):
    """As `lj_forces_matrix`, from precomputed distance vectors (periodic
    images already applied), so neither positions nor indices are read."""
    a = wp.tid()
    e = wp.float64(0.0)
    if a < n:
        f = vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        for k in range(int(count[a])):
            dr = dist[a, k]
            r2 = wp.dot(dr, dr)
            if r2 < cutoff_sq:
                inv_r2 = wp.float64(1.0) / r2
                inv_r6 = inv_r2 * inv_r2 * inv_r2
                f += wp.float64(-24.0) * inv_r2 * inv_r6 * (wp.float64(2.0) * inv_r6 - wp.float64(1.0)) * dr
                e += wp.float64(2.0) * inv_r6 * (inv_r6 - wp.float64(1.0))
        forces[a] = f
    wp.tile_atomic_add(energy, wp.tile_sum(wp.tile(e)))


def launch_forces(kernel, count, inputs, device):
    """Launch a force kernel over ``count`` threads, rounded up to whole blocks
    of `BLOCK_DIM` so that every block takes part in the tile reduction."""
    dim = (count + BLOCK_DIM - 1) // BLOCK_DIM * BLOCK_DIM
    wp.launch(kernel, dim=dim, inputs=inputs, block_dim=BLOCK_DIM,
              device=device)


@wp.kernel
def langevin_drift(pos: wp.array(dtype=vec3d),
                   vel: wp.array(dtype=vec3d),
                   forces: wp.array(dtype=vec3d),
                   c0: wp.float64, c1: wp.float64, c2: wp.float64,
                   dt: wp.float64, mass: wp.float64,
                   sr: wp.float64, sv: wp.float64, crv: wp.float64,
                   seed: wp.int32):
    """First half of a Langevin step (Allen-Tildesley scheme), one thread per
    atom: move the positions; friction, random kicks and the half kick with
    the *old* forces on the velocities."""
    a = wp.tid()
    st = wp.rand_init(seed, a)
    g1 = vec3d(wp.float64(wp.randn(st)), wp.float64(wp.randn(st)), wp.float64(wp.randn(st)))
    g2 = vec3d(wp.float64(wp.randn(st)), wp.float64(wp.randn(st)), wp.float64(wp.randn(st)))
    gr = sr * g1
    gv = sv * (crv * g1 + wp.sqrt(wp.float64(1.0) - crv * crv) * g2)
    v = vel[a]
    fm = forces[a] / mass
    pos[a] = pos[a] + c1 * dt * v + c2 * dt * dt * fm + gr
    vel[a] = c0 * v + (c1 - c2) * dt * fm + gv


@wp.kernel
def langevin_kick(vel: wp.array(dtype=vec3d),
                  forces: wp.array(dtype=vec3d),
                  c2: wp.float64, dt: wp.float64, mass: wp.float64):
    """Second half of a Langevin step: the half kick with the *new* forces."""
    a = wp.tid()
    vel[a] = vel[a] + c2 * dt * forces[a] / mass


# --------------------------------------------------------------------------- #
# Host helpers
# --------------------------------------------------------------------------- #
def fcc_droplet(xp, target_n, lattice):
    """A roughly spherical FCC cluster of *exactly* ``target_n`` atoms (the
    ``target_n`` sites closest to the centre), centred at the origin."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    # Enough cells for a sphere holding >= target_n FCC sites (+ margin).
    ncells = int(math.ceil((3.0 * target_n / (16.0 * math.pi)) ** (1.0 / 3.0))) + 2
    span = range(-ncells, ncells + 1)
    r = np.array([(np.array([ix, iy, iz]) + b) * lattice
                  for ix in span for iy in span for iz in span for b in basis])
    r -= r.mean(axis=0)
    order = np.argsort((r * r).sum(axis=1))
    r = np.ascontiguousarray(r[order[:target_n]], dtype=float)
    return xp.asarray(r)


def fcc_liquid_n(xp, target_n, density):
    """Exactly ``target_n`` atoms on FCC sites filling a periodic cubic box at
    the given number density; returns ``(positions, L)``. Same deterministic
    site selection as the other implementations (see lj_langevin.py)."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    L = (target_n / density) ** (1.0 / 3.0)
    ncells = int(math.ceil((target_n / 4.0) ** (1.0 / 3.0)))
    a = L / ncells
    span = range(ncells)
    r = np.array([(np.array([ix, iy, iz]) + b) * a
                  for ix in span for iy in span for iz in span for b in basis])
    keep = (np.arange(target_n) * r.shape[0]) // target_n
    return xp.asarray(np.ascontiguousarray(r[keep], dtype=float)), L


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


def fixed_box(xp_positions, cutoff):
    """A cubic box centred at the origin enclosing the droplet, with padding so
    it stays valid for the whole (short) run without rebuilding the grid."""
    half = float(abs(xp_positions).max()) + cutoff + 5.0
    L = 2.0 * half
    origin = np.ascontiguousarray(np.full(3, -half))
    cell = np.ascontiguousarray(np.diag([L, L, L]).astype(float))
    return origin, cell


def periodic_box(L):
    """The periodic cubic box [0, L)^3 of the bulk liquid."""
    origin = np.zeros(3)
    cell = np.ascontiguousarray(np.diag([L, L, L]).astype(float))
    return origin, cell


def make_neighbour_builder(kind, xp, on_gpu, cutoff, origin, cell, pbc,
                           max_neighbours):
    """Return ``build(positions) -> (i_int32, j_int32, S_int32, npairs)`` for
    the chosen backend: the device-resident directed pair list, with the cell
    shifts ``S`` (shape (npairs, 3)) for a periodic box and ``None``
    otherwise."""
    quantities = "ijS" if pbc else "ij"

    def pack(i, j, S=None):
        S32 = None if S is None else \
            xp.ascontiguousarray(S.astype(xp.int32, copy=False))
        return (i.astype(xp.int32, copy=False), j.astype(xp.int32, copy=False),
                S32, int(i.shape[0]))

    if kind == "matscipy":
        from matscipy_neighbours import neighbour_list

        def build(positions):
            return pack(*neighbour_list(quantities, positions=positions,
                                        cell=cell, cell_origin=origin, pbc=pbc,
                                        cutoff=cutoff))
        return build

    if kind == "vesin":
        import vesin

        box = xp.asarray(cell)
        nl = vesin.NeighborList(cutoff=cutoff, full_list=True, sorted=True)

        def build(positions):
            return pack(*nl.compute(points=positions, box=box, periodic=pbc,
                                    quantities=quantities))
        return build

    if kind == "matscipy-classic":
        # The classic matscipy package (pinned to 1.2.0). CPU/host only; shift
        # positions by -origin so they sit inside the cell for its C extension.
        from matscipy.neighbours import neighbour_list as ms_nl
        pbc3 = [pbc, pbc, pbc]

        def build(positions):
            return pack(*ms_nl(quantities, positions=positions - origin,
                               cell=cell, pbc=pbc3, cutoff=cutoff))
        return build

    if kind == "alchemi":
        alchemi = alchemi_builder(xp, cutoff, origin, cell, pbc, max_neighbours,
                                  matrix=False)

        def build(positions):
            # COO pairs (2, npairs), CSR row pointer, cell shifts (npairs, 3);
            # zero-copy from PyTorch into CuPy.
            nl, _, S = (xp.from_dlpack(t) for t in alchemi(positions))
            return pack(nl[0], nl[1], S if pbc else None)
        return build

    raise SystemExit(f"unknown neighbour backend: {kind}")


def alchemi_builder(xp, cutoff, origin, cell, pbc, max_neighbours, matrix):
    """NVIDIA ALCHEMI's cell list through its documented PyTorch entry point,
    ``nvalchemiops.torch.neighbors.neighbor_list`` (which runs Warp kernels on
    the tensors). Returns ``build(positions) -> tuple of torch tensors``:
    ``(neighbor_matrix, num_neighbors, shifts)`` if ``matrix`` else
    ``(neighbor_list, neighbor_ptr, shifts)``."""
    import torch
    from nvalchemiops.torch.neighbors import neighbor_list

    cell_t = torch.as_tensor(cell, device="cuda")
    pbc_t = torch.tensor([pbc] * 3, device="cuda")
    origin_d = xp.asarray(origin)

    def build(positions):
        # ALCHEMI has no cell origin: shift the positions into the cell (the
        # list is translation invariant).
        p = torch.from_dlpack(positions - origin_d)
        return neighbor_list(p, cutoff, cell=cell_t, pbc=pbc_t,
                             method="cell_list", max_neighbors=max_neighbours,
                             return_neighbor_list=not matrix)
    return build


def make_matrix_builder(kind, xp, cutoff, origin, cell, pbc, max_neighbours):
    """Return ``build(positions) -> (nbr, count, extra)`` for the neighbour
    matrix: ``extra`` is the distance vectors ``D`` (matscipy-neighbours), the
    cell shifts of a periodic box (ALCHEMI), or ``None``. Capacity overflow is
    checked by :func:`check_capacity`, outside the timed loop."""
    if kind == "matscipy":
        from matscipy_neighbours import neighbour_matrix

        def build(positions):
            nbr, D, count = neighbour_matrix(
                positions=positions, cell=cell, cell_origin=origin, pbc=pbc,
                cutoff=cutoff, max_neighbours=max_neighbours)
            return nbr, count, D
        return build

    if kind == "alchemi":
        alchemi = alchemi_builder(xp, cutoff, origin, cell, pbc, max_neighbours,
                                  matrix=True)

        def build(positions):
            nbr, count, S = (xp.from_dlpack(t) for t in alchemi(positions))
            return nbr, count, (S if pbc else None)
        return build

    raise SystemExit(f"--format matrix supports matscipy and alchemi, not {kind}")


def check_capacity(count, max_neighbours):
    """ALCHEMI's matrix output truncates rows that overflow the capacity
    without raising (the count stays exact; its list output does raise), so
    check the counts explicitly."""
    most = int(count.max())
    if most > max_neighbours:
        raise SystemExit(f"--max-neighbours={max_neighbours} is too small "
                         f"(an atom has {most} neighbours)")


def write_xyz(handle, positions_host, comment):
    n = positions_host.shape[0]
    handle.write(f"{n}\n{comment}\n")
    for p in positions_host:
        handle.write(f"Ar {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--neighbours",
                    choices=["matscipy", "matscipy-classic", "vesin", "alchemi"],
                    default="matscipy",
                    help="neighbour-list builder (matscipy = this library; "
                         "matscipy-classic = the matscipy 1.2.0 package, CPU "
                         "only; alchemi = NVIDIA ALCHEMI, GPU only)")
    ap.add_argument("--format", choices=["list", "matrix"], default="list",
                    help="pair list (one thread per pair) or fixed-capacity "
                         "neighbour matrix (one thread per atom); matrix needs "
                         "--neighbours matscipy or alchemi")
    ap.add_argument("--max-neighbours", type=int, default=96,
                    help="row capacity of the neighbour matrix (also sizes "
                         "ALCHEMI's internal matrix in list format)")
    ap.add_argument("--system", choices=["droplet", "liquid"], default="droplet",
                    help="droplet in vacuum (non-periodic) or bulk liquid in a "
                         "periodic box")
    ap.add_argument("--atoms", type=int, default=2048,
                    help="number of atoms")
    ap.add_argument("--lattice", type=float, default=1.6,
                    help="droplet FCC lattice constant")
    ap.add_argument("--density", type=float, default=0.8442,
                    help="liquid number density (reduced units); the default "
                         "with kT=0.7 is the Verlet (1967) liquid state point")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kT", type=float, default=0.7)
    ap.add_argument("--cutoff", type=float, default=2.5)
    ap.add_argument("--out", default=None, help="optional XYZ trajectory")
    ap.add_argument("--write-every", type=int, default=50,
                    help="write a trajectory frame every N steps (0: none)")
    args = ap.parse_args()

    if args.neighbours == "matscipy-classic" and args.device == "gpu":
        raise SystemExit("matscipy-classic (the matscipy 1.2.0 package) is "
                         "CPU only; use --device cpu")
    if args.neighbours == "alchemi" and args.device == "cpu":
        raise SystemExit("alchemi is benchmarked on the GPU only; use "
                         "--device gpu")
    on_gpu = args.device == "gpu"
    wp_device = "cuda:0" if on_gpu else "cpu"
    wp.init()
    wp.set_device(wp_device)
    if on_gpu:
        import cupy as xp
    else:
        xp = np

    # Positions: a single device buffer, shared zero-copy with Warp.
    if args.system == "liquid":
        pos_host, L = fcc_liquid_n(xp, args.atoms, args.density)
        origin, cell = periodic_box(L)
        pbc = True
    else:
        pos_host = fcc_droplet(xp, args.atoms, args.lattice)
        origin, cell = fixed_box(pos_host, args.cutoff)
        pbc = False
    positions = xp.ascontiguousarray(pos_host)
    n = int(positions.shape[0])
    velocities = xp.zeros_like(positions)

    pos_wp = wp.from_dlpack(positions, dtype=vec3d)
    vel_wp = wp.from_dlpack(velocities, dtype=vec3d)
    forces_wp = wp.zeros(n, dtype=vec3d, device=wp_device)
    energy_wp = wp.zeros(1, dtype=wp.float64, device=wp_device)

    lc = langevin_constants(args.dt, args.gamma, args.kT)
    matrix = args.format == "matrix"
    if matrix:
        build_matrix = make_matrix_builder(args.neighbours, xp, args.cutoff,
                                           origin, cell, pbc,
                                           args.max_neighbours)

        def build_nl(positions):
            nbr, count, extra = build_matrix(positions)
            return nbr, count, extra, None
    else:
        build_nl = make_neighbour_builder(args.neighbours, xp, on_gpu,
                                          args.cutoff, origin, cell, pbc,
                                          args.max_neighbours)
    cutoff_sq = wp.float64(args.cutoff ** 2)
    cell_t = wp.mat33d(np.ascontiguousarray(cell.T))   # S @ cell == cell.T @ S

    def matrix_force_pass(nbr, count, extra):
        energy_wp.zero_()
        count_wp = wp.from_dlpack(count)
        if extra is None:
            launch_forces(lj_forces_matrix, n,
                          [pos_wp, wp.from_dlpack(nbr), count_wp, n, cutoff_sq,
                           forces_wp, energy_wp],
                          wp_device)
        elif args.neighbours == "matscipy":        # extra = distance vectors D
            launch_forces(lj_forces_matrix_D, n,
                          [wp.from_dlpack(extra, dtype=vec3d), count_wp, n,
                           cutoff_sq, forces_wp, energy_wp],
                          wp_device)
        else:                                      # extra = cell shifts S
            launch_forces(lj_forces_matrix_pbc, n,
                          [pos_wp, wp.from_dlpack(nbr), count_wp,
                           wp.from_dlpack(extra, dtype=wp.vec3i), n, cell_t,
                           cutoff_sq, forces_wp, energy_wp],
                          wp_device)

    def force_pass(i_wp, j_wp, s_wp, npairs):
        if npairs is None:                         # neighbour-matrix format
            matrix_force_pass(i_wp, j_wp, s_wp)
            return
        forces_wp.zero_()
        energy_wp.zero_()
        if s_wp is None:
            launch_forces(lj_forces, npairs,
                          [pos_wp, i_wp, j_wp, npairs, cutoff_sq, forces_wp,
                           energy_wp],
                          wp_device)
        else:
            shift_wp = wp.from_dlpack(s_wp, dtype=wp.vec3i)
            launch_forces(lj_forces_pbc, npairs,
                          [pos_wp, i_wp, j_wp, shift_wp, npairs, cell_t,
                           cutoff_sq, forces_wp, energy_wp],
                          wp_device)

    def drift(step):
        wp.launch(langevin_drift, dim=n,
                  inputs=[pos_wp, vel_wp, forces_wp,
                          wp.float64(lc["c0"]), wp.float64(lc["c1"]),
                          wp.float64(lc["c2"]), wp.float64(lc["dt"]),
                          wp.float64(lc["mass"]), wp.float64(lc["sr"]),
                          wp.float64(lc["sv"]), wp.float64(lc["crv"]),
                          wp.int32(step + 1)],
                  device=wp_device)

    def kick():
        wp.launch(langevin_kick, dim=n,
                  inputs=[vel_wp, forces_wp, wp.float64(lc["c2"]),
                          wp.float64(lc["dt"]), wp.float64(lc["mass"])],
                  device=wp_device)

    print(f"device={args.device}  neighbours={args.neighbours}  "
          f"format={args.format}  system={args.system}  atoms={n}")

    # Warm-up: build once and trigger the one-time Warp kernel compilation.
    iw, jw, sw, npairs = build_nl(positions)
    force_pass(iw, jw, sw, npairs)
    wp.synchronize_device(wp_device)
    if matrix:
        check_capacity(jw, args.max_neighbours)
        npairs_report = int(jw.sum())
    else:
        npairs_report = npairs
    # The kernel already carries the 1/2 factor for directed pairs.
    print(f"pairs~{npairs_report}  E_pot={float(energy_wp.numpy()[0]):.6f}")
    # One complete (untimed) step so both integrator kernels are compiled too.
    drift(0)
    iw, jw, sw, npairs = build_nl(positions)
    force_pass(iw, jw, sw, npairs)
    kick()
    wp.synchronize_device(wp_device)

    out = open(args.out, "w") if args.out else None
    timer = Timer()
    for step in range(args.steps):
        with timer("integrate"):
            drift(step + 1)
            wp.synchronize_device(wp_device)
        with timer("neighbour list"):
            iw, jw, sw, npairs = build_nl(positions)
            wp.synchronize_device(wp_device)
        with timer("LJ forces"):
            force_pass(iw, jw, sw, npairs)
            wp.synchronize_device(wp_device)
        with timer("integrate"):
            kick()
            wp.synchronize_device(wp_device)
        if (out is not None and args.write_every > 0
                and step % args.write_every == 0):
            e = float(energy_wp.numpy()[0])
            host = xp.asnumpy(positions) if on_gpu else np.asarray(positions)
            vel_host = xp.asnumpy(velocities) if on_gpu else np.asarray(velocities)
            T = float((vel_host * vel_host).sum()) / (3.0 * n)
            write_xyz(out, host, f"step={step} E_pot={e:.4f} T={T:.4f}")
    if out is not None:
        out.close()

    if matrix:
        check_capacity(jw, args.max_neighbours)
        npairs = int(jw.sum())
    timer.print_summary()
    nl_ms = timer.get_time("neighbour list") / args.steps * 1e3
    force_ms = timer.get_time("LJ forces") / args.steps * 1e3
    int_ms = timer.get_time("integrate") / args.steps * 1e3
    total_ms = nl_ms + force_ms + int_ms
    print(f"nl_ms={nl_ms:.4f}  force_ms={force_ms:.4f}  integrate_ms={int_ms:.4f}")
    print(f"steps={args.steps}  {total_ms:.3f} ms/step  "
          f"{total_ms * 1e6 / npairs:.1f} ns/pair")


if __name__ == "__main__":
    main()
