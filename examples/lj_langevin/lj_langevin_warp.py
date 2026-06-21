#!/usr/bin/env python3
"""Lennard-Jones liquid droplet with a Langevin thermostat — NVIDIA Warp.

The Lennard-Jones force/energy and the Langevin integrator are written as
`warp` kernels (compiled once, then launched every step), but the neighbour
list is built by *this* library — `matscipy_neighbours` — or, for comparison,
by `vesin` (https://github.com/luthaf/vesin). Select the builder with
``--neighbours {matscipy,vesin}``.

The point is interop: positions live in a single device buffer (CuPy on the
GPU, NumPy on the CPU); Warp wraps it zero-copy through DLPack, and the
neighbour-list builder reads the very same buffer. The builder returns the
directed pair list ``(i, j)`` (full list, sorted by ``i``); the fused Warp
kernel recomputes the distance for each pair and atomically accumulates the
force on ``i`` — never materialising per-pair arrays.

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

import numpy as np
import warp as wp
from muTimer import Timer

vec3d = wp.vec3d


# --------------------------------------------------------------------------- #
# Warp kernels
# --------------------------------------------------------------------------- #
@wp.kernel
def lj_forces(pos: wp.array(dtype=vec3d),
              ii: wp.array(dtype=wp.int32),
              jj: wp.array(dtype=wp.int32),
              cutoff_sq: wp.float64,
              forces: wp.array(dtype=vec3d),
              energy: wp.array(dtype=wp.float64)):
    """One thread per directed pair. ``forces`` and ``energy`` must be zeroed
    beforehand; the full (both-directions) list means the force is accumulated
    on ``i`` only and the energy carries the 1/2 double-counting factor."""
    p = wp.tid()
    i = ii[p]
    j = jj[p]
    dr = pos[j] - pos[i]
    r2 = wp.dot(dr, dr)
    if r2 < cutoff_sq:
        inv_r2 = wp.float64(1.0) / r2
        inv_r6 = inv_r2 * inv_r2 * inv_r2
        coef = wp.float64(-24.0) * inv_r2 * inv_r6 * (wp.float64(2.0) * inv_r6 - wp.float64(1.0))
        wp.atomic_add(forces, i, coef * dr)
        wp.atomic_add(energy, 0, wp.float64(2.0) * inv_r6 * (inv_r6 - wp.float64(1.0)))


@wp.kernel
def langevin(pos: wp.array(dtype=vec3d),
             vel: wp.array(dtype=vec3d),
             forces: wp.array(dtype=vec3d),
             c0: wp.float64, c1: wp.float64, c2: wp.float64,
             dt: wp.float64, mass: wp.float64,
             sr: wp.float64, sv: wp.float64, crv: wp.float64,
             seed: wp.int32):
    """One Langevin update (Allen-Tildesley scheme), one thread per atom."""
    a = wp.tid()
    st = wp.rand_init(seed, a)
    g1 = vec3d(wp.float64(wp.randn(st)), wp.float64(wp.randn(st)), wp.float64(wp.randn(st)))
    g2 = vec3d(wp.float64(wp.randn(st)), wp.float64(wp.randn(st)), wp.float64(wp.randn(st)))
    gr = sr * g1
    gv = sv * (crv * g1 + wp.sqrt(wp.float64(1.0) - crv * crv) * g2)
    v = vel[a]
    fm = forces[a] / mass
    pos[a] = pos[a] + c1 * dt * v + c2 * dt * dt * fm + gr
    vel[a] = v + (c0 - wp.float64(1.0)) * v + c1 * dt * fm + gv


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


def make_neighbour_builder(kind, xp, on_gpu, cutoff, origin, cell):
    """Return ``build(positions) -> (i_int32, j_int32, npairs)`` for the chosen
    backend. Both return the device-resident directed pair list."""
    if kind == "matscipy":
        from matscipy_neighbours import neighbour_list

        def build(positions):
            i, j = neighbour_list("ij", positions=positions, cell=cell,
                                  cell_origin=origin, pbc=False, cutoff=cutoff)
            return i.astype(xp.int32), j.astype(xp.int32), int(i.shape[0])
        return build

    if kind == "vesin":
        import vesin

        box = xp.asarray(cell)
        nl = vesin.NeighborList(cutoff=cutoff, full_list=True, sorted=True)

        def build(positions):
            i, j = nl.compute(points=positions, box=box, periodic=False,
                              quantities="ij")
            return i.astype(xp.int32), j.astype(xp.int32), int(i.shape[0])
        return build

    if kind == "matscipy-classic":
        # The classic matscipy package (pinned to 1.2.0). CPU/host only; shift
        # positions by -origin so they sit inside the cell for its C extension.
        from matscipy.neighbours import neighbour_list as ms_nl
        pbc = [False, False, False]

        def build(positions):
            i, j = ms_nl("ij", positions=positions - origin, cell=cell,
                         pbc=pbc, cutoff=cutoff)
            return i.astype(xp.int32), j.astype(xp.int32), int(i.shape[0])
        return build

    raise SystemExit(f"unknown neighbour backend: {kind}")


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
                    choices=["matscipy", "matscipy-classic", "vesin"],
                    default="matscipy",
                    help="neighbour-list builder (matscipy = this library; "
                         "matscipy-classic = the matscipy 1.2.0 package, CPU only)")
    ap.add_argument("--atoms", type=int, default=2048,
                    help="target number of atoms in the droplet")
    ap.add_argument("--lattice", type=float, default=1.6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kT", type=float, default=0.7)
    ap.add_argument("--cutoff", type=float, default=2.5)
    ap.add_argument("--out", default=None, help="optional XYZ trajectory")
    ap.add_argument("--write-every", type=int, default=50)
    args = ap.parse_args()

    if args.neighbours == "matscipy-classic" and args.device == "gpu":
        raise SystemExit("matscipy-classic (the matscipy 1.2.0 package) is "
                         "CPU only; use --device cpu")
    on_gpu = args.device == "gpu"
    wp_device = "cuda:0" if on_gpu else "cpu"
    wp.init()
    wp.set_device(wp_device)
    if on_gpu:
        import cupy as xp
    else:
        xp = np

    # Positions: a single device buffer, shared zero-copy with Warp.
    pos_host = fcc_droplet(xp, args.atoms, args.lattice)
    positions = xp.ascontiguousarray(pos_host)
    n = int(positions.shape[0])
    velocities = xp.zeros_like(positions)

    pos_wp = wp.from_dlpack(positions, dtype=vec3d)
    vel_wp = wp.from_dlpack(velocities, dtype=vec3d)
    forces_wp = wp.zeros(n, dtype=vec3d, device=wp_device)
    energy_wp = wp.zeros(1, dtype=wp.float64, device=wp_device)

    lc = langevin_constants(args.dt, args.gamma, args.kT)
    origin, cell = fixed_box(pos_host, args.cutoff)
    build_nl = make_neighbour_builder(args.neighbours, xp, on_gpu, args.cutoff,
                                      origin, cell)
    cutoff_sq = wp.float64(args.cutoff ** 2)

    def force_pass(i_wp, j_wp, npairs):
        forces_wp.zero_()
        energy_wp.zero_()
        wp.launch(lj_forces, dim=npairs,
                  inputs=[pos_wp, i_wp, j_wp, cutoff_sq, forces_wp, energy_wp],
                  device=wp_device)

    def integrate(step):
        wp.launch(langevin, dim=n,
                  inputs=[pos_wp, vel_wp, forces_wp,
                          wp.float64(lc["c0"]), wp.float64(lc["c1"]),
                          wp.float64(lc["c2"]), wp.float64(lc["dt"]),
                          wp.float64(lc["mass"]), wp.float64(lc["sr"]),
                          wp.float64(lc["sv"]), wp.float64(lc["crv"]),
                          wp.int32(step + 1)],
                  device=wp_device)

    print(f"device={args.device}  neighbours={args.neighbours}  atoms={n}")

    # Warm-up: build once and trigger the one-time Warp kernel compilation.
    iw, jw, npairs = build_nl(positions)
    force_pass(iw, jw, npairs)
    integrate(0)
    wp.synchronize_device(wp_device)
    print(f"pairs~{npairs}")

    out = open(args.out, "w") if args.out else None
    timer = Timer()
    for step in range(args.steps):
        with timer("neighbour list"):
            iw, jw, npairs = build_nl(positions)
            wp.synchronize_device(wp_device)
        with timer("LJ forces"):
            force_pass(iw, jw, npairs)
            wp.synchronize_device(wp_device)
        with timer("integrate"):
            integrate(step)
            wp.synchronize_device(wp_device)
        if out is not None and step % args.write_every == 0:
            e = 0.5 * float(energy_wp.numpy()[0])
            host = xp.asnumpy(positions) if on_gpu else np.asarray(positions)
            write_xyz(out, host, f"step={step} E_pot={e:.4f}")
    if out is not None:
        out.close()

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
