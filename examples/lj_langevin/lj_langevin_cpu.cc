/*
 * Lennard-Jones Langevin dynamics (droplet or periodic liquid) — CPU.
 *
 * Uses the neighbour list for connectivity (the `ij` index pairs) and computes
 * the LJ forces in a single fused pass that recomputes the distance vectors,
 * accumulating each pair's force onto atom i (the list is grouped by i, so the
 * per-atom loop parallelises without races). For the periodic liquid the list
 * also supplies the cell shift `S` of each pair, so a pair across the boundary
 * is evaluated at its periodic image: D = r[j] - r[i] + S @ cell. Output is an
 * XYZ trajectory.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "error.hh"
#include "lj_common.hh"
#include "neighbour_list.hh"

using namespace matscipy;
using clock_type = std::chrono::steady_clock;

static double argd(int argc, char **argv, const char *key, double def) {
    for (int i = 1; i + 1 < argc; i++)
        if (std::strcmp(argv[i], key) == 0) return std::atof(argv[i + 1]);
    return def;
}
static const char *args(int argc, char **argv, const char *key, const char *def) {
    for (int i = 1; i + 1 < argc; i++)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}

int main(int argc, char **argv) {
    const std::string system = args(argc, argv, "--system", "droplet");
    const int ncells = (int)argd(argc, argv, "--ncells", 6);
    const index_t atoms = (index_t)argd(argc, argv, "--atoms", 0);
    const real_t lattice = argd(argc, argv, "--lattice", 1.6);
    const real_t density = argd(argc, argv, "--density", 0.8442);
    const int steps = (int)argd(argc, argv, "--steps", 300);
    const real_t dt = argd(argc, argv, "--dt", 0.005);
    const real_t gamma = argd(argc, argv, "--gamma", 1.0);
    const real_t kT = argd(argc, argv, "--kT", 0.7);
    const real_t cutoff = argd(argc, argv, "--cutoff", 2.5);
    const int write_every = (int)argd(argc, argv, "--write-every", 50);
    const char *outfile = args(argc, argv, "--out", "traj_cpu.xyz");

    std::vector<real_t> pos;
    real_t origin[3], cell[9], inv_cell[9];
    bool periodic;
    const index_t n = lj::make_system(system, atoms, ncells, lattice, density,
                                      cutoff, pos, origin, cell, inv_cell,
                                      periodic);
    std::vector<real_t> vel(3 * n, 0.0), f(3 * n, 0.0);

    const bool pbc[3] = {periodic, periodic, periodic};
    const int quantities = QUANTITY_FIRST | QUANTITY_SECOND |
                           (periodic ? QUANTITY_SHIFT : 0);
    const lj::Langevin lc = lj::langevin_constants(dt, gamma, kT);
    const real_t rc2 = cutoff * cutoff;

    std::vector<index_t> off(n + 1);
    index_t npairs = 0;
    real_t energy = 0.0;
    auto compute_forces = [&]() {
        NeighbourList nl;
        if (neighbour_list(quantities, origin, cell, inv_cell, pbc, n, pos.data(),
                           cutoff, nullptr, nullptr, 0, nullptr,
                           nl) != NL_SUCCESS) {
            std::fprintf(stderr, "neighbour list failed: %s\n", error_string);
            std::exit(1);
        }
        npairs = nl.npairs;
        /* Shifts are only requested (and only non-zero) for the periodic box. */
        const index_t *shift = periodic ? nl.shift.data() : nullptr;
        /* CSR offsets from the i-sorted pair list, so the force loop is one
           thread per atom (no atomic accumulation). */
        std::fill(off.begin(), off.end(), 0);
        for (index_t p = 0; p < nl.npairs; p++) off[nl.first[p] + 1]++;
        for (index_t a = 0; a < n; a++) off[a + 1] += off[a];

        std::fill(f.begin(), f.end(), 0.0);
        real_t epot = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : epot)
#endif
        for (index_t i = 0; i < n; i++) {
            real_t fx = 0, fy = 0, fz = 0;
            const real_t xi = pos[3 * i], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
            for (index_t p = off[i]; p < off[i + 1]; p++) {
                const index_t j = nl.secnd[p];
                real_t dx = pos[3 * j] - xi, dy = pos[3 * j + 1] - yi,
                       dz = pos[3 * j + 2] - zi;
                if (shift) {
                    const real_t s0 = shift[3 * p], s1 = shift[3 * p + 1],
                                 s2 = shift[3 * p + 2];
                    dx += s0 * cell[0] + s1 * cell[3] + s2 * cell[6];
                    dy += s0 * cell[1] + s1 * cell[4] + s2 * cell[7];
                    dz += s0 * cell[2] + s1 * cell[5] + s2 * cell[8];
                }
                const real_t r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc2) continue;
                const real_t ir2 = 1.0 / r2, ir6 = ir2 * ir2 * ir2;
                const real_t coef = -24.0 * ir2 * ir6 * (2.0 * ir6 - 1.0);
                fx += coef * dx;
                fy += coef * dy;
                fz += coef * dz;
                epot += 2.0 * ir6 * (ir6 - 1.0);  /* 4 eps (...) / 2: directed pairs */
            }
            f[3 * i] = fx;
            f[3 * i + 1] = fy;
            f[3 * i + 2] = fz;
        }
        energy = epot;
    };

    /* Allen-Tildesley Langevin step: drift (positions, friction, noise and the
       half kick with the old forces), then forces at the new positions, then
       the second half kick. Reduces to velocity Verlet for gamma -> 0. */
    std::mt19937 rng(12345);
    std::normal_distribution<real_t> gauss(0.0, 1.0);
    auto drift = [&]() {
        const real_t k = std::sqrt(1.0 - lc.crv * lc.crv);
        for (index_t a = 0; a < 3 * n; a++) {
            const real_t g1 = gauss(rng), g2 = gauss(rng);
            const real_t gr = lc.sr * g1;
            const real_t gv = lc.sv * (lc.crv * g1 + k * g2);
            const real_t fm = f[a] / lc.mass;
            pos[a] += lc.c1 * lc.dt * vel[a] + lc.c2 * lc.dt * lc.dt * fm + gr;
            vel[a] = lc.c0 * vel[a] + (lc.c1 - lc.c2) * lc.dt * fm + gv;
        }
    };
    auto kick = [&]() {
        for (index_t a = 0; a < 3 * n; a++)
            vel[a] += lc.c2 * lc.dt * f[a] / lc.mass;
    };
    auto temperature = [&]() {
        real_t k2 = 0.0;
        for (index_t a = 0; a < 3 * n; a++) k2 += vel[a] * vel[a];
        return k2 / (3.0 * n);
    };

    compute_forces();
    std::printf("device=cpu  system=%s  atoms=%lld  pairs~%lld  E_pot=%.6f\n",
                system.c_str(), (long long)n, (long long)npairs, energy);

    std::ofstream out(outfile);
    const auto t0 = clock_type::now();
    for (int step = 0; step < steps; step++) {
        drift();
        compute_forces();
        kick();
        if (write_every > 0 && step % write_every == 0)  /* 0: no trajectory */
            lj::write_xyz(out, pos.data(), n,
                          "step=" + std::to_string(step) +
                              " E_pot=" + std::to_string(energy) +
                              " T=" + std::to_string(temperature()));
    }
    const double elapsed =
        std::chrono::duration<double>(clock_type::now() - t0).count();
    const double per_step = elapsed / std::max(steps, 1);
    std::printf("steps=%d  total=%.3fs  %.3f ms/step  %.1f ns/pair\n", steps,
                elapsed, per_step * 1e3, per_step * 1e9 / npairs);
    return 0;
}
