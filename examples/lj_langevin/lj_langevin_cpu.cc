/*
 * Lennard-Jones liquid droplet with a Langevin thermostat — CPU.
 *
 * Uses the neighbour list for connectivity (the `ij` index pairs) and computes
 * the LJ forces in a single fused pass that recomputes the distance vectors,
 * accumulating each pair's force onto atom i (the list is grouped by i, so the
 * per-atom loop parallelises without races). Output is an XYZ trajectory.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

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
    const int ncells = (int)argd(argc, argv, "--ncells", 6);
    const real_t lattice = argd(argc, argv, "--lattice", 1.6);
    const int steps = (int)argd(argc, argv, "--steps", 300);
    const real_t dt = argd(argc, argv, "--dt", 0.005);
    const real_t gamma = argd(argc, argv, "--gamma", 1.0);
    const real_t kT = argd(argc, argv, "--kT", 0.7);
    const real_t cutoff = argd(argc, argv, "--cutoff", 2.5);
    const int write_every = (int)argd(argc, argv, "--write-every", 50);
    const char *outfile = args(argc, argv, "--out", "traj_cpu.xyz");

    std::vector<real_t> pos;
    const index_t n = lj::fcc_droplet(ncells, lattice, pos);
    std::vector<real_t> vel(3 * n, 0.0), f(3 * n, 0.0);

    real_t origin[3], cell[9], inv_cell[9];
    lj::fixed_box(ncells, lattice, cutoff, origin, cell, inv_cell);
    const bool pbc[3] = {false, false, false};
    const lj::Langevin lc = lj::langevin_constants(dt, gamma, kT);
    const real_t rc2 = cutoff * cutoff;

    std::vector<index_t> off(n + 1);
    index_t npairs = 0;
    auto compute_forces = [&]() {
        NeighbourList nl;
        neighbour_list(QUANTITY_FIRST | QUANTITY_SECOND, origin, cell, inv_cell,
                       pbc, n, pos.data(), cutoff, nullptr, nullptr, 0, nullptr,
                       nl);
        npairs = nl.npairs;
        /* CSR offsets from the i-sorted pair list, so the force loop is one
           thread per atom (no atomic accumulation). */
        std::fill(off.begin(), off.end(), 0);
        for (index_t p = 0; p < nl.npairs; p++) off[nl.first[p] + 1]++;
        for (index_t a = 0; a < n; a++) off[a + 1] += off[a];

        std::fill(f.begin(), f.end(), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
        for (index_t i = 0; i < n; i++) {
            real_t fx = 0, fy = 0, fz = 0;
            const real_t xi = pos[3 * i], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
            for (index_t p = off[i]; p < off[i + 1]; p++) {
                const index_t j = nl.secnd[p];
                const real_t dx = pos[3 * j] - xi, dy = pos[3 * j + 1] - yi,
                             dz = pos[3 * j + 2] - zi;
                const real_t r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= rc2) continue;
                const real_t ir2 = 1.0 / r2, ir6 = ir2 * ir2 * ir2;
                const real_t coef = -24.0 * ir2 * ir6 * (2.0 * ir6 - 1.0);
                fx += coef * dx;
                fy += coef * dy;
                fz += coef * dz;
            }
            f[3 * i] = fx;
            f[3 * i + 1] = fy;
            f[3 * i + 2] = fz;
        }
    };

    std::mt19937 rng(12345);
    std::normal_distribution<real_t> gauss(0.0, 1.0);
    auto langevin = [&]() {
        const real_t k = std::sqrt(1.0 - lc.crv * lc.crv);
        for (index_t a = 0; a < 3 * n; a++) {
            const real_t g1 = gauss(rng), g2 = gauss(rng);
            const real_t gr = lc.sr * g1;
            const real_t gv = lc.sv * (lc.crv * g1 + k * g2);
            const real_t fm = f[a] / lc.mass;
            pos[a] += lc.c1 * lc.dt * vel[a] + lc.c2 * lc.dt * lc.dt * fm + gr;
            vel[a] += (lc.c0 - 1.0) * vel[a] + lc.c1 * lc.dt * fm + gv;
        }
    };

    compute_forces();
    std::printf("device=cpu  atoms=%d  pairs~%d\n", (int)n, (int)npairs);

    std::ofstream out(outfile);
    const auto t0 = clock_type::now();
    for (int step = 0; step < steps; step++) {
        langevin();
        compute_forces();
        if (step % write_every == 0)
            lj::write_xyz(out, pos.data(), n, "step=" + std::to_string(step));
    }
    const double elapsed =
        std::chrono::duration<double>(clock_type::now() - t0).count();
    const double per_step = elapsed / steps;
    std::printf("steps=%d  total=%.3fs  %.3f ms/step  %.1f ns/pair\n", steps,
                elapsed, per_step * 1e3, per_step * 1e9 / npairs);
    return 0;
}
