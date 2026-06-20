/*
 * Shared helpers for the Lennard-Jones Langevin droplet examples: an FCC
 * droplet generator, the Allen-Tildesley Langevin coefficients, and an XYZ
 * writer. Reduced LJ units (epsilon = sigma = mass = kB = 1).
 */

#ifndef LJ_COMMON_HH
#define LJ_COMMON_HH

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "types.hh"

namespace lj {

using matscipy::index_t;
using matscipy::real_t;

/* A roughly spherical FCC cluster centred at the origin (positions as 3*n). */
inline index_t fcc_droplet(int ncells, real_t lattice,
                           std::vector<real_t> &pos) {
    const real_t basis[4][3] = {
        {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
    std::vector<real_t> cand;
    for (int ix = -ncells; ix <= ncells; ix++)
        for (int iy = -ncells; iy <= ncells; iy++)
            for (int iz = -ncells; iz <= ncells; iz++)
                for (auto &b : basis) {
                    cand.push_back((ix + b[0]) * lattice);
                    cand.push_back((iy + b[1]) * lattice);
                    cand.push_back((iz + b[2]) * lattice);
                }
    real_t c[3] = {0, 0, 0};
    const index_t m = (index_t)(cand.size() / 3);
    for (index_t a = 0; a < m; a++)
        for (int k = 0; k < 3; k++) c[k] += cand[3 * a + k];
    for (int k = 0; k < 3; k++) c[k] /= m;

    const real_t radius = ncells * lattice;
    pos.clear();
    for (index_t a = 0; a < m; a++) {
        real_t x = cand[3 * a] - c[0], y = cand[3 * a + 1] - c[1],
               z = cand[3 * a + 2] - c[2];
        if (x * x + y * y + z * z <= radius * radius) {
            pos.push_back(x);
            pos.push_back(y);
            pos.push_back(z);
        }
    }
    return (index_t)(pos.size() / 3);
}

/* Allen-Tildesley Langevin coefficients for timestep dt, friction gamma and
   temperature kT. Plain data so it can be passed to a device kernel by value. */
struct Langevin {
    real_t c0, c1, c2, sr, sv, crv, dt, mass;
};

inline Langevin langevin_constants(real_t dt, real_t gamma, real_t kT,
                                   real_t mass = 1.0) {
    Langevin l;
    l.dt = dt;
    l.mass = mass;
    const real_t D = kT / (mass * gamma);
    l.c0 = std::exp(-gamma * dt);
    l.c1 = (1.0 - l.c0) / (gamma * dt);
    l.c2 = (1.0 - l.c1) / (gamma * dt);
    l.sr = std::sqrt(dt * D * (2.0 - (3.0 - 4.0 * l.c0 + l.c0 * l.c0) /
                                         (gamma * dt)));
    l.sv = std::sqrt(gamma * D * (1.0 - l.c0 * l.c0));
    l.crv = D * (1.0 - l.c0) * (1.0 - l.c0) / (l.sr * l.sv);
    return l;
}

inline void write_xyz(std::ofstream &o, const real_t *pos, index_t n,
                      const std::string &comment) {
    o << n << "\n" << comment << "\n";
    for (index_t a = 0; a < n; a++)
        o << "Ar " << pos[3 * a] << " " << pos[3 * a + 1] << " "
          << pos[3 * a + 2] << "\n";
}

/* A cubic box centred at the origin large enough to enclose the droplet for the
   run (non-periodic; the droplet is self-bound, so a generous fixed box is
   sufficient). Fills origin/cell/inv_cell. */
inline void fixed_box(int ncells, real_t lattice, real_t cutoff,
                      real_t origin[3], real_t cell[9], real_t inv_cell[9]) {
    const real_t half = ncells * lattice + cutoff + 5.0;
    const real_t L = 2.0 * half;
    for (int k = 0; k < 3; k++) origin[k] = -half;
    for (int k = 0; k < 9; k++) {
        cell[k] = 0;
        inv_cell[k] = 0;
    }
    cell[0] = cell[4] = cell[8] = L;
    inv_cell[0] = inv_cell[4] = inv_cell[8] = 1.0 / L;
}

}  // namespace lj

#endif
