/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#include "tools.hh"

#include <cmath>

namespace matscipy {

void cross_product(const real_t *a, const real_t *b, real_t *c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

void mat_mul_vec(const real_t *mat, const real_t *vin, real_t *vout) {
    for (int i = 0; i < 3; i++) {
        vout[i] = 0.0;
        for (int j = 0; j < 3; j++) {
            vout[i] += mat[3 * i + j] * vin[j];
        }
    }
}

real_t norm(const real_t *a) {
    return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

index_t bin_wrap(index_t i, index_t n) {
    while (i < 0) i += n;
    while (i >= n) i -= n;
    return i;
}

index_t bin_trunc(index_t i, index_t n) {
    if (i < 0)
        i = 0;
    else if (i >= n)
        i = n - 1;
    return i;
}

void position_to_cell_index(const real_t *cell_origin, const real_t *inv_cell,
                            const real_t *ri, index_t n1, index_t n2, index_t n3,
                            index_t *c1, index_t *c2, index_t *c3) {
    real_t dri[3], si[3];
    for (int i = 0; i < 3; i++) {
        dri[i] = ri[i] - cell_origin[i];
    }
    mat_mul_vec(inv_cell, dri, si);
    *c1 = static_cast<index_t>(std::floor(si[0] * n1));
    *c2 = static_cast<index_t>(std::floor(si[1] * n2));
    *c3 = static_cast<index_t>(std::floor(si[2] * n3));
}

}  // namespace matscipy
