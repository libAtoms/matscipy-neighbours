/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#ifndef MATSCIPY_TOOLS_HH
#define MATSCIPY_TOOLS_HH

#include <cmath>

#include "types.hh"

namespace matscipy {

/* c = a x b for 3-vectors. */
void cross_product(const real_t *a, const real_t *b, real_t *c);

/* Euclidean norm (length) of a 3-vector. */
real_t norm(const real_t *a);

/* The cell-indexing leaf helpers are shared by the CPU loops and the GPU
   kernels (host+device, header-inline) — keeping one definition each. */

/* vout = mat . vin, with mat a row-major 3x3 matrix. */
MATSCIPY_HD inline void mat_mul_vec(const real_t *mat, const real_t *vin,
                                    real_t *vout) {
    for (int i = 0; i < 3; i++) {
        vout[i] = 0.0;
        for (int j = 0; j < 3; j++) vout[i] += mat[3 * i + j] * vin[j];
    }
}

/* Map cell index i back into [0, n) by shifting by multiples of n (periodic). */
MATSCIPY_HD inline index_t bin_wrap(index_t i, index_t n) {
    while (i < 0) i += n;
    while (i >= n) i -= n;
    return i;
}

/* Clamp cell index i into [0, n) (non-periodic). */
MATSCIPY_HD inline index_t bin_trunc(index_t i, index_t n) {
    if (i < 0)
        i = 0;
    else if (i >= n)
        i = n - 1;
    return i;
}

/* Map a Cartesian position to (unwrapped) integer cell indices. */
MATSCIPY_HD inline void position_to_cell_index(const real_t *cell_origin,
                                               const real_t *inv_cell,
                                               const real_t *ri, index_t n1,
                                               index_t n2, index_t n3,
                                               index_t *c1, index_t *c2,
                                               index_t *c3) {
    real_t dri[3], si[3];
    for (int i = 0; i < 3; i++) dri[i] = ri[i] - cell_origin[i];
    mat_mul_vec(inv_cell, dri, si);
    *c1 = static_cast<index_t>(std::floor(si[0] * n1));
    *c2 = static_cast<index_t>(std::floor(si[1] * n2));
    *c3 = static_cast<index_t>(std::floor(si[2] * n3));
}

}  // namespace matscipy

#endif
