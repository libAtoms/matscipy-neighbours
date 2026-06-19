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

#include "types.hh"

namespace matscipy {

/* c = a x b for 3-vectors. */
void cross_product(const real_t *a, const real_t *b, real_t *c);

/* vout = mat . vin, with mat a row-major 3x3 matrix. */
void mat_mul_vec(const real_t *mat, const real_t *vin, real_t *vout);

/* Euclidean norm (length) of a 3-vector. */
real_t norm(const real_t *a);

/* Map cell index i back into [0, n) by shifting by multiples of n (periodic). */
index_t bin_wrap(index_t i, index_t n);

/* Clamp cell index i into [0, n) (non-periodic). */
index_t bin_trunc(index_t i, index_t n);

/* Map a Cartesian position to (unwrapped) integer cell indices. */
void position_to_cell_index(const real_t *cell_origin, const real_t *inv_cell,
                            const real_t *ri, index_t n1, index_t n2, index_t n3,
                            index_t *c1, index_t *c2, index_t *c3);

}  // namespace matscipy

#endif
