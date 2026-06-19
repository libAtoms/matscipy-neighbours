/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#ifndef MATSCIPY_FIRST_NEIGHBOURS_HH
#define MATSCIPY_FIRST_NEIGHBOURS_HH

#include <vector>

#include "types.hh"

namespace matscipy {

/*
 * Build the row-start ("seed") array of a neighbour list whose first index
 * array i_n is sorted and non-decreasing.
 *
 * n      number of rows (atoms); seed must have room for n + 1 entries
 * nn     length of i_n
 * i_n    [nn] sorted first-atom indices
 * seed   [n + 1] output: seed[k] is the position in i_n where atom k starts,
 *        seed[n] == nn. Atoms with no entries keep seed value -1.
 *
 * Handles nn == 0 (empty list) without dereferencing i_n.
 */
void first_neighbours(index_t n, index_t nn, const index_t *i_n, index_t *seed);

/*
 * Given a sorted, contiguous array starting at 0, return the array pointing to
 * the index jumps (the row-start array for the implied number of distinct
 * values). Returned vector has length (number of distinct values) + 1.
 */
std::vector<index_t> get_jump_indicies(index_t nn, const index_t *sorted);

}  // namespace matscipy

#endif
