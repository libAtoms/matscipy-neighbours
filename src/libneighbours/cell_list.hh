/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#ifndef MATSCIPY_CELL_LIST_HH
#define MATSCIPY_CELL_LIST_HH

#include <vector>

#include "types.hh"

namespace matscipy {

/*
 * Atoms binned into an n1 x n2 x n3 grid, stored in CSR (compressed sparse row)
 * form. This is the "index sort" data structure (Ihmsen et al. 2010): atoms are
 * grouped by cell so each cell's members are a contiguous slice of
 * `sorted_atom`, which replaces the serial per-cell linked list and is built
 * from data-parallel primitives (histogram, prefix sum, scatter).
 *
 * The atoms in linear cell `c` are
 *     sorted_atom[cell_start[c] .. cell_start[c + 1])
 * with `c = c1 + n1 * (c2 + n2 * c3)`.
 */
struct CellList {
    index_t n1 = 0, n2 = 0, n3 = 0;
    index_t ncells = 0;
    std::vector<index_t> cell_start;   /* size ncells + 1 */
    std::vector<index_t> sorted_atom;  /* size nat */
};

/*
 * Bin atoms into an n1 x n2 x n3 grid via counting sort.
 *
 * Periodic directions wrap the cell index, non-periodic directions clamp it,
 * matching the convention used by neighbour_list(). `cl` is overwritten.
 */
void build_cell_list(const real_t cell_origin[3], const real_t inv_cell[9],
                     const bool pbc[3], index_t nat, const real_t *positions,
                     index_t n1, index_t n2, index_t n3, CellList &cl);

}  // namespace matscipy

#endif
