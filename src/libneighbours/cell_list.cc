/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#include "cell_list.hh"

#include <cassert>

#include "tools.hh"

namespace matscipy {

void build_cell_list(const real_t cell_origin[3], const real_t inv_cell[9],
                     const bool pbc[3], index_t nat, const real_t *positions,
                     index_t n1, index_t n2, index_t n3, CellList &cl) {
    const index_t ncells = n1 * n2 * n3;

    cl.n1 = n1;
    cl.n2 = n2;
    cl.n3 = n3;
    cl.ncells = ncells;

    /* Linear cell index of every atom (with PBC wrap / non-PBC clamp). */
    std::vector<index_t> cell_of(nat);

    /* Histogram of atoms per cell, offset by one so the prefix sum below turns
       it directly into row-start offsets. */
    cl.cell_start.assign(ncells + 1, 0);
    for (index_t i = 0; i < nat; i++) {
        index_t c1, c2, c3;
        position_to_cell_index(cell_origin, inv_cell, &positions[3 * i], n1, n2,
                               n3, &c1, &c2, &c3);

        c1 = pbc[0] ? bin_wrap(c1, n1) : bin_trunc(c1, n1);
        c2 = pbc[1] ? bin_wrap(c2, n2) : bin_trunc(c2, n2);
        c3 = pbc[2] ? bin_wrap(c3, n3) : bin_trunc(c3, n3);

        index_t ci = c1 + n1 * (c2 + n2 * c3);
        assert(ci >= 0 && ci < ncells);

        cell_of[i] = ci;
        cl.cell_start[ci + 1]++;
    }

    /* Prefix sum: cell_start[c] becomes the first index of cell c in
       sorted_atom, and cell_start[ncells] == nat. */
    for (index_t c = 0; c < ncells; c++) {
        cl.cell_start[c + 1] += cl.cell_start[c];
    }

    /* Scatter atoms into their cell's slice. */
    std::vector<index_t> cursor(cl.cell_start.begin(),
                                cl.cell_start.begin() + ncells);
    cl.sorted_atom.resize(nat);
    for (index_t i = 0; i < nat; i++) {
        cl.sorted_atom[cursor[cell_of[i]]++] = i;
    }
}

}  // namespace matscipy
