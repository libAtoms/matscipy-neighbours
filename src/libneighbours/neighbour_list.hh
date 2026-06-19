/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#ifndef MATSCIPY_NEIGHBOUR_LIST_HH
#define MATSCIPY_NEIGHBOUR_LIST_HH

#include <vector>

#include "cell_list.hh"
#include "types.hh"

namespace matscipy {

/* Result of a neighbour-list computation. Only the buffers whose quantity flag
   was requested are filled; the rest stay empty. `npairs` is the number of
   neighbour pairs found. Distance vectors and shifts are stored row-major with
   three entries per pair. */
struct NeighbourList {
    std::vector<index_t> first;    /* [npairs]      i indices */
    std::vector<index_t> secnd;    /* [npairs]      j indices */
    std::vector<real_t> distvec;   /* [3 * npairs]  distance vectors */
    std::vector<real_t> absdist;   /* [npairs]      absolute distances */
    std::vector<index_t> shift;    /* [3 * npairs]  cell shifts */
    index_t npairs = 0;
};

/*
 * Build a neighbour list by spatial binning. Python-free.
 *
 * quantities          bitwise-or of Quantity flags selecting outputs
 * cell_origin[3]      lower-left corner of the simulation cell
 * cell[9]             row-major 3x3 cell matrix (rows are lattice vectors)
 * inv_cell[9]         row-major inverse such that inv_cell . r = fractional
 * pbc[3]              periodicity per lattice direction
 * nat                 number of atoms
 * positions[3 * nat]  Cartesian atom positions
 * cutoff              binning / global cutoff (max pair cutoff for arrays)
 * per_atom_cutoff     [nat] per-atom radii (pair cutoff = sum), or nullptr
 * per_type_cutoff_sq  [ncutoffs * ncutoffs] squared per-type cutoffs, or nullptr
 * ncutoffs            dimension of the per-type matrix
 * types               [nat] per-atom type indices, or nullptr
 * out                 result (cleared and filled)
 *
 * Returns NL_SUCCESS, or NL_ERROR with a message recorded via set_error()
 * (e.g. for a degenerate cell). May record a warning via set_warning().
 */
error_t neighbour_list(int quantities, const real_t cell_origin[3],
                       const real_t cell[9], const real_t inv_cell[9],
                       const bool pbc[3], index_t nat, const real_t *positions,
                       real_t cutoff, const real_t *per_atom_cutoff,
                       const real_t *per_type_cutoff_sq, index_t ncutoffs,
                       const index_t *types, NeighbourList &out,
                       CellOrder order = CellOrder::Linear);

}  // namespace matscipy

#endif
