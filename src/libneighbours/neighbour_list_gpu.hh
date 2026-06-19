/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Phase 3.2 — GPU neighbour-list backend (CUDA/HIP single-source kernels).
 * Same algorithm as the CPU path (binning -> CSR cell list -> two-pass
 * count/fill), expressed as kernels over the Phase-3.3 sort/scan primitives.
 *
 * Available only when a GPU backend is compiled in. The signature mirrors the
 * CPU neighbour_list(); for now the dense periodic / global-or-per-type cutoff
 * path is implemented (the path the benchmark and oracle exercise). Inputs are
 * host pointers; the result is returned on the host (NeighbourList), so this is
 * a drop-in oracle/benchmark target. Device-resident results land in Phase 4.
 */

#ifndef MATSCIPY_NEIGHBOUR_LIST_GPU_HH
#define MATSCIPY_NEIGHBOUR_LIST_GPU_HH

#include "neighbour_list.hh"
#include "types.hh"

namespace matscipy {

error_t neighbour_list_gpu(int quantities, const real_t cell_origin[3],
                            const real_t cell[9], const real_t inv_cell[9],
                            const bool pbc[3], index_t nat, const real_t *r,
                            real_t cutoff, const real_t *per_atom_cutoff,
                            const real_t *per_type_cutoff_sq, index_t ncutoffs,
                            const index_t *types, NeighbourList &out);

}  // namespace matscipy

#endif
