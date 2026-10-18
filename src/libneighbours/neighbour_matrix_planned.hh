/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * EXPERIMENTAL: a "planned" GPU neighbour matrix that runs entirely on the
 * caller's stream, with no host synchronisation, no allocation and no host
 * read-back -- so a compiled program (an XLA FFI call, a CUDA graph) can
 * rebuild its neighbour list on the device without leaving it.
 *
 * Every decision the regular build takes on the host from the cell (cell-grid
 * resolution, dense vs hashed lookup, buffer sizes, scan scratch) is taken
 * once, by plan_neighbour_matrix(), from a reference cell. The build then
 * reads the *current* cell from device memory and derives the per-build
 * geometry (inverse cell, bin vectors, cells to scan) on the device, so the
 * cell may change between builds (NPT); a grid planned for a much larger or
 * smaller cell stays correct but scans more or fewer cells than needed --
 * plan again then.
 *
 * Covers a global or per-type cutoff, the dense or hashed cell lookup (fixed
 * by the plan), float64 or float32 inputs (widened on the device; the search
 * runs in double), linear cell order, cell origin at 0; outputs int32. Status
 * is left on the device (see neighbour_matrix_gpu_planned). Not covered:
 * per-atom radii, Morton order, the pair-list (COO) output.
 */
#ifndef MATSCIPY_NEIGHBOUR_MATRIX_PLANNED_HH
#define MATSCIPY_NEIGHBOUR_MATRIX_PLANNED_HH

#include <cstddef>
#include <cstdint>

#include "types.hh"

namespace matscipy {

struct NeighbourMatrixPlan {
    index_t nat = 0, K = 0;
    index_t n1 = 1, n2 = 1, n3 = 1;     /* cell-grid resolution, fixed */
    bool pbc[3] = {true, true, true};
    real_t cutoff = 0;                   /* the grid's (largest) cutoff */
    index_t ncutoffs = 0;                /* > 0: per-type cutoff matrix [ncutoffs^2] */
    bool single = false;                 /* float32 positions, cell and cutoffs */
    bool hashed = false;                 /* hashed compact cell lookup (sparse grids) */
    std::int64_t hmask = 0;              /* hash capacity - 1 (hashed) */
    std::size_t temp_bytes = 0;          /* scan scratch */
    std::size_t workspace_bytes = 0;     /* device workspace the caller provides */
    /* byte offsets into the workspace */
    std::size_t off_geom = 0, off_raw = 0, off_lin = 0, off_count = 0, off_first = 0,
                off_cursor = 0, off_sorted = 0, off_raw_s = 0, off_rel_s = 0, off_pos_s = 0,
                off_types_s = 0, off_pos64 = 0, off_cell64 = 0, off_pt_sq = 0, off_hkey = 0,
                off_key = 0, off_temp = 0;
};

#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
/* Host only (no device work): fix the grid from `cell` (row-major, rows the
   lattice vectors) and `cutoff` (the largest cutoff: the global one, or the
   largest entry of a per-type matrix), choose the dense or hashed cell lookup as
   the regular build would, and size the workspace. `ncutoffs` > 0 declares a
   per-type cutoff matrix; `single` float32 inputs. NL_INVALID_ARGUMENT for a
   bad cutoff, K or cell. */
error_t plan_neighbour_matrix(index_t nat, const real_t cell[9], const bool pbc[3],
                              real_t cutoff, index_t K, index_t ncutoffs, bool single,
                              NeighbourMatrixPlan &plan);

/* The workspace layout of a plan with a given grid and lookup (what
   plan_neighbour_matrix ends with): lets a caller that kept only these values,
   e.g. as attributes of a compiled call, rebuild the identical plan. */
error_t layout_neighbour_matrix(index_t nat, index_t K, index_t n1, index_t n2, index_t n3,
                                const bool pbc[3], real_t cutoff, index_t ncutoffs,
                                bool single, bool hashed, NeighbourMatrixPlan &plan);

/* Enqueue the build on `stream` (a cudaStream_t / hipStream_t, as void*).
   Device inputs: positions [3 nat] and cell [9] (row-major), float64, or float32
   when plan.single; with plan.ncutoffs > 0, types [nat] (int32, in [0,
   ncutoffs)) and the per-type cutoff matrix [ncutoffs^2] (cutoffs, not squared;
   positions' dtype), otherwise both null. d_workspace [plan.workspace_bytes,
   256-byte aligned]. Device outputs, written on the stream: idx [nat K], shift
   [3 nat K], count [nat] (int32), status [2]: status[0] = some row has more than
   K neighbours (clipped); status[1] = invalid input (a non-finite position, a
   degenerate cell, a type out of range or a per-type cutoff above plan.cutoff),
   counts then 0. Returns once enqueued: nothing is waited for or read back; only
   enqueue errors (a bad launch) are reported. */
error_t neighbour_matrix_gpu_planned(const NeighbourMatrixPlan &plan, const void *d_positions,
                                     const void *d_cell, const std::int32_t *d_types,
                                     const void *d_per_type_cutoff, void *d_workspace,
                                     void *stream, std::int32_t *idx, std::int32_t *shift,
                                     std::int32_t *count, std::int32_t *status);
#endif

}  // namespace matscipy

#endif
