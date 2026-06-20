/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * GPU neighbour-list backend (CUDA/HIP single-source kernels). Binning -> CSR
 * cell list -> two-pass count/fill, expressed as kernels over device sort/scan
 * primitives. The traversal and query are shared with the CPU path.
 *
 * Available only when a GPU backend is compiled in. neighbour_list_gpu() takes
 * the same arguments as the CPU neighbour_list() and returns the result on the
 * host (NeighbourList): inputs are host pointers and the output is copied back.
 */

#ifndef MATSCIPY_NEIGHBOUR_LIST_GPU_HH
#define MATSCIPY_NEIGHBOUR_LIST_GPU_HH

#include "neighbour_list.hh"
#include "types.hh"

namespace matscipy {

/* Host-out: results copied back to host std::vectors (NeighbourList). */
error_t neighbour_list_gpu(int quantities, const real_t cell_origin[3],
                            const real_t cell[9], const real_t inv_cell[9],
                            const bool pbc[3], index_t nat, const real_t *r,
                            real_t cutoff, const real_t *per_atom_cutoff,
                            const real_t *per_type_cutoff_sq, index_t ncutoffs,
                            const index_t *types, NeighbourList &out,
                            CellOrder order = CellOrder::Linear);

#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
}  // namespace matscipy

#include "memory_space.hh"

namespace matscipy {

/*
 * Inputs for a GPU neighbour-list build. Bundles geometry, the positions
 * (host or device), the cutoff specification, and the tuning knobs (order,
 * target device) so the device entry points take one argument instead of ~15.
 * Pointers are non-owning; the caller keeps them alive across the call.
 */
struct NeighbourListRequest {
    int quantities = 0;                          /* Quantity flags (pairs path) */
    const real_t *cell_origin = nullptr;         /* [3] */
    const real_t *cell = nullptr;                /* [9] row-major */
    const real_t *inv_cell = nullptr;            /* [9] */
    const bool *pbc = nullptr;                   /* [3] */
    index_t nat = 0;
    const real_t *positions = nullptr;           /* [3*nat], host or device */
    bool positions_on_device = false;            /* true => use in place (no H2D) */
    real_t cutoff = 0;                           /* binning / global cutoff */
    const real_t *per_atom_cutoff = nullptr;     /* [nat] or null */
    const real_t *per_type_cutoff_sq = nullptr;  /* [ncutoffs^2] or null */
    index_t ncutoffs = 0;
    const index_t *types = nullptr;              /* [nat] or null */
    CellOrder order = CellOrder::Linear;
    int device_id = -1;                          /* GPU to run on; -1 = current */
};

/*
 * Device-resident result. Each requested quantity stays in device memory as an
 * Array<T, DeviceSpace>; unrequested ones are empty. Layout matches
 * NeighbourList: distvec/shift are 3*npairs (logically npairs x 3). The *_view()
 * accessors give typed device Spans (valid as kernel/deep_copy arguments). The
 * Python layer exports these as DLPack tensors for zero-copy hand-off.
 */
struct NeighbourListDevice {
    Array<index_t, DeviceSpace> first;    /* [npairs] */
    Array<index_t, DeviceSpace> secnd;    /* [npairs] */
    Array<real_t, DeviceSpace> distvec;   /* [3*npairs] */
    Array<real_t, DeviceSpace> absdist;   /* [npairs] */
    Array<index_t, DeviceSpace> shift;    /* [3*npairs] */
    Array<index_t, DeviceSpace> counts;   /* [nat] per-atom neighbour count */
    index_t npairs = 0;

    Span<const index_t> first_view() const { return {first.data(), first.size()}; }
    Span<const index_t> secnd_view() const { return {secnd.data(), secnd.size()}; }
    Span<const real_t> distvec_view() const {
        return {distvec.data(), distvec.size()};
    }
    Span<const real_t> absdist_view() const {
        return {absdist.data(), absdist.size()};
    }
    Span<const index_t> shift_view() const { return {shift.data(), shift.size()}; }
    Span<const index_t> counts_view() const {
        return {counts.data(), counts.size()};
    }
};

/* Device-out: results left on the GPU (no D2H copy). */
error_t neighbour_list_gpu_device(const NeighbourListRequest &req,
                                  NeighbourListDevice &out);

/* Leaves per-atom neighbour counts on the device in `out.counts` (size nat)
   without materialising the O(npairs) pair arrays — a GPU coordination number.
   (req.quantities is ignored.) */
error_t neighbour_count_gpu_device(const NeighbourListRequest &req,
                                   NeighbourListDevice &out);

/*
 * Fixed-capacity ("dense") neighbour list on the device. Each atom's neighbours
 * occupy a row of an n x max_neighbours matrix, so the shapes are static —
 * suitable for frameworks that compile for fixed shapes (e.g. JAX). The buffers
 * stay on the device for zero-copy export. `overflow` is true if any atom has
 * more than max_neighbours neighbours (rows clipped; retry with a larger
 * capacity). Unused slots are 0; mask them with `count`.
 */
struct NeighbourMatrixDevice {
    index_t n = 0;
    index_t max_neighbours = 0;
    Array<index_t, DeviceSpace> idx;     /* [n*K] */
    Array<real_t, DeviceSpace> dist;     /* [n*K*3] */
    Array<index_t, DeviceSpace> count;   /* [n] */
    bool overflow = false;
};

error_t neighbour_matrix_gpu_device(const NeighbourListRequest &req,
                                    index_t max_neighbours,
                                    NeighbourMatrixDevice &out);
#endif

}  // namespace matscipy

#endif
