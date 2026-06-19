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
 * Device-resident result (Phase 4). Each requested quantity stays in device
 * memory as an Array<T, DeviceSpace>; unrequested ones are empty. Layout matches
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

/* Phase 3.4 (store vs recompute): per-atom neighbour counts left on the device
   in `out.counts` (size nat), without materialising the O(npairs) pair arrays —
   the basis for a GPU coordination number. (req.quantities is ignored.) */
error_t neighbour_count_gpu_device(const NeighbourListRequest &req,
                                   NeighbourListDevice &out);
#endif

}  // namespace matscipy

#endif
