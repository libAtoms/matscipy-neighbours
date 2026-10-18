/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * GPU cell-list build kernels shared by the regular build (neighbour_list_gpu.cc)
 * and the planned build (neighbour_matrix_planned_gpu.cc): the dense histogram +
 * scatter and the hashed compact table. Included by GPU translation units only;
 * internal linkage (an anonymous namespace), so each unit has its own copy.
 */
#ifndef MATSCIPY_CELL_LIST_GPU_HH
#define MATSCIPY_CELL_LIST_GPU_HH

#include <cstdint>

#include "cell_list.hh"
#include "device.hh"
#include "tools.hh"

namespace matscipy {
namespace {

/* Atomic post-increment of a 64-bit index (histogram / cursor / row counters).
   CUDA and HIP provide 64-bit atomicAdd only for unsigned long long. */
__device__ inline index_t atomic_inc(index_t *p) {
    static_assert(sizeof(index_t) == sizeof(unsigned long long),
                  "atomic_inc assumes a 64-bit index_t");
    return static_cast<index_t>(
        atomicAdd(reinterpret_cast<unsigned long long *>(p), 1ull));
}

__global__ void k_fold_lin_hist(const index_t *raw, int pbc0, int pbc1, int pbc2,
                                index_t n1, index_t n2, index_t n3, index_t nat,
                                index_t *lin, index_t *cell_count) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    index_t c1 = pbc0 ? bin_wrap(raw[3 * a], n1) : bin_trunc(raw[3 * a], n1);
    index_t c2 = pbc1 ? bin_wrap(raw[3 * a + 1], n2) : bin_trunc(raw[3 * a + 1], n2);
    index_t c3 = pbc2 ? bin_wrap(raw[3 * a + 2], n3) : bin_trunc(raw[3 * a + 2], n3);
    index_t l = c1 + n1 * (c2 + n2 * c3);
    lin[a] = l;
    atomic_inc(&cell_count[l]);
}

__global__ void k_scatter(const index_t *lin, index_t nat, index_t *cursor,
                          index_t *sorted_atom) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    index_t pos = atomic_inc(&cursor[lin[a]]);
    sorted_atom[pos] = a;
}

/* --- sparse (hashed compact) cell list, for huge/sparse grids --------------
   Open addressing keyed by the 64-bit linear cell index; built in parallel with
   atomicCAS inserts. Used when the dense histogram/array path would need
   O(ncells) memory (or overflow a 32-bit index). */

__global__ void k_fold_key64(const index_t *raw, int pbc0, int pbc1, int pbc2,
                            index_t n1, index_t n2, index_t n3, index_t nat,
                            std::int64_t *key) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    index_t c1 = pbc0 ? bin_wrap(raw[3 * a], n1) : bin_trunc(raw[3 * a], n1);
    index_t c2 = pbc1 ? bin_wrap(raw[3 * a + 1], n2) : bin_trunc(raw[3 * a + 1], n2);
    index_t c3 = pbc2 ? bin_wrap(raw[3 * a + 2], n3) : bin_trunc(raw[3 * a + 2], n3);
    key[a] = static_cast<std::int64_t>(c1) +
             static_cast<std::int64_t>(n1) *
                 (static_cast<std::int64_t>(c2) + static_cast<std::int64_t>(n2) * c3);
}

__global__ void k_hash_insert(const std::int64_t *key, index_t nat,
                             std::int64_t mask, std::int64_t *hkey,
                             index_t *hcount) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    std::int64_t k = key[a];
    std::int64_t h = cell_hash(k) & mask;
    auto *slot = reinterpret_cast<unsigned long long *>(hkey);
    const unsigned long long empty = static_cast<unsigned long long>(-1);
    while (true) {
        unsigned long long old =
            atomicCAS(&slot[h], empty, static_cast<unsigned long long>(k));
        if (old == empty || static_cast<std::int64_t>(old) == k) {
            atomic_inc(&hcount[h]);
            return;
        }
        h = (h + 1) & mask;
    }
}

__global__ void k_hash_scatter(const std::int64_t *key, index_t nat,
                              std::int64_t mask, const std::int64_t *hkey,
                              index_t *cursor, index_t *sorted_atom) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    std::int64_t k = key[a];
    std::int64_t h = cell_hash(k) & mask;
    while (hkey[h] != k) h = (h + 1) & mask;  /* key is guaranteed present */
    index_t pos = atomic_inc(&cursor[h]);
    sorted_atom[pos] = a;
}

}  // namespace
}  // namespace matscipy

#endif
