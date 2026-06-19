/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Phase 3.3 — data-parallel primitives (radix sort, exclusive scan) on the
 * device, wrapping CUB (CUDA) / rocPRIM (HIP). These are the parallel
 * equivalents of the serial counting-sort prefix-sum + scatter the CPU path
 * uses; they unblock the key-sorted (Morton) build and the count->offset scan.
 *
 * Declarations only; the implementation is in device_primitives.cc, compiled by
 * nvcc/hipcc (CMake assigns it the CUDA/HIP language). Header is plain C++ so
 * host translation units may call these.
 */

#ifndef MATSCIPY_DEVICE_PRIMITIVES_HH
#define MATSCIPY_DEVICE_PRIMITIVES_HH

#include <cstdint>

#include "types.hh"

namespace matscipy {

/*
 * Exclusive prefix sum of `n` int32 values, device pointers in/out (may differ).
 * Returns the grand total (== out[n-1] + in[n-1]), read back to the host.
 */
index_t device_exclusive_scan(const index_t *d_in, index_t *d_out, index_t n);

/*
 * Stable-by-construction key/value radix sort on the device. Sorts `n` pairs by
 * the unsigned 64-bit key ascending; `d_keys`/`d_values` are sorted in place
 * (CUB uses double buffers internally and copies back). Values carry the
 * original atom indices through the sort.
 */
void device_sort_pairs(std::uint64_t *d_keys, index_t *d_values, index_t n);

}  // namespace matscipy

#endif
