/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Data-parallel primitives (radix sort, exclusive scan) on the device,
 * wrapping CUB (CUDA) / rocPRIM (HIP). These are the parallel equivalents of
 * the serial counting-sort prefix-sum + scatter the CPU path uses, and drive
 * the key-sorted (Morton) build and the count->offset scan.
 *
 * Declarations only; the implementation is in device_primitives.cc, compiled by
 * nvcc/hipcc (CMake assigns it the CUDA/HIP language). The header is plain C++
 * so host translation units may call these.
 */

#ifndef MATSCIPY_DEVICE_PRIMITIVES_HH
#define MATSCIPY_DEVICE_PRIMITIVES_HH

#include <cstddef>
#include <cstdint>

#include "memory_space.hh"
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

#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
/*
 * Verlet-skin "update check" primitives. These move the rebuild test that used
 * to run in CuPy onto the device kernels, so a consumer needs no CuPy.
 *
 * device_clone_positions: device->device copy of `n3` (= 3 * n_atoms) reals,
 * snapshotting the build-time positions as the skin reference.
 */
Array<real_t, DeviceSpace> device_clone_positions(const real_t *d_src,
                                                   index_t n3);

/*
 * On-device rebuild test. Returns a 1-element uint8 device array set to 1 iff
 * max_i ||cur_i - ref_i||^2 > skin*skin, else 0. The per-atom squared
 * displacement, its max-reduction, and the threshold all run on the device;
 * nothing is copied to the host (residency is preserved).
 */
Array<std::uint8_t, DeviceSpace> device_needs_rebuild(const real_t *d_cur,
                                                      const real_t *d_ref,
                                                      index_t nat, real_t skin);

/*
 * Copy `nbytes` from a device pointer to host. Used only for the single small
 * scalar read-back when an eager consumer converts the device flag to a Python
 * bool (the documented sync point); not used on the residency hot path.
 */
void device_copy_to_host(void *h_dst, const void *d_src, std::size_t nbytes);
#endif

}  // namespace matscipy

#endif
