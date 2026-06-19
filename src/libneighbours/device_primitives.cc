/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Phase 3.3 — CUB-backed device sort/scan. The HIP build maps these onto
 * rocPRIM (same call shapes); guarded include below.
 */

#include "device_primitives.hh"

#include "device.hh"

#if defined(MATSCIPY_ENABLE_CUDA)
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
namespace gpuprim = cub;
#elif defined(MATSCIPY_ENABLE_HIP)
#include <rocprim/rocprim.hpp>
namespace gpuprim = rocprim;
#endif

namespace matscipy {

index_t device_exclusive_scan(const index_t *d_in, index_t *d_out, index_t n) {
    if (n <= 0) return 0;

    void *d_temp = nullptr;
    std::size_t temp_bytes = 0;
    GPU_CHECK(gpuprim::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out,
                                                n));
    GPU_CHECK(gpuMalloc(&d_temp, temp_bytes));
    GPU_CHECK(gpuprim::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out,
                                                n));
    GPU_CHECK(gpuFree(d_temp));

    /* Grand total = last exclusive entry + last input. Read both back. */
    index_t last_excl = 0, last_in = 0;
    GPU_CHECK(gpuMemcpy(&last_excl, d_out + (n - 1), sizeof(index_t),
                        gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(&last_in, d_in + (n - 1), sizeof(index_t),
                        gpuMemcpyDeviceToHost));
    return last_excl + last_in;
}

void device_sort_pairs(std::uint64_t *d_keys, index_t *d_values, index_t n) {
    if (n <= 1) return;

    /* CUB sorts into double buffers; allocate the alternates and let it pick. */
    std::uint64_t *d_keys_alt = nullptr;
    index_t *d_values_alt = nullptr;
    GPU_CHECK(gpuMalloc(&d_keys_alt, n * sizeof(std::uint64_t)));
    GPU_CHECK(gpuMalloc(&d_values_alt, n * sizeof(index_t)));

    gpuprim::DoubleBuffer<std::uint64_t> keys(d_keys, d_keys_alt);
    gpuprim::DoubleBuffer<index_t> values(d_values, d_values_alt);

    void *d_temp = nullptr;
    std::size_t temp_bytes = 0;
    GPU_CHECK(gpuprim::DeviceRadixSort::SortPairs(d_temp, temp_bytes, keys,
                                                  values, n));
    GPU_CHECK(gpuMalloc(&d_temp, temp_bytes));
    GPU_CHECK(gpuprim::DeviceRadixSort::SortPairs(d_temp, temp_bytes, keys,
                                                  values, n));

    /* If the sorted data ended up in the alternate buffer, copy it back so the
       caller's pointers hold the result. */
    if (keys.Current() != d_keys) {
        GPU_CHECK(gpuMemcpy(d_keys, keys.Current(), n * sizeof(std::uint64_t),
                            gpuMemcpyDeviceToDevice));
    }
    if (values.Current() != d_values) {
        GPU_CHECK(gpuMemcpy(d_values, values.Current(), n * sizeof(index_t),
                            gpuMemcpyDeviceToDevice));
    }

    GPU_CHECK(gpuFree(d_temp));
    GPU_CHECK(gpuFree(d_keys_alt));
    GPU_CHECK(gpuFree(d_values_alt));
}

}  // namespace matscipy
