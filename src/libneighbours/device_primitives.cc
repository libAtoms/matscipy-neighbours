/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * CUB-backed device sort/scan. The CUDA build uses CUB; the HIP build uses
 * hipCUB, AMD's CUB-compatible wrapper over rocPRIM, so the call sites
 * (DeviceScan::ExclusiveSum, DeviceRadixSort::SortPairs, DoubleBuffer) are
 * identical on both backends.
 */

#include "device_primitives.hh"

#include "device.hh"
#include "memory_space.hh"

#include <cstdio>
#include <cstdlib>

#if defined(MATSCIPY_ENABLE_CUDA)
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
namespace gpuprim = cub;
#elif defined(MATSCIPY_ENABLE_HIP)
#include <hipcub/hipcub.hpp>
namespace gpuprim = hipcub;
#endif

namespace matscipy {

/* CUB / hipCUB take the item count as a 32-bit int in all versions; the
   callers guarantee n < 2^31 (see kMaxDeviceAtoms in neighbour_list_gpu.cc). */
static int item_count(index_t n) {
    if (n > 0x7fffffff) {
        std::fprintf(stderr, "[matscipy] device primitive called with %lld items "
                     "(limit 2^31-1)\n", static_cast<long long>(n));
        std::abort();
    }
    return static_cast<int>(n);
}

index_t device_exclusive_scan(const index_t *d_in, index_t *d_out, index_t n) {
    if (n <= 0) return 0;
    const int num_items = item_count(n);

    std::size_t temp_bytes = 0;
    GPU_CHECK(gpuprim::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_in, d_out,
                                                num_items));
    /* At least one byte: a null scratch pointer makes CUB only report the size. */
    Array<unsigned char, DeviceSpace> temp(temp_bytes > 0 ? temp_bytes : 1);
    GPU_CHECK(gpuprim::DeviceScan::ExclusiveSum(temp.data(), temp_bytes, d_in,
                                                d_out, num_items));

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
    const int num_items = item_count(n);

    /* CUB sorts into double buffers; allocate the alternates and let it pick. */
    Array<std::uint64_t, DeviceSpace> keys_alt(n);
    Array<index_t, DeviceSpace> values_alt(n);
    gpuprim::DoubleBuffer<std::uint64_t> keys(d_keys, keys_alt.data());
    gpuprim::DoubleBuffer<index_t> values(d_values, values_alt.data());

    std::size_t temp_bytes = 0;
    GPU_CHECK(gpuprim::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys,
                                                  values, num_items));
    /* At least one byte: a null scratch pointer makes CUB only report the size. */
    Array<unsigned char, DeviceSpace> temp(temp_bytes > 0 ? temp_bytes : 1);
    GPU_CHECK(gpuprim::DeviceRadixSort::SortPairs(temp.data(), temp_bytes, keys,
                                                  values, num_items));

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
}

}  // namespace matscipy
