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

#if defined(MATSCIPY_ENABLE_CUDA)
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
namespace gpuprim = cub;
#elif defined(MATSCIPY_ENABLE_HIP)
#include <hipcub/hipcub.hpp>
namespace gpuprim = hipcub;
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

/* ---------------------------------------------------- Verlet-skin update check */

namespace {

/* Per-atom squared displacement ||cur_i - ref_i||^2, evaluated on host+device so
   the CUB transform iterator can call it on the device. Positions are row-major
   (n, 3), so atom i occupies [3i, 3i+3). */
struct SqDispOp {
    const real_t *cur;
    const real_t *ref;
    MATSCIPY_HD real_t operator()(index_t i) const {
        const real_t dx = cur[3 * i + 0] - ref[3 * i + 0];
        const real_t dy = cur[3 * i + 1] - ref[3 * i + 1];
        const real_t dz = cur[3 * i + 2] - ref[3 * i + 2];
        return dx * dx + dy * dy + dz * dz;
    }
};

}  // namespace

/* One thread: threshold the reduced max against skin^2, writing the uint8 flag.
   Keeps the comparison on the device so no value is read back to the host. */
__global__ void device_skin_threshold_k(const real_t *d_maxsq, real_t skinsq,
                                        std::uint8_t *d_out) {
    *d_out = (*d_maxsq > skinsq) ? std::uint8_t{1} : std::uint8_t{0};
}

Array<real_t, DeviceSpace> device_clone_positions(const real_t *d_src,
                                                   index_t n3) {
    const std::size_t n = n3 > 0 ? static_cast<std::size_t>(n3) : 0;
    Array<real_t, DeviceSpace> out(n);
    if (n > 0)
        GPU_CHECK(gpuMemcpy(out.data(), d_src, n * sizeof(real_t),
                            gpuMemcpyDeviceToDevice));
    return out;
}

Array<std::uint8_t, DeviceSpace> device_needs_rebuild(const real_t *d_cur,
                                                      const real_t *d_ref,
                                                      index_t nat, real_t skin) {
    Array<std::uint8_t, DeviceSpace> out(1);
    if (nat <= 0) {
        GPU_CHECK(gpuMemset(out.data(), 0, sizeof(std::uint8_t)));
        return out;
    }

    /* max over atoms of the squared displacement, via a transform-then-reduce so
       no per-atom temporary array is materialised. */
    Array<real_t, DeviceSpace> maxsq(1);
    gpuprim::CountingInputIterator<index_t> counting(0);
    SqDispOp op{d_cur, d_ref};
    gpuprim::TransformInputIterator<real_t, SqDispOp,
                                    gpuprim::CountingInputIterator<index_t>>
        disp(counting, op);

    void *d_temp = nullptr;
    std::size_t temp_bytes = 0;
    GPU_CHECK(gpuprim::DeviceReduce::Max(d_temp, temp_bytes, disp, maxsq.data(),
                                         nat));
    GPU_CHECK(gpuMalloc(&d_temp, temp_bytes));
    GPU_CHECK(gpuprim::DeviceReduce::Max(d_temp, temp_bytes, disp, maxsq.data(),
                                         nat));
    GPU_CHECK(gpuFree(d_temp));

    GPU_LAUNCH(device_skin_threshold_k, 1, 1, maxsq.data(), skin * skin,
               out.data());
    GPU_CHECK(gpuGetLastError());
    return out;
}

void device_copy_to_host(void *h_dst, const void *d_src, std::size_t nbytes) {
    if (nbytes == 0) return;
    GPU_CHECK(gpuMemcpy(h_dst, d_src, nbytes, gpuMemcpyDeviceToHost));
}

}  // namespace matscipy
