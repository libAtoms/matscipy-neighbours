/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * GPU backend for the memory-space abstraction. The same body serves CUDA
 * (nvcc, -DENABLE_CUDA=ON) and HIP (hipcc, -DENABLE_HIP=ON) via the device.hh
 * runtime aliases; the only difference is which set of detail hooks it defines
 * (alloc_cuda/... vs alloc_hip/...), matched to what memory_space.hh declares
 * for the active backend.
 */

#include "device.hh"
#include "memory_space.hh"

#if defined(MATSCIPY_ENABLE_CUDA)
#define GPU_ALLOC alloc_cuda
#define GPU_FREE free_cuda
#define GPU_COPY copy_cuda
#elif defined(MATSCIPY_ENABLE_HIP)
#define GPU_ALLOC alloc_hip
#define GPU_FREE free_hip
#define GPU_COPY copy_hip
#endif

namespace matscipy {
namespace detail {

void *GPU_ALLOC(std::size_t bytes) {
    void *ptr = nullptr;
    GPU_CHECK(gpuMalloc(&ptr, bytes));
    return ptr;
}

void GPU_FREE(void *ptr) { GPU_CHECK(gpuFree(ptr)); }

void GPU_COPY(void *dst, DeviceType dst_dev, const void *src,
              DeviceType src_dev, std::size_t bytes) {
    const bool dst_host = dst_dev == DeviceType::CPU;
    const bool src_host = src_dev == DeviceType::CPU;
    auto kind = src_host ? (dst_host ? gpuMemcpyHostToHost
                                     : gpuMemcpyHostToDevice)
                         : (dst_host ? gpuMemcpyDeviceToHost
                                     : gpuMemcpyDeviceToDevice);
    GPU_CHECK(gpuMemcpy(dst, src, bytes, kind));
}

}  // namespace detail

int current_device_id() {
    int dev = 0;
    GPU_CHECK(gpuGetDevice(&dev));
    return dev;
}

}  // namespace matscipy
