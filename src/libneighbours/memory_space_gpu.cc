/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * CUDA backend for the memory-space abstraction. Compiled by nvcc only when
 * -DENABLE_CUDA=ON. The HIP backend is the same body with the device.hh
 * aliases pointing at the HIP runtime; share it once HIP is wired.
 */

#include "device.hh"
#include "memory_space.hh"

namespace matscipy {
namespace detail {

void *alloc_cuda(std::size_t bytes) {
    void *ptr = nullptr;
    GPU_CHECK(gpuMalloc(&ptr, bytes));
    return ptr;
}

void free_cuda(void *ptr) { GPU_CHECK(gpuFree(ptr)); }

void copy_cuda(void *dst, DeviceType dst_dev, const void *src,
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
}  // namespace matscipy
