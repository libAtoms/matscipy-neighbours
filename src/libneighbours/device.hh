/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Thin single-source GPU shim (Phase 3.2). The same kernel body compiles under
 * nvcc (CUDA) or hipcc (HIP); this header maps the vendor runtime onto common
 * names so the algorithm code below contains no <<<>>> / hipLaunchKernelGGL
 * spelled out by hand. Included only from the GPU backend sources (the .cc
 * files CMake compiles with nvcc/hipcc).
 */

#ifndef MATSCIPY_DEVICE_HH
#define MATSCIPY_DEVICE_HH

#if !defined(MATSCIPY_ENABLE_CUDA) && !defined(MATSCIPY_ENABLE_HIP)
#error "device.hh included without a GPU backend enabled"
#endif

#include <cstdio>
#include <cstdlib>

#if defined(MATSCIPY_ENABLE_CUDA)
#include <cuda_runtime.h>
using gpuError_t = cudaError_t;
#define gpuSuccess cudaSuccess
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemset cudaMemset
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyHostToHost cudaMemcpyHostToHost
#define gpuGetErrorString cudaGetErrorString
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#elif defined(MATSCIPY_ENABLE_HIP)
#include <hip/hip_runtime.h>
using gpuError_t = hipError_t;
#define gpuSuccess hipSuccess
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemset hipMemset
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyHostToHost hipMemcpyHostToHost
#define gpuGetErrorString hipGetErrorString
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetLastError hipGetLastError
#endif

namespace matscipy {

/* Abort with a diagnostic on a failed runtime call. The core contract forbids
   exceptions on the device path; for unrecoverable runtime/allocation failures
   (out of memory, no device) aborting with a message is the honest option. */
inline void gpu_check(gpuError_t err, const char *file, int line) {
    if (err != gpuSuccess) {
        std::fprintf(stderr, "[matscipy] GPU error at %s:%d: %s\n", file, line,
                     gpuGetErrorString(err));
        std::abort();
    }
}

#define GPU_CHECK(call) ::matscipy::gpu_check((call), __FILE__, __LINE__)

/* Launch `kernel` over `grid` x `block`. One spelling for both backends. */
#if defined(MATSCIPY_ENABLE_CUDA)
#define GPU_LAUNCH(kernel, grid, block, ...) \
    kernel<<<(grid), (block)>>>(__VA_ARGS__)
#elif defined(MATSCIPY_ENABLE_HIP)
#define GPU_LAUNCH(kernel, grid, block, ...) \
    hipLaunchKernelGGL(kernel, (grid), (block), 0, 0, __VA_ARGS__)
#endif

}  // namespace matscipy

#endif
