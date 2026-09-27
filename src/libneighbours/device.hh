/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Thin single-source GPU shim. The same kernel body compiles under nvcc (CUDA)
 * or hipcc (HIP); this header maps the vendor runtime onto common names so the
 * algorithm code contains no <<<>>> / hipLaunchKernelGGL spelled out by hand.
 * Included only from the GPU backend sources (the .cc files CMake compiles with
 * nvcc/hipcc).
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
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuErrorUnloading cudaErrorCudartUnloading
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
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuErrorUnloading hipErrorDeinitialized
#endif

namespace matscipy {

/* Abort with a diagnostic on a failed runtime call. The device path uses no
   exceptions; unrecoverable runtime/allocation failures (out of memory, no
   device) abort with a message. */
inline void gpu_check(gpuError_t err, const char *file, int line) {
    if (err != gpuSuccess) {
        std::fprintf(stderr, "[matscipy] GPU error at %s:%d: %s\n", file, line,
                     gpuGetErrorString(err));
        std::abort();
    }
}

#define GPU_CHECK(call) ::matscipy::gpu_check((call), __FILE__, __LINE__)

/* Check that a kernel launch was accepted (bad configuration, no device, ...);
   a launch error is otherwise silent and every later read sees garbage. Debug
   builds also synchronise, so an asynchronous fault is reported at the
   offending launch rather than at the next runtime call. */
inline void gpu_check_launch(const char *file, int line) {
    gpu_check(gpuGetLastError(), file, line);
#ifndef NDEBUG
    gpu_check(gpuDeviceSynchronize(), file, line);
#endif
}

/* Launch `kernel` over `grid` x `block` and check the launch. One spelling for
   both backends. */
#if defined(MATSCIPY_ENABLE_CUDA)
#define GPU_LAUNCH(kernel, grid, block, ...)                  \
    do {                                                      \
        kernel<<<(grid), (block)>>>(__VA_ARGS__);             \
        ::matscipy::gpu_check_launch(__FILE__, __LINE__);     \
    } while (0)
#elif defined(MATSCIPY_ENABLE_HIP)
#define GPU_LAUNCH(kernel, grid, block, ...)                             \
    do {                                                                 \
        hipLaunchKernelGGL(kernel, (grid), (block), 0, 0, __VA_ARGS__);  \
        ::matscipy::gpu_check_launch(__FILE__, __LINE__);                \
    } while (0)
#endif

}  // namespace matscipy

#endif
