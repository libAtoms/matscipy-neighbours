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
#define gpuErrorMemoryAllocation cudaErrorMemoryAllocation
/* Stream-ordered memory pools (the caching allocator, memory_space_gpu.cc). */
using gpuMemPool_t = cudaMemPool_t;
using gpuMemPoolProps = cudaMemPoolProps;
using gpuPointerAttributes = cudaPointerAttributes;
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#define gpuDevAttrMemoryPoolsSupported cudaDevAttrMemoryPoolsSupported
#define gpuMemAllocationTypePinned cudaMemAllocationTypePinned
#define gpuMemLocationTypeDevice cudaMemLocationTypeDevice
#define gpuMemPoolCreate cudaMemPoolCreate
#define gpuMemPoolSetAttribute cudaMemPoolSetAttribute
#define gpuMemPoolAttrReleaseThreshold cudaMemPoolAttrReleaseThreshold
#define gpuMemPoolTrimTo cudaMemPoolTrimTo
#define gpuMallocFromPoolAsync cudaMallocFromPoolAsync
#define gpuFreeAsync cudaFreeAsync
#define gpuPointerGetAttributes cudaPointerGetAttributes
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
#define gpuErrorMemoryAllocation hipErrorOutOfMemory
using gpuMemPool_t = hipMemPool_t;
using gpuMemPoolProps = hipMemPoolProps;
using gpuPointerAttributes = hipPointerAttribute_t;
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#define gpuDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
#define gpuMemAllocationTypePinned hipMemAllocationTypePinned
#define gpuMemLocationTypeDevice hipMemLocationTypeDevice
#define gpuMemPoolCreate hipMemPoolCreate
#define gpuMemPoolSetAttribute hipMemPoolSetAttribute
#define gpuMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define gpuMemPoolTrimTo hipMemPoolTrimTo
#define gpuMallocFromPoolAsync hipMallocFromPoolAsync
#define gpuFreeAsync hipFreeAsync
#define gpuPointerGetAttributes hipPointerGetAttributes
#endif

namespace matscipy {

/* Abort with a diagnostic on a failed runtime call, for unrecoverable runtime
   failures (no device, a failed launch). Running out of memory is not one of
   them: the allocator (memory_space_gpu.cc) checks allocations itself and
   throws std::bad_alloc, which the public entry points turn into
   NL_OUT_OF_MEMORY. */
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

/* RAII: switch to `dev` for the duration of a call, restore on exit. A
   negative id means "use the current device, don't switch" (host-input path). */
struct DeviceGuard {
    int prev = -1;
    explicit DeviceGuard(int dev) {
        if (dev >= 0) {
            int cur = 0;
            GPU_CHECK(gpuGetDevice(&cur));
            if (dev != cur) {
                GPU_CHECK(gpuSetDevice(dev));
                prev = cur;
            }
        }
    }
    ~DeviceGuard() {
        if (prev < 0) return;
        /* Restoring the caller's device cannot fail meaningfully at this
           point; report rather than abort from a destructor. */
        const gpuError_t err = gpuSetDevice(prev);
        if (err != gpuSuccess) {
            std::fprintf(stderr, "[matscipy] could not restore GPU device %d: %s\n",
                         prev, gpuGetErrorString(err));
        }
    }
};

/* Sum `v` down the lanes of each group of `width` consecutive lanes (a power
   of two up to the warp size); lane 0 of each group holds the group's sum.
   Every lane of the warp must call it. */
template <typename T>
__device__ inline T group_sum(T v, int width) {
    for (int off = width / 2; off > 0; off /= 2) {
#if defined(MATSCIPY_ENABLE_CUDA)
        v += __shfl_down_sync(0xffffffffu, v, off, width);
#else
        v += __shfl_down(v, off, width);
#endif
    }
    return v;
}

}  // namespace matscipy

#endif
