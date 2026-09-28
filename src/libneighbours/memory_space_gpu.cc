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

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <new>

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
namespace {

/*
 * Caching allocator. Every call of the neighbour-list builds allocates its
 * outputs and scratch buffers afresh, and plain cudaMalloc/cudaFree of large
 * buffers costs milliseconds (and synchronises the device) each. Instead each
 * device gets its own stream-ordered memory pool whose release threshold is
 * unlimited, so freed blocks stay cached and the next call of the same size
 * reuses them without a driver round trip. The pool is private to this library:
 * other libraries' allocators (CuPy, PyTorch, XLA) are unaffected, but memory
 * cached here is not available to them until empty_gpu_cache() returns it.
 * Devices without memory-pool support fall back to cudaMalloc/cudaFree.
 */
constexpr int kMaxDevices = 64;

struct DevicePool {
    bool initialised = false;
    gpuMemPool_t pool{};  /* null when the device has no memory-pool support */
};

std::mutex pools_mutex;
DevicePool pools[kMaxDevices];

/* The pool of `dev`, created on first use; null if pools are unsupported. */
gpuMemPool_t pool_for(int dev) {
    if (dev < 0 || dev >= kMaxDevices) return nullptr;
    std::lock_guard<std::mutex> lock(pools_mutex);
    DevicePool &p = pools[dev];
    if (!p.initialised) {
        p.initialised = true;
        int supported = 0;
        if (gpuDeviceGetAttribute(&supported, gpuDevAttrMemoryPoolsSupported,
                                  dev) == gpuSuccess &&
            supported) {
            gpuMemPoolProps props{};
            props.allocType = gpuMemAllocationTypePinned;
            props.location.type = gpuMemLocationTypeDevice;
            props.location.id = dev;
            gpuMemPool_t pool{};
            if (gpuMemPoolCreate(&pool, &props) == gpuSuccess) {
                std::uint64_t keep = UINT64_MAX;
                GPU_CHECK(gpuMemPoolSetAttribute(
                    pool, gpuMemPoolAttrReleaseThreshold, &keep));
                p.pool = pool;
            }
        }
        (void)gpuGetLastError();  /* a failed probe must not linger */
    }
    return p.pool;
}

/* Allocate `bytes` on the current device; nullptr (and a cleared error
   state) if the device is out of memory. */
void *try_alloc(gpuMemPool_t pool, std::size_t bytes) {
    void *ptr = nullptr;
    const gpuError_t err =
        pool ? gpuMallocFromPoolAsync(&ptr, bytes, pool, 0) : gpuMalloc(&ptr, bytes);
    if (err == gpuSuccess) return ptr;
    if (err != gpuErrorMemoryAllocation) GPU_CHECK(err);
    (void)gpuGetLastError();  /* out of memory is not sticky; clear it */
    return nullptr;
}

/* Return the cached blocks of `pool` to the driver. */
void trim(gpuMemPool_t pool) {
    GPU_CHECK(gpuDeviceSynchronize());  /* let pending frees retire first */
    GPU_CHECK(gpuMemPoolTrimTo(pool, 0));
}

}  // namespace

namespace detail {

/* Out of memory throws std::bad_alloc: the public entry points turn it into
   NL_OUT_OF_MEMORY (a MemoryError in Python) instead of aborting. The cache is
   trimmed and the allocation retried once before giving up. */
void *GPU_ALLOC(std::size_t bytes) {
    int dev = 0;
    GPU_CHECK(gpuGetDevice(&dev));
    const gpuMemPool_t pool = pool_for(dev);
    void *ptr = try_alloc(pool, bytes);
    if (!ptr && pool) {
        trim(pool);
        ptr = try_alloc(pool, bytes);
    }
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

/* Freeing runs from destructors, including those of DLPack capsules that a
   consumer drops after using the buffer on its own streams, or during
   interpreter shutdown after the runtime has been torn down. So first wait for
   all work on the buffer's device, as cudaFree does implicitly, then return
   the block to its pool. "Unloading" at shutdown is harmless and must not
   abort the process; any other failure is reported but not fatal either: a
   destructor cannot recover, and aborting would only turn a leak into a crash. */
void GPU_FREE(void *ptr) {
    gpuPointerAttributes attr{};
    gpuError_t err = gpuPointerGetAttributes(&attr, ptr);
    int cur = -1;
    if (err == gpuSuccess) err = gpuGetDevice(&cur);
    const int dev = err == gpuSuccess ? attr.device : -1;
    if (err == gpuSuccess && dev != cur) err = gpuSetDevice(dev);
    if (err == gpuSuccess) err = gpuDeviceSynchronize();
    if (err == gpuSuccess) {
        const gpuMemPool_t pool = pool_for(dev);
        err = pool ? gpuFreeAsync(ptr, 0) : gpuFree(ptr);
    }
    if (dev >= 0 && dev != cur && cur >= 0) (void)gpuSetDevice(cur);
    if (err != gpuSuccess && err != gpuErrorUnloading) {
        std::fprintf(stderr, "[matscipy] GPU free failed: %s\n",
                     gpuGetErrorString(err));
    }
    (void)gpuGetLastError();
}

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

void empty_gpu_cache() {
    int cur = 0;
    GPU_CHECK(gpuGetDevice(&cur));
    for (int dev = 0; dev < kMaxDevices; dev++) {
        gpuMemPool_t pool{};
        {
            std::lock_guard<std::mutex> lock(pools_mutex);
            pool = pools[dev].pool;
        }
        if (!pool) continue;
        GPU_CHECK(gpuSetDevice(dev));
        trim(pool);
    }
    GPU_CHECK(gpuSetDevice(cur));
}

}  // namespace matscipy
