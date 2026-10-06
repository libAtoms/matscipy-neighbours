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
#include <map>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

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
 *
 * HIP keeps its own cache instead (see BlockCache below): ROCm's (6.4) pools
 * do not reliably hand a freed block back to the next request, and on an
 * MI300A every miss maps fresh memory at roughly 15 GB/s, so a call that
 * allocates two gigabytes of outputs spent most of its time in the allocator.
 */
constexpr int kMaxDevices = 64;

#if defined(MATSCIPY_ENABLE_HIP)
/*
 * Per-device cache of hipMalloc blocks: freed blocks are kept in free lists by
 * size and handed to the next request of that size. Requests are rounded up to
 * one of eight sizes per power of two (at most 12.5% padding), so a call whose
 * output is a little larger than last time still finds a cached block. A block
 * is only freed once the device is idle (free_hip synchronises), so there is no
 * stream ordering to respect.
 */
struct BlockCache {
    std::map<std::size_t, std::vector<void *>> free;  /* by block size */
    std::unordered_map<void *, std::size_t> live;     /* block -> size */
};

std::mutex cache_mutex;
BlockCache caches[kMaxDevices];

std::size_t block_size(std::size_t bytes) {
    if (bytes <= 512) return 512;
    int e = 0;  /* 2^e < bytes <= 2^(e+1) */
    while ((std::size_t(2) << e) < bytes) e++;
    const std::size_t step = std::size_t(1) << (e - 3);
    return (bytes + step - 1) / step * step;
}

/* A cached block of `size` bytes on `dev`, or nullptr. */
void *take_cached(int dev, std::size_t size) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    BlockCache &c = caches[dev];
    auto it = c.free.find(size);
    if (it == c.free.end() || it->second.empty()) return nullptr;
    void *ptr = it->second.back();
    it->second.pop_back();
    c.live[ptr] = size;
    return ptr;
}

/* Return the cached blocks of `dev` (the current device) to the driver. */
void release_cached(int dev) {
    std::vector<void *> blocks;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        for (auto &kv : caches[dev].free)
            blocks.insert(blocks.end(), kv.second.begin(), kv.second.end());
        caches[dev].free.clear();
    }
    for (void *ptr : blocks) GPU_CHECK(gpuFree(ptr));
}
#endif

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
#if defined(MATSCIPY_ENABLE_HIP)
    if (dev >= 0 && dev < kMaxDevices) {
        const std::size_t size = block_size(bytes);
        if (void *ptr = take_cached(dev, size)) return ptr;
        void *ptr = try_alloc(nullptr, size);
        if (!ptr) {
            release_cached(dev);
            ptr = try_alloc(nullptr, size);
        }
        if (!ptr) throw std::bad_alloc();
        std::lock_guard<std::mutex> lock(cache_mutex);
        caches[dev].live[ptr] = size;
        return ptr;
    }
#endif
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
#if defined(MATSCIPY_ENABLE_HIP)
    bool cached = false;
    if (err == gpuSuccess && dev >= 0 && dev < kMaxDevices) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        BlockCache &c = caches[dev];
        auto it = c.live.find(ptr);
        if (it != c.live.end()) {
            c.free[it->second].push_back(ptr);
            c.live.erase(it);
            cached = true;
        }
    }
    if (err == gpuSuccess && !cached) err = gpuFree(ptr);
#else
    if (err == gpuSuccess) {
        const gpuMemPool_t pool = pool_for(dev);
        err = pool ? gpuFreeAsync(ptr, 0) : gpuFree(ptr);
    }
#endif
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
#if defined(MATSCIPY_ENABLE_HIP)
        bool any = false;
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            for (const auto &kv : caches[dev].free) any = any || !kv.second.empty();
        }
        if (!any) continue;
        GPU_CHECK(gpuSetDevice(dev));
        release_cached(dev);
        continue;
#endif
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
