/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Device segment sum (segment_sum.hh). No atomics: every output value is
 * summed by one thread, or by one group of lanes that reduces with warp
 * shuffles, in a fixed order. The totals are a two-stage block reduction,
 * also in a fixed order. Compiled by nvcc or hipcc.
 */

#include "segment_sum.hh"

#include <cstdint>

#include "device.hh"
#include "error.hh"
#include "memory_space.hh"

namespace matscipy {
namespace {

constexpr int BLOCK = 256;
constexpr int kMaxTotalBlocks = 1024;

inline int grid_for(index_t n) { return static_cast<int>((n + BLOCK - 1) / BLOCK); }

__device__ inline index_t clamp0(index_t s) { return s < 0 ? 0 : s; }

/* Flags a row start that decreases (after clamping at 0) or points beyond
   nrows. Threads only ever write 1, so their race is benign. */
__global__ void k_check_seed(const index_t *seed, index_t n, index_t nrows,
                             int *bad) {
    const index_t k = blockIdx.x * static_cast<index_t>(blockDim.x) + threadIdx.x;
    if (k > n) return;
    const index_t s = clamp0(seed[k]);
    if (s > nrows || (k > 0 && s < clamp0(seed[k - 1]))) *bad = 1;
}

/* One thread per (segment, value): walks the rows of its segment. Adjacent
   threads read adjacent values of a row. */
template <typename T>
__global__ void k_segment_sum_thread(const index_t *seed, index_t n, index_t d,
                                     const T *values, T *out) {
    const index_t t = blockIdx.x * static_cast<index_t>(blockDim.x) + threadIdx.x;
    if (t >= n * d) return;
    const index_t a = t / d, c = t % d;
    const index_t begin = clamp0(seed[a]), end = clamp0(seed[a + 1]);
    T acc = T(0);
    for (index_t p = begin; p < end; p++) acc += values[p * d + c];
    out[t] = acc;
}

/* A group of L lanes per (segment, value): lane l sums rows begin + l,
   begin + l + L, ..., then the group adds up its lanes with shuffles, so a
   segment is read by L lanes at once. Every lane takes part in the shuffle,
   including those past the end of the output. */
template <typename T, int L>
__global__ void k_segment_sum_group(const index_t *seed, index_t n, index_t d,
                                    const T *values, T *out) {
    const index_t t = blockIdx.x * static_cast<index_t>(blockDim.x) + threadIdx.x;
    const index_t g = t / L;
    const int lane = static_cast<int>(t % L);
    T acc = T(0);
    if (g < n * d) {
        const index_t a = g / d, c = g % d;
        const index_t begin = clamp0(seed[a]), end = clamp0(seed[a + 1]);
        for (index_t p = begin + lane; p < end; p += L) acc += values[p * d + c];
    }
    acc = group_sum(acc, L);
    if (g < n * d && lane == 0) out[g] = acc;
}

/* Sums the threads' values of one block in shared memory (tree, fixed order);
   thread 0 gets the result. */
template <typename T>
__device__ T block_sum(T v) {
    __shared__ T buf[BLOCK];
    buf[threadIdx.x] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s /= 2) {
        if (static_cast<int>(threadIdx.x) < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    return buf[0];
}

/* Stage 1 of the totals: block (b, c) sums value c of the segments
   [b * chunk, (b + 1) * chunk) into partial[c * nblocks + b]. */
template <typename T>
__global__ void k_total_partial(const T *out, index_t n, index_t d,
                                index_t chunk, T *partial) {
    const index_t b = blockIdx.x, c = blockIdx.y;
    const index_t lo = b * chunk;
    const index_t hi = lo + chunk < n ? lo + chunk : n;
    T acc = T(0);
    for (index_t a = lo + threadIdx.x; a < hi; a += BLOCK) acc += out[a * d + c];
    acc = block_sum(acc);
    if (threadIdx.x == 0) partial[c * gridDim.x + b] = acc;
}

/* Stage 2: block c sums the nblocks partials of value c. */
template <typename T>
__global__ void k_total_final(const T *partial, int nblocks, T *total) {
    const index_t c = blockIdx.y;
    T acc = T(0);
    for (int b = threadIdx.x; b < nblocks; b += BLOCK) acc += partial[c * nblocks + b];
    acc = block_sum(acc);
    if (threadIdx.x == 0) total[c] = acc;
}

/* Lanes per (segment, value). The d values of a row are adjacent, so the
   groups of one segment together read L * d consecutive values per step;
   about 8-12 of them was best on an MI300A (wave64), for mean segment lengths
   of 20 and 54 and d = 1, 3, 9. A lane should also get at least two rows. */
int auto_group(index_t n, index_t nrows, index_t d) {
    const double mean = n > 0 ? static_cast<double>(nrows) / n : 0.0;
    int L = 1;
    while (L < 32 && 2 * L * d <= 12 && 2.0 * (2 * L) <= mean) L *= 2;
    return L;
}

template <typename T, int L>
void launch_group(const index_t *seed, index_t n, index_t d, const T *values,
                  T *out) {
    const index_t threads = n * d * L;
    GPU_LAUNCH((k_segment_sum_group<T, L>), grid_for(threads), BLOCK, seed, n, d,
               values, out);
}

}  // namespace

template <typename T>
error_t segment_sum_gpu_device(index_t n, const index_t *seed, index_t nrows,
                               index_t d, const T *values,
                               Array<T, DeviceSpace> &out,
                               typename NonDeduced<Array<T, DeviceSpace>>::type
                                   *total,
                               int device_id,
                               SegmentSumKernel kernel, int group) {
    clear_error();
    if (n < 0 || nrows < 0 || d < 0) {
        return set_invalid_argument(
            "segment_sum: sizes must be non-negative.");
    }
    if (!seed || (nrows * d > 0 && !values)) {
        return set_invalid_argument("segment_sum: invalid array.");
    }
    if (total && d > 65535) {
        return set_invalid_argument(
            "segment_sum: totals support at most 65535 values per row.");
    }
    if (group != 0 && group != 1 && group != 2 && group != 4 && group != 8 &&
        group != 16 && group != 32) {
        return set_invalid_argument(
            "segment_sum: the group size must be 1, 2, 4, 8, 16 or 32.");
    }
    return catch_out_of_memory("GPU out of memory in segment_sum.", [&] {
        DeviceGuard guard(device_id);
        /* Validate before reading values through seed. */
        Array<int, DeviceSpace> bad(1);
        GPU_CHECK(gpuMemset(bad.data(), 0, sizeof(int)));
        GPU_LAUNCH(k_check_seed, grid_for(n + 1), BLOCK, seed, n, nrows,
                   bad.data());
        int host_bad = 0;
        GPU_CHECK(gpuMemcpy(&host_bad, bad.data(), sizeof(int),
                            gpuMemcpyDeviceToHost));
        if (host_bad) {
            return set_invalid_argument(
                "segment_sum: row starts must be non-decreasing and at most "
                "the number of rows.");
        }

        out.resize(static_cast<std::size_t>(n) * d);
        if (n * d > 0) {
            int L = kernel == SegmentSumKernel::ThreadPerRow ? 1
                    : group > 0                              ? group
                                                             : auto_group(n, nrows, d);
            if (kernel == SegmentSumKernel::Group && L == 1 && group == 0) L = 2;
            switch (L) {
                case 1:
                    GPU_LAUNCH(k_segment_sum_thread<T>, grid_for(n * d), BLOCK,
                               seed, n, d, values, out.data());
                    break;
                case 2: launch_group<T, 2>(seed, n, d, values, out.data()); break;
                case 4: launch_group<T, 4>(seed, n, d, values, out.data()); break;
                case 8: launch_group<T, 8>(seed, n, d, values, out.data()); break;
                case 16: launch_group<T, 16>(seed, n, d, values, out.data()); break;
                default: launch_group<T, 32>(seed, n, d, values, out.data()); break;
            }
        }

        if (total) {
            total->resize(static_cast<std::size_t>(d));
            if (d > 0 && n == 0) {
                GPU_CHECK(gpuMemset(total->data(), 0, d * sizeof(T)));
            } else if (d > 0) {
                const int nblocks = grid_for(n) < kMaxTotalBlocks ? grid_for(n)
                                                                  : kMaxTotalBlocks;
                const index_t chunk = (n + nblocks - 1) / nblocks;
                Array<T, DeviceSpace> partial(static_cast<std::size_t>(nblocks) * d);
                GPU_LAUNCH(k_total_partial<T>,
                           dim3(nblocks, static_cast<unsigned>(d)), BLOCK,
                           out.data(), n, d, chunk, partial.data());
                GPU_LAUNCH(k_total_final<T>, dim3(1, static_cast<unsigned>(d)),
                           BLOCK, partial.data(), nblocks, total->data());
            }
        }
        /* The results are handed to consumers on their own streams. */
        GPU_CHECK(gpuDeviceSynchronize());
        return NL_SUCCESS;
    });
}

#define MATSCIPY_SEGMENT_SUM_GPU(T)                                          \
    template error_t segment_sum_gpu_device<T>(                              \
        index_t, const index_t *, index_t, index_t, const T *,               \
        Array<T, DeviceSpace> &, Array<T, DeviceSpace> *, int,               \
        SegmentSumKernel, int);
MATSCIPY_SEGMENT_SUM_GPU(float)
MATSCIPY_SEGMENT_SUM_GPU(double)
MATSCIPY_SEGMENT_SUM_GPU(std::int32_t)
MATSCIPY_SEGMENT_SUM_GPU(std::int64_t)
#undef MATSCIPY_SEGMENT_SUM_GPU

}  // namespace matscipy
