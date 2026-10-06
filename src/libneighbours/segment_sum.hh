/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Per-atom sums over a pair list sorted by its first index: a multi-axis
 * bincount (matscipy's mabincount) without atomics. The pairs of an atom are
 * one contiguous segment of rows, delimited by the row-start array that
 * first_neighbours builds, so each segment is summed on its own. The
 * summation order is fixed, so results are reproducible bit for bit, unlike
 * an atomic scatter.
 */

#ifndef MATSCIPY_SEGMENT_SUM_HH
#define MATSCIPY_SEGMENT_SUM_HH

#include "types.hh"

namespace matscipy {

/*
 * n       number of segments (atoms)
 * seed    [n + 1] row starts, as from first_neighbours: the rows of segment k
 *         are [seed[k], seed[k+1]). Negative values (first_neighbours' -1 for
 *         atoms before the first pair) count as 0.
 * nrows   number of rows of `values`
 * d       values per row (the product of the trailing dimensions)
 * values  [nrows * d]
 * out     [n * d] per-segment sums
 * total   [d] or null: the sum over all segments
 * Returns NL_INVALID_ARGUMENT if seed decreases (after clamping at 0) or
 * points beyond nrows. Instantiated for float, double, int32_t and int64_t.
 */
template <typename T>
error_t segment_sum(index_t n, const index_t *seed, index_t nrows, index_t d,
                    const T *values, T *out, T *total = nullptr);

}  // namespace matscipy

#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
#include "memory_space.hh"

namespace matscipy {

/* Device kernels: one thread per (segment, value), or a group of lanes per
   (segment, value) that strides over the rows and sums with warp shuffles.
   Auto picks the group size from the mean segment length. */
enum class SegmentSumKernel { Auto, ThreadPerRow, Group };

/* Keeps a parameter out of template argument deduction, so that `total` can
   be passed as nullptr. */
template <typename T>
struct NonDeduced {
    using type = T;
};

/*
 * Device version of segment_sum: seed and values are device pointers on GPU
 * `device_id` (-1 = current); `out` is resized to n * d and, if `total` is
 * not null, *total to d, both filled on the device. `group` sets the lanes
 * per (segment, value) for SegmentSumKernel::Group (0 = from the mean
 * segment length).
 */
template <typename T>
error_t segment_sum_gpu_device(index_t n, const index_t *seed, index_t nrows,
                               index_t d, const T *values,
                               Array<T, DeviceSpace> &out,
                               typename NonDeduced<Array<T, DeviceSpace>>::type
                                   *total = nullptr,
                               int device_id = -1,
                               SegmentSumKernel kernel = SegmentSumKernel::Auto,
                               int group = 0);

}  // namespace matscipy
#endif

#endif
