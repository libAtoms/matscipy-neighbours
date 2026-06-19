/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 *
 * Phase 3.2 validation: the GPU neighbour list must agree with the CPU oracle
 * pair-for-pair. Within-row order differs (GPU scatter is unordered), so pairs
 * are compared as sorted (i, j, shift) multisets plus the distance multiset.
 * Built only with the CUDA backend; skips if no device is present.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include "device_primitives.hh"
#include "memory_space.hh"
#include "neighbour_list.hh"
#include "neighbour_list_gpu.hh"

#ifdef MATSCIPY_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

using namespace matscipy;

namespace {

bool cuda_device_present() {
#ifdef MATSCIPY_ENABLE_CUDA
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}

/* Canonicalise a result into a sorted list of (i, j, sx, sy, sz) tuples so two
   runs are comparable regardless of within-row ordering. */
std::vector<std::array<index_t, 5>> canonical(const NeighbourList &nl) {
    std::vector<std::array<index_t, 5>> v(nl.npairs);
    for (index_t p = 0; p < nl.npairs; p++) {
        v[p] = {nl.first[p], nl.secnd[p], nl.shift[3 * p], nl.shift[3 * p + 1],
                nl.shift[3 * p + 2]};
    }
    std::sort(v.begin(), v.end());
    return v;
}

void compare(int N, double L, double cutoff, unsigned seed) {
    const real_t cell[9] = {(real_t)L, 0, 0, 0, (real_t)L, 0, 0, 0, (real_t)L};
    const real_t inv[9] = {(real_t)(1 / L), 0, 0, 0, (real_t)(1 / L), 0,
                           0, 0, (real_t)(1 / L)};
    const real_t origin[3] = {0, 0, 0};
    const bool pbc[3] = {true, true, true};

    std::mt19937 rng(seed);
    std::uniform_real_distribution<real_t> U(0.0, L);
    std::vector<real_t> r(3 * N);
    for (int k = 0; k < 3 * N; k++) r[k] = U(rng);

    const int flags = QUANTITY_FIRST | QUANTITY_SECOND | QUANTITY_DISTVEC |
                      QUANTITY_ABSDIST | QUANTITY_SHIFT;
    NeighbourList cpu, gpu;
    ASSERT_EQ(neighbour_list(flags, origin, cell, inv, pbc, N, r.data(), cutoff,
                             nullptr, nullptr, 0, nullptr, cpu),
              NL_SUCCESS);
    ASSERT_EQ(neighbour_list_gpu(flags, origin, cell, inv, pbc, N, r.data(),
                                 cutoff, nullptr, nullptr, 0, nullptr, gpu),
              NL_SUCCESS);

    ASSERT_EQ(gpu.npairs, cpu.npairs);
    EXPECT_EQ(canonical(gpu), canonical(cpu));

    /* Distance multisets must match too (catches a wrong D with right i,j). */
    std::vector<real_t> dc(cpu.absdist), dg(gpu.absdist);
    std::sort(dc.begin(), dc.end());
    std::sort(dg.begin(), dg.end());
    ASSERT_EQ(dc.size(), dg.size());
    for (size_t k = 0; k < dc.size(); k++) EXPECT_NEAR(dc[k], dg[k], 1e-12);
}

}  // namespace

TEST(NeighbourListGpu, MatchesCpuDense) {
    if (!cuda_device_present()) GTEST_SKIP() << "no CUDA device";
    compare(/*N=*/2000, /*L=*/12.0, /*cutoff=*/1.0, /*seed=*/1);
}

TEST(NeighbourListGpu, MatchesCpuFewLargeCells) {
    if (!cuda_device_present()) GTEST_SKIP() << "no CUDA device";
    /* Small box: only a few cells per side, so the periodic wrap and multiple
       images per neighbour cell are exercised. */
    compare(/*N=*/500, /*L=*/3.5, /*cutoff=*/1.0, /*seed=*/2);
}

TEST(NeighbourListGpu, MatchesCpuLargerCutoff) {
    if (!cuda_device_present()) GTEST_SKIP() << "no CUDA device";
    compare(/*N=*/3000, /*L=*/15.0, /*cutoff=*/2.0, /*seed=*/3);
}

/* --- Phase 3.3 primitives -------------------------------------------------- */

TEST(DevicePrimitives, ExclusiveScan) {
    if (!cuda_device_present()) GTEST_SKIP() << "no CUDA device";
    const index_t n = 1000;
    Array<index_t> h_in(n), h_out(n);
    for (index_t i = 0; i < n; i++) h_in.data()[i] = i % 7;
    Array<index_t, CudaSpace> d_in(n), d_out(n);
    deep_copy(d_in, h_in);

    index_t total = device_exclusive_scan(d_in.data(), d_out.data(), n);
    deep_copy(h_out, d_out);

    index_t acc = 0;
    for (index_t i = 0; i < n; i++) {
        ASSERT_EQ(h_out.data()[i], acc) << "at " << i;
        acc += h_in.data()[i];
    }
    EXPECT_EQ(total, acc);
}

TEST(DevicePrimitives, RadixSortPairs) {
    if (!cuda_device_present()) GTEST_SKIP() << "no CUDA device";
    const index_t n = 4096;
    Array<std::uint64_t> h_keys(n);
    Array<index_t> h_vals(n);
    /* Reverse-sorted keys; values are the original positions. */
    for (index_t i = 0; i < n; i++) {
        h_keys.data()[i] = static_cast<std::uint64_t>(n - 1 - i) * 2654435761u;
        h_vals.data()[i] = i;
    }
    Array<std::uint64_t, CudaSpace> d_keys(n);
    Array<index_t, CudaSpace> d_vals(n);
    deep_copy(d_keys, h_keys);
    deep_copy(d_vals, h_vals);

    device_sort_pairs(d_keys.data(), d_vals.data(), n);
    deep_copy(h_keys, d_keys);
    deep_copy(h_vals, d_vals);

    for (index_t i = 1; i < n; i++)
        ASSERT_LE(h_keys.data()[i - 1], h_keys.data()[i]) << "unsorted at " << i;
    /* The reversed input means the value carried to slot i is (n-1-i). */
    for (index_t i = 0; i < n; i++)
        ASSERT_EQ(h_vals.data()[i], n - 1 - i) << "value mispaired at " << i;
}
