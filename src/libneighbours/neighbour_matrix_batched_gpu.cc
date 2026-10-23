/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * EXPERIMENTAL batched neighbour matrix on the device (see neighbour_matrix_batched.hh):
 * one thread per row of the whole batch, the same row search as the host build
 * (batched_row). Three kernel launches on the caller's stream (zero the status,
 * fill, clear invalid structures), nothing synchronised, allocated or read back,
 * kernel nodes only (as the planned build), so it can sit in a CUDA graph.
 */

#include "neighbour_matrix_batched.hh"

#include <algorithm>
#include <cmath>

#include "device.hh"
#include "error.hh"

#if defined(MATSCIPY_ENABLE_CUDA)
using gpuStream_t = cudaStream_t;
#elif defined(MATSCIPY_ENABLE_HIP)
using gpuStream_t = hipStream_t;
#endif

namespace matscipy {
namespace {

constexpr int BLOCK = 128;

__global__ void k_zero(std::int32_t *p, index_t n) {
    for (index_t a = blockIdx.x * index_t(blockDim.x) + threadIdx.x; a < n; a += index_t(blockDim.x) * gridDim.x)
        p[a] = 0;
}

template <typename T>
__global__ void k_batched(index_t B, index_t N, const std::int32_t *n_atoms, const T *pos, const T *cells,
                          const std::int32_t *pbc, real_t cutoff, const std::int32_t *types, const T *pt,
                          index_t m, index_t K, std::int32_t *idx, std::int32_t *shift, std::int32_t *count,
                          std::int32_t *status) {
    const index_t t = blockIdx.x * index_t(blockDim.x) + threadIdx.x;
    if (t >= B * N) return;
    const index_t b = t / N, i = t % N;
    count[t] = 0;
    const index_t n = n_atoms[b];
    if (n < 0 || n > N) {                                     /* a bad count: the structure is invalid */
        if (i == 0) status[2 * b + 1] = 1;
        return;
    }
    if (i >= n) return;                                       /* a padding row */
    real_t c[9];
    for (int k = 0; k < 9; k++) c[k] = static_cast<real_t>(cells[9 * b + k]);
    const BatchedCell g = batched_cell(c, pbc[b], cutoff);
    bool invalid = !g.ok;
    if (g.ok) {
        const index_t w = batched_row(g, pbc[b], pos + 3 * N * b, n, i, cutoff, types ? types + N * b : nullptr,
                                      pt, m, K, idx + K * t, shift + 3 * K * t, invalid);
        count[t] = static_cast<std::int32_t>(w);
        if (w > K) status[2 * b] = 1;                         /* racing stores of 1: benign */
    }
    if (invalid) status[2 * b + 1] = 1;
}

__global__ void k_clear_invalid(index_t B, index_t N, const std::int32_t *status, std::int32_t *count) {
    const index_t t = blockIdx.x * index_t(blockDim.x) + threadIdx.x;
    if (t < B * N && status[2 * (t / N) + 1]) count[t] = 0;
}

}  // namespace

error_t neighbour_matrix_batched_gpu(index_t B, index_t N, const std::int32_t *n_atoms, const void *positions,
                                     const void *cells, bool single, const std::int32_t *pbc, real_t cutoff,
                                     const std::int32_t *types, const void *per_type_cutoff,
                                     index_t ncutoffs, index_t K, void *stream_ptr, std::int32_t *idx,
                                     std::int32_t *shift, std::int32_t *count, std::int32_t *status) {
    clear_error();
    if (!(cutoff > 0) || !std::isfinite(cutoff))
        return set_invalid_argument("cutoff must be positive and finite.");
    if (K < 1) return set_invalid_argument("max_neighbours must be positive.");
    if (B < 0 || N < 0 || ncutoffs < 0) return set_invalid_argument("negative batch, atom or type count.");
    if (ncutoffs > 0 && (!types || !per_type_cutoff))
        return set_invalid_argument("per-type cutoffs need types and the cutoff matrix.");
    const gpuStream_t s = static_cast<gpuStream_t>(stream_ptr);
    const index_t rows = B * N;
    k_zero<<<std::max<index_t>(1, std::min<index_t>((2 * B + BLOCK - 1) / BLOCK, 65535)), BLOCK, 0, s>>>(status,
                                                                                                         2 * B);
    if (rows > 0) {
        const int grid = static_cast<int>((rows + BLOCK - 1) / BLOCK);
        const std::int32_t *ty = ncutoffs > 0 ? types : nullptr;
        if (single)
            k_batched<float><<<grid, BLOCK, 0, s>>>(B, N, n_atoms, static_cast<const float *>(positions),
                                                    static_cast<const float *>(cells), pbc, cutoff, ty,
                                                    ncutoffs > 0 ? static_cast<const float *>(per_type_cutoff)
                                                                 : nullptr,
                                                    ncutoffs, K, idx, shift, count, status);
        else
            k_batched<double><<<grid, BLOCK, 0, s>>>(B, N, n_atoms, static_cast<const double *>(positions),
                                                     static_cast<const double *>(cells), pbc, cutoff, ty,
                                                     ncutoffs > 0
                                                         ? static_cast<const double *>(per_type_cutoff)
                                                         : nullptr,
                                                     ncutoffs, K, idx, shift, count, status);
        k_clear_invalid<<<grid, BLOCK, 0, s>>>(B, N, status, count);
    }
    const gpuError_t e = gpuGetLastError();   /* enqueue errors only: no synchronisation */
    if (e != gpuSuccess) return set_error(gpuGetErrorString(e));
    return NL_SUCCESS;
}

}  // namespace matscipy
