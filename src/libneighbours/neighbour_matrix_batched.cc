/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * EXPERIMENTAL batched neighbour matrix on the host (see neighbour_matrix_batched.hh).
 */

#include "neighbour_matrix_batched.hh"

#include <cmath>

#include "error.hh"

namespace matscipy {
namespace {

template <typename T>
void batched_host(index_t B, index_t N, const std::int32_t *n_atoms, const T *pos, const T *cells,
                  const std::int32_t *pbc, real_t cutoff, const std::int32_t *types, const T *pt, index_t m,
                  index_t K, std::int32_t *idx, std::int32_t *shift, std::int32_t *count,
                  std::int32_t *status) {
    for (index_t a = 0; a < 2 * B; a++) status[a] = 0;
    /* threads only for enough work: a wake-up costs more than a small batch's whole search */
    const bool par = double(B) * double(N) * double(N) > 65536.0;
#pragma omp parallel for schedule(dynamic, 16) if (par)
    for (index_t t = 0; t < B * N; t++) {
        const index_t b = t / N, i = t % N;
        count[t] = 0;
        if (i >= n_atoms[b]) continue;                       /* a padding row */
        real_t c[9];
        for (int k = 0; k < 9; k++) c[k] = static_cast<real_t>(cells[9 * b + k]);
        const BatchedCell g = batched_cell(c, pbc[b], cutoff);
        bool invalid = !g.ok, clipped = false;
        if (g.ok) {
            const index_t w = batched_row(g, pbc[b], pos + 3 * N * b, n_atoms[b], i, cutoff,
                                          types ? types + N * b : nullptr, pt, m, K, idx + K * t,
                                          shift + 3 * K * t, invalid);
            count[t] = static_cast<std::int32_t>(w);
            clipped = w > K;
        }
        if (clipped) {
#pragma omp atomic write
            status[2 * b] = 1;
        }
        if (invalid) {
#pragma omp atomic write
            status[2 * b + 1] = 1;
        }
    }
    /* an invalid structure has no neighbours at all, not a partial list */
    for (index_t b = 0; b < B; b++)
        if (status[2 * b + 1])
            for (index_t i = 0; i < N; i++) count[N * b + i] = 0;
}

}  // namespace

error_t neighbour_matrix_batched(index_t B, index_t N, const std::int32_t *n_atoms, const void *positions,
                                 const void *cells, bool single, const std::int32_t *pbc, real_t cutoff,
                                 const std::int32_t *types, const void *per_type_cutoff, index_t ncutoffs,
                                 index_t K, std::int32_t *idx, std::int32_t *shift, std::int32_t *count,
                                 std::int32_t *status) {
    clear_error();
    if (!(cutoff > 0) || !std::isfinite(cutoff))
        return set_invalid_argument("cutoff must be positive and finite.");
    if (K < 1) return set_invalid_argument("max_neighbours must be positive.");
    if (B < 0 || N < 0 || ncutoffs < 0) return set_invalid_argument("negative batch, atom or type count.");
    if (ncutoffs > 0 && (!types || !per_type_cutoff))
        return set_invalid_argument("per-type cutoffs need types and the cutoff matrix.");
    for (index_t b = 0; b < B; b++)
        if (n_atoms[b] < 0 || n_atoms[b] > N)
            return set_invalid_argumentf("n_atoms[%ld] = %d is outside [0, %ld].", long(b), int(n_atoms[b]),
                                         long(N));
    const std::int32_t *ty = ncutoffs > 0 ? types : nullptr;
    if (single)
        batched_host(B, N, n_atoms, static_cast<const float *>(positions), static_cast<const float *>(cells),
                     pbc, cutoff, ty, ncutoffs > 0 ? static_cast<const float *>(per_type_cutoff) : nullptr,
                     ncutoffs, K, idx, shift, count, status);
    else
        batched_host(B, N, n_atoms, static_cast<const double *>(positions),
                     static_cast<const double *>(cells), pbc, cutoff, ty,
                     ncutoffs > 0 ? static_cast<const double *>(per_type_cutoff) : nullptr, ncutoffs, K, idx,
                     shift, count, status);
    return NL_SUCCESS;
}

}  // namespace matscipy
