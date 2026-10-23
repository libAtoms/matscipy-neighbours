/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * EXPERIMENTAL: the neighbour matrix of a *batch* of independent structures in
 * one call -- many small cells (unit cells, small supercells) at once, where a
 * per-structure build is dominated by its fixed cost (a cell grid, a scan and
 * ~15 kernel launches per structure on the GPU).
 *
 * The search is brute force: every atom of a structure against every atom of
 * the same structure, over the periodic images within the cutoff, O(n^2 S) per
 * structure (S images). That is the right algorithm for the small cells
 * batching is for; for large cells use the cell-list builds. Positions are
 * wrapped into the cell before the search (the wrap folded into the returned
 * shift), so they may lie outside it; images of any order are found, so the
 * cell may be smaller than the cutoff.
 *
 * Layout: B structures padded to N atoms. Inputs positions [B, N, 3] and cells
 * [B, 9] (row-major, rows the lattice vectors), float64 or float32 (the search
 * runs in double); n_atoms [B] (live atoms per structure, the first n_atoms[b]
 * rows); pbc [B] (bit k = periodic along k); per-type cutoffs as the other
 * builds (types [B, N] int32 in [0, ncutoffs), the cutoff matrix
 * [ncutoffs^2], not squared, positions' dtype), or none. Outputs, int32: idx
 * [B, N, K], shift [B, N, K, 3] (r[j] - r[i] + shift @ cell is the pair
 * vector), count [B, N] (the true count, which may exceed K; 0 on padding rows),
 * status [B, 2] per structure: (some row clipped at K, invalid input: a
 * non-finite position, a degenerate cell, a type out of range, or a per-type
 * cutoff above `cutoff`; that structure's counts are then 0). Slots past
 * min(count, K) are unspecified. A pair is a neighbour when |d| < cutoff (and
 * < its per-type cutoff), as the cell-list builds decide; row order is by j, then
 * by image, and differs from theirs.
 */
#ifndef MATSCIPY_NEIGHBOUR_MATRIX_BATCHED_HH
#define MATSCIPY_NEIGHBOUR_MATRIX_BATCHED_HH

#include <cmath>
#include <cstdint>

#include "tools.hh"
#include "types.hh"

namespace matscipy {

/* One structure's geometry for the search: its cell, inv(cell^T) (fractional =
   inv @ r, rows the reciprocal vectors), the interplanar spacings h and the
   most images to visit per axis. */
struct BatchedCell {
    real_t cell[9];
    real_t inv[9];
    real_t h[3];
    index_t n[3];
    bool ok;
};

MATSCIPY_HD inline BatchedCell batched_cell(const real_t *c, int pbc_mask, real_t cutoff) {
    BatchedCell g;
    for (int k = 0; k < 9; k++) g.cell[k] = c[k];
    const real_t *c1 = &c[0], *c2 = &c[3], *c3 = &c[6];
    real_t nrm[3][3];
    auto cross = [](const real_t *a, const real_t *b, real_t *o) {
        o[0] = a[1] * b[2] - a[2] * b[1];
        o[1] = a[2] * b[0] - a[0] * b[2];
        o[2] = a[0] * b[1] - a[1] * b[0];
    };
    auto norm = [](const real_t *a) { return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); };
    cross(c2, c3, nrm[0]);
    cross(c3, c1, nrm[1]);
    cross(c1, c2, nrm[2]);
    const real_t det = c1[0] * nrm[0][0] + c1[1] * nrm[0][1] + c1[2] * nrm[0][2];
    const real_t volume = fabs(det);
    g.ok = volume > 1e-12 * norm(c1) * norm(c2) * norm(c3) && is_finite(volume);
    for (int k = 0; k < 3; k++) {
        g.n[k] = 0;
        for (int m = 0; m < 3; m++) g.inv[3 * k + m] = g.ok ? nrm[k][m] / det : 0;
        g.h[k] = g.ok ? volume / norm(nrm[k]) : 0;
        if (g.ok && (pbc_mask >> k & 1))   /* images within the cutoff of the (wrapped) cell */
            g.n[k] = static_cast<index_t>(ceil(cutoff / g.h[k]));
    }
    return g;
}

/* Position a of a structure's padded rows, as double. */
template <typename T>
MATSCIPY_HD inline void batched_load(const T *pos, index_t a, real_t *r) {
    for (int k = 0; k < 3; k++) r[k] = static_cast<real_t>(pos[3 * a + k]);
}

/* r wrapped into the cell along its periodic axes: r' = r - w @ cell. */
MATSCIPY_HD inline void batched_wrap(const BatchedCell &g, int pbc_mask, const real_t *r, real_t *rw,
                                     index_t *w) {
    for (int k = 0; k < 3; k++) {
        const real_t f = g.inv[3 * k] * r[0] + g.inv[3 * k + 1] * r[1] + g.inv[3 * k + 2] * r[2];
        w[k] = (pbc_mask >> k & 1) ? static_cast<index_t>(floor(f)) : 0;
    }
    for (int m = 0; m < 3; m++)
        rw[m] = r[m] - w[0] * g.cell[m] - w[1] * g.cell[3 + m] - w[2] * g.cell[6 + m];
}

/* The neighbours of atom i of one structure (n live atoms): writes the first K
   to idx_row [K] and shift_row [3 K], returns the true count. `invalid` is set
   for a non-finite position of i, a type out of range or a per-type cutoff
   above `cutoff` (other atoms' bad positions are flagged by their own rows and
   their pairs skipped). pt: the per-type cutoff matrix [m^2] (not squared, the
   positions' dtype), or null. */
template <typename T>
MATSCIPY_HD inline index_t batched_row(const BatchedCell &g, int pbc_mask, const T *pos, index_t n,
                                       index_t i, real_t cutoff, const std::int32_t *types, const T *pt,
                                       index_t m, index_t K, std::int32_t *idx_row,
                                       std::int32_t *shift_row, bool &invalid) {
    const real_t cutoff_sq = cutoff * cutoff;
    real_t ri[3], rwi[3];
    index_t wi[3];
    batched_load(pos, i, ri);
    if (!is_finite(ri[0]) || !is_finite(ri[1]) || !is_finite(ri[2])) {
        invalid = true;
        return 0;
    }
    const index_t ti = types ? types[i] : 0;
    if (types && (ti < 0 || ti >= m)) {
        invalid = true;
        return 0;
    }
    batched_wrap(g, pbc_mask, ri, rwi, wi);
    index_t w = 0;
    for (index_t j = 0; j < n; j++) {
        real_t rj[3], rwj[3];
        index_t wj[3];
        batched_load(pos, j, rj);
        if (!is_finite(rj[0]) || !is_finite(rj[1]) || !is_finite(rj[2])) continue;
        const index_t tj = types ? types[j] : 0;
        if (types && (tj < 0 || tj >= m)) continue;
        batched_wrap(g, pbc_mask, rj, rwj, wj);
        real_t cut_ij = cutoff_sq;
        if (pt) {
            const real_t c = static_cast<real_t>(pt[ti * m + tj]);
            if (!(c <= cutoff) || !is_finite(c)) {
                invalid = true;
                continue;
            }
            cut_ij = c * c;
        }
        const real_t d0[3] = {rwj[0] - rwi[0], rwj[1] - rwi[1], rwj[2] - rwi[2]};
        /* |d| >= |df_k + s_k| h_k (the distance across the planes of axis k), so only the s_k
           with that below the cutoff can hold a neighbour: ~2 per axis, not 2 n_k + 1. The bound is
           widened by a relative 1e-12; the distance test below is the exact one. */
        index_t lo[3], hi[3];
        for (int k = 0; k < 3; k++) {
            const real_t df = g.inv[3 * k] * d0[0] + g.inv[3 * k + 1] * d0[1] + g.inv[3 * k + 2] * d0[2];
            const real_t r = g.n[k] ? cutoff / g.h[k] * (1 + 1e-12) : 0;
            lo[k] = g.n[k] ? static_cast<index_t>(ceil(-r - df)) : 0;   /* not clamped to n_k: wrap */
            hi[k] = g.n[k] ? static_cast<index_t>(floor(r - df)) : 0;   /* rounding can put |df| > 1 */
        }
        for (index_t s0 = lo[0]; s0 <= hi[0]; s0++)
            for (index_t s1 = lo[1]; s1 <= hi[1]; s1++)
                for (index_t s2 = lo[2]; s2 <= hi[2]; s2++) {
                    if (j == i && s0 == 0 && s1 == 0 && s2 == 0) continue;
                    real_t d2 = 0;
                    for (int k = 0; k < 3; k++) {
                        const real_t d = d0[k] + s0 * g.cell[k] + s1 * g.cell[3 + k] + s2 * g.cell[6 + k];
                        d2 += d * d;
                    }
                    /* !(d < c), so a NaN distance is never a neighbour (as visit_neighbours) */
                    if (!(d2 < cutoff_sq) || !(d2 < cut_ij)) continue;
                    if (w < K) {
                        idx_row[w] = static_cast<std::int32_t>(j);
                        shift_row[3 * w + 0] = static_cast<std::int32_t>(s0 - wj[0] + wi[0]);
                        shift_row[3 * w + 1] = static_cast<std::int32_t>(s1 - wj[1] + wi[1]);
                        shift_row[3 * w + 2] = static_cast<std::int32_t>(s2 - wj[2] + wi[2]);
                    }
                    w++;
                }
    }
    return w;
}

/* The batch on the host (OpenMP over all rows). `single`: float32 positions,
   cells and per-type cutoffs. NL_INVALID_ARGUMENT for a bad cutoff, K, or
   n_atoms outside [0, N]; per-structure bad inputs are flagged in status. */
error_t neighbour_matrix_batched(index_t B, index_t N, const std::int32_t *n_atoms, const void *positions,
                                 const void *cells, bool single, const std::int32_t *pbc, real_t cutoff,
                                 const std::int32_t *types, const void *per_type_cutoff, index_t ncutoffs,
                                 index_t K, std::int32_t *idx, std::int32_t *shift, std::int32_t *count,
                                 std::int32_t *status);

#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
/* The same on the device, enqueued on `stream` (a cudaStream_t / hipStream_t as
   void*): every pointer is device memory; two kernel launches, no host
   synchronisation, allocation or read-back (so it can sit in a CUDA graph).
   n_atoms outside [0, N] marks that structure invalid. Only enqueue errors are
   reported. */
error_t neighbour_matrix_batched_gpu(index_t B, index_t N, const std::int32_t *n_atoms, const void *positions,
                                     const void *cells, bool single, const std::int32_t *pbc, real_t cutoff,
                                     const std::int32_t *types, const void *per_type_cutoff,
                                     index_t ncutoffs, index_t K, void *stream, std::int32_t *idx,
                                     std::int32_t *shift, std::int32_t *count, std::int32_t *status);
#endif

}  // namespace matscipy

#endif
