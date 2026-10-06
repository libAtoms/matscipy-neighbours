/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Host segment sum: the segments are independent, so they are summed in
 * parallel (OpenMP) without changing the order within one.
 */

#include "segment_sum.hh"

#include <cstdint>

#include "error.hh"

namespace matscipy {

template <typename T>
error_t segment_sum(index_t n, const index_t *seed, index_t nrows, index_t d,
                    const T *values, T *out, T *total) {
    clear_error();
    if (n < 0 || nrows < 0 || d < 0) {
        return set_invalid_argument(
            "segment_sum: sizes must be non-negative.");
    }
    if (!seed || (nrows * d > 0 && !values) || (n * d > 0 && !out)) {
        return set_invalid_argument("segment_sum: invalid array.");
    }
    index_t prev = 0;
    for (index_t k = 0; k <= n; k++) {
        const index_t s = seed[k] < 0 ? 0 : seed[k];
        if (s < prev || s > nrows) {
            return set_invalid_argumentf(
                "segment_sum: row starts must be non-decreasing and at most "
                "the number of rows (%lld at position %lld).",
                static_cast<long long>(seed[k]), static_cast<long long>(k));
        }
        prev = s;
    }

#pragma omp parallel for schedule(static)
    for (index_t a = 0; a < n; a++) {
        const index_t begin = seed[a] < 0 ? 0 : seed[a];
        const index_t end = seed[a + 1] < 0 ? 0 : seed[a + 1];
        T *o = out + a * d;
        for (index_t c = 0; c < d; c++) o[c] = T(0);
        for (index_t p = begin; p < end; p++) {
            const T *v = values + p * d;
            for (index_t c = 0; c < d; c++) o[c] += v[c];
        }
    }

    if (total) {
        for (index_t c = 0; c < d; c++) total[c] = T(0);
        for (index_t a = 0; a < n; a++)
            for (index_t c = 0; c < d; c++) total[c] += out[a * d + c];
    }
    return NL_SUCCESS;
}

#define MATSCIPY_SEGMENT_SUM(T)                                             \
    template error_t segment_sum<T>(index_t, const index_t *, index_t,      \
                                    index_t, const T *, T *, T *);
MATSCIPY_SEGMENT_SUM(float)
MATSCIPY_SEGMENT_SUM(double)
MATSCIPY_SEGMENT_SUM(std::int32_t)
MATSCIPY_SEGMENT_SUM(std::int64_t)
#undef MATSCIPY_SEGMENT_SUM

}  // namespace matscipy
