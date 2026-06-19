/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#include "first_neighbours.hh"

namespace matscipy {

void first_neighbours(index_t n, index_t nn, const index_t *i_n,
                      index_t *seed) {
    for (index_t k = 0; k <= n; k++) {
        seed[k] = -1;
    }

    /* Empty neighbour list: every row starts (and ends) at 0. */
    if (nn <= 0) {
        for (index_t k = 0; k <= n; k++) seed[k] = 0;
        return;
    }

    seed[i_n[0]] = 0;

    for (index_t k = 1; k < nn; k++) {
        if (i_n[k] != i_n[k - 1]) {
            for (index_t l = i_n[k - 1] + 1; l <= i_n[k]; l++) {
                seed[l] = k;
            }
        }
    }

    for (index_t k = i_n[nn - 1] + 1; k <= n; k++) {
        seed[k] = nn;
    }
}

std::vector<index_t> get_jump_indicies(index_t nn, const index_t *sorted) {
    /* Number of distinct values = number of jumps + 1. */
    index_t n = 0;
    for (index_t i = 0; i < nn - 1; i++) {
        if (sorted[i] != sorted[i + 1]) {
            n++;
        }
    }
    n++;

    std::vector<index_t> seed(n + 1);
    first_neighbours(n, nn, sorted, seed.data());
    return seed;
}

}  // namespace matscipy
