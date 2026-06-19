/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#include "triplet_list.hh"

namespace matscipy {

void triplet_list(index_t n_first, const index_t *first_i,
                  const real_t *absdist, real_t cutoff,
                  std::vector<index_t> &ij_t, std::vector<index_t> &ik_t) {
    ij_t.clear();
    ik_t.clear();

    for (index_t r = 0; r < n_first - 1; r++) {
        for (index_t ij = first_i[r]; ij < first_i[r + 1]; ij++) {
            for (index_t ik = first_i[r]; ik < first_i[r + 1]; ik++) {
                if (ij == ik) continue;
                if (absdist &&
                    (absdist[ij] >= cutoff || absdist[ik] >= cutoff)) {
                    continue;
                }
                ij_t.push_back(ij);
                ik_t.push_back(ik);
            }
        }
    }
}

}  // namespace matscipy
