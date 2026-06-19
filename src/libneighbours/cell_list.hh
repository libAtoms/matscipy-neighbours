/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#ifndef MATSCIPY_CELL_LIST_HH
#define MATSCIPY_CELL_LIST_HH

#include <cstdint>
#include <vector>

#include "types.hh"

namespace matscipy {

/* Order in which atoms of different cells are laid out in `sorted_atom`.
   Linear is z-major (counting sort); Morton interleaves the cell coordinates
   (Z-curve) for tighter 3D spatial locality. Both group each cell contiguously;
   only the order of cells differs. */
enum class CellOrder { Linear, Morton };

/*
 * Atoms binned into an n1 x n2 x n3 grid (Ihmsen et al. 2010, Band et al.
 * 2019). Atoms of one cell are a contiguous slice of `sorted_atom`. Two
 * backends:
 *
 *   dense  — `cell_first[c]` / `cell_count[c]` indexed by linear cell index
 *            c = c1 + n1*(c2 + n2*c3). O(ncells) memory. Used when the grid is
 *            not too large.
 *   sparse — an open-addressing hash table keyed by the (64-bit) linear cell
 *            index. O(#non-empty cells) memory. Used for huge/sparse grids
 *            (lots of vacuum), where a dense array would be wasteful or
 *            overflow.
 */
struct CellList {
    index_t n1 = 0, n2 = 0, n3 = 0;
    bool sparse = false;
    std::vector<index_t> sorted_atom;  /* size nat */

    /* Dense backend. */
    std::vector<index_t> cell_first;   /* size ncells */
    std::vector<index_t> cell_count;   /* size ncells */

    /* Sparse backend (open addressing, power-of-two capacity). */
    std::vector<std::int64_t> hkey;    /* size cap, -1 = empty */
    std::vector<index_t> hfirst;       /* size cap */
    std::vector<index_t> hcount;       /* size cap */
    std::int64_t hmask = 0;            /* cap - 1 */
};

/* 64-bit hash used by both the sparse build and query (Fibonacci hashing). */
inline std::int64_t cell_hash(std::int64_t key) {
    std::uint64_t h = static_cast<std::uint64_t>(key) * 0x9E3779B97F4A7C15ull;
    return static_cast<std::int64_t>(h >> 1);  /* keep non-negative */
}

/* Dense lookup: neighbour cell coords -> [b, e) slice of sorted_atom. */
struct DenseQuery {
    index_t n1, n2;
    const index_t *cell_first;
    const index_t *cell_count;
    inline void slice(index_t c1, index_t c2, index_t c3, index_t &b,
                      index_t &e) const {
        index_t c = c1 + n1 * (c2 + n2 * c3);
        b = cell_first[c];
        e = b + cell_count[c];
    }
};

/* Sparse lookup: hash the 64-bit linear cell key, probe linearly. */
struct SparseQuery {
    index_t n1, n2;
    std::int64_t mask;
    const std::int64_t *hkey;
    const index_t *hfirst;
    const index_t *hcount;
    inline void slice(index_t c1, index_t c2, index_t c3, index_t &b,
                      index_t &e) const {
        std::int64_t key = static_cast<std::int64_t>(c1) +
                           static_cast<std::int64_t>(n1) *
                               (static_cast<std::int64_t>(c2) +
                                static_cast<std::int64_t>(n2) * c3);
        std::int64_t h = cell_hash(key) & mask;
        while (hkey[h] != -1) {
            if (hkey[h] == key) {
                b = hfirst[h];
                e = b + hcount[h];
                return;
            }
            h = (h + 1) & mask;
        }
        b = 0;
        e = 0;  /* empty cell */
    }
};

/*
 * Bin atoms into an n1 x n2 x n3 grid. `raw_cell` holds the per-atom (un-wrapped)
 * integer cell coordinates (3*nat, original atom order), as produced by
 * position_to_cell_index(); periodic directions are wrapped and non-periodic
 * directions clamped here. The dense/sparse backend is chosen automatically.
 */
void build_cell_list(const index_t *raw_cell, const bool *pbc, index_t nat,
                     index_t n1, index_t n2, index_t n3, CellOrder order,
                     CellList &cl);

}  // namespace matscipy

#endif
