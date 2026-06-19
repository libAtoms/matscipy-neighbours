/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#include "cell_list.hh"

#include <algorithm>
#include <cstdint>
#include <numeric>

#include "tools.hh"

namespace matscipy {

namespace {

/* Wrap (periodic) / clamp (non-periodic) a raw cell coordinate into [0, n). */
inline index_t fold(index_t c, index_t n, bool periodic) {
    return periodic ? bin_wrap(c, n) : bin_trunc(c, n);
}

/* Spread the low 21 bits of x so they occupy every third bit (for Morton). */
inline std::uint64_t part1by2(std::uint64_t x) {
    x &= 0x1fffffull;
    x = (x | (x << 32)) & 0x1f00000000ffffull;
    x = (x | (x << 16)) & 0x1f0000ff0000ffull;
    x = (x | (x << 8)) & 0x100f00f00f00f00full;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
    x = (x | (x << 2)) & 0x1249249249249249ull;
    return x;
}

inline std::uint64_t morton3(index_t a, index_t b, index_t c) {
    return part1by2(a) | (part1by2(b) << 1) | (part1by2(c) << 2);
}

/* Wrapped cell coordinates of every atom (used by both backends). */
void folded_coords(const index_t *raw_cell, const bool *pbc, index_t nat,
                   index_t n1, index_t n2, index_t n3,
                   std::vector<index_t> &cc) {
    cc.resize(3 * nat);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (index_t a = 0; a < nat; a++) {
        cc[3 * a + 0] = fold(raw_cell[3 * a + 0], n1, pbc[0]);
        cc[3 * a + 1] = fold(raw_cell[3 * a + 1], n2, pbc[1]);
        cc[3 * a + 2] = fold(raw_cell[3 * a + 2], n3, pbc[2]);
    }
}

void build_dense(const std::vector<index_t> &cc, index_t nat, index_t n1,
                 index_t n2, index_t n3, CellOrder order, CellList &cl) {
    const index_t ncells = n1 * n2 * n3;
    cl.cell_first.assign(ncells, 0);
    cl.cell_count.assign(ncells, 0);
    cl.sorted_atom.resize(nat);

    std::vector<index_t> lin(nat);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (index_t a = 0; a < nat; a++) {
        lin[a] = cc[3 * a] + n1 * (cc[3 * a + 1] + n2 * cc[3 * a + 2]);
    }

    /* Histogram of atoms per cell. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (index_t a = 0; a < nat; a++) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        cl.cell_count[lin[a]]++;
    }

    if (order == CellOrder::Linear) {
        /* Prefix sum -> first index of each cell. */
        index_t acc = 0;
        for (index_t c = 0; c < ncells; c++) {
            cl.cell_first[c] = acc;
            acc += cl.cell_count[c];
        }
        /* Scatter atoms into their cell's slice. */
        std::vector<index_t> cursor(cl.cell_first);
        for (index_t a = 0; a < nat; a++) {
            cl.sorted_atom[cursor[lin[a]]++] = a;
        }
    } else {
        /* Morton order: sort atoms by Z-curve key, then record each cell's
           contiguous run. */
        std::vector<std::uint64_t> key(nat);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index_t a = 0; a < nat; a++) {
            key[a] = morton3(cc[3 * a], cc[3 * a + 1], cc[3 * a + 2]);
        }
        std::iota(cl.sorted_atom.begin(), cl.sorted_atom.end(), 0);
        std::sort(cl.sorted_atom.begin(), cl.sorted_atom.end(),
                  [&](index_t x, index_t y) { return key[x] < key[y]; });
        /* Each cell is one contiguous run (the Morton key is unique per cell);
           record where each run starts. cell_count already holds the lengths. */
        for (index_t s = 0; s < nat; s++) {
            index_t c = lin[cl.sorted_atom[s]];
            if (s == 0 || lin[cl.sorted_atom[s - 1]] != c) {
                cl.cell_first[c] = s;
            }
        }
    }
}

void build_sparse(const std::vector<index_t> &cc, index_t nat, index_t n1,
                  index_t n2, CellList &cl) {
    /* Power-of-two capacity, load factor <= 0.5. */
    std::int64_t cap = 1;
    while (cap < 2 * static_cast<std::int64_t>(nat) + 1) cap <<= 1;
    cl.hmask = cap - 1;
    cl.hkey.assign(cap, -1);
    cl.hfirst.assign(cap, 0);
    cl.hcount.assign(cap, 0);
    cl.sorted_atom.resize(nat);

    auto key_of = [&](index_t a) -> std::int64_t {
        return static_cast<std::int64_t>(cc[3 * a]) +
               static_cast<std::int64_t>(n1) *
                   (static_cast<std::int64_t>(cc[3 * a + 1]) +
                    static_cast<std::int64_t>(n2) * cc[3 * a + 2]);
    };

    /* Insert keys and count occupants (serial; sparse is the rare path). */
    for (index_t a = 0; a < nat; a++) {
        std::int64_t key = key_of(a);
        std::int64_t h = cell_hash(key) & cl.hmask;
        while (cl.hkey[h] != -1 && cl.hkey[h] != key) h = (h + 1) & cl.hmask;
        cl.hkey[h] = key;
        cl.hcount[h]++;
    }

    /* Assign each occupied slot its slice start (prefix sum over slots). */
    index_t acc = 0;
    for (std::int64_t h = 0; h <= cl.hmask; h++) {
        if (cl.hkey[h] != -1) {
            cl.hfirst[h] = acc;
            acc += cl.hcount[h];
        }
    }

    /* Scatter atoms. */
    std::vector<index_t> cursor(cl.hfirst);
    for (index_t a = 0; a < nat; a++) {
        std::int64_t key = key_of(a);
        std::int64_t h = cell_hash(key) & cl.hmask;
        while (cl.hkey[h] != key) h = (h + 1) & cl.hmask;
        cl.sorted_atom[cursor[h]++] = a;
    }
}

}  // namespace

void build_cell_list(const index_t *raw_cell, const bool *pbc, index_t nat,
                     index_t n1, index_t n2, index_t n3, CellOrder order,
                     CellList &cl) {
    cl.n1 = n1;
    cl.n2 = n2;
    cl.n3 = n3;

    std::vector<index_t> cc;
    folded_coords(raw_cell, pbc, nat, n1, n2, n3, cc);

    /* Choose the backend: dense unless the grid is far larger than the atom
       count (or would overflow a 32-bit cell index). */
    const std::int64_t ncells = static_cast<std::int64_t>(n1) * n2 * n3;
    const std::int64_t budget =
        std::max<std::int64_t>(1 << 20, 8 * static_cast<std::int64_t>(nat));
    cl.sparse = ncells > budget;

    if (!cl.sparse) {
        build_dense(cc, nat, n1, n2, n3, order, cl);
    } else {
        build_sparse(cc, nat, n1, n2, cl);
    }
}

}  // namespace matscipy
