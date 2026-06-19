/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Binning neighbour-list construction as data-parallel primitives (Ihmsen et
 * al. 2010, Band et al. 2019):
 *   - atoms are sorted into a CSR cell list (cell_list.{hh,cc});
 *   - per-atom data (cell index, bin-relative position, cutoff/type) is
 *     precomputed and stored in cell-sorted order, so the inner pair loop reads
 *     contiguous memory and does no matrix-vector products;
 *   - pairs are produced in two passes (count, then fill into exactly-sized
 *     buffers), so each atom's work is independent — OpenMP here, GPU later.
 */

#include "neighbour_list.hh"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>

#include "cell_list.hh"
#include "error.hh"
#include "tools.hh"

namespace matscipy {

namespace {

/*
 * All per-atom arrays below are in *cell-sorted* order: index `s` refers to the
 * atom `cl->sorted_atom[s]`. The CSR `cell_start` indexes the same order, so the
 * inner loop walks each cell as a contiguous run.
 */
struct NeighbourContext {
    const bool *pbc;
    index_t n1, n2, n3;
    index_t nx, ny, nz;
    real_t bin1[3], bin2[3], bin3[3];
    real_t cutoff_sq;
    const real_t *per_type_cutoff_sq;
    index_t ncutoffs;
    const CellList *cl;
    const index_t *raw_cell;    /* 3*nat, sorted: raw cell index */
    const index_t *rel_cell;    /* 3*nat, sorted: trunc-if-non-periodic index */
    const real_t *rel_pos;      /* 3*nat, sorted: bin-relative position */
    const real_t *per_atom;     /* nat, sorted, or nullptr */
    const index_t *types;       /* nat, sorted, or nullptr */
};

/*
 * Visit every neighbour of the atom at sorted position `si`, invoking
 *     emit(s_j, dr[3], abs_dr_sq, shift[3])
 * where `s_j` is the neighbour's sorted position. Used by both passes.
 */
template <typename Emit>
inline void visit_neighbours(const NeighbourContext &c, index_t si,
                             Emit &&emit) {
    const CellList &cl = *c.cl;
    const index_t n1 = c.n1, n2 = c.n2, n3 = c.n3;

    const index_t *raw_i = &c.raw_cell[3 * si];
    const real_t *dri = &c.rel_pos[3 * si];

    /* Home-cell index used to step through neighbouring bins. */
    const index_t ci1 = c.pbc[0] ? bin_wrap(raw_i[0], n1) : bin_trunc(raw_i[0], n1);
    const index_t ci2 = c.pbc[1] ? bin_wrap(raw_i[1], n2) : bin_trunc(raw_i[1], n2);
    const index_t ci3 = c.pbc[2] ? bin_wrap(raw_i[2], n3) : bin_trunc(raw_i[2], n3);

    for (index_t z = -c.nz; z <= c.nz; z++) {
        index_t cj3 = ci3 + z;
        if (c.pbc[2]) cj3 = bin_wrap(cj3, n3);
        if (cj3 < 0 || cj3 >= n3) continue;
        index_t ncj3 = n2 * cj3;
        real_t off3[3] = {z * c.bin3[0], z * c.bin3[1], z * c.bin3[2]};

        for (index_t y = -c.ny; y <= c.ny; y++) {
            index_t cj2 = ci2 + y;
            if (c.pbc[1]) cj2 = bin_wrap(cj2, n2);
            if (cj2 < 0 || cj2 >= n2) continue;
            index_t ncj2 = n1 * (cj2 + ncj3);
            real_t off2[3] = {off3[0] + y * c.bin2[0], off3[1] + y * c.bin2[1],
                              off3[2] + y * c.bin2[2]};

            for (index_t x = -c.nx; x <= c.nx; x++) {
                index_t cj1 = ci1 + x;
                if (c.pbc[0]) cj1 = bin_wrap(cj1, n1);
                if (cj1 < 0 || cj1 >= n1) continue;
                index_t ncj = cj1 + ncj2;

                real_t off[3] = {off2[0] + x * c.bin1[0], off2[1] + x * c.bin1[1],
                                 off2[2] + x * c.bin1[2]};

                /* Atoms in the neighbouring bin: a contiguous CSR slice. */
                const index_t begin = cl.cell_start[ncj];
                const index_t end = cl.cell_start[ncj + 1];
                for (index_t sj = begin; sj < end; sj++) {
                    /* Same atom in the home image is not its own neighbour.
                       sorted positions are unique, so si == sj identifies it. */
                    if (sj == si && x == 0 && y == 0 && z == 0) continue;

                    const real_t *drj = &c.rel_pos[3 * sj];
                    real_t dr[3] = {drj[0] - dri[0] + off[0],
                                    drj[1] - dri[1] + off[1],
                                    drj[2] - dri[2] + off[2]};
                    real_t abs_dr_sq =
                        dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];

                    if (abs_dr_sq >= c.cutoff_sq) continue;

                    bool inside_cutoff = true;
                    if (c.per_atom) {
                        real_t cc = c.per_atom[si] + c.per_atom[sj];
                        inside_cutoff = abs_dr_sq < cc * cc;
                    } else if (c.per_type_cutoff_sq && c.types) {
                        index_t ti = c.types[si], tj = c.types[sj];
                        if (ti >= 0 && ti < c.ncutoffs && tj >= 0 &&
                            tj < c.ncutoffs) {
                            inside_cutoff =
                                abs_dr_sq <
                                c.per_type_cutoff_sq[ti * c.ncutoffs + tj];
                        }
                    }
                    if (!inside_cutoff) continue;

                    const index_t *crj = &c.rel_cell[3 * sj];
                    index_t shift[3] = {(raw_i[0] - crj[0] + x) / n1,
                                        (raw_i[1] - crj[1] + y) / n2,
                                        (raw_i[2] - crj[2] + z) / n3};
                    emit(sj, dr, abs_dr_sq, shift);
                }
            }
        }
    }
}

}  // namespace

error_t neighbour_list(int quantities, const real_t cell_origin[3],
                       const real_t cell[9], const real_t inv_cell[9],
                       const bool pbc[3], index_t nat, const real_t *r,
                       real_t cutoff, const real_t *per_atom_cutoff,
                       const real_t *per_type_cutoff_sq, index_t ncutoffs,
                       const index_t *types, NeighbourList &out) {
    clear_error();

    out.first.clear();
    out.secnd.clear();
    out.distvec.clear();
    out.absdist.clear();
    out.shift.clear();
    out.npairs = 0;

    const real_t *cell1 = &cell[0], *cell2 = &cell[3], *cell3 = &cell[6];

    /* Vectors normal to the cell faces; their lengths give the face areas. */
    real_t normal1[3], normal2[3], normal3[3];
    cross_product(cell2, cell3, normal1);
    cross_product(cell3, cell1, normal2);
    cross_product(cell1, cell2, normal3);
    real_t volume = std::fabs(cell3[0] * normal3[0] + cell3[1] * normal3[1] +
                              cell3[2] * normal3[2]);

    if (volume < 1e-12) {
        return set_error("Zero cell volume.");
    }

    /* Distances between opposite cell faces. */
    real_t len1 = volume / norm(normal1);
    real_t len2 = volume / norm(normal2);
    real_t len3 = volume / norm(normal3);

    /* Number of cells for the spatial subdivision. */
    index_t n1 = std::max(static_cast<index_t>(std::floor(len1 / cutoff)), 1);
    index_t n2 = std::max(static_cast<index_t>(std::floor(len2 / cutoff)), 1);
    index_t n3 = std::max(static_cast<index_t>(std::floor(len3 / cutoff)), 1);

    /* Avoid overflow in the total number of cells (e.g. lots of vacuum). */
    bool warned = false;
    while (static_cast<double>(n1) * n2 * n3 > INT_MAX) {
        if (!warned) {
            set_warning(
                "Ratio of simulation cell size to cutoff is very large; "
                "reducing number of bins for neighbour list search, but this "
                "may be slow. Are you using a cell with lots of vacuum?");
            warned = true;
        }
        n1 /= 2;
        if (n1 <= 0) n1 = 1;
        n2 /= 2;
        if (n2 <= 0) n2 = 1;
        n3 /= 2;
        if (n3 <= 0) n3 = 1;
    }

    assert(n1 > 0 && n2 > 0 && n3 > 0);

    /* Bin shape. */
    real_t bin1[3], bin2[3], bin3[3];
    for (int k = 0; k < 3; k++) {
        bin1[k] = cell1[k] / n1;
        bin2[k] = cell2[k] / n2;
        bin3[k] = cell3[k] / n3;
    }

    /* Bin the atoms (counting-sort CSR cell list). */
    CellList cl;
    build_cell_list(cell_origin, inv_cell, pbc, nat, r, n1, n2, n3, cl);

    /* Precompute per-atom data directly in cell-sorted order so the inner pair
       loop reads contiguous memory (the only scattered read is r[3*a] here,
       once per atom). */
    std::vector<index_t> raw_cell(3 * nat);
    std::vector<index_t> rel_cell(3 * nat);
    std::vector<real_t> rel_pos(3 * nat);
    std::vector<real_t> per_atom_s(per_atom_cutoff ? nat : 0);
    std::vector<index_t> types_s(types ? nat : 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (index_t s = 0; s < nat; s++) {
        const index_t a = cl.sorted_atom[s];
        index_t c1, c2, c3;
        position_to_cell_index(cell_origin, inv_cell, &r[3 * a], n1, n2, n3, &c1,
                               &c2, &c3);
        raw_cell[3 * s + 0] = c1;
        raw_cell[3 * s + 1] = c2;
        raw_cell[3 * s + 2] = c3;

        const index_t r1 = pbc[0] ? c1 : bin_trunc(c1, n1);
        const index_t r2 = pbc[1] ? c2 : bin_trunc(c2, n2);
        const index_t r3 = pbc[2] ? c3 : bin_trunc(c3, n3);
        rel_cell[3 * s + 0] = r1;
        rel_cell[3 * s + 1] = r2;
        rel_cell[3 * s + 2] = r3;

        for (int k = 0; k < 3; k++) {
            rel_pos[3 * s + k] =
                r[3 * a + k] - r1 * bin1[k] - r2 * bin2[k] - r3 * bin3[k];
        }
        if (per_atom_cutoff) per_atom_s[s] = per_atom_cutoff[a];
        if (types) types_s[s] = types[a];
    }

    /* Assemble the context shared by both passes. */
    NeighbourContext ctx;
    ctx.pbc = pbc;
    ctx.n1 = n1;
    ctx.n2 = n2;
    ctx.n3 = n3;
    ctx.nx = static_cast<index_t>(std::ceil(cutoff * n1 / len1));
    ctx.ny = static_cast<index_t>(std::ceil(cutoff * n2 / len2));
    ctx.nz = static_cast<index_t>(std::ceil(cutoff * n3 / len3));
    for (int k = 0; k < 3; k++) {
        ctx.bin1[k] = bin1[k];
        ctx.bin2[k] = bin2[k];
        ctx.bin3[k] = bin3[k];
    }
    ctx.cutoff_sq = cutoff * cutoff;
    ctx.per_type_cutoff_sq = per_type_cutoff_sq;
    ctx.ncutoffs = ncutoffs;
    ctx.cl = &cl;
    ctx.raw_cell = raw_cell.data();
    ctx.rel_cell = rel_cell.data();
    ctx.rel_pos = rel_pos.data();
    ctx.per_atom = per_atom_cutoff ? per_atom_s.data() : nullptr;
    ctx.types = types ? types_s.data() : nullptr;

    /* Pass 1: count neighbours per atom (independent per atom). Counts are
       indexed by the *original* atom so the output ends up sorted by i. */
    std::vector<index_t> offset(nat + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
    for (index_t si = 0; si < nat; si++) {
        index_t count = 0;
        visit_neighbours(ctx, si, [&count](index_t, const real_t *, real_t,
                                           const index_t *) { count++; });
        offset[cl.sorted_atom[si] + 1] = count;
    }

    /* Exclusive prefix sum of the per-atom counts -> write offsets. */
    for (index_t i = 0; i < nat; i++) {
        offset[i + 1] += offset[i];
    }
    const index_t npairs = offset[nat];
    out.npairs = npairs;

    /* Allocate exactly-sized output buffers for the requested quantities. */
    const bool want_first = quantities & QUANTITY_FIRST;
    const bool want_secnd = quantities & QUANTITY_SECOND;
    const bool want_distvec = quantities & QUANTITY_DISTVEC;
    const bool want_absdist = quantities & QUANTITY_ABSDIST;
    const bool want_shift = quantities & QUANTITY_SHIFT;
    if (want_first) out.first.resize(npairs);
    if (want_secnd) out.secnd.resize(npairs);
    if (want_distvec) out.distvec.resize(3 * npairs);
    if (want_absdist) out.absdist.resize(npairs);
    if (want_shift) out.shift.resize(3 * npairs);

    /* Pass 2: write each atom's pairs into its own slice (no shared counter). */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
    for (index_t si = 0; si < nat; si++) {
        const index_t i = cl.sorted_atom[si];
        index_t w = offset[i];
        visit_neighbours(ctx, si,
                         [&](index_t sj, const real_t *dr, real_t abs_dr_sq,
                             const index_t *shift) {
                             if (want_first) out.first[w] = i;
                             if (want_secnd) out.secnd[w] = cl.sorted_atom[sj];
                             if (want_distvec) {
                                 out.distvec[3 * w + 0] = dr[0];
                                 out.distvec[3 * w + 1] = dr[1];
                                 out.distvec[3 * w + 2] = dr[2];
                             }
                             if (want_absdist)
                                 out.absdist[w] = std::sqrt(abs_dr_sq);
                             if (want_shift) {
                                 out.shift[3 * w + 0] = shift[0];
                                 out.shift[3 * w + 1] = shift[1];
                                 out.shift[3 * w + 2] = shift[2];
                             }
                             w++;
                         });
    }

    return NL_SUCCESS;
}

}  // namespace matscipy
