/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Binning neighbour-list construction expressed as data-parallel primitives
 * (Ihmsen et al. 2010, Band et al. 2019): atoms are sorted into a CSR cell list
 * (cell_list.{hh,cc}), then the pairs are produced in two passes — count, then
 * fill into exactly-sized buffers. There is no per-cell linked list and no
 * dynamic reallocation, so each atom's work is independent (ready for OpenMP
 * here and a GPU backend later).
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

/* Geometry and cutoff data shared by both passes over the atoms. */
struct NeighbourContext {
    const real_t *cell_origin;
    const real_t *inv_cell;
    const bool *pbc;
    const real_t *r;
    index_t n1, n2, n3;
    index_t nx, ny, nz;
    real_t bin1[3], bin2[3], bin3[3];
    real_t cutoff_sq;
    const real_t *per_atom_cutoff;
    const real_t *per_type_cutoff_sq;
    index_t ncutoffs;
    const index_t *types;
    const CellList *cl;
};

/*
 * Visit every neighbour pair of atom i, invoking
 *     emit(j, dr[3], abs_dr_sq, shift[3])
 * for each accepted pair. Both the counting and filling passes go through this,
 * so the geometry and cutoff logic lives in exactly one place.
 */
template <typename Emit>
inline void visit_neighbours(const NeighbourContext &c, index_t i, Emit &&emit) {
    const real_t *ri = &c.r[3 * i];
    const CellList &cl = *c.cl;
    const index_t n1 = c.n1, n2 = c.n2, n3 = c.n3;

    index_t ci01, ci02, ci03;
    position_to_cell_index(c.cell_origin, c.inv_cell, ri, n1, n2, n3, &ci01,
                           &ci02, &ci03);

    /* Truncate if non-periodic and outside the simulation domain. */
    index_t ci1 = c.pbc[0] ? ci01 : bin_trunc(ci01, n1);
    index_t ci2 = c.pbc[1] ? ci02 : bin_trunc(ci02, n2);
    index_t ci3 = c.pbc[2] ? ci03 : bin_trunc(ci03, n3);

    /* Position relative to the lower-left corner of the bin. */
    real_t dri[3];
    dri[0] = ri[0] - ci1 * c.bin1[0] - ci2 * c.bin2[0] - ci3 * c.bin3[0];
    dri[1] = ri[1] - ci1 * c.bin1[1] - ci2 * c.bin2[1] - ci3 * c.bin3[1];
    dri[2] = ri[2] - ci1 * c.bin1[2] - ci2 * c.bin2[2] - ci3 * c.bin3[2];

    /* Home-cell index used to step through neighbouring bins. */
    ci1 = c.pbc[0] ? bin_wrap(ci01, n1) : bin_trunc(ci01, n1);
    ci2 = c.pbc[1] ? bin_wrap(ci02, n2) : bin_trunc(ci02, n2);
    ci3 = c.pbc[2] ? bin_wrap(ci03, n3) : bin_trunc(ci03, n3);

    for (index_t z = -c.nz; z <= c.nz; z++) {
        index_t cj3 = ci3 + z;
        if (c.pbc[2]) cj3 = bin_wrap(cj3, n3);
        if (cj3 < 0 || cj3 >= n3) continue;
        cj3 = bin_trunc(cj3, n3);
        index_t ncj3 = n2 * cj3;
        real_t off3[3] = {z * c.bin3[0], z * c.bin3[1], z * c.bin3[2]};

        for (index_t y = -c.ny; y <= c.ny; y++) {
            index_t cj2 = ci2 + y;
            if (c.pbc[1]) cj2 = bin_wrap(cj2, n2);
            if (cj2 < 0 || cj2 >= n2) continue;
            cj2 = bin_trunc(cj2, n2);
            index_t ncj2 = n1 * (cj2 + ncj3);
            real_t off2[3] = {off3[0] + y * c.bin2[0], off3[1] + y * c.bin2[1],
                              off3[2] + y * c.bin2[2]};

            for (index_t x = -c.nx; x <= c.nx; x++) {
                index_t cj1 = ci1 + x;
                if (c.pbc[0]) cj1 = bin_wrap(cj1, n1);
                if (cj1 < 0 || cj1 >= n1) continue;
                cj1 = bin_trunc(cj1, n1);
                index_t ncj = cj1 + ncj2;

                assert(ncj == cj1 + n1 * (cj2 + n2 * cj3));

                real_t off[3] = {off2[0] + x * c.bin1[0], off2[1] + x * c.bin1[1],
                                 off2[2] + x * c.bin1[2]};

                /* Atoms in the neighbouring bin: a contiguous CSR slice. */
                for (index_t idx = cl.cell_start[ncj]; idx < cl.cell_start[ncj + 1];
                     idx++) {
                    index_t j = cl.sorted_atom[idx];

                    if (i != j || x != 0 || y != 0 || z != 0) {
                        const real_t *rj = &c.r[3 * j];

                        index_t cjj1, cjj2, cjj3;
                        position_to_cell_index(c.cell_origin, c.inv_cell, rj, n1,
                                               n2, n3, &cjj1, &cjj2, &cjj3);
                        if (!c.pbc[0]) cjj1 = bin_trunc(cjj1, n1);
                        if (!c.pbc[1]) cjj2 = bin_trunc(cjj2, n2);
                        if (!c.pbc[2]) cjj3 = bin_trunc(cjj3, n3);

                        real_t drj[3];
                        drj[0] = rj[0] - cjj1 * c.bin1[0] - cjj2 * c.bin2[0] -
                                 cjj3 * c.bin3[0];
                        drj[1] = rj[1] - cjj1 * c.bin1[1] - cjj2 * c.bin2[1] -
                                 cjj3 * c.bin3[1];
                        drj[2] = rj[2] - cjj1 * c.bin1[2] - cjj2 * c.bin2[2] -
                                 cjj3 * c.bin3[2];

                        real_t dr[3] = {drj[0] - dri[0] + off[0],
                                        drj[1] - dri[1] + off[1],
                                        drj[2] - dri[2] + off[2]};
                        real_t abs_dr_sq =
                            dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];

                        if (abs_dr_sq < c.cutoff_sq) {
                            bool inside_cutoff = true;
                            if (c.per_atom_cutoff) {
                                real_t cc =
                                    c.per_atom_cutoff[i] + c.per_atom_cutoff[j];
                                inside_cutoff = abs_dr_sq < cc * cc;
                            } else if (c.per_type_cutoff_sq && c.types) {
                                if (c.types[i] >= 0 && c.types[i] < c.ncutoffs &&
                                    c.types[j] >= 0 && c.types[j] < c.ncutoffs) {
                                    real_t c_sq =
                                        c.per_type_cutoff_sq[c.types[i] *
                                                                 c.ncutoffs +
                                                             c.types[j]];
                                    inside_cutoff = abs_dr_sq < c_sq;
                                }
                            }

                            if (inside_cutoff) {
                                index_t shift[3] = {(ci01 - cjj1 + x) / n1,
                                                    (ci02 - cjj2 + y) / n2,
                                                    (ci03 - cjj3 + z) / n3};
                                emit(j, dr, abs_dr_sq, shift);
                            }
                        }
                    }
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

    /* Bin the atoms (counting-sort CSR cell list). */
    CellList cl;
    build_cell_list(cell_origin, inv_cell, pbc, nat, r, n1, n2, n3, cl);

    /* Assemble the geometry/cutoff context shared by both passes. */
    NeighbourContext ctx;
    ctx.cell_origin = cell_origin;
    ctx.inv_cell = inv_cell;
    ctx.pbc = pbc;
    ctx.r = r;
    ctx.n1 = n1;
    ctx.n2 = n2;
    ctx.n3 = n3;
    ctx.nx = static_cast<index_t>(std::ceil(cutoff * n1 / len1));
    ctx.ny = static_cast<index_t>(std::ceil(cutoff * n2 / len2));
    ctx.nz = static_cast<index_t>(std::ceil(cutoff * n3 / len3));
    for (int k = 0; k < 3; k++) {
        ctx.bin1[k] = cell1[k] / n1;
        ctx.bin2[k] = cell2[k] / n2;
        ctx.bin3[k] = cell3[k] / n3;
    }
    ctx.cutoff_sq = cutoff * cutoff;
    ctx.per_atom_cutoff = per_atom_cutoff;
    ctx.per_type_cutoff_sq = per_type_cutoff_sq;
    ctx.ncutoffs = ncutoffs;
    ctx.types = types;
    ctx.cl = &cl;

    /* Pass 1: count neighbours per atom (independent per atom). */
    std::vector<index_t> offset(nat + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (index_t i = 0; i < nat; i++) {
        index_t count = 0;
        visit_neighbours(ctx, i, [&count](index_t, const real_t *, real_t,
                                          const index_t *) { count++; });
        offset[i + 1] = count;
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
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (index_t i = 0; i < nat; i++) {
        index_t w = offset[i];
        visit_neighbours(ctx, i,
                         [&](index_t j, const real_t *dr, real_t abs_dr_sq,
                             const index_t *shift) {
                             if (want_first) out.first[w] = i;
                             if (want_secnd) out.secnd[w] = j;
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
