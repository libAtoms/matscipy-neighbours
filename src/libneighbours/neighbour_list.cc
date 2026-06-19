/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#include "neighbour_list.hh"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>

#include "error.hh"
#include "tools.hh"

namespace matscipy {

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

    /* Number of neighbouring cells to scan in each direction. */
    index_t nx = static_cast<index_t>(std::ceil(cutoff * n1 / len1));
    index_t ny = static_cast<index_t>(std::ceil(cutoff * n2 / len2));
    index_t nz = static_cast<index_t>(std::ceil(cutoff * n3 / len3));

    /* Sort atoms into bins using a per-cell linked list. */
    index_t ncells = n1 * n2 * n3;
    std::vector<index_t> seed(ncells, -1);
    std::vector<index_t> last(ncells, -1);
    std::vector<index_t> next(nat, -1);
    for (index_t i = 0; i < nat; i++) {
        index_t c1, c2, c3;
        position_to_cell_index(cell_origin, inv_cell, &r[3 * i], n1, n2, n3, &c1,
                               &c2, &c3);

        if (pbc[0])
            c1 = bin_wrap(c1, n1);
        else
            c1 = bin_trunc(c1, n1);
        if (pbc[1])
            c2 = bin_wrap(c2, n2);
        else
            c2 = bin_trunc(c2, n2);
        if (pbc[2])
            c3 = bin_wrap(c3, n3);
        else
            c3 = bin_trunc(c3, n3);

        index_t ci = c1 + n1 * (c2 + n2 * c3);
        assert(ci >= 0 && ci < ncells);

        if (seed[ci] < 0) {
            next[i] = -1;
            seed[ci] = i;
            last[ci] = i;
        } else {
            next[i] = -1;
            next[last[ci]] = i;
            last[ci] = i;
        }
    }

    /* Shape of a single bin. */
    real_t bin1[3], bin2[3], bin3[3];
    for (int k = 0; k < 3; k++) {
        bin1[k] = cell1[k] / n1;
        bin2[k] = cell2[k] / n2;
        bin3[k] = cell3[k] / n3;
    }

    real_t cutoff_sq = cutoff * cutoff;

    /* Loop over atoms. */
    for (index_t i = 0; i < nat; i++) {
        const real_t *ri = &r[3 * i];

        index_t ci01, ci02, ci03;
        position_to_cell_index(cell_origin, inv_cell, ri, n1, n2, n3, &ci01,
                               &ci02, &ci03);

        /* Truncate if non-periodic and outside the simulation domain. */
        index_t ci1, ci2, ci3;
        ci1 = pbc[0] ? ci01 : bin_trunc(ci01, n1);
        ci2 = pbc[1] ? ci02 : bin_trunc(ci02, n2);
        ci3 = pbc[2] ? ci03 : bin_trunc(ci03, n3);

        /* Position relative to the lower-left corner of the bin. */
        real_t dri[3];
        dri[0] = ri[0] - ci1 * bin1[0] - ci2 * bin2[0] - ci3 * bin3[0];
        dri[1] = ri[1] - ci1 * bin1[1] - ci2 * bin2[1] - ci3 * bin3[1];
        dri[2] = ri[2] - ci1 * bin1[2] - ci2 * bin2[2] - ci3 * bin3[2];

        /* Apply periodic boundary conditions to the home cell index. */
        ci1 = pbc[0] ? bin_wrap(ci01, n1) : bin_trunc(ci01, n1);
        ci2 = pbc[1] ? bin_wrap(ci02, n2) : bin_trunc(ci02, n2);
        ci3 = pbc[2] ? bin_wrap(ci03, n3) : bin_trunc(ci03, n3);

        /* Loop over neighbouring bins. */
        for (index_t z = -nz; z <= nz; z++) {
            index_t cj3 = ci3 + z;
            if (pbc[2]) cj3 = bin_wrap(cj3, n3);
            if (cj3 < 0 || cj3 >= n3) continue;
            cj3 = bin_trunc(cj3, n3);
            index_t ncj3 = n2 * cj3;

            real_t off3[3] = {z * bin3[0], z * bin3[1], z * bin3[2]};

            for (index_t y = -ny; y <= ny; y++) {
                index_t cj2 = ci2 + y;
                if (pbc[1]) cj2 = bin_wrap(cj2, n2);
                if (cj2 < 0 || cj2 >= n2) continue;
                cj2 = bin_trunc(cj2, n2);
                index_t ncj2 = n1 * (cj2 + ncj3);

                real_t off2[3] = {off3[0] + y * bin2[0], off3[1] + y * bin2[1],
                                  off3[2] + y * bin2[2]};

                for (index_t x = -nx; x <= nx; x++) {
                    index_t cj1 = ci1 + x;
                    if (pbc[0]) cj1 = bin_wrap(cj1, n1);
                    if (cj1 < 0 || cj1 >= n1) continue;
                    cj1 = bin_trunc(cj1, n1);
                    index_t ncj = cj1 + ncj2;

                    assert(ncj == cj1 + n1 * (cj2 + n2 * cj3));

                    real_t off[3] = {off2[0] + x * bin1[0],
                                     off2[1] + x * bin1[1],
                                     off2[2] + x * bin1[2]};

                    /* Loop over all atoms in the neighbouring bin. */
                    index_t j = seed[ncj];
                    while (j >= 0) {
                        if (i != j || x != 0 || y != 0 || z != 0) {
                            const real_t *rj = &r[3 * j];

                            index_t cjj1, cjj2, cjj3;
                            position_to_cell_index(cell_origin, inv_cell, rj, n1,
                                                   n2, n3, &cjj1, &cjj2, &cjj3);
                            if (!pbc[0]) cjj1 = bin_trunc(cjj1, n1);
                            if (!pbc[1]) cjj2 = bin_trunc(cjj2, n2);
                            if (!pbc[2]) cjj3 = bin_trunc(cjj3, n3);

                            real_t drj[3];
                            drj[0] = rj[0] - cjj1 * bin1[0] - cjj2 * bin2[0] -
                                     cjj3 * bin3[0];
                            drj[1] = rj[1] - cjj1 * bin1[1] - cjj2 * bin2[1] -
                                     cjj3 * bin3[1];
                            drj[2] = rj[2] - cjj1 * bin1[2] - cjj2 * bin2[2] -
                                     cjj3 * bin3[2];

                            real_t dr[3] = {drj[0] - dri[0] + off[0],
                                            drj[1] - dri[1] + off[1],
                                            drj[2] - dri[2] + off[2]};
                            real_t abs_dr_sq =
                                dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];

                            if (abs_dr_sq < cutoff_sq) {
                                bool inside_cutoff = true;
                                if (per_atom_cutoff) {
                                    real_t c =
                                        per_atom_cutoff[i] + per_atom_cutoff[j];
                                    inside_cutoff = abs_dr_sq < c * c;
                                } else if (per_type_cutoff_sq && types) {
                                    if (types[i] >= 0 && types[i] < ncutoffs &&
                                        types[j] >= 0 && types[j] < ncutoffs) {
                                        real_t c_sq =
                                            per_type_cutoff_sq[types[i] *
                                                                   ncutoffs +
                                                               types[j]];
                                        inside_cutoff = abs_dr_sq < c_sq;
                                    }
                                }

                                if (inside_cutoff) {
                                    if (quantities & QUANTITY_FIRST)
                                        out.first.push_back(i);
                                    if (quantities & QUANTITY_SECOND)
                                        out.secnd.push_back(j);
                                    if (quantities & QUANTITY_DISTVEC) {
                                        out.distvec.push_back(dr[0]);
                                        out.distvec.push_back(dr[1]);
                                        out.distvec.push_back(dr[2]);
                                    }
                                    if (quantities & QUANTITY_ABSDIST)
                                        out.absdist.push_back(
                                            std::sqrt(abs_dr_sq));
                                    if (quantities & QUANTITY_SHIFT) {
                                        out.shift.push_back((ci01 - cjj1 + x) /
                                                            n1);
                                        out.shift.push_back((ci02 - cjj2 + y) /
                                                            n2);
                                        out.shift.push_back((ci03 - cjj3 + z) /
                                                            n3);
                                    }
                                    out.npairs++;
                                }
                            }
                        }

                        j = next[j];
                    }
                }
            }
        }
    }

    return NL_SUCCESS;
}

}  // namespace matscipy
