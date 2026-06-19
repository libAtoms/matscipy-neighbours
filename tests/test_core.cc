/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 *
 * GoogleTest unit tests for the Python-free C++ core. These prove the core
 * links and runs with no Python at all. Exhaustive brute-force validation of
 * the algorithm over many cell shapes lives in the Python suite.
 */

#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "first_neighbours.hh"
#include "neighbour_list.hh"
#include "triplet_list.hh"

using namespace matscipy;

namespace {

const real_t kOrigin[3] = {0, 0, 0};
const real_t kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

}  // namespace

TEST(NeighbourList, SingleAtomUnitCubeCounts) {
    const bool pbc[3] = {true, true, true};
    const real_t pos[3] = {0.5, 0.5, 0.5};
    NeighbourList nl;

    // 6 face images within 1.1.
    ASSERT_EQ(neighbour_list(QUANTITY_FIRST | QUANTITY_SECOND, kOrigin,
                             kIdentity, kIdentity, pbc, 1, pos, 1.1, nullptr,
                             nullptr, 0, nullptr, nl),
              NL_SUCCESS);
    EXPECT_EQ(nl.npairs, 6);
    EXPECT_EQ(nl.first.size(), 6u);
    EXPECT_EQ(nl.secnd.size(), 6u);

    // +12 edge images within 1.5.
    ASSERT_EQ(neighbour_list(QUANTITY_FIRST, kOrigin, kIdentity, kIdentity, pbc,
                             1, pos, 1.5, nullptr, nullptr, 0, nullptr, nl),
              NL_SUCCESS);
    EXPECT_EQ(nl.npairs, 18);
}

TEST(NeighbourList, NonPeriodicSingleAtomHasNoNeighbours) {
    const bool nopbc[3] = {false, false, false};
    const real_t pos[3] = {0.5, 0.5, 0.5};
    NeighbourList nl;
    ASSERT_EQ(neighbour_list(QUANTITY_FIRST, kOrigin, kIdentity, kIdentity,
                             nopbc, 1, pos, 1.1, nullptr, nullptr, 0, nullptr,
                             nl),
              NL_SUCCESS);
    EXPECT_EQ(nl.npairs, 0);
}

TEST(NeighbourList, DegenerateCellReturnsErrorNotCrash) {
    const bool pbc[3] = {true, true, true};
    const real_t pos[3] = {0.5, 0.5, 0.5};
    // Two collinear lattice vectors -> zero volume.
    const real_t bad_cell[9] = {1, 0, 0, 2, 0, 0, 0, 0, 1};
    NeighbourList nl;
    EXPECT_EQ(neighbour_list(QUANTITY_FIRST, kOrigin, bad_cell, kIdentity, pbc,
                             1, pos, 1.0, nullptr, nullptr, 0, nullptr, nl),
              NL_ERROR);
}

TEST(NeighbourList, DistanceVectorAndShiftAreConsistent) {
    // 2x2x2 simple-cubic lattice in a periodic 2x2x2 box.
    const real_t cell[9] = {2, 0, 0, 0, 2, 0, 0, 0, 2};
    const bool pbc[3] = {true, true, true};
    std::vector<real_t> r;
    for (int x = 0; x < 2; x++)
        for (int y = 0; y < 2; y++)
            for (int z = 0; z < 2; z++) {
                r.push_back(0.5 + x);
                r.push_back(0.5 + y);
                r.push_back(0.5 + z);
            }
    const index_t nat = 8;
    const real_t inv[9] = {0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5};

    NeighbourList nl;
    ASSERT_EQ(neighbour_list(QUANTITY_FIRST | QUANTITY_SECOND |
                                 QUANTITY_DISTVEC | QUANTITY_ABSDIST |
                                 QUANTITY_SHIFT,
                             kOrigin, cell, inv, pbc, nat, r.data(), 1.1,
                             nullptr, nullptr, 0, nullptr, nl),
              NL_SUCCESS);

    // Simple cubic, nearest-neighbour cutoff: 6 neighbours each, 48 pairs.
    EXPECT_EQ(nl.npairs, 6 * nat);

    std::vector<int> count_i(nat, 0), count_j(nat, 0);
    for (index_t p = 0; p < nl.npairs; p++) {
        index_t i = nl.first[p], j = nl.secnd[p];
        count_i[i]++;
        count_j[j]++;

        // D == r[j] - r[i] + S . cell  (cell is diagonal with 2 on diagonal).
        for (int c = 0; c < 3; c++) {
            real_t expected =
                r[3 * j + c] - r[3 * i + c] + nl.shift[3 * p + c] * cell[4 * c];
            EXPECT_NEAR(nl.distvec[3 * p + c], expected, 1e-12);
        }
        // |D| matches the recorded absolute distance.
        real_t d2 = 0;
        for (int c = 0; c < 3; c++) d2 += nl.distvec[3 * p + c] * nl.distvec[3 * p + c];
        EXPECT_NEAR(nl.absdist[p], std::sqrt(d2), 1e-12);
    }
    for (index_t a = 0; a < nat; a++) {
        EXPECT_EQ(count_i[a], 6);
        EXPECT_EQ(count_i[a], count_j[a]);  // symmetry of the pair list
    }
}

TEST(NeighbourList, PerTypeCutoffs) {
    const real_t cell[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
    const real_t inv[9] = {0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1};
    const bool nopbc[3] = {false, false, false};
    const real_t r[9] = {0, 0, 0, 1, 0, 0, 2, 0, 0};
    const index_t types[3] = {0, 1, 0};
    // Squared 2x2 cutoff matrix: only the 0-1 interaction reaches d=1.
    const real_t cutoff_sq[4] = {0.25, 2.25, 2.25, 0.25};
    NeighbourList nl;
    ASSERT_EQ(neighbour_list(QUANTITY_FIRST | QUANTITY_SECOND, kOrigin, cell,
                             inv, nopbc, 3, r, 1.5, nullptr, cutoff_sq, 2, types,
                             nl),
              NL_SUCCESS);
    // 0-1 and 1-2 (both d=1, type pair 0-1) included; 0-2 (d=2) excluded.
    EXPECT_EQ(nl.npairs, 4);
}

TEST(FirstNeighbours, ReferenceValues) {
    std::vector<index_t> i_n = {1, 1, 1, 1, 3, 3, 3};
    std::vector<index_t> seed(6);
    first_neighbours(5, static_cast<index_t>(i_n.size()), i_n.data(),
                     seed.data());
    EXPECT_EQ(seed, (std::vector<index_t>{-1, 0, 4, 4, 7, 7}));
}

TEST(FirstNeighbours, EmptyListDoesNotReadOutOfBounds) {
    std::vector<index_t> seed(6);
    first_neighbours(5, 0, nullptr, seed.data());
    EXPECT_EQ(seed, (std::vector<index_t>{0, 0, 0, 0, 0, 0}));
}

TEST(GetJumpIndicies, Basic) {
    std::vector<index_t> sorted = {0, 0, 0, 1, 1, 1, 1, 1, 2, 2,
                                   2, 3, 3, 3, 3, 4, 4, 4, 4};
    auto jumps = get_jump_indicies(static_cast<index_t>(sorted.size()),
                                   sorted.data());
    EXPECT_EQ(jumps, (std::vector<index_t>{0, 3, 8, 11, 15, 19}));
}

TEST(TripletList, WithAndWithoutCutoff) {
    std::vector<index_t> first_i = {0, 2, 6, 10};
    std::vector<index_t> ij, ik;

    triplet_list(static_cast<index_t>(first_i.size()), first_i.data(), nullptr,
                 0.0, ij, ik);
    EXPECT_EQ(ij.size(), 26u);
    EXPECT_EQ(ik.size(), 26u);

    std::vector<real_t> absdist = {2.2, 2.2, 2.2, 2.2, 3.0,
                                   3.0, 2.0, 2.0, 2.0, 2.0};
    triplet_list(static_cast<index_t>(first_i.size()), first_i.data(),
                 absdist.data(), 2.6, ij, ik);
    EXPECT_EQ(ij.size(), 16u);
    EXPECT_EQ(ik.size(), 16u);
}
