/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 *
 * GoogleTest unit tests for the Python-free C++ core. They check that the core
 * links and runs without Python. Exhaustive brute-force validation of the
 * algorithm over many cell shapes lives in the Python suite.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "cell_list.hh"
#include "error.hh"
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

TEST(NeighbourList, TypedViewsMatchBuffers) {
    const real_t cell[9] = {2, 0, 0, 0, 2, 0, 0, 0, 2};
    const real_t inv[9] = {0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5};
    const bool pbc[3] = {true, true, true};
    std::vector<real_t> r;
    for (int x = 0; x < 2; x++)
        for (int y = 0; y < 2; y++)
            for (int z = 0; z < 2; z++) {
                r.push_back(0.5 + x);
                r.push_back(0.5 + y);
                r.push_back(0.5 + z);
            }
    NeighbourList nl;
    ASSERT_EQ(neighbour_list(QUANTITY_FIRST | QUANTITY_SECOND | QUANTITY_DISTVEC |
                                 QUANTITY_SHIFT,
                             kOrigin, cell, inv, pbc, 8, r.data(), 1.1, nullptr,
                             nullptr, 0, nullptr, nl),
              NL_SUCCESS);

    // Span views alias the underlying buffers (no copy).
    auto fi = nl.first_view();
    ASSERT_EQ(fi.size(), (std::size_t)nl.npairs);
    EXPECT_EQ(fi.data(), nl.first.data());
    for (index_t p = 0; p < nl.npairs; p++) EXPECT_EQ(fi[p], nl.first[p]);

    // 3-vector accessors read one pair's distance vector / shift.
    for (index_t p = 0; p < nl.npairs; p++) {
        auto D = nl.distvec_at(p);
        EXPECT_EQ(D.x, nl.distvec[3 * p + 0]);
        EXPECT_EQ(D.y, nl.distvec[3 * p + 1]);
        EXPECT_EQ(D.z, nl.distvec[3 * p + 2]);
        auto S = nl.shift_at(p);
        EXPECT_EQ(S.x, nl.shift[3 * p + 0]);
        EXPECT_EQ(S.z, nl.shift[3 * p + 2]);
    }
    // An unrequested quantity has an empty view.
    EXPECT_TRUE(nl.absdist_view().empty());
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
    ASSERT_EQ(first_neighbours(5, static_cast<index_t>(i_n.size()), i_n.data(),
                               seed.data()),
              NL_SUCCESS);
    EXPECT_EQ(seed, (std::vector<index_t>{-1, 0, 4, 4, 7, 7}));
}

TEST(FirstNeighbours, EmptyListDoesNotReadOutOfBounds) {
    std::vector<index_t> seed(6);
    ASSERT_EQ(first_neighbours(5, 0, nullptr, seed.data()), NL_SUCCESS);
    EXPECT_EQ(seed, (std::vector<index_t>{0, 0, 0, 0, 0, 0}));
}

TEST(FirstNeighbours, RejectsBadInputWithoutWriting) {
    std::vector<index_t> seed(4, 99);
    std::vector<index_t> too_large = {0, 1, 7};
    EXPECT_EQ(first_neighbours(3, 3, too_large.data(), seed.data()),
              NL_INVALID_ARGUMENT);
    EXPECT_TRUE(has_error);
    std::vector<index_t> negative = {-2, 1};
    EXPECT_EQ(first_neighbours(3, 2, negative.data(), seed.data()),
              NL_INVALID_ARGUMENT);
    std::vector<index_t> unsorted = {2, 1};
    EXPECT_EQ(first_neighbours(3, 2, unsorted.data(), seed.data()),
              NL_INVALID_ARGUMENT);
    EXPECT_EQ(first_neighbours(-5, 0, nullptr, seed.data()),
              NL_INVALID_ARGUMENT);
    EXPECT_EQ(seed, (std::vector<index_t>{99, 99, 99, 99}));
}

TEST(GetJumpIndicies, Basic) {
    std::vector<index_t> sorted = {0, 0, 0, 1, 1, 1, 1, 1, 2, 2,
                                   2, 3, 3, 3, 3, 4, 4, 4, 4};
    std::vector<index_t> jumps;
    ASSERT_EQ(get_jump_indicies(static_cast<index_t>(sorted.size()),
                                sorted.data(), jumps),
              NL_SUCCESS);
    EXPECT_EQ(jumps, (std::vector<index_t>{0, 3, 8, 11, 15, 19}));
}

TEST(GetJumpIndicies, RejectsGapsAndNonZeroStart) {
    std::vector<index_t> jumps;
    std::vector<index_t> gap = {0, 0, 2};
    EXPECT_EQ(get_jump_indicies(3, gap.data(), jumps), NL_INVALID_ARGUMENT);
    std::vector<index_t> start = {1, 1, 2};
    EXPECT_EQ(get_jump_indicies(3, start.data(), jumps), NL_INVALID_ARGUMENT);
}

TEST(TripletList, WithAndWithoutCutoff) {
    std::vector<index_t> first_i = {0, 2, 6, 10};
    std::vector<index_t> ij, ik;

    ASSERT_EQ(triplet_list(static_cast<index_t>(first_i.size()), first_i.data(),
                           0, nullptr, 0.0, ij, ik),
              NL_SUCCESS);
    EXPECT_EQ(ij.size(), 26u);
    EXPECT_EQ(ik.size(), 26u);

    std::vector<real_t> absdist = {2.2, 2.2, 2.2, 2.2, 3.0,
                                   3.0, 2.0, 2.0, 2.0, 2.0};
    ASSERT_EQ(triplet_list(static_cast<index_t>(first_i.size()), first_i.data(),
                           static_cast<index_t>(absdist.size()), absdist.data(),
                           2.6, ij, ik),
              NL_SUCCESS);
    EXPECT_EQ(ij.size(), 16u);
    EXPECT_EQ(ik.size(), 16u);
}

TEST(TripletList, LeadingMinusOneIsNoEntries) {
    // first_neighbours() emits -1 for atoms before the first pair.
    std::vector<index_t> first_i = {-1, -1, 0, 2};
    std::vector<real_t> absdist = {1.0, 1.0};
    std::vector<index_t> ij, ik;
    ASSERT_EQ(triplet_list(4, first_i.data(), 2, absdist.data(), 2.0, ij, ik),
              NL_SUCCESS);
    EXPECT_EQ(ij, (std::vector<index_t>{0, 1}));
    EXPECT_EQ(ik, (std::vector<index_t>{1, 0}));
}

TEST(TripletList, RejectsBadRowStarts) {
    std::vector<index_t> ij, ik;
    std::vector<real_t> absdist = {1.0, 1.0, 1.0};
    std::vector<index_t> beyond = {0, 10};
    EXPECT_EQ(triplet_list(2, beyond.data(), 3, absdist.data(), 1.0, ij, ik),
              NL_INVALID_ARGUMENT);
    std::vector<index_t> decreasing = {5, 0};
    EXPECT_EQ(triplet_list(2, decreasing.data(), 0, nullptr, 0.0, ij, ik),
              NL_INVALID_ARGUMENT);
    std::vector<index_t> minus_one_then_three = {-1, 3};
    EXPECT_EQ(triplet_list(2, minus_one_then_three.data(), 0, nullptr, 0.0, ij,
                           ik),
              NL_INVALID_ARGUMENT);
    // A slice of a row-start array (no -1, not starting at 0) is fine.
    std::vector<index_t> slice = {3, 5};
    EXPECT_EQ(triplet_list(2, slice.data(), 0, nullptr, 0.0, ij, ik), NL_SUCCESS);
    EXPECT_EQ(ij.size(), 2u);
}

TEST(NeighbourList, RejectsInvalidArguments) {
    const bool pbc[3] = {true, true, true};
    const real_t pos[6] = {0.2, 0.2, 0.2, 0.7, 0.7, 0.7};
    NeighbourList nl;
    auto call = [&](index_t nat, real_t cutoff, const real_t *per_atom,
                    const real_t *per_type_sq, index_t ncutoffs,
                    const index_t *types) {
        return neighbour_list(QUANTITY_FIRST, kOrigin, kIdentity, kIdentity, pbc,
                              nat, pos, cutoff, per_atom, per_type_sq, ncutoffs,
                              types, nl);
    };
    EXPECT_EQ(call(2, 0.0, nullptr, nullptr, 0, nullptr), NL_INVALID_ARGUMENT);
    EXPECT_EQ(call(2, -1.0, nullptr, nullptr, 0, nullptr), NL_INVALID_ARGUMENT);
    EXPECT_EQ(call(2, std::nan(""), nullptr, nullptr, 0, nullptr),
              NL_INVALID_ARGUMENT);
    EXPECT_EQ(call(2, INFINITY, nullptr, nullptr, 0, nullptr),
              NL_INVALID_ARGUMENT);
    EXPECT_EQ(call(-1, 1.0, nullptr, nullptr, 0, nullptr), NL_INVALID_ARGUMENT);

    const real_t bad_radius[2] = {0.5, -0.5};
    EXPECT_EQ(call(2, 1.0, bad_radius, nullptr, 0, nullptr), NL_INVALID_ARGUMENT);

    const real_t per_type_sq[4] = {1.0, 1.0, 1.0, 1.0};
    const index_t out_of_range[2] = {0, 5};
    EXPECT_EQ(call(2, 1.0, nullptr, per_type_sq, 2, out_of_range),
              NL_INVALID_ARGUMENT);
    EXPECT_EQ(call(2, 1.0, nullptr, per_type_sq, 2, nullptr),
              NL_INVALID_ARGUMENT);

    const real_t nan_pos[6] = {0.2, 0.2, 0.2, NAN, 0.7, 0.7};
    EXPECT_EQ(neighbour_list(QUANTITY_FIRST, kOrigin, kIdentity, kIdentity, pbc,
                             2, nan_pos, 1.0, nullptr, nullptr, 0, nullptr, nl),
              NL_INVALID_ARGUMENT);
    EXPECT_EQ(nl.npairs, 0);
}

TEST(NeighbourList, HugeCoordinatesDoNotHang) {
    // 1e30 cell widths away: the raw cell index is clamped, not UB, and the
    // wrap is O(1). The far atom simply has no neighbours.
    const bool pbc[3] = {true, true, true};
    const real_t pos[6] = {0.5, 0.5, 0.5, 1e30, 0.5, 0.5};
    NeighbourList nl;
    ASSERT_EQ(neighbour_list(QUANTITY_FIRST | QUANTITY_SECOND, kOrigin,
                             kIdentity, kIdentity, pbc, 2, pos, 1.1, nullptr,
                             nullptr, 0, nullptr, nl),
              NL_SUCCESS);
    for (index_t p = 0; p < nl.npairs; p++) {
        EXPECT_EQ(nl.first[p], nl.secnd[p]);  // only self-images
    }
}

TEST(NeighbourList, MortonMatchesLinearOnHugeGrid) {
    // 2.2M cells requested along x exceed the 21-bit Morton key range; the
    // resolution is clamped so both orders bin identically.
    const index_t nat = 2000;
    const real_t L = 2.2e6;
    const real_t cell[9] = {L, 0, 0, 0, 1, 0, 0, 0, 1};
    const real_t inv[9] = {1 / L, 0, 0, 0, 1, 0, 0, 0, 1};
    const bool pbc[3] = {true, true, true};
    std::mt19937 rng(7);
    std::uniform_real_distribution<real_t> ux(0, L), u1(0, 1);
    std::vector<real_t> r(3 * nat);
    for (index_t a = 0; a < nat; a++) {
        r[3 * a] = ux(rng);
        r[3 * a + 1] = u1(rng);
        r[3 * a + 2] = u1(rng);
    }
    NeighbourList lin, mor;
    ASSERT_EQ(neighbour_list(QUANTITY_FIRST, kOrigin, cell, inv, pbc, nat,
                             r.data(), 1.0, nullptr, nullptr, 0, nullptr, lin,
                             CellOrder::Linear),
              NL_SUCCESS);
    ASSERT_EQ(neighbour_list(QUANTITY_FIRST, kOrigin, cell, inv, pbc, nat,
                             r.data(), 1.0, nullptr, nullptr, 0, nullptr, mor,
                             CellOrder::Morton),
              NL_SUCCESS);
    EXPECT_EQ(lin.npairs, mor.npairs);
    EXPECT_EQ(lin.first, mor.first);
}

TEST(CellHash, SpreadsKeysThatDifferOnlyInHighBits) {
    // A wire along z in a 1024^3 grid: keys k << 20. With an 18-bit mask these
    // must not all collide.
    const std::int64_t mask = (1 << 18) - 1;
    std::vector<bool> used(mask + 1, false);
    int collisions = 0;
    for (std::int64_t k = 0; k < 1024; k++) {
        std::int64_t h = cell_hash(k << 20) & mask;
        if (used[h]) collisions++;
        used[h] = true;
    }
    EXPECT_LT(collisions, 16);
}

namespace {
// Random cubic periodic config for the dense-matrix tests.
void random_cubic(int N, double L, unsigned seed, std::vector<real_t> &r) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<real_t> U(0.0, L);
    r.resize(3 * N);
    for (int k = 0; k < 3 * N; k++) r[k] = U(rng);
}
}  // namespace

TEST(NeighbourMatrix, MatchesPairList) {
    const int N = 1500;
    const double L = 12.0, cutoff = 1.5;
    const real_t cell[9] = {(real_t)L, 0, 0, 0, (real_t)L, 0, 0, 0, (real_t)L};
    const real_t inv[9] = {(real_t)(1 / L), 0, 0, 0, (real_t)(1 / L), 0,
                           0, 0, (real_t)(1 / L)};
    const bool pbc[3] = {true, true, true};
    std::vector<real_t> r;
    random_cubic(N, L, 3, r);

    NeighbourList nl;
    ASSERT_EQ(neighbour_list(QUANTITY_FIRST | QUANTITY_SECOND | QUANTITY_DISTVEC,
                             kOrigin, cell, inv, pbc, N, r.data(), cutoff, nullptr,
                             nullptr, 0, nullptr, nl),
              NL_SUCCESS);

    const index_t K = 64;
    NeighbourMatrix nm;
    ASSERT_EQ(neighbour_matrix(kOrigin, cell, inv, pbc, N, r.data(), cutoff,
                               nullptr, nullptr, 0, nullptr, K, nm),
              NL_SUCCESS);
    ASSERT_FALSE(nm.overflow);

    // Per-atom degree must match the pair list.
    std::vector<index_t> deg(N, 0);
    for (index_t p = 0; p < nl.npairs; p++) deg[nl.first[p]]++;
    for (int i = 0; i < N; i++) EXPECT_EQ(nm.count[i], deg[i]);

    // Each pair must appear in its atom's row with the matching distance vector.
    for (index_t p = 0; p < nl.npairs; p++) {
        index_t i = nl.first[p], j = nl.secnd[p];
        // find j in row i of the matrix
        bool found = false;
        for (index_t s = 0; s < nm.count[i]; s++) {
            if (nm.idx[(size_t)i * K + s] == j) {
                for (int k = 0; k < 3; k++)
                    EXPECT_NEAR(nm.dist[((size_t)i * K + s) * 3 + k],
                                nl.distvec[3 * p + k], 1e-12);
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }
}

/* With both per-slot extras requested, every slot satisfies the distance
   contract D == r[j] - r[i] + S @ cell; the shifts are non-trivial here. */
TEST(NeighbourMatrix, ShiftsSatisfyContract) {
    const int N = 600;
    const double L = 6.0, cutoff = 1.4;
    const real_t cell[9] = {(real_t)L, 0, 0, 0, (real_t)L, 0, 0, 0, (real_t)L};
    const real_t inv[9] = {(real_t)(1 / L), 0, 0, 0, (real_t)(1 / L), 0,
                           0, 0, (real_t)(1 / L)};
    const bool pbc[3] = {true, true, true};
    std::vector<real_t> r;
    random_cubic(N, L, 6, r);

    const index_t K = 64;
    NeighbourMatrix nm;
    ASSERT_EQ(neighbour_matrix(kOrigin, cell, inv, pbc, N, r.data(), cutoff,
                               nullptr, nullptr, 0, nullptr, K, nm,
                               CellOrder::Linear,
                               QUANTITY_DISTVEC | QUANTITY_SHIFT),
              NL_SUCCESS);
    ASSERT_FALSE(nm.overflow);
    ASSERT_EQ(nm.shift.size(), (size_t)N * K * 3);
    bool any_shift = false;
    for (int i = 0; i < N; i++) {
        for (index_t s = 0; s < nm.count[i]; s++) {
            const size_t slot = (size_t)i * K + s;
            const index_t j = nm.idx[slot];
            for (int k = 0; k < 3; k++) {
                const index_t sk = nm.shift[3 * slot + k];
                any_shift = any_shift || sk != 0;
                /* cell is diagonal: (S @ cell)_k == S_k * L */
                EXPECT_NEAR(nm.dist[3 * slot + k],
                            r[3 * j + k] - r[3 * i + k] + sk * L, 1e-12);
            }
        }
    }
    EXPECT_TRUE(any_shift);

    NeighbourMatrix bare;  /* no extras: only indices and counts */
    ASSERT_EQ(neighbour_matrix(kOrigin, cell, inv, pbc, N, r.data(), cutoff,
                               nullptr, nullptr, 0, nullptr, K, bare,
                               CellOrder::Linear, 0),
              NL_SUCCESS);
    EXPECT_TRUE(bare.dist.empty());
    EXPECT_TRUE(bare.shift.empty());
    EXPECT_EQ(bare.count, nm.count);
}

TEST(NeighbourMatrix, OverflowFlag) {
    const int N = 800;
    const double L = 9.0, cutoff = 1.5;
    const real_t cell[9] = {(real_t)L, 0, 0, 0, (real_t)L, 0, 0, 0, (real_t)L};
    const real_t inv[9] = {(real_t)(1 / L), 0, 0, 0, (real_t)(1 / L), 0,
                           0, 0, (real_t)(1 / L)};
    const bool pbc[3] = {true, true, true};
    std::vector<real_t> r;
    random_cubic(N, L, 5, r);

    NeighbourMatrix nm;
    ASSERT_EQ(neighbour_matrix(kOrigin, cell, inv, pbc, N, r.data(), cutoff,
                               nullptr, nullptr, 0, nullptr, /*K=*/2, nm),
              NL_SUCCESS);
    EXPECT_TRUE(nm.overflow);  // 2 slots is far too few for this density
}
