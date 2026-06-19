/*
 * matscipy-neighbours — neighbour-list core benchmark
 * SPDX-License-Identifier: MIT
 *
 * Times neighbour_list() on random configurations (no Python). Reports median
 * wall-clock per call. Honours OMP_NUM_THREADS when built with OpenMP.
 *
 *   ./bench_core [quantities] [linear|morton]
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cell_list.hh"
#include "neighbour_list.hh"
#ifdef MATSCIPY_ENABLE_CUDA
#include "memory_space.hh"
#include "neighbour_list_gpu.hh"
#endif

using namespace matscipy;
using clock_type = std::chrono::steady_clock;

static int g_reps(int N) { return N >= 1000000 ? 5 : (N >= 200000 ? 8 : 20); }

static int quantity_flags(const char *q) {
    int f = 0;
    for (; *q; q++)
        switch (*q) {
            case 'i': f |= QUANTITY_FIRST; break;
            case 'j': f |= QUANTITY_SECOND; break;
            case 'D': f |= QUANTITY_DISTVEC; break;
            case 'd': f |= QUANTITY_ABSDIST; break;
            case 'S': f |= QUANTITY_SHIFT; break;
        }
    return f;
}

/* Run a single configuration and print one row. */
static void run(const char *label, int flags, int N, const real_t cell[9],
                const real_t inv[9], std::vector<real_t> &r, CellOrder order) {
    const real_t origin[3] = {0, 0, 0};
    const bool pbc[3] = {true, true, true};

    NeighbourList nl;
    neighbour_list(flags, origin, cell, inv, pbc, N, r.data(), 1.0, nullptr,
                   nullptr, 0, nullptr, nl, order);  // warm up
    const index_t pairs = nl.npairs;

    const int reps = g_reps(N);
    std::vector<double> ms;
    for (int rep = 0; rep < reps; rep++) {
        auto t0 = clock_type::now();
        neighbour_list(flags, origin, cell, inv, pbc, N, r.data(), 1.0, nullptr,
                       nullptr, 0, nullptr, nl, order);
        auto t1 = clock_type::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    std::printf("%-10s %10d %12lld %10.2f %12.2f\n", label, N,
                (long long)pairs, med, med * 1e6 / (double)pairs);
}

#ifdef MATSCIPY_ENABLE_CUDA
/* GPU end-to-end (H2D positions -> kernels -> D2H pairs), same config as run().
   Reports the full call cost — the honest equivalent of the CPU number. */
static index_t run_gpu(const char *label, int flags, int N, const real_t cell[9],
                       const real_t inv[9], std::vector<real_t> &r) {
    const real_t origin[3] = {0, 0, 0};
    const bool pbc[3] = {true, true, true};

    NeighbourList nl;
    neighbour_list_gpu(flags, origin, cell, inv, pbc, N, r.data(), 1.0, nullptr,
                       nullptr, 0, nullptr, nl);  // warm up (GPU ctx + alloc)
    const index_t pairs = nl.npairs;

    const int reps = g_reps(N);
    std::vector<double> ms;
    for (int rep = 0; rep < reps; rep++) {
        auto t0 = clock_type::now();
        neighbour_list_gpu(flags, origin, cell, inv, pbc, N, r.data(), 1.0,
                           nullptr, nullptr, 0, nullptr, nl);
        auto t1 = clock_type::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    std::printf("%-10s %10d %12lld %10.2f %12.2f\n", label, N,
                (long long)pairs, med, med * 1e6 / (double)pairs);
    return pairs;
}

/* Phase-3.1 baseline: just the PCIe round-trip for one call's data
   (H2D 3N positions + D2H 3*pairs distance components). No compute. */
static void run_transfer(int N, index_t pairs) {
    Array<real_t> h_pos(3 * N), h_out(3 * (size_t)pairs);
    Array<real_t, CudaSpace> d_pos(3 * N), d_out(3 * (size_t)pairs);
    for (int k = 0; k < 3 * N; k++) h_pos.data()[k] = 0.1 * k;

    const int reps = g_reps(N);
    std::vector<double> ms;
    for (int rep = 0; rep < reps; rep++) {
        auto t0 = clock_type::now();
        deep_copy(d_pos, h_pos);   // H2D positions
        deep_copy(h_out, d_out);   // D2H pairs
        auto t1 = clock_type::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    std::printf("%-10s %10d %12lld %10.2f %12.2f\n", "  xfer-only", N,
                (long long)pairs, med, med * 1e6 / (double)pairs);
}
#endif

int main(int argc, char **argv) {
    const char *quantities = argc > 1 ? argv[1] : "ijD";
    const int flags = quantity_flags(quantities);
    const CellOrder order =
        (argc > 2 && std::strcmp(argv[2], "morton") == 0) ? CellOrder::Morton
                                                          : CellOrder::Linear;
    const double rho = 12.0;  // ~50 neighbours per atom within cutoff 1.0

    std::printf("quantities=\"%s\"  order=%s  cutoff=1.0  density=%.1f\n",
                quantities, order == CellOrder::Morton ? "morton" : "linear",
                rho);
    std::printf("%-10s %10s %12s %10s %12s\n", "scenario", "natoms", "pairs",
                "med[ms]", "ns/pair");

    /* Dense scenarios: uniform fill of a cubic periodic box. */
    for (int N : {10000, 50000, 200000, 1000000}) {
        const double L = std::cbrt(N / rho);
        const real_t cell[9] = {(real_t)L, 0, 0, 0, (real_t)L, 0, 0, 0, (real_t)L};
        const real_t inv[9] = {(real_t)(1 / L), 0, 0, 0, (real_t)(1 / L),
                               0, 0, 0, (real_t)(1 / L)};
        std::mt19937 rng(42);
        std::uniform_real_distribution<real_t> U(0.0, L);
        std::vector<real_t> r(3 * N);
        for (int k = 0; k < 3 * N; k++) r[k] = U(rng);
        run("cpu-dense", flags, N, cell, inv, r, order);
#ifdef MATSCIPY_ENABLE_CUDA
        index_t pairs = run_gpu("gpu-dense", flags, N, cell, inv, r);
        run_transfer(N, pairs);
#endif
    }

    /* Sparse scenario: atoms clustered in a corner of a box with lots of
       vacuum (50x larger per side). The grid has ~1.2e5x more cells than the
       dense case, so the hashed compact backend is selected; the old code would
       have collapsed the bin count and degraded toward O(N^2). For reference we
       run the *same atoms* in a tight box (dense) too. */
    {
        const int N = 50000;
        const double Lc = std::cbrt(N / rho);  // cluster extent
        std::mt19937 rng(7);
        std::uniform_real_distribution<real_t> U(0.0, Lc);
        std::vector<real_t> r(3 * N);
        for (int k = 0; k < 3 * N; k++) r[k] = U(rng);

        const real_t tight[9] = {(real_t)Lc, 0, 0, 0, (real_t)Lc, 0, 0, 0, (real_t)Lc};
        const real_t tinv[9] = {(real_t)(1 / Lc), 0, 0, 0, (real_t)(1 / Lc),
                                0, 0, 0, (real_t)(1 / Lc)};
        run("tight", flags, N, tight, tinv, r, order);

        const double Lb = Lc * 50.0;  // mostly vacuum
        const real_t big[9] = {(real_t)Lb, 0, 0, 0, (real_t)Lb, 0, 0, 0, (real_t)Lb};
        const real_t binv[9] = {(real_t)(1 / Lb), 0, 0, 0, (real_t)(1 / Lb),
                                0, 0, 0, (real_t)(1 / Lb)};
        run("vacuum", flags, N, big, binv, r, order);
    }
    return 0;
}
