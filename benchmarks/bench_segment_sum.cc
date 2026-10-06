/*
 * matscipy-neighbours — segment_sum GPU benchmark
 * SPDX-License-Identifier: MIT
 *
 * Times segment_sum_gpu_device() for every kernel variant on synthetic sorted
 * pair lists (10^6 atoms, Poisson-distributed segment lengths) and d values
 * per pair, to check the group-size rule of SegmentSumKernel::Auto on a given
 * GPU. Reports the mean wall-clock per call in ms, including the row-start
 * validation and the final device synchronisation.
 *
 *   ./bench_segment_sum
 */

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "first_neighbours.hh"
#include "memory_space.hh"
#include "segment_sum.hh"

using namespace matscipy;
using clock_type = std::chrono::steady_clock;

int main() {
    const index_t n = 1000000;
    const int reps = 10;
    struct Variant {
        const char *name;
        SegmentSumKernel kernel;
        int group;
    };
    const Variant variants[] = {
        {"thread", SegmentSumKernel::ThreadPerRow, 0},
        {"g2", SegmentSumKernel::Group, 2},
        {"g4", SegmentSumKernel::Group, 4},
        {"g8", SegmentSumKernel::Group, 8},
        {"g16", SegmentSumKernel::Group, 16},
        {"g32", SegmentSumKernel::Group, 32},
        {"auto", SegmentSumKernel::Auto, 0},
    };
    std::printf("%-6s %-3s %8s", "mean", "d", "GB");
    for (const Variant &v : variants) std::printf(" %8s", v.name);
    std::printf(" %10s\n", "auto+total");

    for (int mean : {20, 54}) {
        std::mt19937 rng(1);
        std::poisson_distribution<int> length(mean);
        std::vector<index_t> i_n;
        for (index_t a = 0; a < n; a++) i_n.insert(i_n.end(), length(rng), a);
        const index_t nn = static_cast<index_t>(i_n.size());
        std::vector<index_t> seed(n + 1);
        if (first_neighbours(n, nn, i_n.data(), seed.data()) != NL_SUCCESS) return 1;
        Array<index_t> h_seed(n + 1);
        std::copy(seed.begin(), seed.end(), h_seed.data());
        Array<index_t, DeviceSpace> d_seed(n + 1);
        deep_copy(d_seed, h_seed);

        for (index_t d : {1, 3, 9}) {
            Array<double> h_v(nn * d);
            for (index_t k = 0; k < nn * d; k++) h_v.data()[k] = 0.1 * (k % 17);
            Array<double, DeviceSpace> d_v(nn * d), out, total;
            deep_copy(d_v, h_v);
            std::printf("%-6d %-3lld %8.2f", mean, static_cast<long long>(d),
                        nn * d * 8e-9);
            auto time = [&](SegmentSumKernel k, int g, bool with_total) {
                Array<double, DeviceSpace> *t = with_total ? &total : nullptr;
                segment_sum_gpu_device<double>(n, d_seed.data(), nn, d, d_v.data(),
                                               out, t, -1, k, g);
                const auto t0 = clock_type::now();
                for (int r = 0; r < reps; r++)
                    segment_sum_gpu_device<double>(n, d_seed.data(), nn, d,
                                                   d_v.data(), out, t, -1, k, g);
                return std::chrono::duration<double, std::milli>(
                           clock_type::now() - t0).count() / reps;
            };
            for (const Variant &v : variants)
                std::printf(" %8.2f", time(v.kernel, v.group, false));
            std::printf(" %10.2f\n", time(SegmentSumKernel::Auto, 0, true));
        }
    }
    return 0;
}
