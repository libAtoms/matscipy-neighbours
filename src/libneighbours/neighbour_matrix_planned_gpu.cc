/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * EXPERIMENTAL planned GPU neighbour matrix (see neighbour_matrix_planned.hh).
 * The same pipeline as the regular build in neighbour_list_gpu.cc (cell index,
 * dense histogram or hashed table, scan, scatter, gather, fill), the same cell
 * list kernels (cell_list_gpu.hh) and the same traversal (visit_neighbours), but:
 * every launch is on the caller's stream; all scratch lives in the caller's
 * workspace; the geometry derived from the cell is computed on the device
 * (k_geometry) and read by the later kernels from device memory; flags stay on
 * the device. Nothing here synchronises, allocates or reads back, and the graph
 * it records holds kernel nodes only (no memset/memcpy nodes: one inside an XLA
 * conditional crashed cuGraphInstantiate on driver 560 / CUDA 12.6), so it can
 * be captured in a CUDA graph, conditional bodies included.
 */

#include "neighbour_matrix_planned.hh"

#include <algorithm>
#include <cmath>

#include "cell_list.hh"
#include "cell_list_gpu.hh"
#include "device.hh"
#include "error.hh"
#include "neighbour_visit.hh"
#include "tools.hh"

#if defined(MATSCIPY_ENABLE_CUDA)
#include <cub/device/device_scan.cuh>
namespace gpuprim = cub;
using gpuStream_t = cudaStream_t;
#elif defined(MATSCIPY_ENABLE_HIP)
#include <hipcub/hipcub.hpp>
namespace gpuprim = hipcub;
using gpuStream_t = hipStream_t;
#endif

namespace matscipy {
namespace {

constexpr int BLOCK = 256;
constexpr std::size_t ALIGN = 256;
inline int grid_for(index_t n) { return static_cast<int>((n + BLOCK - 1) / BLOCK); }
inline int grid_capped(index_t n) { return static_cast<int>(std::min<index_t>((n + BLOCK - 1) / BLOCK, 65535)); }
inline std::size_t align_up(std::size_t x) { return (x + ALIGN - 1) / ALIGN * ALIGN; }

/* Per-build geometry, written by k_geometry from the current cell. */
struct Geometry {
    real_t origin[3];     /* zero: the planned build has no cell origin */
    real_t inv[9];        /* inv(cell^T): fractional = inv @ r */
    real_t bin1[3], bin2[3], bin3[3];
    index_t nx, ny, nz;   /* cells to scan each way */
};

/* Fill / copy / widen as kernels rather than memset or memcpy nodes (see top). */
template <typename T>
__global__ void k_fill_value(T *p, T v, index_t n) {
    for (index_t a = blockIdx.x * index_t(blockDim.x) + threadIdx.x; a < n; a += index_t(blockDim.x) * gridDim.x)
        p[a] = v;
}

__global__ void k_copy_index(const index_t *src, index_t *dst, index_t n) {
    for (index_t a = blockIdx.x * index_t(blockDim.x) + threadIdx.x; a < n; a += index_t(blockDim.x) * gridDim.x)
        dst[a] = src[a];
}

__global__ void k_widen(const float *src, real_t *dst, index_t n) {
    for (index_t a = blockIdx.x * index_t(blockDim.x) + threadIdx.x; a < n; a += index_t(blockDim.x) * gridDim.x)
        dst[a] = static_cast<real_t>(src[a]);
}

/* Squared per-type cutoffs; a cutoff above the grid's (or not finite) is invalid.
   In place is fine (src == sq): each thread reads and writes its own entries. */
__global__ void k_square_cutoffs(const real_t *c, real_t *sq, index_t n, real_t cutoff,
                                 std::int32_t *status) {
    for (index_t a = blockIdx.x * index_t(blockDim.x) + threadIdx.x; a < n; a += index_t(blockDim.x) * gridDim.x) {
        const real_t v = c[a];
        if (!(v <= cutoff) || !is_finite(v)) status[1] = 1;
        sq[a] = v * v;
    }
}

/* One thread: the regular build's host-side cell_grid_geometry, with the grid
   resolution fixed by the plan. A degenerate cell sets status[1]. */
__global__ void k_geometry(const real_t *cell, index_t n1, index_t n2, index_t n3,
                           real_t cutoff, Geometry *g, std::int32_t *status) {
    const real_t *c1 = &cell[0], *c2 = &cell[3], *c3 = &cell[6];
    real_t nrm1[3], nrm2[3], nrm3[3];
    auto cross = [](const real_t *a, const real_t *b, real_t *o) {
        o[0] = a[1] * b[2] - a[2] * b[1];
        o[1] = a[2] * b[0] - a[0] * b[2];
        o[2] = a[0] * b[1] - a[1] * b[0];
    };
    auto nrm = [](const real_t *a) { return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); };
    cross(c2, c3, nrm1);
    cross(c3, c1, nrm2);
    cross(c1, c2, nrm3);
    const real_t det = c1[0] * nrm1[0] + c1[1] * nrm1[1] + c1[2] * nrm1[2];
    const real_t volume = fabs(det);
    if (!(volume > 1e-12 * nrm(c1) * nrm(c2) * nrm(c3)) || !is_finite(volume)) {
        status[1] = 1;
        return;
    }
    /* inv(cell^T): its rows are the reciprocal vectors (c2 x c3, ...) / det */
    for (int k = 0; k < 3; k++) {
        g->inv[0 + k] = nrm1[k] / det;
        g->inv[3 + k] = nrm2[k] / det;
        g->inv[6 + k] = nrm3[k] / det;
        g->origin[k] = 0;
        g->bin1[k] = c1[k] / n1;
        g->bin2[k] = c2[k] / n2;
        g->bin3[k] = c3[k] / n3;
    }
    g->nx = static_cast<index_t>(ceil(cutoff * n1 / (volume / nrm(nrm1))));
    g->ny = static_cast<index_t>(ceil(cutoff * n2 / (volume / nrm(nrm2))));
    g->nz = static_cast<index_t>(ceil(cutoff * n3 / (volume / nrm(nrm3))));
}

/* Raw cell index per atom; a non-finite position, or a type out of range, is
   invalid (the atom is parked in cell 0 so later indexing stays in range). */
__global__ void k_cell_index(const Geometry *g, const real_t *r, const std::int32_t *types,
                             index_t ncutoffs, index_t n1, index_t n2, index_t n3, index_t nat,
                             index_t *raw, std::int32_t *status) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    const bool bad_type = types && (types[a] < 0 || types[a] >= ncutoffs);
    if (!is_finite(r[3 * a]) || !is_finite(r[3 * a + 1]) || !is_finite(r[3 * a + 2]) || bad_type) {
        status[1] = 1;
        raw[3 * a] = raw[3 * a + 1] = raw[3 * a + 2] = 0;
        return;
    }
    position_to_cell_index(g->origin, g->inv, &r[3 * a], n1, n2, n3, &raw[3 * a],
                           &raw[3 * a + 1], &raw[3 * a + 2]);
}

__global__ void k_gather(const Geometry *g, const index_t *sorted, const index_t *raw,
                         const real_t *r, const std::int32_t *types, int pbc0, int pbc1, int pbc2,
                         index_t n1, index_t n2, index_t n3, index_t nat, index_t *raw_s,
                         index_t *rel_s, real_t *pos_s, index_t *types_s) {
    index_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nat) return;
    index_t a = sorted[s];
    index_t c1 = raw[3 * a], c2 = raw[3 * a + 1], c3 = raw[3 * a + 2];
    raw_s[3 * s] = c1;
    raw_s[3 * s + 1] = c2;
    raw_s[3 * s + 2] = c3;
    index_t r1 = pbc0 ? c1 : bin_trunc(c1, n1);
    index_t r2 = pbc1 ? c2 : bin_trunc(c2, n2);
    index_t r3 = pbc2 ? c3 : bin_trunc(c3, n3);
    rel_s[3 * s] = r1;
    rel_s[3 * s + 1] = r2;
    rel_s[3 * s + 2] = r3;
    for (int k = 0; k < 3; k++)
        pos_s[3 * s + k] = r[3 * a + k] - r1 * g->bin1[k] - r2 * g->bin2[k] - r3 * g->bin3[k];
    if (types_s) types_s[s] = types[a];
}

struct Filler32 {
    index_t i, K, w = 0;
    const index_t *sorted_atom;
    std::int32_t *idx, *shift;
    MATSCIPY_HD void operator()(index_t sj, const real_t *, real_t, const index_t *sh) {
        if (w < K) {
            const std::size_t slot = static_cast<std::size_t>(i) * K + w;
            idx[slot] = static_cast<std::int32_t>(sorted_atom[sj]);
            shift[3 * slot + 0] = static_cast<std::int32_t>(sh[0]);
            shift[3 * slot + 1] = static_cast<std::int32_t>(sh[1]);
            shift[3 * slot + 2] = static_cast<std::int32_t>(sh[2]);
        }
        w++;
    }
};

/* The fill pass. The context's geometry comes from device memory (k_geometry);
   the rest is fixed by the plan. Rows are skipped (count 0) when the input was
   invalid: the geometry may be unwritten. */
template <typename Query>
__global__ void k_fill(const Geometry *g, NeighbourContext c, Query q, index_t nat, index_t K,
                       std::int32_t *idx, std::int32_t *shift, std::int32_t *count,
                       std::int32_t *status) {
    index_t si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= nat) return;
    const index_t i = c.sorted_atom[si];
    if (status[1]) {
        count[i] = 0;
        return;
    }
    c.nx = g->nx;
    c.ny = g->ny;
    c.nz = g->nz;
    for (int k = 0; k < 3; k++) {
        c.bin1[k] = g->bin1[k];
        c.bin2[k] = g->bin2[k];
        c.bin3[k] = g->bin3[k];
    }
    Filler32 f;
    f.i = i;
    f.K = K;
    f.sorted_atom = c.sorted_atom;
    f.idx = idx;
    f.shift = shift;
    visit_neighbours(c, q, si, f);
    count[i] = static_cast<std::int32_t>(f.w);
    if (f.w > K) status[0] = 1;
}

template <typename T>
T *at(void *ws, std::size_t off) {
    return reinterpret_cast<T *>(static_cast<char *>(ws) + off);
}

/* Hash capacity of the hashed lookup: a power of two, load factor <= 0.5 (as the
   regular build). */
std::int64_t hash_capacity(index_t nat) {
    std::int64_t cap = 1;
    while (cap < 2 * static_cast<std::int64_t>(nat) + 1) cap <<= 1;
    return cap;
}

template <typename Query>
void enqueue_fill(const NeighbourMatrixPlan &p, const Geometry *g, const NeighbourContext &c,
                  const Query &q, gpuStream_t s, std::int32_t *idx, std::int32_t *shift,
                  std::int32_t *count, std::int32_t *status) {
    k_fill<Query><<<grid_for(p.nat), BLOCK, 0, s>>>(g, c, q, p.nat, p.K, idx, shift, count, status);
}

}  // namespace

error_t plan_neighbour_matrix(index_t nat, const real_t cell[9], const bool pbc[3], real_t cutoff,
                              index_t K, index_t ncutoffs, bool single, NeighbourMatrixPlan &plan) {
    clear_error();
    if (!(cutoff > 0) || !std::isfinite(cutoff))
        return set_invalid_argument("cutoff must be positive and finite.");
    if (K < 1) return set_invalid_argument("max_neighbours must be positive.");
    if (nat < 0 || ncutoffs < 0) return set_invalid_argument("negative atom or type count.");
    real_t inv[9] = {0}, origin[3] = {0, 0, 0};   /* the grid ignores inv_cell */
    CellGrid grid;
    if (!cell_grid_geometry(origin, cell, inv, pbc, cutoff, grid))
        return set_invalid_argument("Zero cell volume.");
    const std::int64_t ncells = static_cast<std::int64_t>(grid.n1) * grid.n2 * grid.n3;
    const std::int64_t budget = std::max<std::int64_t>(1 << 20, 8 * static_cast<std::int64_t>(nat));
    return layout_neighbour_matrix(nat, K, grid.n1, grid.n2, grid.n3, pbc, cutoff, ncutoffs, single,
                                   ncells > budget, plan);
}

error_t layout_neighbour_matrix(index_t nat, index_t K, index_t n1, index_t n2, index_t n3,
                                const bool pbc[3], real_t cutoff, index_t ncutoffs, bool single,
                                bool hashed, NeighbourMatrixPlan &plan) {
    if (nat < 0 || K < 1 || n1 < 1 || n2 < 1 || n3 < 1 || !(cutoff > 0) || ncutoffs < 0)
        return set_invalid_argument("planned build: bad plan.");
    if (nat >= (index_t(1) << 29)) return set_error("GPU backend supports at most 2^29 atoms per call.");
    plan.nat = nat;
    plan.K = K;
    plan.n1 = n1;
    plan.n2 = n2;
    plan.n3 = n3;
    for (int k = 0; k < 3; k++) plan.pbc[k] = pbc[k];
    plan.cutoff = cutoff;
    plan.ncutoffs = ncutoffs;
    plan.single = single;
    plan.hashed = hashed;
    const std::int64_t cap = hashed ? hash_capacity(nat) : 0;
    plan.hmask = hashed ? cap - 1 : 0;
    /* the scanned table: cells (dense) or hash slots (hashed) */
    const std::int64_t nt = hashed ? cap : static_cast<std::int64_t>(n1) * n2 * n3;
    if (nt >= (std::int64_t(1) << 31)) return set_error("planned build: cell table too large.");
    std::size_t temp = 0;   /* size query only: no device work */
    GPU_CHECK(gpuprim::DeviceScan::ExclusiveSum(nullptr, temp, static_cast<index_t *>(nullptr),
                                                static_cast<index_t *>(nullptr), static_cast<int>(nt)));
    plan.temp_bytes = temp > 0 ? temp : 1;
    std::size_t off = 0;
    auto take = [&off](std::size_t bytes) {
        const std::size_t o = off;
        off += align_up(bytes > 0 ? bytes : 1);
        return o;
    };
    const std::size_t I = sizeof(index_t), R = sizeof(real_t), n = static_cast<std::size_t>(nat);
    const std::size_t T = static_cast<std::size_t>(nt);
    plan.off_geom = take(sizeof(Geometry));
    plan.off_raw = take(3 * n * I);
    plan.off_lin = take(hashed ? 0 : n * I);
    plan.off_count = take(T * I);
    plan.off_first = take(T * I);
    plan.off_cursor = take(T * I);
    plan.off_sorted = take(n * I);
    plan.off_raw_s = take(3 * n * I);
    plan.off_rel_s = take(3 * n * I);
    plan.off_pos_s = take(3 * n * R);
    plan.off_types_s = take(ncutoffs > 0 ? n * I : 0);
    plan.off_pos64 = take(single ? 3 * n * R : 0);
    plan.off_cell64 = take(9 * R);
    plan.off_pt_sq = take(static_cast<std::size_t>(ncutoffs * ncutoffs) * R);
    plan.off_hkey = take(hashed ? T * sizeof(std::int64_t) : 0);
    plan.off_key = take(hashed ? n * sizeof(std::int64_t) : 0);
    plan.off_temp = take(plan.temp_bytes);
    plan.workspace_bytes = off;
    return NL_SUCCESS;
}

error_t neighbour_matrix_gpu_planned(const NeighbourMatrixPlan &p, const void *d_positions,
                                     const void *d_cell, const std::int32_t *d_types,
                                     const void *d_per_type_cutoff, void *ws, void *stream_ptr,
                                     std::int32_t *idx, std::int32_t *shift, std::int32_t *count,
                                     std::int32_t *status) {
    clear_error();
    const gpuStream_t s = static_cast<gpuStream_t>(stream_ptr);
    const index_t n = p.nat, nc2 = p.ncutoffs * p.ncutoffs;
    if (p.ncutoffs > 0 && (!d_types || !d_per_type_cutoff))
        return set_invalid_argument("planned build: a per-type plan needs types and cutoffs.");
    k_fill_value<std::int32_t><<<1, 32, 0, s>>>(status, 0, 2);
    if (n == 0) return NL_SUCCESS;
    Geometry *g = at<Geometry>(ws, p.off_geom);
    index_t *raw = at<index_t>(ws, p.off_raw);
    index_t *cnt = at<index_t>(ws, p.off_count), *first = at<index_t>(ws, p.off_first);
    index_t *cursor = at<index_t>(ws, p.off_cursor), *sorted = at<index_t>(ws, p.off_sorted);
    index_t *raw_s = at<index_t>(ws, p.off_raw_s), *rel_s = at<index_t>(ws, p.off_rel_s);
    real_t *pos_s = at<real_t>(ws, p.off_pos_s);
    index_t *types_s = p.ncutoffs > 0 ? at<index_t>(ws, p.off_types_s) : nullptr;
    real_t *pt_sq = p.ncutoffs > 0 ? at<real_t>(ws, p.off_pt_sq) : nullptr;
    const int g_at = grid_for(n);
    const int pb0 = p.pbc[0], pb1 = p.pbc[1], pb2 = p.pbc[2];

    /* 0. inputs as double: float32 positions, cell and cutoffs widened here */
    const real_t *r = static_cast<const real_t *>(d_positions);
    const real_t *cell = static_cast<const real_t *>(d_cell);
    const real_t *ptc = static_cast<const real_t *>(d_per_type_cutoff);
    if (p.single) {
        real_t *r64 = at<real_t>(ws, p.off_pos64), *cell64 = at<real_t>(ws, p.off_cell64);
        k_widen<<<grid_capped(3 * n), BLOCK, 0, s>>>(static_cast<const float *>(d_positions), r64, 3 * n);
        k_widen<<<1, 32, 0, s>>>(static_cast<const float *>(d_cell), cell64, 9);
        r = r64;
        cell = cell64;
        if (p.ncutoffs > 0) {   /* widened into pt_sq, then squared in place */
            k_widen<<<grid_capped(nc2), BLOCK, 0, s>>>(static_cast<const float *>(d_per_type_cutoff), pt_sq, nc2);
            ptc = pt_sq;
        }
    }
    if (p.ncutoffs > 0)
        k_square_cutoffs<<<grid_capped(nc2), BLOCK, 0, s>>>(ptc, pt_sq, nc2, p.cutoff, status);

    /* 1. geometry from the current cell; raw cell index per atom */
    k_geometry<<<1, 1, 0, s>>>(cell, p.n1, p.n2, p.n3, p.cutoff, g, status);
    k_cell_index<<<g_at, BLOCK, 0, s>>>(g, r, p.ncutoffs > 0 ? d_types : nullptr, p.ncutoffs, p.n1,
                                        p.n2, p.n3, n, raw, status);

    /* 2. the cell list: dense histogram or hashed table, scan, scatter */
    std::size_t temp = p.temp_bytes;
    void *d_temp = at<void>(ws, p.off_temp);
    if (!p.hashed) {
        const index_t nc = p.n1 * p.n2 * p.n3;
        index_t *lin = at<index_t>(ws, p.off_lin);
        k_fill_value<index_t><<<grid_capped(nc), BLOCK, 0, s>>>(cnt, 0, nc);
        k_fold_lin_hist<<<g_at, BLOCK, 0, s>>>(raw, pb0, pb1, pb2, p.n1, p.n2, p.n3, n, lin, cnt);
        gpuprim::DeviceScan::ExclusiveSum(d_temp, temp, cnt, first, static_cast<int>(nc), s);
        k_copy_index<<<grid_capped(nc), BLOCK, 0, s>>>(first, cursor, nc);
        k_scatter<<<g_at, BLOCK, 0, s>>>(lin, n, cursor, sorted);
    } else {
        const index_t cap = static_cast<index_t>(p.hmask + 1);
        std::int64_t *hkey = at<std::int64_t>(ws, p.off_hkey), *key = at<std::int64_t>(ws, p.off_key);
        k_fill_value<std::int64_t><<<grid_capped(cap), BLOCK, 0, s>>>(hkey, -1, cap);
        k_fill_value<index_t><<<grid_capped(cap), BLOCK, 0, s>>>(cnt, 0, cap);
        k_fold_key64<<<g_at, BLOCK, 0, s>>>(raw, pb0, pb1, pb2, p.n1, p.n2, p.n3, n, key);
        k_hash_insert<<<g_at, BLOCK, 0, s>>>(key, n, p.hmask, hkey, cnt);
        gpuprim::DeviceScan::ExclusiveSum(d_temp, temp, cnt, first, static_cast<int>(cap), s);
        k_copy_index<<<grid_capped(cap), BLOCK, 0, s>>>(first, cursor, cap);
        k_hash_scatter<<<g_at, BLOCK, 0, s>>>(key, n, p.hmask, hkey, cursor, sorted);
    }

    /* 3. gather per-atom data into cell order; 4. fill */
    k_gather<<<g_at, BLOCK, 0, s>>>(g, sorted, raw, r, p.ncutoffs > 0 ? d_types : nullptr, pb0, pb1,
                                    pb2, p.n1, p.n2, p.n3, n, raw_s, rel_s, pos_s, types_s);
    NeighbourContext c{};
    c.n1 = p.n1;
    c.n2 = p.n2;
    c.n3 = p.n3;
    c.pbc0 = pb0;
    c.pbc1 = pb1;
    c.pbc2 = pb2;
    c.cutoff_sq = p.cutoff * p.cutoff;
    c.per_type_cutoff_sq = pt_sq;
    c.ncutoffs = p.ncutoffs;
    c.sorted_atom = sorted;
    c.raw_cell = raw_s;
    c.rel_cell = rel_s;
    c.rel_pos = pos_s;
    c.per_atom = nullptr;
    c.types = types_s;
    if (!p.hashed) {
        enqueue_fill(p, g, c, DenseQuery{p.n1, p.n2, first, cnt}, s, idx, shift, count, status);
    } else {
        enqueue_fill(p, g, c,
                     SparseQuery{p.n1, p.n2, p.hmask, at<std::int64_t>(ws, p.off_hkey), first, cnt}, s,
                     idx, shift, count, status);
    }
    const gpuError_t e = gpuGetLastError();   /* enqueue errors only: no synchronisation */
    if (e != gpuSuccess) return set_error(gpuGetErrorString(e));
    return NL_SUCCESS;
}

}  // namespace matscipy
