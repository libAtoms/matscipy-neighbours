/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Phase 3.2 — GPU neighbour-list kernels. The CPU pipeline (neighbour_list.cc)
 * ported one-to-one onto the device: compute cell index, atomic-histogram the
 * cells, exclusive-scan to a CSR (Phase-3.3 primitive), scatter atoms into
 * cell-sorted order, gather per-atom data, then the two independent passes
 * (count, scan, fill). Every per-atom pass is one thread per atom.
 */

#include "neighbour_list_gpu.hh"

#include <cmath>

#include "device.hh"
#include "device_primitives.hh"
#include "error.hh"
#include "memory_space.hh"
#include "tools.hh"

namespace matscipy {
namespace {

constexpr int BLOCK = 256;
inline int grid_for(index_t n) { return static_cast<int>((n + BLOCK - 1) / BLOCK); }

/* --- device-callable mirrors of the host helpers (must match exactly) ----- */

__device__ inline int d_bin_wrap(int i, int n) {
    while (i < 0) i += n;
    while (i >= n) i -= n;
    return i;
}
__device__ inline int d_bin_trunc(int i, int n) {
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}
__device__ inline void d_pos_to_cell(const real_t *origin, const real_t *inv,
                                     const real_t *ri, int n1, int n2, int n3,
                                     int *c1, int *c2, int *c3) {
    real_t d0 = ri[0] - origin[0], d1 = ri[1] - origin[1], d2 = ri[2] - origin[2];
    real_t s0 = inv[0] * d0 + inv[1] * d1 + inv[2] * d2;
    real_t s1 = inv[3] * d0 + inv[4] * d1 + inv[5] * d2;
    real_t s2 = inv[6] * d0 + inv[7] * d1 + inv[8] * d2;
    *c1 = static_cast<int>(floor(s0 * n1));
    *c2 = static_cast<int>(floor(s1 * n2));
    *c3 = static_cast<int>(floor(s2 * n3));
}

/* Morton (Z-curve) key of a cell, mirroring cell_list.cc's host version. */
__device__ inline std::uint64_t d_part1by2(std::uint64_t x) {
    x &= 0x1fffffull;
    x = (x | (x << 32)) & 0x1f00000000ffffull;
    x = (x | (x << 16)) & 0x1f0000ff0000ffull;
    x = (x | (x << 8)) & 0x100f00f00f00f00full;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
    x = (x | (x << 2)) & 0x1249249249249249ull;
    return x;
}
__device__ inline std::uint64_t d_morton3(int a, int b, int c) {
    return d_part1by2(a) | (d_part1by2(b) << 1) | (d_part1by2(c) << 2);
}

/* RAII: switch to `dev` for the duration of the build, restore on exit. A
   negative id means "use the current device, don't switch" (host-input path). */
struct DeviceGuard {
    int prev = -1;
    explicit DeviceGuard(int dev) {
        if (dev >= 0) {
            int cur = 0;
            GPU_CHECK(gpuGetDevice(&cur));
            if (dev != cur) {
                GPU_CHECK(gpuSetDevice(dev));
                prev = cur;
            }
        }
    }
    ~DeviceGuard() {
        if (prev >= 0) gpuSetDevice(prev);
    }
};

/* 64-bit cell hash (Fibonacci), mirroring cell_list.cc's host version. */
__device__ inline std::int64_t d_cell_hash(std::int64_t key) {
    std::uint64_t h = static_cast<std::uint64_t>(key) * 0x9E3779B97F4A7C15ull;
    return static_cast<std::int64_t>(h >> 1);
}

/* Cell-lookup policies: map neighbour-cell coords -> [b, e) slice of
   sorted_atom. Dense indexes arrays by linear cell; sparse probes a hash table
   keyed by the 64-bit linear cell index (for huge/sparse grids). The kernels
   are templated on the policy, exactly like the CPU visit_neighbours. */
struct DenseQueryDev {
    int n1, n2;
    const int *cell_first;
    const int *cell_count;
    __device__ void slice(int c1, int c2, int c3, int &b, int &e) const {
        int c = c1 + n1 * (c2 + n2 * c3);
        b = cell_first[c];
        e = b + cell_count[c];
    }
};
struct SparseQueryDev {
    int n1, n2;
    std::int64_t mask;
    const std::int64_t *hkey;
    const int *hfirst;
    const int *hcount;
    __device__ void slice(int c1, int c2, int c3, int &b, int &e) const {
        std::int64_t key = static_cast<std::int64_t>(c1) +
                           static_cast<std::int64_t>(n1) *
                               (static_cast<std::int64_t>(c2) +
                                static_cast<std::int64_t>(n2) * c3);
        std::int64_t h = d_cell_hash(key) & mask;
        while (hkey[h] != -1) {
            if (hkey[h] == key) {
                b = hfirst[h];
                e = b + hcount[h];
                return;
            }
            h = (h + 1) & mask;
        }
        b = 0;
        e = 0;
    }
};

/* Per-kernel context, passed by value (small, all device pointers). The cell
   lookup lives in a separate Query (dense or sparse) passed alongside. */
struct DevCtx {
    int n1, n2, n3, nx, ny, nz;
    int pbc0, pbc1, pbc2;
    real_t bin1[3], bin2[3], bin3[3];
    real_t cutoff_sq;
    const real_t *per_type_cutoff_sq;
    int ncutoffs;
    const real_t *rel_pos;     /* 3*nat sorted */
    const int *raw_cell;       /* 3*nat sorted (unwrapped) */
    const int *rel_cell;       /* 3*nat sorted */
    const int *sorted_atom;    /* nat */
    const real_t *per_atom;    /* nat sorted or null */
    const int *types;          /* nat sorted or null */
};

/* Visit every neighbour of sorted atom `si`, calling f(sj, dr[3], r2, shift[3]).
   Byte-for-byte the same traversal/pair test as the host visit_neighbours. */
template <typename Query, typename F>
__device__ inline void d_visit(const DevCtx &c, const Query &q, int si, F &f) {
    const int n1 = c.n1, n2 = c.n2, n3 = c.n3;
    const int *raw_i = &c.raw_cell[3 * si];
    const real_t *dri = &c.rel_pos[3 * si];
    const int ci1 = c.pbc0 ? d_bin_wrap(raw_i[0], n1) : d_bin_trunc(raw_i[0], n1);
    const int ci2 = c.pbc1 ? d_bin_wrap(raw_i[1], n2) : d_bin_trunc(raw_i[1], n2);
    const int ci3 = c.pbc2 ? d_bin_wrap(raw_i[2], n3) : d_bin_trunc(raw_i[2], n3);

    for (int z = -c.nz; z <= c.nz; z++) {
        int cj3 = ci3 + z;
        if (c.pbc2) cj3 = d_bin_wrap(cj3, n3);
        if (cj3 < 0 || cj3 >= n3) continue;
        real_t off3[3] = {z * c.bin3[0], z * c.bin3[1], z * c.bin3[2]};

        for (int y = -c.ny; y <= c.ny; y++) {
            int cj2 = ci2 + y;
            if (c.pbc1) cj2 = d_bin_wrap(cj2, n2);
            if (cj2 < 0 || cj2 >= n2) continue;
            real_t off2[3] = {off3[0] + y * c.bin2[0], off3[1] + y * c.bin2[1],
                              off3[2] + y * c.bin2[2]};

            for (int x = -c.nx; x <= c.nx; x++) {
                int cj1 = ci1 + x;
                if (c.pbc0) cj1 = d_bin_wrap(cj1, n1);
                if (cj1 < 0 || cj1 >= n1) continue;
                real_t off[3] = {off2[0] + x * c.bin1[0], off2[1] + x * c.bin1[1],
                                 off2[2] + x * c.bin1[2]};

                int begin, end;
                q.slice(cj1, cj2, cj3, begin, end);
                for (int sj = begin; sj < end; sj++) {
                    if (sj == si && x == 0 && y == 0 && z == 0) continue;
                    const real_t *drj = &c.rel_pos[3 * sj];
                    real_t dr[3] = {drj[0] - dri[0] + off[0],
                                    drj[1] - dri[1] + off[1],
                                    drj[2] - dri[2] + off[2]};
                    real_t r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if (r2 >= c.cutoff_sq) continue;

                    bool inside = true;
                    if (c.per_atom) {
                        real_t s = c.per_atom[si] + c.per_atom[sj];
                        inside = r2 < s * s;
                    } else if (c.per_type_cutoff_sq && c.types) {
                        int ti = c.types[si], tj = c.types[sj];
                        if (ti >= 0 && ti < c.ncutoffs && tj >= 0 &&
                            tj < c.ncutoffs)
                            inside = r2 < c.per_type_cutoff_sq[ti * c.ncutoffs + tj];
                    }
                    if (!inside) continue;

                    const int *crj = &c.rel_cell[3 * sj];
                    int shift[3] = {(raw_i[0] - crj[0] + x) / n1,
                                    (raw_i[1] - crj[1] + y) / n2,
                                    (raw_i[2] - crj[2] + z) / n3};
                    f(sj, dr, r2, shift);
                }
            }
        }
    }
}

struct Counter {
    int n = 0;
    __device__ void operator()(int, const real_t *, real_t, const int *) { n++; }
};

struct Filler {
    int i, w;
    const int *sorted_atom;
    int *first, *secnd, *shift;
    real_t *distvec, *absdist;
    __device__ void operator()(int sj, const real_t *dr, real_t r2,
                               const int *sh) {
        if (first) first[w] = i;
        if (secnd) secnd[w] = sorted_atom[sj];
        if (distvec) {
            distvec[3 * w + 0] = dr[0];
            distvec[3 * w + 1] = dr[1];
            distvec[3 * w + 2] = dr[2];
        }
        if (absdist) absdist[w] = sqrt(r2);
        if (shift) {
            shift[3 * w + 0] = sh[0];
            shift[3 * w + 1] = sh[1];
            shift[3 * w + 2] = sh[2];
        }
        w++;
    }
};

/* --- kernels --------------------------------------------------------------- */

__global__ void k_cell_index(const real_t *origin, const real_t *inv,
                             const real_t *r, int n1, int n2, int n3,
                             index_t nat, int *raw) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    d_pos_to_cell(origin, inv, &r[3 * a], n1, n2, n3, &raw[3 * a],
                  &raw[3 * a + 1], &raw[3 * a + 2]);
}

__global__ void k_fold_lin_hist(const int *raw, int pbc0, int pbc1, int pbc2,
                                int n1, int n2, int n3, index_t nat, int *lin,
                                int *cell_count) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    int c1 = pbc0 ? d_bin_wrap(raw[3 * a], n1) : d_bin_trunc(raw[3 * a], n1);
    int c2 = pbc1 ? d_bin_wrap(raw[3 * a + 1], n2) : d_bin_trunc(raw[3 * a + 1], n2);
    int c3 = pbc2 ? d_bin_wrap(raw[3 * a + 2], n3) : d_bin_trunc(raw[3 * a + 2], n3);
    int l = c1 + n1 * (c2 + n2 * c3);
    lin[a] = l;
    atomicAdd(&cell_count[l], 1);
}

__global__ void k_scatter(const int *lin, index_t nat, int *cursor,
                          int *sorted_atom) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    int pos = atomicAdd(&cursor[lin[a]], 1);
    sorted_atom[pos] = a;
}

__global__ void k_iota(int *v, index_t n) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < n) v[a] = static_cast<int>(a);
}

/* Morton key (Z-curve) of each atom's wrapped cell — the radix-sort key for the
   Morton (coalesced) layout. */
__global__ void k_morton_key(const int *raw, int pbc0, int pbc1, int pbc2,
                            int n1, int n2, int n3, index_t nat,
                            std::uint64_t *key) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    int c1 = pbc0 ? d_bin_wrap(raw[3 * a], n1) : d_bin_trunc(raw[3 * a], n1);
    int c2 = pbc1 ? d_bin_wrap(raw[3 * a + 1], n2) : d_bin_trunc(raw[3 * a + 1], n2);
    int c3 = pbc2 ? d_bin_wrap(raw[3 * a + 2], n3) : d_bin_trunc(raw[3 * a + 2], n3);
    key[a] = d_morton3(c1, c2, c3);
}

/* After a Morton sort, atoms of one cell are a contiguous run in `sorted_atom`
   (the key is unique per cell). Record where each run starts: cell_first[lin]. */
__global__ void k_cell_first_from_runs(const int *sorted_atom, const int *lin,
                                       index_t nat, int *cell_first) {
    index_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nat) return;
    int c = lin[sorted_atom[s]];
    if (s == 0 || lin[sorted_atom[s - 1]] != c) cell_first[c] = s;
}

/* --- sparse (hashed compact) cell list, for huge/sparse grids --------------
   Open addressing keyed by the 64-bit linear cell index; built in parallel with
   atomicCAS inserts. The dense histogram/array path would need O(ncells) memory
   (or overflow a 32-bit index) here. Mirrors cell_list.cc's build_sparse. */

__global__ void k_fold_key64(const int *raw, int pbc0, int pbc1, int pbc2,
                            int n1, int n2, int n3, index_t nat,
                            std::int64_t *key) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    int c1 = pbc0 ? d_bin_wrap(raw[3 * a], n1) : d_bin_trunc(raw[3 * a], n1);
    int c2 = pbc1 ? d_bin_wrap(raw[3 * a + 1], n2) : d_bin_trunc(raw[3 * a + 1], n2);
    int c3 = pbc2 ? d_bin_wrap(raw[3 * a + 2], n3) : d_bin_trunc(raw[3 * a + 2], n3);
    key[a] = static_cast<std::int64_t>(c1) +
             static_cast<std::int64_t>(n1) *
                 (static_cast<std::int64_t>(c2) + static_cast<std::int64_t>(n2) * c3);
}

__global__ void k_hash_insert(const std::int64_t *key, index_t nat,
                             std::int64_t mask, std::int64_t *hkey, int *hcount) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    std::int64_t k = key[a];
    std::int64_t h = d_cell_hash(k) & mask;
    auto *slot = reinterpret_cast<unsigned long long *>(hkey);
    const unsigned long long empty = static_cast<unsigned long long>(-1);
    while (true) {
        unsigned long long old =
            atomicCAS(&slot[h], empty, static_cast<unsigned long long>(k));
        if (old == empty || static_cast<std::int64_t>(old) == k) {
            atomicAdd(&hcount[h], 1);
            return;
        }
        h = (h + 1) & mask;
    }
}

__global__ void k_hash_scatter(const std::int64_t *key, index_t nat,
                              std::int64_t mask, const std::int64_t *hkey,
                              int *cursor, int *sorted_atom) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    std::int64_t k = key[a];
    std::int64_t h = d_cell_hash(k) & mask;
    while (hkey[h] != k) h = (h + 1) & mask;  /* key is guaranteed present */
    int pos = atomicAdd(&cursor[h], 1);
    sorted_atom[pos] = static_cast<int>(a);
}

__global__ void k_gather(const int *sorted_atom, const int *raw, const real_t *r,
                        int pbc0, int pbc1, int pbc2, int n1, int n2, int n3,
                        real_t b1x, real_t b1y, real_t b1z, real_t b2x,
                        real_t b2y, real_t b2z, real_t b3x, real_t b3y,
                        real_t b3z, const real_t *per_atom, const int *types,
                        index_t nat, int *raw_s, int *rel_s, real_t *pos_s,
                        real_t *per_atom_s, int *types_s) {
    index_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nat) return;
    int a = sorted_atom[s];
    int c1 = raw[3 * a], c2 = raw[3 * a + 1], c3 = raw[3 * a + 2];
    raw_s[3 * s] = c1;
    raw_s[3 * s + 1] = c2;
    raw_s[3 * s + 2] = c3;
    int r1 = pbc0 ? c1 : d_bin_trunc(c1, n1);
    int r2 = pbc1 ? c2 : d_bin_trunc(c2, n2);
    int r3 = pbc2 ? c3 : d_bin_trunc(c3, n3);
    rel_s[3 * s] = r1;
    rel_s[3 * s + 1] = r2;
    rel_s[3 * s + 2] = r3;
    pos_s[3 * s] = r[3 * a] - r1 * b1x - r2 * b2x - r3 * b3x;
    pos_s[3 * s + 1] = r[3 * a + 1] - r1 * b1y - r2 * b2y - r3 * b3y;
    pos_s[3 * s + 2] = r[3 * a + 2] - r1 * b1z - r2 * b2z - r3 * b3z;
    if (per_atom_s) per_atom_s[s] = per_atom[a];
    if (types_s) types_s[s] = types[a];
}

template <typename Query>
__global__ void k_count(DevCtx c, Query q, index_t nat, int *cnt) {
    index_t si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= nat) return;
    Counter f;
    d_visit(c, q, static_cast<int>(si), f);
    cnt[c.sorted_atom[si]] = f.n;
}

template <typename Query>
__global__ void k_fill(DevCtx c, Query q, index_t nat, const int *offset,
                      int *first, int *secnd, real_t *distvec, real_t *absdist,
                      int *shift) {
    index_t si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= nat) return;
    int i = c.sorted_atom[si];
    Filler f;
    f.i = i;
    f.w = offset[i];
    f.sorted_atom = c.sorted_atom;
    f.first = first;
    f.secnd = secnd;
    f.distvec = distvec;
    f.absdist = absdist;
    f.shift = shift;
    d_visit(c, q, static_cast<int>(si), f);
}

/* Device buffer alias (RAII via the Phase-3.1 Array). */
template <typename T>
using DBuf = Array<T, DeviceSpace>;

/* Two-pass output, parameterised on the cell-lookup policy (dense or sparse).
   Always fills `dev.counts` (per-atom neighbour count — Phase 3.4 coordination,
   no pair materialisation); fills the pair arrays only when `want_pairs`. */
template <typename Query>
static error_t count_and_fill(const DevCtx &ctx, const Query &q, index_t nat,
                              int quantities, bool want_pairs,
                              NeighbourListDevice &dev) {
    const int g_at = grid_for(nat);

    /* pass 1: count neighbours per (original) atom; cnt[nat] = 0 sentinel. */
    DBuf<int> d_cnt(nat + 1), d_offset(nat + 1);
    GPU_CHECK(gpuMemset(d_cnt.data() + nat, 0, sizeof(int)));
    GPU_LAUNCH(k_count, g_at, BLOCK, ctx, q, nat, d_cnt.data());

    dev.counts.resize(nat);
    GPU_CHECK(gpuMemcpy(dev.counts.data(), d_cnt.data(), nat * sizeof(index_t),
                        gpuMemcpyDeviceToDevice));

    /* scan counts -> per-atom write offset; total pairs. */
    index_t npairs = device_exclusive_scan(d_cnt.data(), d_offset.data(), nat + 1);
    dev.npairs = npairs;
    if (!want_pairs || npairs == 0) {
        GPU_CHECK(gpuDeviceSynchronize());
        return NL_SUCCESS;
    }

    /* allocate exact-size outputs into `dev`, pass 2: fill. */
    const bool wf = quantities & QUANTITY_FIRST;
    const bool ws = quantities & QUANTITY_SECOND;
    const bool wD = quantities & QUANTITY_DISTVEC;
    const bool wd = quantities & QUANTITY_ABSDIST;
    const bool wS = quantities & QUANTITY_SHIFT;
    if (wf) dev.first.resize(npairs);
    if (ws) dev.secnd.resize(npairs);
    if (wD) dev.distvec.resize(3 * npairs);
    if (wd) dev.absdist.resize(npairs);
    if (wS) dev.shift.resize(3 * npairs);
    GPU_LAUNCH(k_fill, g_at, BLOCK, ctx, q, nat, d_offset.data(),
               wf ? dev.first.data() : nullptr, ws ? dev.secnd.data() : nullptr,
               wD ? dev.distvec.data() : nullptr,
               wd ? dev.absdist.data() : nullptr,
               wS ? dev.shift.data() : nullptr);
    GPU_CHECK(gpuDeviceSynchronize());
    return NL_SUCCESS;
}

/* Shared kernel pipeline. Runs the whole build and leaves the requested output
   quantities in `dev` (device memory). Both public entry points wrap this. */
static error_t build_device(int quantities, const real_t cell_origin[3],
                            const real_t cell[9], const real_t inv_cell[9],
                            const bool pbc[3], index_t nat, const real_t *r,
                            bool r_is_device, real_t cutoff,
                            const real_t *per_atom_cutoff,
                            const real_t *per_type_cutoff_sq, index_t ncutoffs,
                            const index_t *types, CellOrder order, int device_id,
                            bool want_pairs, NeighbourListDevice &dev) {
    clear_error();
    dev.npairs = 0;
    if (nat <= 0) return NL_SUCCESS;
    DeviceGuard guard(device_id);  /* run on the input's device; restore on exit */

    /* Geometry: identical to the CPU path. */
    const real_t *c1 = &cell[0], *c2 = &cell[3], *c3 = &cell[6];
    real_t nrm1[3], nrm2[3], nrm3[3];
    cross_product(c2, c3, nrm1);
    cross_product(c3, c1, nrm2);
    cross_product(c1, c2, nrm3);
    real_t volume = std::fabs(c3[0] * nrm3[0] + c3[1] * nrm3[1] + c3[2] * nrm3[2]);
    if (volume < 1e-12) return set_error("Zero cell volume.");
    real_t len1 = volume / norm(nrm1), len2 = volume / norm(nrm2),
           len3 = volume / norm(nrm3);
    int n1 = std::max(static_cast<int>(std::floor(len1 / cutoff)), 1);
    int n2 = std::max(static_cast<int>(std::floor(len2 / cutoff)), 1);
    int n3 = std::max(static_cast<int>(std::floor(len3 / cutoff)), 1);
    real_t bin1[3], bin2[3], bin3[3];
    for (int k = 0; k < 3; k++) {
        bin1[k] = c1[k] / n1;
        bin2[k] = c2[k] / n2;
        bin3[k] = c3[k] / n3;
    }
    /* Cell count in 64-bit: a huge/sparse grid can exceed a 32-bit index, which
       is exactly when the hashed compact backend is used instead of dense
       arrays (mirrors cell_list.cc's threshold). */
    const std::int64_t ncells = static_cast<std::int64_t>(n1) * n2 * n3;
    const std::int64_t budget =
        std::max<std::int64_t>(1 << 20, 8 * static_cast<std::int64_t>(nat));
    const bool sparse = ncells > budget;
    const int g_at = grid_for(nat);

    /* Upload geometry (always tiny host arrays). Positions are either uploaded
       (host input) or used in place (already-on-device input, e.g. cupy) — the
       latter avoids the input H2D round-trip for a device-native caller. */
    DBuf<real_t> d_origin(3), d_inv(9);
    GPU_CHECK(gpuMemcpy(d_origin.data(), cell_origin, 3 * sizeof(real_t),
                        gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_inv.data(), inv_cell, 9 * sizeof(real_t),
                        gpuMemcpyHostToDevice));
    DBuf<real_t> d_r_owned;
    const real_t *d_r;
    if (r_is_device) {
        d_r = r;
    } else {
        d_r_owned.resize(3 * nat);
        GPU_CHECK(gpuMemcpy(d_r_owned.data(), r, 3 * nat * sizeof(real_t),
                            gpuMemcpyHostToDevice));
        d_r = d_r_owned.data();
    }

    /* 1. raw (unwrapped) cell index per atom. */
    DBuf<int> d_raw(3 * nat);
    GPU_LAUNCH(k_cell_index, g_at, BLOCK, d_origin.data(), d_inv.data(), d_r,
               n1, n2, n3, nat, d_raw.data());

    /* 2. build the CSR cell list -> d_sorted plus a dense or sparse lookup.
          The dense backends (Linear scan+scatter / Morton radix-sort+runs) need
          O(ncells) arrays; for a huge/sparse grid we build a hashed compact
          table instead. */
    DBuf<int> d_sorted(nat);
    /* Dense backend buffers (empty for the sparse path). */
    DBuf<int> d_cell_first, d_cell_count;
    /* Sparse backend buffers (empty for the dense path). */
    DBuf<std::int64_t> d_hkey;
    DBuf<int> d_hfirst, d_hcount;
    std::int64_t hmask = 0;

    if (!sparse) {
        const index_t nc = static_cast<index_t>(ncells);
        DBuf<int> d_lin(nat);
        d_cell_count.resize(nc);
        d_cell_first.resize(nc);
        GPU_CHECK(gpuMemset(d_cell_count.data(), 0, nc * sizeof(int)));
        GPU_LAUNCH(k_fold_lin_hist, g_at, BLOCK, d_raw.data(), pbc[0], pbc[1],
                   pbc[2], n1, n2, n3, nat, d_lin.data(), d_cell_count.data());
        if (order == CellOrder::Morton) {
            DBuf<std::uint64_t> d_key(nat);
            GPU_LAUNCH(k_morton_key, g_at, BLOCK, d_raw.data(), pbc[0], pbc[1],
                       pbc[2], n1, n2, n3, nat, d_key.data());
            GPU_LAUNCH(k_iota, g_at, BLOCK, d_sorted.data(), nat);
            device_sort_pairs(d_key.data(), d_sorted.data(), nat);  /* Phase 3.3 */
            /* Empty cells keep count 0, so their cell_first is unused. */
            GPU_LAUNCH(k_cell_first_from_runs, g_at, BLOCK, d_sorted.data(),
                       d_lin.data(), nat, d_cell_first.data());
        } else {
            device_exclusive_scan(d_cell_count.data(), d_cell_first.data(), nc);
            DBuf<int> d_cursor(nc);
            GPU_CHECK(gpuMemcpy(d_cursor.data(), d_cell_first.data(),
                                nc * sizeof(int), gpuMemcpyDeviceToDevice));
            GPU_LAUNCH(k_scatter, g_at, BLOCK, d_lin.data(), nat,
                       d_cursor.data(), d_sorted.data());
        }
    } else {
        /* Hashed compact: power-of-two capacity, load factor <= 0.5. */
        std::int64_t cap = 1;
        while (cap < 2 * static_cast<std::int64_t>(nat) + 1) cap <<= 1;
        hmask = cap - 1;
        d_hkey.resize(cap);
        d_hfirst.resize(cap);
        d_hcount.resize(cap);
        GPU_CHECK(gpuMemset(d_hkey.data(), 0xFF, cap * sizeof(std::int64_t)));
        GPU_CHECK(gpuMemset(d_hcount.data(), 0, cap * sizeof(int)));
        DBuf<std::int64_t> d_key(nat);
        GPU_LAUNCH(k_fold_key64, g_at, BLOCK, d_raw.data(), pbc[0], pbc[1], pbc[2],
                   n1, n2, n3, nat, d_key.data());
        GPU_LAUNCH(k_hash_insert, g_at, BLOCK, d_key.data(), nat, hmask,
                   d_hkey.data(), d_hcount.data());
        device_exclusive_scan(d_hcount.data(), d_hfirst.data(),
                              static_cast<index_t>(cap));
        DBuf<int> d_cursor(cap);
        GPU_CHECK(gpuMemcpy(d_cursor.data(), d_hfirst.data(),
                            cap * sizeof(int), gpuMemcpyDeviceToDevice));
        GPU_LAUNCH(k_hash_scatter, g_at, BLOCK, d_key.data(), nat, hmask,
                   d_hkey.data(), d_cursor.data(), d_sorted.data());
    }

    /* 3. gather per-atom data into cell-sorted order. */
    DBuf<int> d_raw_s(3 * nat), d_rel_s(3 * nat);
    DBuf<real_t> d_pos_s(3 * nat);
    DBuf<real_t> d_per_atom_s(per_atom_cutoff ? nat : 0);
    DBuf<int> d_types_s(types ? nat : 0);
    DBuf<real_t> d_per_atom(per_atom_cutoff ? nat : 0);
    DBuf<int> d_types(types ? nat : 0);
    DBuf<real_t> d_pt_sq(per_type_cutoff_sq ? ncutoffs * ncutoffs : 0);
    if (per_atom_cutoff)
        GPU_CHECK(gpuMemcpy(d_per_atom.data(), per_atom_cutoff,
                            nat * sizeof(real_t), gpuMemcpyHostToDevice));
    if (types)
        GPU_CHECK(gpuMemcpy(d_types.data(), types, nat * sizeof(int),
                            gpuMemcpyHostToDevice));
    if (per_type_cutoff_sq)
        GPU_CHECK(gpuMemcpy(d_pt_sq.data(), per_type_cutoff_sq,
                            ncutoffs * ncutoffs * sizeof(real_t),
                            gpuMemcpyHostToDevice));
    GPU_LAUNCH(k_gather, g_at, BLOCK, d_sorted.data(), d_raw.data(), d_r,
               pbc[0], pbc[1], pbc[2], n1, n2, n3, bin1[0], bin1[1], bin1[2],
               bin2[0], bin2[1], bin2[2], bin3[0], bin3[1], bin3[2],
               per_atom_cutoff ? d_per_atom.data() : nullptr,
               types ? d_types.data() : nullptr, nat, d_raw_s.data(),
               d_rel_s.data(), d_pos_s.data(),
               per_atom_cutoff ? d_per_atom_s.data() : nullptr,
               types ? d_types_s.data() : nullptr);

    /* Assemble the device context. */
    DevCtx ctx;
    ctx.n1 = n1; ctx.n2 = n2; ctx.n3 = n3;
    ctx.nx = static_cast<int>(std::ceil(cutoff * n1 / len1));
    ctx.ny = static_cast<int>(std::ceil(cutoff * n2 / len2));
    ctx.nz = static_cast<int>(std::ceil(cutoff * n3 / len3));
    ctx.pbc0 = pbc[0]; ctx.pbc1 = pbc[1]; ctx.pbc2 = pbc[2];
    for (int k = 0; k < 3; k++) {
        ctx.bin1[k] = bin1[k]; ctx.bin2[k] = bin2[k]; ctx.bin3[k] = bin3[k];
    }
    ctx.cutoff_sq = cutoff * cutoff;
    ctx.per_type_cutoff_sq = per_type_cutoff_sq ? d_pt_sq.data() : nullptr;
    ctx.ncutoffs = ncutoffs;
    ctx.rel_pos = d_pos_s.data();
    ctx.raw_cell = d_raw_s.data();
    ctx.rel_cell = d_rel_s.data();
    ctx.sorted_atom = d_sorted.data();
    ctx.per_atom = per_atom_cutoff ? d_per_atom_s.data() : nullptr;
    ctx.types = types ? d_types_s.data() : nullptr;

    /* 4. two-pass count/fill, dispatched on the chosen cell-lookup policy. */
    if (!sparse) {
        DenseQueryDev q{n1, n2, d_cell_first.data(), d_cell_count.data()};
        return count_and_fill(ctx, q, nat, quantities, want_pairs, dev);
    }
    SparseQueryDev q{n1,  n2,           hmask, d_hkey.data(),
                     d_hfirst.data(), d_hcount.data()};
    return count_and_fill(ctx, q, nat, quantities, want_pairs, dev);
}

}  // namespace

error_t neighbour_list_gpu_device(int quantities, const real_t cell_origin[3],
                                  const real_t cell[9], const real_t inv_cell[9],
                                  const bool pbc[3], index_t nat, const real_t *r,
                                  bool r_is_device, real_t cutoff,
                                  const real_t *per_atom_cutoff,
                                  const real_t *per_type_cutoff_sq,
                                  index_t ncutoffs, const index_t *types,
                                  CellOrder order, int device_id,
                                  NeighbourListDevice &out) {
    return build_device(quantities, cell_origin, cell, inv_cell, pbc, nat, r,
                        r_is_device, cutoff, per_atom_cutoff, per_type_cutoff_sq,
                        ncutoffs, types, order, device_id, /*want_pairs=*/true,
                        out);
}

error_t neighbour_count_gpu_device(const real_t cell_origin[3],
                                   const real_t cell[9], const real_t inv_cell[9],
                                   const bool pbc[3], index_t nat, const real_t *r,
                                   bool r_is_device, real_t cutoff,
                                   const real_t *per_atom_cutoff,
                                   const real_t *per_type_cutoff_sq,
                                   index_t ncutoffs, const index_t *types,
                                   int device_id, NeighbourListDevice &out) {
    /* Phase 3.4: per-atom neighbour counts without materialising the pairs. */
    return build_device(0, cell_origin, cell, inv_cell, pbc, nat, r, r_is_device,
                        cutoff, per_atom_cutoff, per_type_cutoff_sq, ncutoffs,
                        types, CellOrder::Linear, device_id, /*want_pairs=*/false,
                        out);
}

error_t neighbour_list_gpu(int quantities, const real_t cell_origin[3],
                           const real_t cell[9], const real_t inv_cell[9],
                           const bool pbc[3], index_t nat, const real_t *r,
                           real_t cutoff, const real_t *per_atom_cutoff,
                           const real_t *per_type_cutoff_sq, index_t ncutoffs,
                           const index_t *types, NeighbourList &out,
                           CellOrder order) {
    out.first.clear();
    out.secnd.clear();
    out.distvec.clear();
    out.absdist.clear();
    out.shift.clear();
    out.npairs = 0;

    NeighbourListDevice dev;
    error_t e = build_device(quantities, cell_origin, cell, inv_cell, pbc, nat, r,
                             /*r_is_device=*/false, cutoff, per_atom_cutoff,
                             per_type_cutoff_sq, ncutoffs, types, order,
                             /*device_id=*/-1, /*want_pairs=*/true, dev);
    if (e != NL_SUCCESS) return e;
    out.npairs = dev.npairs;
    if (dev.npairs == 0) return NL_SUCCESS;

    /* D2H the requested quantities (presence inferred from buffer size). */
    auto d2h_i = [](std::vector<index_t> &h, const Array<index_t, DeviceSpace> &d) {
        if (d.size() == 0) return;
        h.resize(d.size());
        GPU_CHECK(gpuMemcpy(h.data(), d.data(), d.size() * sizeof(index_t),
                            gpuMemcpyDeviceToHost));
    };
    auto d2h_r = [](std::vector<real_t> &h, const Array<real_t, DeviceSpace> &d) {
        if (d.size() == 0) return;
        h.resize(d.size());
        GPU_CHECK(gpuMemcpy(h.data(), d.data(), d.size() * sizeof(real_t),
                            gpuMemcpyDeviceToHost));
    };
    d2h_i(out.first, dev.first);
    d2h_i(out.secnd, dev.secnd);
    d2h_r(out.distvec, dev.distvec);
    d2h_r(out.absdist, dev.absdist);
    d2h_i(out.shift, dev.shift);
    return NL_SUCCESS;
}

}  // namespace matscipy
