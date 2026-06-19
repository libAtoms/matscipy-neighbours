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

/* Per-kernel context, passed by value (small, all device pointers). */
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
    const int *cell_first;     /* ncells */
    const int *cell_count;     /* ncells */
};

/* Visit every neighbour of sorted atom `si`, calling f(sj, dr[3], r2, shift[3]).
   Byte-for-byte the same traversal/pair test as the host visit_neighbours. */
template <typename F>
__device__ inline void d_visit(const DevCtx &c, int si, F &f) {
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

                int cc = cj1 + n1 * (cj2 + n2 * cj3);
                int begin = c.cell_first[cc];
                int end = begin + c.cell_count[cc];
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

__global__ void k_count(DevCtx c, index_t nat, int *cnt) {
    index_t si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= nat) return;
    Counter f;
    d_visit(c, static_cast<int>(si), f);
    cnt[c.sorted_atom[si]] = f.n;
}

__global__ void k_fill(DevCtx c, index_t nat, const int *offset, int *first,
                      int *secnd, real_t *distvec, real_t *absdist, int *shift) {
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
    d_visit(c, static_cast<int>(si), f);
}

/* Device buffer alias (RAII via the Phase-3.1 Array). */
template <typename T>
using DBuf = Array<T, CudaSpace>;

}  // namespace

error_t neighbour_list_gpu(int quantities, const real_t cell_origin[3],
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
    if (nat <= 0) return NL_SUCCESS;

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
    const index_t ncells = static_cast<index_t>(n1) * n2 * n3;
    const int g_at = grid_for(nat);

    /* Upload geometry + positions. */
    DBuf<real_t> d_origin(3), d_inv(9), d_r(3 * nat);
    GPU_CHECK(gpuMemcpy(d_origin.data(), cell_origin, 3 * sizeof(real_t),
                        gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_inv.data(), inv_cell, 9 * sizeof(real_t),
                        gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_r.data(), r, 3 * nat * sizeof(real_t),
                        gpuMemcpyHostToDevice));

    /* 1. raw (unwrapped) cell index per atom. */
    DBuf<int> d_raw(3 * nat);
    GPU_LAUNCH(k_cell_index, g_at, BLOCK, d_origin.data(), d_inv.data(),
               d_r.data(), n1, n2, n3, nat, d_raw.data());

    /* 2. fold -> linear cell + histogram, scan to CSR first index, scatter. */
    DBuf<int> d_lin(nat), d_cell_count(ncells), d_cell_first(ncells);
    GPU_CHECK(gpuMemset(d_cell_count.data(), 0, ncells * sizeof(int)));
    GPU_LAUNCH(k_fold_lin_hist, g_at, BLOCK, d_raw.data(), pbc[0], pbc[1], pbc[2],
               n1, n2, n3, nat, d_lin.data(), d_cell_count.data());
    device_exclusive_scan(d_cell_count.data(), d_cell_first.data(), ncells);

    DBuf<int> d_cursor(ncells), d_sorted(nat);
    GPU_CHECK(gpuMemcpy(d_cursor.data(), d_cell_first.data(),
                        ncells * sizeof(int), gpuMemcpyDeviceToDevice));
    GPU_LAUNCH(k_scatter, g_at, BLOCK, d_lin.data(), nat, d_cursor.data(),
               d_sorted.data());

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
    GPU_LAUNCH(k_gather, g_at, BLOCK, d_sorted.data(), d_raw.data(), d_r.data(),
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
    ctx.cell_first = d_cell_first.data();
    ctx.cell_count = d_cell_count.data();

    /* 4. pass 1: count neighbours per (original) atom; cnt[nat] = 0 sentinel. */
    DBuf<int> d_cnt(nat + 1), d_offset(nat + 1);
    GPU_CHECK(gpuMemset(d_cnt.data() + nat, 0, sizeof(int)));
    GPU_LAUNCH(k_count, g_at, BLOCK, ctx, nat, d_cnt.data());

    /* 5. scan counts -> per-atom write offset; total pairs. */
    index_t npairs = device_exclusive_scan(d_cnt.data(), d_offset.data(), nat + 1);
    out.npairs = npairs;
    if (npairs == 0) return NL_SUCCESS;

    /* 6. allocate exact-size outputs, pass 2: fill. */
    const bool wf = quantities & QUANTITY_FIRST;
    const bool ws = quantities & QUANTITY_SECOND;
    const bool wD = quantities & QUANTITY_DISTVEC;
    const bool wd = quantities & QUANTITY_ABSDIST;
    const bool wS = quantities & QUANTITY_SHIFT;
    DBuf<int> d_first(wf ? npairs : 0), d_secnd(ws ? npairs : 0),
        d_shift(wS ? 3 * npairs : 0);
    DBuf<real_t> d_distvec(wD ? 3 * npairs : 0), d_absdist(wd ? npairs : 0);
    GPU_LAUNCH(k_fill, g_at, BLOCK, ctx, nat, d_offset.data(),
               wf ? d_first.data() : nullptr, ws ? d_secnd.data() : nullptr,
               wD ? d_distvec.data() : nullptr, wd ? d_absdist.data() : nullptr,
               wS ? d_shift.data() : nullptr);
    GPU_CHECK(gpuDeviceSynchronize());

    /* Copy requested outputs back to the host result. */
    if (wf) {
        out.first.resize(npairs);
        GPU_CHECK(gpuMemcpy(out.first.data(), d_first.data(),
                            npairs * sizeof(int), gpuMemcpyDeviceToHost));
    }
    if (ws) {
        out.secnd.resize(npairs);
        GPU_CHECK(gpuMemcpy(out.secnd.data(), d_secnd.data(),
                            npairs * sizeof(int), gpuMemcpyDeviceToHost));
    }
    if (wD) {
        out.distvec.resize(3 * npairs);
        GPU_CHECK(gpuMemcpy(out.distvec.data(), d_distvec.data(),
                            3 * npairs * sizeof(real_t), gpuMemcpyDeviceToHost));
    }
    if (wd) {
        out.absdist.resize(npairs);
        GPU_CHECK(gpuMemcpy(out.absdist.data(), d_absdist.data(),
                            npairs * sizeof(real_t), gpuMemcpyDeviceToHost));
    }
    if (wS) {
        out.shift.resize(3 * npairs);
        GPU_CHECK(gpuMemcpy(out.shift.data(), d_shift.data(),
                            3 * npairs * sizeof(int), gpuMemcpyDeviceToHost));
    }
    return NL_SUCCESS;
}

}  // namespace matscipy
