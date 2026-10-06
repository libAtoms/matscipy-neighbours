/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * GPU neighbour-list kernels: compute cell index, atomic-histogram the cells,
 * exclusive-scan to a CSR, scatter atoms into cell-sorted order, gather per-atom
 * data, then the two passes (count, scan, fill). Every per-atom pass is one
 * thread per atom. The traversal and query are shared with the CPU path; the
 * requested quantities are left in device memory unless copied back to the host.
 */

#include "neighbour_list_gpu.hh"

#include <cmath>
#include <cstdio>

#include "cell_list.hh"          /* shared cell_hash, morton3, Dense/SparseQuery */
#include "device.hh"
#include "device_primitives.hh"
#include "error.hh"
#include "memory_space.hh"
#include "neighbour_visit.hh"     /* shared NeighbourContext + visit_neighbours */
#include "tools.hh"               /* shared bin_wrap/bin_trunc/position_to_cell_index */

namespace matscipy {
namespace {

constexpr int BLOCK = 256;
constexpr index_t kMaxDeviceAtoms = index_t(1) << 29;
inline int grid_for(index_t n) { return static_cast<int>((n + BLOCK - 1) / BLOCK); }

/* The cell-index helpers (bin_wrap/bin_trunc/position_to_cell_index), the hash
   and Morton keys (cell_hash/morton3), the Dense/SparseQuery lookups, and the
   NeighbourContext + visit_neighbours traversal are all host+device functions
   shared with the CPU path (tools.hh / cell_list.hh / neighbour_visit.hh). The
   kernels below add the GPU-specific build (atomic histogram / hash) and the
   two-pass driver. */



/* Per-pair sinks for the shared visit_neighbours traversal. HD so the
   host+device template that calls them is valid for both instantiations. */
struct Counter {
    index_t n = 0;
    MATSCIPY_HD void operator()(index_t, const real_t *, real_t, const index_t *) {
        n++;
    }
};

struct Filler {
    index_t i, w;
    const index_t *sorted_atom;
    index_t *first, *secnd, *shift;
    real_t *distvec, *absdist;
    MATSCIPY_HD void operator()(index_t sj, const real_t *dr, real_t r2,
                                const index_t *sh) {
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

/* Writes each neighbour into the next slot of the atom's row of the dense
   matrix; `w` keeps counting past the capacity K so the true degree is known. */
struct MatrixFiller {
    index_t i, K, w = 0;
    const index_t *sorted_atom;
    index_t *idx, *shift;
    real_t *dist;
    MATSCIPY_HD void operator()(index_t sj, const real_t *dr, real_t,
                                const index_t *sh) {
        if (w < K) {
            const std::size_t slot = static_cast<std::size_t>(i) * K + w;
            idx[slot] = sorted_atom[sj];
            if (dist) {
                dist[3 * slot + 0] = dr[0];
                dist[3 * slot + 1] = dr[1];
                dist[3 * slot + 2] = dr[2];
            }
            if (shift) {
                shift[3 * slot + 0] = sh[0];
                shift[3 * slot + 1] = sh[1];
                shift[3 * slot + 2] = sh[2];
            }
        }
        w++;
    }
};

/* Atomic post-increment of a 64-bit index (histogram / cursor / row counters).
   CUDA and HIP provide 64-bit atomicAdd only for unsigned long long. */
__device__ inline index_t atomic_inc(index_t *p) {
    static_assert(sizeof(index_t) == sizeof(unsigned long long),
                  "atomic_inc assumes a 64-bit index_t");
    return static_cast<index_t>(
        atomicAdd(reinterpret_cast<unsigned long long *>(p), 1ull));
}

/* --- kernels --------------------------------------------------------------- */

/* Also flags any non-finite coordinate (every offending thread stores the same
   value, so a plain store suffices); the host turns the flag into an error. */
__global__ void k_cell_index(const real_t *origin, const real_t *inv,
                             const real_t *r, index_t n1, index_t n2, index_t n3,
                             index_t nat, index_t *raw, int *nonfinite) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    if (!is_finite(r[3 * a]) || !is_finite(r[3 * a + 1]) ||
        !is_finite(r[3 * a + 2])) {
        *nonfinite = 1;
    }
    position_to_cell_index(origin, inv, &r[3 * a], n1, n2, n3, &raw[3 * a],
                  &raw[3 * a + 1], &raw[3 * a + 2]);
}

__global__ void k_fold_lin_hist(const index_t *raw, int pbc0, int pbc1, int pbc2,
                                index_t n1, index_t n2, index_t n3, index_t nat,
                                index_t *lin, index_t *cell_count) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    index_t c1 = pbc0 ? bin_wrap(raw[3 * a], n1) : bin_trunc(raw[3 * a], n1);
    index_t c2 = pbc1 ? bin_wrap(raw[3 * a + 1], n2) : bin_trunc(raw[3 * a + 1], n2);
    index_t c3 = pbc2 ? bin_wrap(raw[3 * a + 2], n3) : bin_trunc(raw[3 * a + 2], n3);
    index_t l = c1 + n1 * (c2 + n2 * c3);
    lin[a] = l;
    atomic_inc(&cell_count[l]);
}

__global__ void k_scatter(const index_t *lin, index_t nat, index_t *cursor,
                          index_t *sorted_atom) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    index_t pos = atomic_inc(&cursor[lin[a]]);
    sorted_atom[pos] = a;
}

__global__ void k_iota(index_t *v, index_t n) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < n) v[a] = a;
}

/* Morton key (Z-curve) of each atom's wrapped cell — the radix-sort key for the
   Morton (coalesced) layout. */
__global__ void k_morton_key(const index_t *raw, int pbc0, int pbc1, int pbc2,
                            index_t n1, index_t n2, index_t n3, index_t nat,
                            std::uint64_t *key) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    index_t c1 = pbc0 ? bin_wrap(raw[3 * a], n1) : bin_trunc(raw[3 * a], n1);
    index_t c2 = pbc1 ? bin_wrap(raw[3 * a + 1], n2) : bin_trunc(raw[3 * a + 1], n2);
    index_t c3 = pbc2 ? bin_wrap(raw[3 * a + 2], n3) : bin_trunc(raw[3 * a + 2], n3);
    key[a] = morton3(c1, c2, c3);
}

/* After a Morton sort, atoms of one cell are a contiguous run in `sorted_atom`
   (the key is unique per cell). Record where each run starts: cell_first[lin]. */
__global__ void k_cell_first_from_runs(const index_t *sorted_atom,
                                       const index_t *lin, index_t nat,
                                       index_t *cell_first) {
    index_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nat) return;
    index_t c = lin[sorted_atom[s]];
    if (s == 0 || lin[sorted_atom[s - 1]] != c) cell_first[c] = s;
}

/* --- sparse (hashed compact) cell list, for huge/sparse grids --------------
   Open addressing keyed by the 64-bit linear cell index; built in parallel with
   atomicCAS inserts. Used when the dense histogram/array path would need
   O(ncells) memory (or overflow a 32-bit index). */

__global__ void k_fold_key64(const index_t *raw, int pbc0, int pbc1, int pbc2,
                            index_t n1, index_t n2, index_t n3, index_t nat,
                            std::int64_t *key) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    index_t c1 = pbc0 ? bin_wrap(raw[3 * a], n1) : bin_trunc(raw[3 * a], n1);
    index_t c2 = pbc1 ? bin_wrap(raw[3 * a + 1], n2) : bin_trunc(raw[3 * a + 1], n2);
    index_t c3 = pbc2 ? bin_wrap(raw[3 * a + 2], n3) : bin_trunc(raw[3 * a + 2], n3);
    key[a] = static_cast<std::int64_t>(c1) +
             static_cast<std::int64_t>(n1) *
                 (static_cast<std::int64_t>(c2) + static_cast<std::int64_t>(n2) * c3);
}

__global__ void k_hash_insert(const std::int64_t *key, index_t nat,
                             std::int64_t mask, std::int64_t *hkey,
                             index_t *hcount) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    std::int64_t k = key[a];
    std::int64_t h = cell_hash(k) & mask;
    auto *slot = reinterpret_cast<unsigned long long *>(hkey);
    const unsigned long long empty = static_cast<unsigned long long>(-1);
    while (true) {
        unsigned long long old =
            atomicCAS(&slot[h], empty, static_cast<unsigned long long>(k));
        if (old == empty || static_cast<std::int64_t>(old) == k) {
            atomic_inc(&hcount[h]);
            return;
        }
        h = (h + 1) & mask;
    }
}

__global__ void k_hash_scatter(const std::int64_t *key, index_t nat,
                              std::int64_t mask, const std::int64_t *hkey,
                              index_t *cursor, index_t *sorted_atom) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nat) return;
    std::int64_t k = key[a];
    std::int64_t h = cell_hash(k) & mask;
    while (hkey[h] != k) h = (h + 1) & mask;  /* key is guaranteed present */
    index_t pos = atomic_inc(&cursor[h]);
    sorted_atom[pos] = a;
}

__global__ void k_gather(const index_t *sorted_atom, const index_t *raw,
                        const real_t *r,
                        int pbc0, int pbc1, int pbc2, index_t n1, index_t n2, index_t n3,
                        real_t b1x, real_t b1y, real_t b1z, real_t b2x,
                        real_t b2y, real_t b2z, real_t b3x, real_t b3y,
                        real_t b3z, const real_t *per_atom, const index_t *types,
                        index_t nat, index_t *raw_s, index_t *rel_s, real_t *pos_s,
                        real_t *per_atom_s, index_t *types_s) {
    index_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nat) return;
    index_t a = sorted_atom[s];
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
    pos_s[3 * s] = r[3 * a] - r1 * b1x - r2 * b2x - r3 * b3x;
    pos_s[3 * s + 1] = r[3 * a + 1] - r1 * b1y - r2 * b2y - r3 * b3y;
    pos_s[3 * s + 2] = r[3 * a + 2] - r1 * b1z - r2 * b2z - r3 * b3z;
    if (per_atom_s) per_atom_s[s] = per_atom[a];
    if (types_s) types_s[s] = types[a];
}

template <typename Query>
__global__ void k_count(NeighbourContext c, Query q, index_t nat, index_t *cnt) {
    index_t si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= nat) return;
    Counter f;
    visit_neighbours(c, q, si, f);
    cnt[c.sorted_atom[si]] = f.n;
}

template <typename Query>
__global__ void k_fill(NeighbourContext c, Query q, index_t nat,
                      const index_t *offset, index_t *first, index_t *secnd,
                      real_t *distvec, real_t *absdist, index_t *shift) {
    index_t si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= nat) return;
    index_t i = c.sorted_atom[si];
    Filler f;
    f.i = i;
    f.w = offset[i];
    f.sorted_atom = c.sorted_atom;
    f.first = first;
    f.secnd = secnd;
    f.distvec = distvec;
    f.absdist = absdist;
    f.shift = shift;
    visit_neighbours(c, q, si, f);
}

/* Dense matrix in a single pass: one thread per atom fills its own row while
   visiting the neighbours, so there is no counting pass, no intermediate pair
   list and no atomic slot assignment. */
template <typename Query>
__global__ void k_fill_matrix(NeighbourContext c, Query q, index_t nat,
                              index_t K, index_t *idx, real_t *dist,
                              index_t *shift, index_t *count, int *overflow) {
    index_t si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= nat) return;
    MatrixFiller f;
    f.i = c.sorted_atom[si];
    f.K = K;
    f.sorted_atom = c.sorted_atom;
    f.idx = idx;
    f.dist = dist;
    f.shift = shift;
    visit_neighbours(c, q, si, f);
    count[f.i] = f.w;
    if (f.w > K) *overflow = 1;
}

/* Device buffer alias (RAII via Array). */
template <typename T>
using DBuf = Array<T, DeviceSpace>;

/* Two-pass output, parameterised on the cell-lookup policy (dense or sparse).
   Always fills `dev.counts` (per-atom neighbour count, no pair materialisation);
   fills the pair arrays only when `want_pairs`. */
template <typename Query>
static error_t count_and_fill(const NeighbourContext &ctx, const Query &q, index_t nat,
                              int quantities, bool want_pairs,
                              NeighbourListDevice &dev) {
    const int g_at = grid_for(nat);

    /* pass 1: count neighbours per (original) atom; cnt[nat] = 0 sentinel. */
    DBuf<index_t> d_cnt(nat + 1), d_offset(nat + 1);
    GPU_CHECK(gpuMemset(d_cnt.data() + nat, 0, sizeof(index_t)));
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

/* Shared kernel pipeline: validates the request, builds the cell list and the
   traversal context in device memory, then hands the context and the chosen
   cell lookup (dense or sparse) to `run(ctx, query)` for the final pass. The
   context's buffers live until `run` returns. */
template <typename Run>
static error_t with_context(const NeighbourListRequest &req, Run &&run) {
    /* Unpack the request into the local names the body uses. */
    const real_t *cell_origin = req.cell_origin;
    const real_t *cell = req.cell;
    const real_t *inv_cell = req.inv_cell;
    const bool *pbc = req.pbc;
    const index_t nat = req.nat;
    const real_t *r = req.positions;
    const bool r_is_device = req.positions_on_device;
    const real_t cutoff = req.cutoff;
    const real_t *per_atom_cutoff = req.per_atom_cutoff;
    const real_t *per_type_cutoff_sq = req.per_type_cutoff_sq;
    const index_t ncutoffs = req.ncutoffs;
    const index_t *types = req.types;
    const CellOrder order = req.order;
    const int device_id = req.device_id;

    error_t status = validate_neighbour_args(nat, cutoff, per_atom_cutoff,
                                             per_type_cutoff_sq, ncutoffs, types);
    if (status != NL_SUCCESS) return status;
    if (nat == 0) return NL_SUCCESS;
    /* The device primitives (CUB/hipCUB) take 32-bit item counts, and the sparse
       hash table has capacity ~2*nat; keep both comfortably below 2^31. */
    if (nat >= kMaxDeviceAtoms)
        return set_error("GPU backend supports at most 2^29 atoms per call.");
    DeviceGuard guard(device_id);  /* run on the input's device; restore on exit */

    /* Grid definition (resolution, bins, box widths) via the shared helper. The
       binning vectors of `grid` stay empty; the GPU builds its cell list in
       device memory below. */
    CellGrid grid;
    if (!cell_grid_geometry(cell_origin, cell, inv_cell, pbc, cutoff, grid))
        return set_error("Zero cell volume.");
    const index_t n1 = grid.n1, n2 = grid.n2, n3 = grid.n3;
    const real_t *bin1 = grid.bin1, *bin2 = grid.bin2, *bin3 = grid.bin3;
    const real_t len1 = grid.len[0], len2 = grid.len[1], len3 = grid.len[2];

    /* Cell count in 64-bit: a huge/sparse grid can exceed a 32-bit index, which
       is exactly when the hashed compact backend is used instead of dense
       arrays. */
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

    /* 1. raw (unwrapped) cell index per atom, rejecting non-finite positions
          (the CPU path does the same; device input can only be checked here). */
    DBuf<index_t> d_raw(3 * nat);
    DBuf<int> d_nonfinite(1);
    GPU_CHECK(gpuMemset(d_nonfinite.data(), 0, sizeof(int)));
    GPU_LAUNCH(k_cell_index, g_at, BLOCK, d_origin.data(), d_inv.data(), d_r,
               n1, n2, n3, nat, d_raw.data(), d_nonfinite.data());
    int nonfinite = 0;
    GPU_CHECK(gpuMemcpy(&nonfinite, d_nonfinite.data(), sizeof(int),
                        gpuMemcpyDeviceToHost));
    if (nonfinite) return set_invalid_argument("Positions must be finite.");

    /* 2. build the CSR cell list -> d_sorted plus a dense or sparse lookup.
          The dense backends (Linear scan+scatter / Morton radix-sort+runs) need
          O(ncells) arrays; for a huge/sparse grid we build a hashed compact
          table instead. */
    DBuf<index_t> d_sorted(nat);
    /* Dense backend buffers (empty for the sparse path). */
    DBuf<index_t> d_cell_first, d_cell_count;
    /* Sparse backend buffers (empty for the dense path). */
    DBuf<std::int64_t> d_hkey;
    DBuf<index_t> d_hfirst, d_hcount;
    std::int64_t hmask = 0;

    if (!sparse) {
        const index_t nc = static_cast<index_t>(ncells);
        DBuf<index_t> d_lin(nat);
        d_cell_count.resize(nc);
        d_cell_first.resize(nc);
        GPU_CHECK(gpuMemset(d_cell_count.data(), 0, nc * sizeof(index_t)));
        GPU_LAUNCH(k_fold_lin_hist, g_at, BLOCK, d_raw.data(), pbc[0], pbc[1],
                   pbc[2], n1, n2, n3, nat, d_lin.data(), d_cell_count.data());
        if (order == CellOrder::Morton) {
            DBuf<std::uint64_t> d_key(nat);
            GPU_LAUNCH(k_morton_key, g_at, BLOCK, d_raw.data(), pbc[0], pbc[1],
                       pbc[2], n1, n2, n3, nat, d_key.data());
            GPU_LAUNCH(k_iota, g_at, BLOCK, d_sorted.data(), nat);
            device_sort_pairs(d_key.data(), d_sorted.data(), nat);
            /* Empty cells keep count 0, so their cell_first is unused. */
            GPU_LAUNCH(k_cell_first_from_runs, g_at, BLOCK, d_sorted.data(),
                       d_lin.data(), nat, d_cell_first.data());
        } else {
            device_exclusive_scan(d_cell_count.data(), d_cell_first.data(), nc);
            DBuf<index_t> d_cursor(nc);
            GPU_CHECK(gpuMemcpy(d_cursor.data(), d_cell_first.data(),
                                nc * sizeof(index_t), gpuMemcpyDeviceToDevice));
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
        GPU_CHECK(gpuMemset(d_hcount.data(), 0, cap * sizeof(index_t)));
        DBuf<std::int64_t> d_key(nat);
        GPU_LAUNCH(k_fold_key64, g_at, BLOCK, d_raw.data(), pbc[0], pbc[1], pbc[2],
                   n1, n2, n3, nat, d_key.data());
        GPU_LAUNCH(k_hash_insert, g_at, BLOCK, d_key.data(), nat, hmask,
                   d_hkey.data(), d_hcount.data());
        device_exclusive_scan(d_hcount.data(), d_hfirst.data(),
                              static_cast<index_t>(cap));
        DBuf<index_t> d_cursor(cap);
        GPU_CHECK(gpuMemcpy(d_cursor.data(), d_hfirst.data(),
                            cap * sizeof(index_t), gpuMemcpyDeviceToDevice));
        GPU_LAUNCH(k_hash_scatter, g_at, BLOCK, d_key.data(), nat, hmask,
                   d_hkey.data(), d_cursor.data(), d_sorted.data());
    }

    /* 3. gather per-atom data into cell-sorted order. */
    DBuf<index_t> d_raw_s(3 * nat), d_rel_s(3 * nat);
    DBuf<real_t> d_pos_s(3 * nat);
    DBuf<real_t> d_per_atom_s(per_atom_cutoff ? nat : 0);
    DBuf<index_t> d_types_s(types ? nat : 0);
    DBuf<real_t> d_per_atom(per_atom_cutoff ? nat : 0);
    DBuf<index_t> d_types(types ? nat : 0);
    DBuf<real_t> d_pt_sq(per_type_cutoff_sq ? ncutoffs * ncutoffs : 0);
    if (per_atom_cutoff)
        GPU_CHECK(gpuMemcpy(d_per_atom.data(), per_atom_cutoff,
                            nat * sizeof(real_t), gpuMemcpyHostToDevice));
    if (types)
        GPU_CHECK(gpuMemcpy(d_types.data(), types, nat * sizeof(index_t),
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
    NeighbourContext ctx;
    ctx.n1 = n1; ctx.n2 = n2; ctx.n3 = n3;
    ctx.nx = static_cast<index_t>(std::ceil(cutoff * n1 / len1));
    ctx.ny = static_cast<index_t>(std::ceil(cutoff * n2 / len2));
    ctx.nz = static_cast<index_t>(std::ceil(cutoff * n3 / len3));
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

    /* 4. the final pass, dispatched on the chosen cell-lookup policy. */
    if (!sparse) {
        DenseQuery q{n1, n2, d_cell_first.data(), d_cell_count.data()};
        return run(ctx, q);
    }
    SparseQuery q{n1,  n2,           hmask, d_hkey.data(),
                     d_hfirst.data(), d_hcount.data()};
    return run(ctx, q);
}

/* Pair list (or, without `want_pairs`, only the per-atom counts) in `dev`. */
static error_t build_device(const NeighbourListRequest &req, bool want_pairs,
                            NeighbourListDevice &dev) {
    dev.npairs = 0;
    return with_context(req, [&](const NeighbourContext &ctx, const auto &q) {
        return count_and_fill(ctx, q, req.nat, req.quantities, want_pairs, dev);
    });
}

}  // namespace

error_t neighbour_list_gpu_device(const NeighbourListRequest &req,
                                  NeighbourListDevice &out) {
    clear_error();
    return catch_out_of_memory("GPU out of memory building the neighbour list.",
                               [&] { return build_device(req, true, out); });
}

error_t neighbour_count_gpu_device(const NeighbourListRequest &req,
                                   NeighbourListDevice &out) {
    /* Per-atom neighbour counts without materialising the pairs. */
    clear_error();
    return catch_out_of_memory("GPU out of memory counting neighbours.",
                               [&] { return build_device(req, false, out); });
}

static error_t neighbour_matrix_body(const NeighbourListRequest &req, index_t K,
                                     NeighbourMatrixDevice &out, int quantities) {
    const index_t n = req.nat;
    const bool wD = quantities & QUANTITY_DISTVEC;
    const bool wS = quantities & QUANTITY_SHIFT;
    if (K < 0) return set_invalid_argument("max_neighbours must be non-negative.");
    if (n >= kMaxDeviceAtoms)
        return set_error("GPU backend supports at most 2^29 atoms per call.");
    DeviceGuard guard(req.device_id);  /* allocate + fill on the input device */
    out.n = n;
    out.max_neighbours = K;
    out.overflow = false;
    /* No clearing: only the first min(count, K) slots of a row are defined. */
    const std::size_t slots = static_cast<std::size_t>(n) * K;
    out.idx.resize(slots);
    out.dist.resize(wD ? 3 * slots : 0);
    out.shift.resize(wS ? 3 * slots : 0);
    out.count.resize(n);
    if (n <= 0) return NL_SUCCESS;

    DBuf<int> d_overflow(1);
    GPU_CHECK(gpuMemset(d_overflow.data(), 0, sizeof(int)));
    error_t e = with_context(req, [&](const NeighbourContext &ctx, const auto &q) {
        GPU_LAUNCH(k_fill_matrix, grid_for(n), BLOCK, ctx, q, n, K,
                   out.idx.data(), wD ? out.dist.data() : nullptr,
                   wS ? out.shift.data() : nullptr, out.count.data(),
                   d_overflow.data());
        return NL_SUCCESS;
    });
    if (e != NL_SUCCESS) return e;
    int host_overflow = 0;  /* the blocking copy also waits for the fill */
    GPU_CHECK(gpuMemcpy(&host_overflow, d_overflow.data(), sizeof(int),
                        gpuMemcpyDeviceToHost));
    out.overflow = host_overflow != 0;
    return NL_SUCCESS;
}

error_t neighbour_matrix_gpu_device(const NeighbourListRequest &req, index_t K,
                                    NeighbourMatrixDevice &out, int quantities) {
    clear_error();
    out.idx.resize(0);
    out.dist.resize(0);
    out.shift.resize(0);
    out.count.resize(0);
    return catch_out_of_memory(
        "GPU out of memory building the neighbour matrix.",
        [&] { return neighbour_matrix_body(req, K, out, quantities); });
}

static error_t neighbour_list_gpu_body(int quantities, const real_t cell_origin[3],
                                       const real_t cell[9],
                                       const real_t inv_cell[9], const bool pbc[3],
                                       index_t nat, const real_t *r, real_t cutoff,
                                       const real_t *per_atom_cutoff,
                                       const real_t *per_type_cutoff_sq,
                                       index_t ncutoffs, const index_t *types,
                                       NeighbourList &out, CellOrder order) {
    out.first.clear();
    out.secnd.clear();
    out.distvec.clear();
    out.absdist.clear();
    out.shift.clear();
    out.npairs = 0;

    NeighbourListRequest req;
    req.quantities = quantities;
    req.cell_origin = cell_origin;
    req.cell = cell;
    req.inv_cell = inv_cell;
    req.pbc = pbc;
    req.nat = nat;
    req.positions = r;
    req.positions_on_device = false;
    req.cutoff = cutoff;
    req.per_atom_cutoff = per_atom_cutoff;
    req.per_type_cutoff_sq = per_type_cutoff_sq;
    req.ncutoffs = ncutoffs;
    req.types = types;
    req.order = order;
    req.device_id = -1;

    NeighbourListDevice dev;
    error_t e = build_device(req, /*want_pairs=*/true, dev);
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

error_t neighbour_list_gpu(int quantities, const real_t cell_origin[3],
                           const real_t cell[9], const real_t inv_cell[9],
                           const bool pbc[3], index_t nat, const real_t *r,
                           real_t cutoff, const real_t *per_atom_cutoff,
                           const real_t *per_type_cutoff_sq, index_t ncutoffs,
                           const index_t *types, NeighbourList &out,
                           CellOrder order) {
    clear_error();
    /* Host std::vector outputs throw std::bad_alloc too. */
    return catch_out_of_memory("Out of memory building the neighbour list.", [&] {
        return neighbour_list_gpu_body(quantities, cell_origin, cell, inv_cell,
                                       pbc, nat, r, cutoff, per_atom_cutoff,
                                       per_type_cutoff_sq, ncutoffs, types, out,
                                       order);
    });
}

/* --- first_neighbours on the device --------------------------------------- */

namespace {

/* Flags an index outside [0, n) or a decrease. Threads only ever write 1, so
   their race is benign. */
__global__ void k_check_sorted(const index_t *i_n, index_t nn, index_t n,
                               int *bad) {
    const index_t p = blockIdx.x * static_cast<index_t>(blockDim.x) + threadIdx.x;
    if (p >= nn) return;
    const index_t a = i_n[p];
    if (a < 0 || a >= n || (p > 0 && a < i_n[p - 1])) *bad = 1;
}

/* The rows outside (i_n[0], i_n[nn-1]]: -1 before the first pair (matscipy's
   convention), 0 for the first row, nn after the last pair. */
__global__ void k_seed_ends(const index_t *i_n, index_t nn, index_t n,
                            index_t *seed) {
    const index_t k = blockIdx.x * static_cast<index_t>(blockDim.x) + threadIdx.x;
    if (k > n) return;
    const index_t first = i_n[0], last = i_n[nn - 1];
    if (k < first) seed[k] = -1;
    else if (k == first) seed[k] = 0;
    else if (k > last) seed[k] = nn;
}

/* Every row in (i_n[p-1], i_n[p]] starts at pair p: the thread of a pair where
   the index jumps writes the rows the jump covers (empty rows start where the
   next row does). */
__global__ void k_seed_jumps(const index_t *i_n, index_t nn, index_t *seed) {
    const index_t p = blockIdx.x * static_cast<index_t>(blockDim.x) + threadIdx.x;
    if (p < 1 || p >= nn) return;
    for (index_t l = i_n[p - 1] + 1; l <= i_n[p]; l++) seed[l] = p;
}

}  // namespace

error_t first_neighbours_gpu_device(index_t n, index_t nn, const index_t *i_n,
                                    Array<index_t, DeviceSpace> &seed,
                                    int device_id) {
    clear_error();
    if (n < 0) {
        return set_invalid_argument(
            "first_neighbours: number of atoms must be non-negative.");
    }
    if (nn < 0 || (nn > 0 && !i_n)) {
        return set_invalid_argument(
            "first_neighbours: invalid neighbour index array.");
    }
    return catch_out_of_memory("GPU out of memory in first_neighbours.", [&] {
        DeviceGuard guard(device_id);
        /* Validate before writing anything; the fill indexes seed by i_n. */
        if (nn > 0) {
            DBuf<int> d_bad(1);
            GPU_CHECK(gpuMemset(d_bad.data(), 0, sizeof(int)));
            GPU_LAUNCH(k_check_sorted, grid_for(nn), BLOCK, i_n, nn, n,
                       d_bad.data());
            int bad = 0;
            GPU_CHECK(gpuMemcpy(&bad, d_bad.data(), sizeof(int),
                                gpuMemcpyDeviceToHost));
            if (bad) {
                return set_invalid_argument(
                    "first_neighbours: index array must be sorted, with every "
                    "index in [0, n).");
            }
        }
        seed.resize(static_cast<std::size_t>(n) + 1);
        if (nn == 0) {  /* empty list: every row starts (and ends) at 0 */
            GPU_CHECK(gpuMemset(seed.data(), 0, seed.size() * sizeof(index_t)));
        } else {
            GPU_LAUNCH(k_seed_ends, grid_for(n + 1), BLOCK, i_n, nn, n,
                       seed.data());
            GPU_LAUNCH(k_seed_jumps, grid_for(nn), BLOCK, i_n, nn, seed.data());
        }
        /* The result is handed to consumers on their own streams. */
        GPU_CHECK(gpuDeviceSynchronize());
        return NL_SUCCESS;
    });
}

}  // namespace matscipy
