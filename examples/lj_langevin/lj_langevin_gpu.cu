/*
 * Lennard-Jones Langevin dynamics (droplet or periodic liquid) — GPU (CUDA).
 *
 * Mirrors the CPU example: the neighbour list provides the device-resident `ij`
 * connectivity (plus the cell shift `S` for the periodic liquid), then a fused
 * kernel recomputes the distance vectors D = r[j] - r[i] + S @ cell and
 * accumulates the LJ force per atom. The Langevin update runs in a second kernel
 * with a per-atom cuRAND stream. Positions stay on the device; only the XYZ
 * frames are copied back.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "error.hh"
#include "lj_common.hh"
#include "neighbour_list_gpu.hh"

using namespace matscipy;
using clock_type = std::chrono::steady_clock;

/* The status variable has a name no caller uses: a plain `e` would shadow a
   caller's `e` inside `call`, e.g. in CUDA_CHECK(cudaMemcpy(&e, ...)). */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t cuda_check_err_ = (call);                              \
        if (cuda_check_err_ != cudaSuccess) {                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(cuda_check_err_));    \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

/* Double-precision atomicAdd. GTX-era (sm_52) cards lack a hardware version. */
__device__ inline double atomicAddD(double *addr, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(addr, val);
#else
    auto *p = reinterpret_cast<unsigned long long *>(addr);
    unsigned long long old = *p, assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void k_init_rng(curandState *st, index_t n, unsigned long long seed) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < n) curand_init(seed, a, 0, &st[a]);
}

/* Threads per block of the force kernel (a multiple of the warp size). */
constexpr int FORCE_BLOCK = 256;

/* Sum over the 32 lanes of a warp by register shuffles (16, 8, 4, 2, 1); the
   total ends up in lane 0. CUDA only: the example is not built for HIP, where
   a wavefront has 64 lanes. */
__device__ inline double warp_sum(double v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

/* Sum over the thread block, valid in thread 0: each warp reduces its values,
   lane 0 of every warp parks the partial sum in shared memory, and the first
   warp reduces the partials. Every thread of the block must call this. */
__device__ inline double block_sum(double v) {
    __shared__ double partial[FORCE_BLOCK / 32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_sum(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < FORCE_BLOCK / 32 ? partial[lane] : 0.0;
        v = warp_sum(v);
    }
    return v;
}

/* Fused LJ force pass: recompute the distance from positions (plus the
   periodic shift when `shift` is non-null) and accumulate onto atom i (each
   directed pair contributes to its own i). The potential energy is reduced
   within the block and added with one atomic per block rather than one per
   pair, which would serialise the whole launch on a single address; so no
   thread returns early, a thread without a pair in range contributes 0. */
__global__ void k_lj_forces(const index_t *first, const index_t *secnd,
                            const index_t *shift, lj::Cell cell,
                            index_t npairs, const double *pos, double *f,
                            double *epot, double rc2) {
    const index_t p = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0;
    if (p < npairs) {
        const index_t i = first[p], j = secnd[p];
        double dx = pos[3 * j] - pos[3 * i], dy = pos[3 * j + 1] - pos[3 * i + 1],
               dz = pos[3 * j + 2] - pos[3 * i + 2];
        if (shift) {
            const double s0 = shift[3 * p], s1 = shift[3 * p + 1],
                         s2 = shift[3 * p + 2];
            dx += s0 * cell.m[0] + s1 * cell.m[3] + s2 * cell.m[6];
            dy += s0 * cell.m[1] + s1 * cell.m[4] + s2 * cell.m[7];
            dz += s0 * cell.m[2] + s1 * cell.m[5] + s2 * cell.m[8];
        }
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < rc2) {
            const double ir2 = 1.0 / r2, ir6 = ir2 * ir2 * ir2;
            const double coef = -24.0 * ir2 * ir6 * (2.0 * ir6 - 1.0);
            atomicAddD(&f[3 * i], coef * dx);
            atomicAddD(&f[3 * i + 1], coef * dy);
            atomicAddD(&f[3 * i + 2], coef * dz);
            e = 2.0 * ir6 * (ir6 - 1.0);  /* 4 eps (...) / 2: directed pairs */
        }
    }
    e = block_sum(e);
    if (threadIdx.x == 0) atomicAddD(epot, e);
}

/* First half of an Allen-Tildesley Langevin step: move the positions; friction,
   noise and the half kick with the *old* forces on the velocities. */
__global__ void k_langevin_drift(double *pos, double *vel, const double *f,
                                 index_t n, lj::Langevin lc, curandState *st) {
    index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n) return;
    curandState s = st[a];
    const double kk = sqrt(1.0 - lc.crv * lc.crv);
    for (int d = 0; d < 3; d++) {
        const int q = 3 * a + d;
        const double g1 = curand_normal_double(&s), g2 = curand_normal_double(&s);
        const double gr = lc.sr * g1;
        const double gv = lc.sv * (lc.crv * g1 + kk * g2);
        const double fm = f[q] / lc.mass;
        pos[q] += lc.c1 * lc.dt * vel[q] + lc.c2 * lc.dt * lc.dt * fm + gr;
        vel[q] = lc.c0 * vel[q] + (lc.c1 - lc.c2) * lc.dt * fm + gv;
    }
    st[a] = s;
}

/* Second half: the half kick with the *new* forces. */
__global__ void k_langevin_kick(double *vel, const double *f, index_t n,
                                lj::Langevin lc) {
    index_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= 3 * n) return;
    vel[q] += lc.c2 * lc.dt * f[q] / lc.mass;
}

static double argd(int argc, char **argv, const char *key, double def) {
    for (int i = 1; i + 1 < argc; i++)
        if (std::strcmp(argv[i], key) == 0) return std::atof(argv[i + 1]);
    return def;
}
static const char *args(int argc, char **argv, const char *key, const char *def) {
    for (int i = 1; i + 1 < argc; i++)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}

int main(int argc, char **argv) {
    const std::string system = args(argc, argv, "--system", "droplet");
    const int ncells = (int)argd(argc, argv, "--ncells", 6);
    const index_t atoms = (index_t)argd(argc, argv, "--atoms", 0);
    const real_t lattice = argd(argc, argv, "--lattice", 1.6);
    const real_t density = argd(argc, argv, "--density", 0.8442);
    const int steps = (int)argd(argc, argv, "--steps", 300);
    const real_t dt = argd(argc, argv, "--dt", 0.005);
    const real_t gamma = argd(argc, argv, "--gamma", 1.0);
    const real_t kT = argd(argc, argv, "--kT", 0.7);
    const real_t cutoff = argd(argc, argv, "--cutoff", 2.5);
    const int write_every = (int)argd(argc, argv, "--write-every", 50);
    const char *outfile = args(argc, argv, "--out", "traj_gpu.xyz");

    std::vector<real_t> pos;
    real_t origin[3], cell[9], inv_cell[9];
    bool periodic;
    const index_t n = lj::make_system(system, atoms, ncells, lattice, density,
                                      cutoff, pos, origin, cell, inv_cell,
                                      periodic);
    std::vector<real_t> host_pos(3 * n);

    const bool pbc[3] = {periodic, periodic, periodic};
    lj::Cell cell_val;
    for (int k = 0; k < 9; k++) cell_val.m[k] = cell[k];
    const lj::Langevin lc = lj::langevin_constants(dt, gamma, kT);
    const real_t rc2 = cutoff * cutoff;
    const int BLK = 256;

    double *d_pos, *d_vel, *d_f, *d_e;
    curandState *d_st;
    CUDA_CHECK(cudaMalloc(&d_pos, 3 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel, 3 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f, 3 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_e, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_st, n * sizeof(curandState)));
    CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), 3 * n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vel, 0, 3 * n * sizeof(double)));
    k_init_rng<<<(n + BLK - 1) / BLK, BLK>>>(d_st, n, 12345ULL);

    NeighbourListRequest req;
    req.quantities = QUANTITY_FIRST | QUANTITY_SECOND |
                     (periodic ? QUANTITY_SHIFT : 0);
    req.cell_origin = origin;
    req.cell = cell;
    req.inv_cell = inv_cell;
    req.pbc = pbc;
    req.nat = n;
    req.positions = d_pos;            /* device pointer, used in place */
    req.positions_on_device = true;
    req.cutoff = cutoff;
    req.device_id = -1;               /* current device */

    index_t npairs = 0;
    auto compute_forces = [&]() {
        NeighbourListDevice dev;
        if (neighbour_list_gpu_device(req, dev) != NL_SUCCESS) {
            std::fprintf(stderr, "neighbour list failed: %s\n", error_string);
            std::exit(1);
        }
        npairs = dev.npairs;
        CUDA_CHECK(cudaMemset(d_f, 0, 3 * n * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_e, 0, sizeof(double)));
        /* 64-bit grid arithmetic: large lists exceed 2^31 - FORCE_BLOCK pairs. */
        const auto blocks =
            static_cast<unsigned>((npairs + FORCE_BLOCK - 1) / FORCE_BLOCK);
        k_lj_forces<<<blocks, FORCE_BLOCK>>>(
            dev.first.data(), dev.secnd.data(),
            periodic ? dev.shift.data() : nullptr, cell_val, npairs, d_pos,
            d_f, d_e, rc2);
    };
    auto energy = [&]() {
        double e;
        CUDA_CHECK(cudaMemcpy(&e, d_e, sizeof(double), cudaMemcpyDeviceToHost));
        return e;
    };

    compute_forces();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("device=gpu  system=%s  atoms=%lld  pairs~%lld  E_pot=%.6f\n",
                system.c_str(), (long long)n, (long long)npairs, energy());

    std::ofstream out(outfile);
    const auto t0 = clock_type::now();
    for (int step = 0; step < steps; step++) {
        k_langevin_drift<<<(n + BLK - 1) / BLK, BLK>>>(d_pos, d_vel, d_f, n, lc,
                                                       d_st);
        compute_forces();
        k_langevin_kick<<<(3 * n + BLK - 1) / BLK, BLK>>>(d_vel, d_f, n, lc);
        if (write_every > 0 && step % write_every == 0) {  /* 0: no trajectory */
            CUDA_CHECK(cudaMemcpy(host_pos.data(), d_pos, 3 * n * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            lj::write_xyz(out, host_pos.data(), n,
                          "step=" + std::to_string(step) +
                              " E_pot=" + std::to_string(energy()));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed =
        std::chrono::duration<double>(clock_type::now() - t0).count();
    const double per_step = elapsed / std::max(steps, 1);
    std::printf("steps=%d  total=%.3fs  %.3f ms/step  %.1f ns/pair\n", steps,
                elapsed, per_step * 1e3, per_step * 1e9 / npairs);

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_f);
    cudaFree(d_e);
    cudaFree(d_st);
    return 0;
}
