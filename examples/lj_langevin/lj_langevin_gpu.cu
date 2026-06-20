/*
 * Lennard-Jones liquid droplet with a Langevin thermostat — GPU (CUDA).
 *
 * Mirrors the CPU example: the neighbour list provides the device-resident `ij`
 * connectivity, then a fused kernel recomputes the distance vectors and
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

#include "lj_common.hh"
#include "neighbour_list_gpu.hh"

using namespace matscipy;
using clock_type = std::chrono::steady_clock;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(e));                  \
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

/* Fused LJ force pass: recompute the distance from positions, accumulate onto
   atom i (each directed pair contributes to its own i). */
__global__ void k_lj_forces(const index_t *first, const index_t *secnd,
                            index_t npairs, const double *pos, double *f,
                            double rc2) {
    index_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npairs) return;
    const index_t i = first[p], j = secnd[p];
    const double dx = pos[3 * j] - pos[3 * i], dy = pos[3 * j + 1] - pos[3 * i + 1],
                 dz = pos[3 * j + 2] - pos[3 * i + 2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 >= rc2) return;
    const double ir2 = 1.0 / r2, ir6 = ir2 * ir2 * ir2;
    const double coef = -24.0 * ir2 * ir6 * (2.0 * ir6 - 1.0);
    atomicAddD(&f[3 * i], coef * dx);
    atomicAddD(&f[3 * i + 1], coef * dy);
    atomicAddD(&f[3 * i + 2], coef * dz);
}

__global__ void k_langevin(double *pos, double *vel, const double *f, index_t n,
                           lj::Langevin lc, curandState *st) {
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
        vel[q] += (lc.c0 - 1.0) * vel[q] + lc.c1 * lc.dt * fm + gv;
    }
    st[a] = s;
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
    const int ncells = (int)argd(argc, argv, "--ncells", 6);
    const real_t lattice = argd(argc, argv, "--lattice", 1.6);
    const int steps = (int)argd(argc, argv, "--steps", 300);
    const real_t dt = argd(argc, argv, "--dt", 0.005);
    const real_t gamma = argd(argc, argv, "--gamma", 1.0);
    const real_t kT = argd(argc, argv, "--kT", 0.7);
    const real_t cutoff = argd(argc, argv, "--cutoff", 2.5);
    const int write_every = (int)argd(argc, argv, "--write-every", 50);
    const char *outfile = args(argc, argv, "--out", "traj_gpu.xyz");

    std::vector<real_t> pos;
    const index_t n = lj::fcc_droplet(ncells, lattice, pos);
    std::vector<real_t> host_pos(3 * n);

    real_t origin[3], cell[9], inv_cell[9];
    lj::fixed_box(ncells, lattice, cutoff, origin, cell, inv_cell);
    const bool pbc[3] = {false, false, false};
    const lj::Langevin lc = lj::langevin_constants(dt, gamma, kT);
    const real_t rc2 = cutoff * cutoff;
    const int BLK = 256;

    double *d_pos, *d_vel, *d_f;
    curandState *d_st;
    CUDA_CHECK(cudaMalloc(&d_pos, 3 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel, 3 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f, 3 * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_st, n * sizeof(curandState)));
    CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), 3 * n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_vel, 0, 3 * n * sizeof(double)));
    k_init_rng<<<(n + BLK - 1) / BLK, BLK>>>(d_st, n, 12345ULL);

    NeighbourListRequest req;
    req.quantities = QUANTITY_FIRST | QUANTITY_SECOND;
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
        neighbour_list_gpu_device(req, dev);
        npairs = dev.npairs;
        CUDA_CHECK(cudaMemset(d_f, 0, 3 * n * sizeof(double)));
        k_lj_forces<<<((int)npairs + BLK - 1) / BLK, BLK>>>(
            dev.first.data(), dev.secnd.data(), npairs, d_pos, d_f, rc2);
    };

    compute_forces();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("device=gpu  atoms=%d  pairs~%d\n", (int)n, (int)npairs);

    std::ofstream out(outfile);
    const auto t0 = clock_type::now();
    for (int step = 0; step < steps; step++) {
        k_langevin<<<(n + BLK - 1) / BLK, BLK>>>(d_pos, d_vel, d_f, n, lc, d_st);
        compute_forces();
        if (step % write_every == 0) {
            CUDA_CHECK(cudaMemcpy(host_pos.data(), d_pos, 3 * n * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            lj::write_xyz(out, host_pos.data(), n, "step=" + std::to_string(step));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed =
        std::chrono::duration<double>(clock_type::now() - t0).count();
    const double per_step = elapsed / steps;
    std::printf("steps=%d  total=%.3fs  %.3f ms/step  %.1f ns/pair\n", steps,
                elapsed, per_step * 1e3, per_step * 1e9 / npairs);

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_f);
    cudaFree(d_st);
    return 0;
}
