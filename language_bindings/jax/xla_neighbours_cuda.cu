/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 *
 * CUDA XLA FFI handlers (inputs and outputs as xla_neighbours.cc).
 *
 * MatscipyNeighbourMatrixCuda: the regular GPU build. The core runs on the
 * legacy default stream, allocates from its pool, reads flags back and ends
 * with a device synchronisation, and takes the cell and the per-type data on
 * the host; XLA's stream is non-blocking. So: wait for XLA's stream, read the
 * small inputs to the host, build, narrow the core's int64 outputs into XLA's
 * int32 buffers on XLA's stream, and wait before the core's buffers return to
 * its pool. Several host synchronisations per build; not CUDA-graph capturable.
 *
 * MatscipyNeighbourMatrixPlannedCuda (EXPERIMENTAL, neighbour_matrix_planned.hh):
 * the planned build, enqueued on XLA's stream with no host synchronisation,
 * allocation or read-back, and declared command-buffer (CUDA graph)
 * compatible, so under jit a rebuild can sit inside a device-side conditional.
 * Its grid (n1, n2, n3) and lookup come from MatscipyPlanNeighbourMatrix as
 * attributes; its workspace is an extra output XLA allocates (and the wrapper
 * drops); status [2] = (overflow, invalid).
 */
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "xla_neighbours.hh"

#include "error.hh"
#include "neighbour_list_gpu.hh"
#include "neighbour_matrix_batched.hh"
#include "neighbour_matrix_planned.hh"
#include "types.hh"

using namespace xla_neighbours;
using matscipy::index_t;
using matscipy::real_t;

namespace {

__global__ void narrow(const index_t *__restrict__ src, int32_t *__restrict__ dst, index_t n) {
    for (index_t a = blockIdx.x * index_t(blockDim.x) + threadIdx.x; a < n;
         a += index_t(blockDim.x) * gridDim.x)
        dst[a] = int32_t(src[a]);
}

__global__ void widen(const float *__restrict__ src, double *__restrict__ dst, index_t n) {
    for (index_t a = blockIdx.x * index_t(blockDim.x) + threadIdx.x; a < n;
         a += index_t(blockDim.x) * gridDim.x)
        dst[a] = double(src[a]);
}

__global__ void set_scalar(int32_t *dst, int32_t v) { *dst = v; }

int grid_for(index_t n) { return int(std::min<index_t>((n + 255) / 256, 65535)); }

ffi::Error cuda_error(cudaError_t e, const char *what) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("matscipy-neighbours: ") + what + ": " + cudaGetErrorString(e));
}

/* A device float32 / float64 buffer as host doubles (blocking on `stream`). */
cudaError_t to_host_double(const ffi::AnyBuffer &b, std::vector<double> &out, cudaStream_t stream) {
    const std::size_t n = b.element_count();
    out.resize(n);
    if (n == 0) return cudaSuccess;
    cudaError_t e;
    if (b.element_type() == ffi::DataType::F64) {
        e = cudaMemcpyAsync(out.data(), b.untyped_data(), n * sizeof(double), cudaMemcpyDeviceToHost, stream);
    } else {
        std::vector<float> f(n);
        e = cudaMemcpyAsync(f.data(), b.untyped_data(), n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        for (std::size_t a = 0; a < n; a++) out[a] = f[a];
        return e;
    }
    return e == cudaSuccess ? cudaStreamSynchronize(stream) : e;
}

}  // namespace

/* ------------------------------------------------------------------ regular build */
static ffi::Error NeighbourMatrixCuda(cudaStream_t stream, ffi::AnyBuffer pos, ffi::AnyBuffer cell,
                                      ffi::Buffer<ffi::S32> types, ffi::AnyBuffer cutoffs,
                                      double cutoff, int64_t pbc_mask,
                                      ffi::ResultBuffer<ffi::S32> idx,
                                      ffi::ResultBuffer<ffi::S32> shift,
                                      ffi::ResultBuffer<ffi::S32> count,
                                      ffi::ResultBuffer<ffi::S32> overflow) {
    const index_t n = idx->dimensions()[0], K = idx->dimensions()[1];
    int64_t m = 0;
    if (auto e = check_inputs(pos, cell, types, cutoffs, n, K, m); e.failure()) return e;
    cudaError_t ce;
    std::vector<double> c, pt;
    if ((ce = to_host_double(cell, c, stream)) != cudaSuccess)   /* also: positions are ready */
        return cuda_error(ce, "reading the cell");
    std::vector<index_t> ty;
    if (m > 0) {
        if ((ce = to_host_double(cutoffs, pt, stream)) != cudaSuccess) return cuda_error(ce, "reading cutoffs");
        for (double &v : pt) v *= v;
        std::vector<int32_t> t32(n);
        if ((ce = cudaMemcpyAsync(t32.data(), types.typed_data(), n * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, stream)) != cudaSuccess ||
            (ce = cudaStreamSynchronize(stream)) != cudaSuccess)
            return cuda_error(ce, "reading types");
        ty.assign(t32.begin(), t32.end());
    }
    double *r64 = nullptr;                                       /* float32 positions, widened */
    if (pos.element_type() == ffi::DataType::F32) {
        if ((ce = cudaMalloc(&r64, 3 * n * sizeof(double))) != cudaSuccess) return cuda_error(ce, "allocating");
        widen<<<grid_for(3 * n), 256, 0, stream>>>(pos.typed_data<float>(), r64, 3 * n);
        if ((ce = cudaStreamSynchronize(stream)) != cudaSuccess) {
            cudaFree(r64);
            return cuda_error(ce, "widening positions");
        }
    }
    double inv[9];
    inverse_transpose3(c.data(), inv);
    const real_t origin[3] = {0, 0, 0};
    const bool pbc[3] = {bool(pbc_mask & 1), bool(pbc_mask & 2), bool(pbc_mask & 4)};
    int dev = 0;
    cudaGetDevice(&dev);
    matscipy::NeighbourListRequest req;
    req.cell_origin = origin;
    req.cell = c.data();
    req.inv_cell = inv;
    req.pbc = pbc;
    req.nat = n;
    req.positions = r64 ? r64 : pos.typed_data<double>();
    req.positions_on_device = true;
    req.cutoff = cutoff;
    req.per_type_cutoff_sq = m > 0 ? pt.data() : nullptr;
    req.ncutoffs = m;
    req.types = m > 0 ? ty.data() : nullptr;
    req.device_id = dev;
    matscipy::NeighbourMatrixDevice out;
    auto e = matscipy::neighbour_matrix_gpu_device(req, K, out, matscipy::QUANTITY_SHIFT);
    if (r64) cudaFree(r64);                                      /* the build synchronised the device */
    if (e != matscipy::NL_SUCCESS) return core_error();
    narrow<<<grid_for(n * K), 256, 0, stream>>>(out.idx.data(), idx->typed_data(), n * K);
    narrow<<<grid_for(3 * n * K), 256, 0, stream>>>(out.shift.data(), shift->typed_data(), 3 * n * K);
    narrow<<<grid_for(n), 256, 0, stream>>>(out.count.data(), count->typed_data(), n);
    set_scalar<<<1, 1, 0, stream>>>(overflow->typed_data(), out.overflow ? 1 : 0);
    if ((ce = cudaGetLastError()) != cudaSuccess) return cuda_error(ce, "narrowing kernels");
    if ((ce = cudaStreamSynchronize(stream)) != cudaSuccess)    /* before `out` returns to the pool */
        return cuda_error(ce, "narrowing");
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(MatscipyNeighbourMatrixCuda, NeighbourMatrixCuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::AnyBuffer>()          // positions
                                  .Arg<ffi::AnyBuffer>()          // cell
                                  .Arg<ffi::Buffer<ffi::S32>>()   // types (or [0])
                                  .Arg<ffi::AnyBuffer>()          // cutoffs (or [0, 0])
                                  .Attr<double>("cutoff")
                                  .Attr<int64_t>("pbc")
                                  .Ret<ffi::Buffer<ffi::S32>>()   // idx
                                  .Ret<ffi::Buffer<ffi::S32>>()   // shift
                                  .Ret<ffi::Buffer<ffi::S32>>()   // count
                                  .Ret<ffi::Buffer<ffi::S32>>()); // overflow

/* ------------------------------------------------------------------ planned build */
static ffi::Error NeighbourMatrixPlannedCuda(cudaStream_t stream, ffi::AnyBuffer pos,
                                             ffi::AnyBuffer cell, ffi::Buffer<ffi::S32> types,
                                             ffi::AnyBuffer cutoffs, double cutoff, int64_t pbc_mask,
                                             int64_t n1, int64_t n2, int64_t n3, int64_t hashed,
                                             ffi::ResultBuffer<ffi::S32> idx,
                                             ffi::ResultBuffer<ffi::S32> shift,
                                             ffi::ResultBuffer<ffi::S32> count,
                                             ffi::ResultBuffer<ffi::S32> status,
                                             ffi::ResultBuffer<ffi::U8> workspace) {
    const index_t n = idx->dimensions()[0], K = idx->dimensions()[1];
    int64_t m = 0;
    if (auto e = check_inputs(pos, cell, types, cutoffs, n, K, m); e.failure()) return e;
    const bool pbc[3] = {bool(pbc_mask & 1), bool(pbc_mask & 2), bool(pbc_mask & 4)};
    const bool single = pos.element_type() == ffi::DataType::F32;
    matscipy::NeighbourMatrixPlan plan;
    if (matscipy::layout_neighbour_matrix(n, K, n1, n2, n3, pbc, cutoff, m, single, hashed != 0, plan) !=
        matscipy::NL_SUCCESS)
        return core_error();
    if (static_cast<std::size_t>(workspace->element_count()) < plan.workspace_bytes)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "matscipy-neighbours: workspace smaller than the plan's (plan again?)");
    if (matscipy::neighbour_matrix_gpu_planned(plan, pos.untyped_data(), cell.untyped_data(),
                                               m > 0 ? types.typed_data() : nullptr,
                                               m > 0 ? cutoffs.untyped_data() : nullptr,
                                               workspace->untyped_data(), stream, idx->typed_data(),
                                               shift->typed_data(), count->typed_data(),
                                               status->typed_data()) != matscipy::NL_SUCCESS)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("matscipy-neighbours: ") + matscipy::error_string);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(MatscipyNeighbourMatrixPlannedCuda, NeighbourMatrixPlannedCuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::AnyBuffer>()          // positions
                                  .Arg<ffi::AnyBuffer>()          // cell
                                  .Arg<ffi::Buffer<ffi::S32>>()   // types (or [0])
                                  .Arg<ffi::AnyBuffer>()          // cutoffs (or [0, 0])
                                  .Attr<double>("cutoff")
                                  .Attr<int64_t>("pbc")
                                  .Attr<int64_t>("n1")
                                  .Attr<int64_t>("n2")
                                  .Attr<int64_t>("n3")
                                  .Attr<int64_t>("hashed")
                                  .Ret<ffi::Buffer<ffi::S32>>()   // idx
                                  .Ret<ffi::Buffer<ffi::S32>>()   // shift
                                  .Ret<ffi::Buffer<ffi::S32>>()   // count
                                  .Ret<ffi::Buffer<ffi::S32>>()   // status [overflow, invalid]
                                  .Ret<ffi::Buffer<ffi::U8>>(),   // workspace (dropped)
                              {ffi::Traits::kCmdBufferCompatible});

/* ------------------------------------------------------------------ batched build (EXPERIMENTAL) */
/* As MatscipyNeighbourMatrixBatchedCpu, enqueued on XLA's stream (three kernels, no host
   synchronisation): CUDA-graph compatible. A bad n_atoms marks its structure invalid. */
static ffi::Error NeighbourMatrixBatchedCuda(cudaStream_t stream, ffi::AnyBuffer pos, ffi::AnyBuffer cell,
                                             ffi::Buffer<ffi::S32> n_atoms, ffi::Buffer<ffi::S32> pbc,
                                             ffi::Buffer<ffi::S32> types, ffi::AnyBuffer cutoffs,
                                             double cutoff, ffi::ResultBuffer<ffi::S32> idx,
                                             ffi::ResultBuffer<ffi::S32> shift,
                                             ffi::ResultBuffer<ffi::S32> count,
                                             ffi::ResultBuffer<ffi::S32> status) {
    const auto od = idx->dimensions();
    const int64_t B = od[0], N = od[1], K = od[2];
    int64_t m = 0;
    if (auto e = check_batched_inputs(pos, cell, n_atoms, pbc, types, cutoffs, B, N, K, m); e.failure())
        return e;
    if (matscipy::neighbour_matrix_batched_gpu(B, N, n_atoms.typed_data(), pos.untyped_data(),
                                               cell.untyped_data(), pos.element_type() == ffi::DataType::F32,
                                               pbc.typed_data(), cutoff, m > 0 ? types.typed_data() : nullptr,
                                               m > 0 ? cutoffs.untyped_data() : nullptr, m, K, stream,
                                               idx->typed_data(), shift->typed_data(), count->typed_data(),
                                               status->typed_data()) != matscipy::NL_SUCCESS)
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("matscipy-neighbours: ") + matscipy::error_string);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(MatscipyNeighbourMatrixBatchedCuda, NeighbourMatrixBatchedCuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::AnyBuffer>()          // positions [B, N, 3]
                                  .Arg<ffi::AnyBuffer>()          // cell [B, 3, 3]
                                  .Arg<ffi::Buffer<ffi::S32>>()   // n_atoms [B]
                                  .Arg<ffi::Buffer<ffi::S32>>()   // pbc [B]
                                  .Arg<ffi::Buffer<ffi::S32>>()   // types [B, N] (or [0])
                                  .Arg<ffi::AnyBuffer>()          // cutoffs (or [0, 0])
                                  .Attr<double>("cutoff")
                                  .Ret<ffi::Buffer<ffi::S32>>()   // idx [B, N, K]
                                  .Ret<ffi::Buffer<ffi::S32>>()   // shift [B, N, K, 3]
                                  .Ret<ffi::Buffer<ffi::S32>>()   // count [B, N]
                                  .Ret<ffi::Buffer<ffi::S32>>(),  // status [B, 2]
                              {ffi::Traits::kCmdBufferCompatible});

/* Plan on the host (ctypes): out = {n1, n2, n3, hashed, workspace_bytes}. Returns the core's
   error_t (0 on success); the message is MatscipyLastError() (per thread). */
extern "C" int MatscipyPlanNeighbourMatrix(int64_t nat, const double *cell, int64_t pbc_mask,
                                           double cutoff, int64_t K, int64_t ncutoffs, int64_t single,
                                           int64_t *out) {
    const bool pbc[3] = {bool(pbc_mask & 1), bool(pbc_mask & 2), bool(pbc_mask & 4)};
    matscipy::NeighbourMatrixPlan plan;
    const auto e = matscipy::plan_neighbour_matrix(nat, cell, pbc, cutoff, K, ncutoffs, single != 0, plan);
    if (e != matscipy::NL_SUCCESS) return int(e);
    out[0] = plan.n1;
    out[1] = plan.n2;
    out[2] = plan.n3;
    out[3] = plan.hashed ? 1 : 0;
    out[4] = int64_t(plan.workspace_bytes);
    return 0;
}

extern "C" const char *MatscipyLastError() { return matscipy::error_string; }
