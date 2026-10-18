/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 *
 * XLA FFI handlers for the fixed-capacity neighbour matrix, so a JAX program can
 * build its neighbour list inside `jit` (e.g. under `lax.cond` when atoms have
 * moved half the skin) without leaving the device.
 *
 *   inputs   positions [n, 3], cell [3, 3] (rows: lattice vectors), float32 or
 *            float64 (the same); types [n] int32 and cutoffs [m, m] (positions'
 *            dtype) for per-type cutoffs, or both empty ([0], [0, 0])
 *   attrs    cutoff (f64: the global cutoff, or the largest per-type one),
 *            pbc (i64 bit mask: bit k = periodic along k)
 *   outputs  idx [n, K] i32, shift [n, K, 3] i32, count [n] i32, overflow [] i32
 *
 * K is the output's second dimension. Slots past min(count, K) are unspecified;
 * overflow = 1 when some count > K (rows clipped). D = r[j] - r[i] + S @ cell is
 * left to the caller, from the positions it differentiates through. The search
 * runs in float64 whatever the input dtype.
 *
 * This file: the CPU handler, and the helpers shared with the CUDA handlers
 * (xla_neighbours_cuda.cu). Errors from the core (non-finite positions,
 * degenerate cell, out of memory) become ffi::Error, which JAX raises as
 * XlaRuntimeError; the core's error state is per thread, so concurrent calls
 * from XLA's threads do not mix messages.
 */
#include "xla_neighbours.hh"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "error.hh"
#include "neighbour_list.hh"

namespace xla_neighbours {

void inverse_transpose3(const double *a, double *inv) {
    /* inv(a^T) as the core takes inv_cell (the Python layer passes
       np.linalg.inv(cell.T)): the transpose of inv(a) */
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                       a[2] * (a[3] * a[7] - a[4] * a[6]);
    inv[0] = (a[4] * a[8] - a[5] * a[7]) / det;
    inv[1] = (a[2] * a[7] - a[1] * a[8]) / det;
    inv[2] = (a[1] * a[5] - a[2] * a[4]) / det;
    inv[3] = (a[5] * a[6] - a[3] * a[8]) / det;
    inv[4] = (a[0] * a[8] - a[2] * a[6]) / det;
    inv[5] = (a[2] * a[3] - a[0] * a[5]) / det;
    inv[6] = (a[3] * a[7] - a[4] * a[6]) / det;
    inv[7] = (a[1] * a[6] - a[0] * a[7]) / det;
    inv[8] = (a[0] * a[4] - a[1] * a[3]) / det;
    for (int r = 0; r < 3; r++)
        for (int c = r + 1; c < 3; c++) std::swap(inv[3 * r + c], inv[3 * c + r]);
}

ffi::Error core_error() {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string("matscipy-neighbours: ") + matscipy::error_string);
}

ffi::Error check_inputs(const ffi::AnyBuffer &pos, const ffi::AnyBuffer &cell,
                        const ffi::Buffer<ffi::S32> &types, const ffi::AnyBuffer &cutoffs,
                        int64_t n, int64_t K, int64_t &ncutoffs) {
    auto pd = pos.dimensions(), cd = cell.dimensions(), td = types.dimensions(),
         md = cutoffs.dimensions();
    if (pd.size() != 2 || pd[0] != n || pd[1] != 3 || cd.size() != 2 || cd[0] != 3 || cd[1] != 3)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "matscipy-neighbours: positions must be [n, 3], cell [3, 3]");
    const auto dt = pos.element_type();
    if ((dt != ffi::DataType::F64 && dt != ffi::DataType::F32) || cell.element_type() != dt)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "matscipy-neighbours: positions and cell must both be float32 or float64");
    if (K < 1)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "matscipy-neighbours: max_neighbours must be positive");
    ncutoffs = md.size() == 2 ? md[0] : 0;
    const bool none = td.size() == 1 && td[0] == 0 && ncutoffs == 0;
    const bool some = td.size() == 1 && td[0] == n && md.size() == 2 && md[0] == md[1] &&
                      ncutoffs > 0 && cutoffs.element_type() == dt;
    if (!none && !some)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "matscipy-neighbours: per-type cutoffs need types [n] int32 and a square "
                          "cutoff matrix of the positions' dtype (or both empty)");
    return ffi::Error::Success();
}

void to_double(const ffi::AnyBuffer &b, std::vector<double> &out) {
    const std::size_t n = b.element_count();
    out.resize(n);
    if (b.element_type() == ffi::DataType::F64) {
        const double *p = b.typed_data<double>();
        out.assign(p, p + n);
    } else {
        const float *p = b.typed_data<float>();
        for (std::size_t a = 0; a < n; a++) out[a] = p[a];
    }
}

}  // namespace xla_neighbours

using namespace xla_neighbours;
using matscipy::index_t;

static ffi::Error NeighbourMatrixCpu(ffi::AnyBuffer pos, ffi::AnyBuffer cell,
                                     ffi::Buffer<ffi::S32> types, ffi::AnyBuffer cutoffs,
                                     double cutoff, int64_t pbc_mask,
                                     ffi::ResultBuffer<ffi::S32> idx,
                                     ffi::ResultBuffer<ffi::S32> shift,
                                     ffi::ResultBuffer<ffi::S32> count,
                                     ffi::ResultBuffer<ffi::S32> overflow) {
    const index_t n = idx->dimensions()[0], K = idx->dimensions()[1];
    int64_t m = 0;
    if (auto e = check_inputs(pos, cell, types, cutoffs, n, K, m); e.failure()) return e;
    std::vector<double> r, c, pt;
    to_double(pos, r);
    to_double(cell, c);
    double inv[9];
    inverse_transpose3(c.data(), inv);
    std::vector<index_t> ty;
    if (m > 0) {
        to_double(cutoffs, pt);
        for (double &v : pt) v *= v;                /* the core takes squared cutoffs */
        ty.assign(types.typed_data(), types.typed_data() + n);
    }
    const double origin[3] = {0, 0, 0};
    const bool pbc[3] = {bool(pbc_mask & 1), bool(pbc_mask & 2), bool(pbc_mask & 4)};
    matscipy::NeighbourMatrix out;
    auto e = matscipy::neighbour_matrix(origin, c.data(), inv, pbc, n, r.data(), cutoff, nullptr,
                                        m > 0 ? pt.data() : nullptr, m, m > 0 ? ty.data() : nullptr, K,
                                        out, matscipy::CellOrder::Linear, matscipy::QUANTITY_SHIFT);
    if (e != matscipy::NL_SUCCESS) return core_error();
    int32_t *pi = idx->typed_data(), *ps = shift->typed_data(), *pc = count->typed_data();
    for (index_t a = 0; a < n * K; a++) pi[a] = int32_t(out.idx[a]);
    for (index_t a = 0; a < 3 * n * K; a++) ps[a] = int32_t(out.shift[a]);
    for (index_t a = 0; a < n; a++) pc[a] = int32_t(out.count[a]);
    overflow->typed_data()[0] = out.overflow ? 1 : 0;
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(MatscipyNeighbourMatrixCpu, NeighbourMatrixCpu,
                              ffi::Ffi::Bind()
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
