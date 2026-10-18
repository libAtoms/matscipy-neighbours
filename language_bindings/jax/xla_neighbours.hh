/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 *
 * Helpers shared by the XLA FFI handlers (xla_neighbours.cc, the CPU handler;
 * xla_neighbours_cuda.cu, the CUDA handlers).
 */
#ifndef MATSCIPY_XLA_NEIGHBOURS_HH
#define MATSCIPY_XLA_NEIGHBOURS_HH

#include <cstdint>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace xla_neighbours {

/* inv(cell^T), the inverse cell as the core takes it. */
void inverse_transpose3(const double *cell, double *inv);

/* The core's last error (per thread) as an ffi::Error. */
ffi::Error core_error();

/* Shapes and dtypes of the inputs; ncutoffs = the per-type matrix's side, 0 if none. */
ffi::Error check_inputs(const ffi::AnyBuffer &pos, const ffi::AnyBuffer &cell,
                        const ffi::Buffer<ffi::S32> &types, const ffi::AnyBuffer &cutoffs,
                        int64_t n, int64_t K, int64_t &ncutoffs);

/* A host float32 or float64 buffer as doubles. */
void to_double(const ffi::AnyBuffer &b, std::vector<double> &out);

}  // namespace xla_neighbours

#endif
