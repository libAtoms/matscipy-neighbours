/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#ifndef MATSCIPY_TYPES_HH
#define MATSCIPY_TYPES_HH

/* Mark a small leaf function callable from both host and device so the CPU and
   GPU paths share one definition. Expands to nothing for the host compiler and
   to `__host__ __device__` when nvcc/hipcc compiles the translation unit. */
#if defined(__CUDACC__) || defined(__HIPCC__)
#define MATSCIPY_HD __host__ __device__
#else
#define MATSCIPY_HD
#endif

namespace matscipy {

/* Integer and floating-point types used throughout the core. Kept as plain
   aliases so the core never needs to include the NumPy headers. `index_t` must
   match the integer type the Python layer hands across (NumPy NPY_INT). */
using index_t = int;
using real_t = double;

/* Error code returned by core routines. */
using error_t = int;
constexpr error_t NL_SUCCESS = 0;
constexpr error_t NL_ERROR = -1;

/* Bit flags selecting which per-pair quantities a neighbour-list call computes.
   The Python layer maps the "ijdDS" quantity string onto these. */
enum Quantity : int {
    QUANTITY_FIRST = 1 << 0,    /* i: first atom index */
    QUANTITY_SECOND = 1 << 1,   /* j: second atom index */
    QUANTITY_DISTVEC = 1 << 2,  /* D: distance vector */
    QUANTITY_ABSDIST = 1 << 3,  /* d: absolute distance */
    QUANTITY_SHIFT = 1 << 4,    /* S: cell shift */
};

}  // namespace matscipy

#endif
