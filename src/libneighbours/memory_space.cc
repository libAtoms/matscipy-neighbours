/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Host backend for the memory-space abstraction. Device backends (CUDA/HIP)
 * live in their own translation units, compiled only when enabled.
 */

#include <cstdlib>

#include "memory_space.hh"

namespace matscipy {
namespace detail {

void *alloc_host(std::size_t bytes) { return std::malloc(bytes); }

void free_host(void *ptr) { std::free(ptr); }

}  // namespace detail
}  // namespace matscipy
