/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#ifndef BIND_PY_NEIGHBOURS_HH
#define BIND_PY_NEIGHBOURS_HH

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NumPy <-> core glue. Each function marshals NumPy arrays into the
   Python-free core in src/libneighbours and wraps the results back up. */
PyObject *py_neighbour_list(PyObject *self, PyObject *args);
PyObject *py_first_neighbours(PyObject *self, PyObject *args);
PyObject *py_triplet_list(PyObject *self, PyObject *args);
PyObject *py_get_jump_indicies(PyObject *self, PyObject *args);

#ifdef __cplusplus
}
#endif

#endif
