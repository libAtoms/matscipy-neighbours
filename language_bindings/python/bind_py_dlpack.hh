/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#ifndef BIND_PY_DLPACK_HH
#define BIND_PY_DLPACK_HH

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Phase 4: compute a neighbour list and return each requested quantity as a
   DLPack "dltensor" PyCapsule (zero-copy). The CPU backend yields host (kDLCPU)
   capsules for numpy.from_dlpack; the GPU backend yields device (kDLCUDA)
   capsules for cupy.from_dlpack, with results left on the device. */
PyObject *py_neighbour_list_dlpack(PyObject *self, PyObject *args);

#ifdef __cplusplus
}
#endif

#endif
