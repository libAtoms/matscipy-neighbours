/* ======================================================================
 * matscipy-neigbours - Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * Copyright (2014-2024) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL MATSCIPY_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

#include <stdbool.h>
#include <stddef.h>

#include "bind_py_neighbours.hh"
#include "bind_py_dlpack.hh"

#include "module.h"

/*
 * Method declaration
 */

static PyMethodDef module_methods[] = {
    { "neighbour_list", (PyCFunction) py_neighbour_list, METH_VARARGS,
      "Compute a neighbour list for an atomic configuration." },
    { "neighbour_list_dlpack", (PyCFunction) py_neighbour_list_dlpack,
      METH_VARARGS,
      "Compute a neighbour list, returning each quantity as a DLPack capsule "
      "(zero-copy; CPU host or CUDA device)." },
    { "first_neighbours", (PyCFunction) py_first_neighbours, METH_VARARGS,
      "Compute indices of first neighbours in neighbour list array." },
    { "triplet_list", (PyCFunction) py_triplet_list, METH_VARARGS,
      "Compute a triplet list for a first_neighbour list." },
    { "get_jump_indicies", (PyCFunction) py_get_jump_indicies, METH_VARARGS,
      "Get jump indicies of an ordered list. Does not need list's length \
       as an argument - only the ordered list." },
    { NULL, NULL, 0, NULL }  /* Sentinel */
};

/*
 * Module initialization
 */

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

/*
 * Module declaration
 */

#if PY_MAJOR_VERSION >= 3
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
    #define MOD_DEF(ob, name, methods, doc) \
        static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&moduledef);
#else
    #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
    #define MOD_DEF(ob, name, methods, doc) \
        ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(_matscipy_neighbours)
{
    PyObject* m;

    import_array();

    MOD_DEF(m, "_matscipy_neighbours", module_methods,
            "Neigbour list for particle simulations.");

    /* Whether this build has the GPU (CUDA/HIP) backend compiled in. The macro
       is inherited from the `neighbours` core target's interface definitions. */
#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
    PyModule_AddIntConstant(m, "_has_gpu", 1);
#else
    PyModule_AddIntConstant(m, "_has_gpu", 0);
#endif

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
