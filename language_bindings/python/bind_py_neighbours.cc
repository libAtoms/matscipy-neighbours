/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Thin NumPy <-> C++ glue. All algorithmic work lives in the Python-free core
 * in src/libneighbours; this file only marshals arrays in and out.
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL MATSCIPY_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "bind_py_neighbours.hh"

#include "error.hh"
#include "first_neighbours.hh"
#include "neighbour_list.hh"
#include "triplet_list.hh"
#include "types.hh"

using namespace matscipy;

/* ------------------------------------------------------------------ helpers */

/* Wrap a core result buffer in a freshly-allocated NumPy array (copying). */
static PyObject *array_1d_int(const std::vector<index_t> &v) {
    npy_intp dims[1] = {static_cast<npy_intp>(v.size())};
    PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT);
    if (a && !v.empty()) {
        std::memcpy(PyArray_DATA((PyArrayObject *)a), v.data(),
                    v.size() * sizeof(index_t));
    }
    return a;
}

static PyObject *array_1d_double(const std::vector<real_t> &v) {
    npy_intp dims[1] = {static_cast<npy_intp>(v.size())};
    PyObject *a = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (a && !v.empty()) {
        std::memcpy(PyArray_DATA((PyArrayObject *)a), v.data(),
                    v.size() * sizeof(real_t));
    }
    return a;
}

static PyObject *array_2d_int(const std::vector<index_t> &v, npy_intp ncols) {
    npy_intp dims[2] = {static_cast<npy_intp>(v.size()) / ncols, ncols};
    PyObject *a = PyArray_SimpleNew(2, dims, NPY_INT);
    if (a && !v.empty()) {
        std::memcpy(PyArray_DATA((PyArrayObject *)a), v.data(),
                    v.size() * sizeof(index_t));
    }
    return a;
}

static PyObject *array_2d_double(const std::vector<real_t> &v, npy_intp ncols) {
    npy_intp dims[2] = {static_cast<npy_intp>(v.size()) / ncols, ncols};
    PyObject *a = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (a && !v.empty()) {
        std::memcpy(PyArray_DATA((PyArrayObject *)a), v.data(),
                    v.size() * sizeof(real_t));
    }
    return a;
}

/* Raise the core error (if any) as a Python exception. Returns true if raised. */
static bool raise_if_core_error() {
    if (has_error) {
        PyErr_SetString(PyExc_RuntimeError, error_string);
        return true;
    }
    return false;
}

/* ----------------------------------------------------------- neighbour_list */

PyObject *py_neighbour_list(PyObject *self, PyObject *args) {
    PyObject *py_cell_origin, *py_cell, *py_inv_cell, *py_pbc, *py_r;
    PyObject *py_quantities, *py_cutoffs, *py_types = NULL;

    if (!PyArg_ParseTuple(args, "O!OOOOOO|O", &PyUnicode_Type, &py_quantities,
                          &py_cell_origin, &py_cell, &py_inv_cell, &py_pbc,
                          &py_r, &py_cutoffs, &py_types))
        return NULL;

    /* Borrowed-to-owned conversions; all initialised so cleanup is safe. */
    PyObject *a_cell_origin = NULL, *a_cell = NULL, *a_inv_cell = NULL;
    PyObject *a_pbc = NULL, *a_r = NULL, *a_cutoffs = NULL, *a_types = NULL;
    PyObject *py_bquantities = NULL;
    PyObject *py_ret = NULL;

    a_cell_origin =
        PyArray_FROMANY(py_cell_origin, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    a_cell = PyArray_FROMANY(py_cell, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    a_inv_cell =
        PyArray_FROMANY(py_inv_cell, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    a_pbc = PyArray_FROMANY(py_pbc, NPY_BOOL, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    a_r = PyArray_FROMANY(py_r, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!a_cell_origin || !a_cell || !a_inv_cell || !a_pbc || !a_r) goto fail;
    if (py_types) {
        a_types =
            PyArray_FROMANY(py_types, NPY_INT, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
        if (!a_types) goto fail;
    }

    {
        index_t nat = (index_t)PyArray_DIM((PyArrayObject *)a_r, 0);

        if (a_types &&
            (index_t)PyArray_DIM((PyArrayObject *)a_types, 0) != nat) {
            PyErr_SetString(PyExc_TypeError,
                            "Position and type arrays must have identical first "
                            "dimension.");
            goto fail;
        }

        /* Interpret the cutoff argument: scalar, per-atom (1d) or per-type
           (2d) matrix. */
        real_t cutoff = 0.0;
        const real_t *per_atom = NULL;
        const real_t *per_type_sq = NULL;
        index_t ncutoffs = 0;
        std::vector<real_t> per_type_storage;

        if (PyFloat_Check(py_cutoffs)) {
            cutoff = PyFloat_AsDouble(py_cutoffs);
        } else {
            a_cutoffs = PyArray_FROMANY(py_cutoffs, NPY_DOUBLE, 1, 2,
                                        NPY_ARRAY_C_CONTIGUOUS);
            if (!a_cutoffs) goto fail;
            int ndim = PyArray_NDIM((PyArrayObject *)a_cutoffs);
            npy_intp dim0 = PyArray_DIM((PyArrayObject *)a_cutoffs, 0);
            const real_t *cdata =
                (const real_t *)PyArray_DATA((PyArrayObject *)a_cutoffs);

            if (ndim == 1) {
                if ((index_t)dim0 != nat) {
                    PyErr_SetString(PyExc_TypeError,
                                    "One-dimensional cutoff array must have "
                                    "length that corresponds to position "
                                    "array.");
                    goto fail;
                }
                for (npy_intp k = 0; k < dim0; k++)
                    cutoff = std::max(cutoff, 2 * cdata[k]);
                per_atom = cdata;
            } else {
                ncutoffs = (index_t)dim0;
                if (PyArray_DIM((PyArrayObject *)a_cutoffs, 1) != dim0) {
                    PyErr_SetString(PyExc_TypeError,
                                    "Two-dimensional cutoff array must be "
                                    "square.");
                    goto fail;
                }
                per_type_storage.resize((size_t)ncutoffs * ncutoffs);
                for (size_t k = 0; k < per_type_storage.size(); k++) {
                    cutoff = std::max(cutoff, cdata[k]);
                    per_type_storage[k] = cdata[k] * cdata[k];
                }
                per_type_sq = per_type_storage.data();
            }
        }

        /* Parse the quantity string into flags (the order is reused below). */
        py_bquantities = PyUnicode_AsASCIIString(py_quantities);
        if (!py_bquantities) {
            PyErr_SetString(PyExc_TypeError, "Conversion to ASCII string failed.");
            goto fail;
        }
        const char *quantities = PyBytes_AS_STRING(py_bquantities);
        int flags = 0;
        for (const char *q = quantities; *q; q++) {
            switch (*q) {
                case 'i': flags |= QUANTITY_FIRST; break;
                case 'j': flags |= QUANTITY_SECOND; break;
                case 'D': flags |= QUANTITY_DISTVEC; break;
                case 'd': flags |= QUANTITY_ABSDIST; break;
                case 'S': flags |= QUANTITY_SHIFT; break;
                default:
                    PyErr_SetString(PyExc_ValueError,
                                    "Unsupported quantity specified.");
                    goto fail;
            }
        }

        /* Gather raw pointers and call the core. */
        const real_t *cell_origin =
            (const real_t *)PyArray_DATA((PyArrayObject *)a_cell_origin);
        const real_t *cell =
            (const real_t *)PyArray_DATA((PyArrayObject *)a_cell);
        const real_t *inv_cell =
            (const real_t *)PyArray_DATA((PyArrayObject *)a_inv_cell);
        const npy_bool *pbc_data =
            (const npy_bool *)PyArray_DATA((PyArrayObject *)a_pbc);
        bool pbc[3] = {(bool)pbc_data[0], (bool)pbc_data[1], (bool)pbc_data[2]};
        const real_t *r = (const real_t *)PyArray_DATA((PyArrayObject *)a_r);
        const index_t *types =
            a_types ? (const index_t *)PyArray_DATA((PyArrayObject *)a_types)
                    : NULL;

        NeighbourList nl;
        error_t status =
            neighbour_list(flags, cell_origin, cell, inv_cell, pbc, nat, r,
                           cutoff, per_atom, per_type_sq, ncutoffs, types, nl);

        if (status != NL_SUCCESS) {
            raise_if_core_error();
            goto fail;
        }

        /* Build the output tuple in the requested order. */
        py_ret = PyTuple_New(strlen(quantities));
        if (!py_ret) goto fail;
        int pos = 0;
        for (const char *q = quantities; *q; q++, pos++) {
            PyObject *item = NULL;
            switch (*q) {
                case 'i': item = array_1d_int(nl.first); break;
                case 'j': item = array_1d_int(nl.secnd); break;
                case 'D': item = array_2d_double(nl.distvec, 3); break;
                case 'd': item = array_1d_double(nl.absdist); break;
                case 'S': item = array_2d_int(nl.shift, 3); break;
            }
            if (!item) goto fail;
            PyTuple_SET_ITEM(py_ret, pos, item);  /* steals reference */
        }

        if (strlen(quantities) == 1) {
            PyObject *only = PyTuple_GET_ITEM(py_ret, 0);
            Py_INCREF(only);
            Py_DECREF(py_ret);
            py_ret = only;
        }
    }

    Py_XDECREF(py_bquantities);
    Py_XDECREF(a_cutoffs);
    Py_XDECREF(a_cell_origin);
    Py_XDECREF(a_cell);
    Py_XDECREF(a_inv_cell);
    Py_XDECREF(a_pbc);
    Py_XDECREF(a_r);
    Py_XDECREF(a_types);
    return py_ret;

fail:
    Py_XDECREF(py_ret);
    Py_XDECREF(py_bquantities);
    Py_XDECREF(a_cutoffs);
    Py_XDECREF(a_cell_origin);
    Py_XDECREF(a_cell);
    Py_XDECREF(a_inv_cell);
    Py_XDECREF(a_pbc);
    Py_XDECREF(a_r);
    Py_XDECREF(a_types);
    return NULL;
}

/* ---------------------------------------------------------- first_neighbours */

PyObject *py_first_neighbours(PyObject *self, PyObject *args) {
    index_t n;
    PyObject *py_i;

    if (!PyArg_ParseTuple(args, "iO", &n, &py_i)) return NULL;

    PyObject *a_i =
        PyArray_FROMANY(py_i, NPY_INT, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    if (!a_i) return NULL;

    index_t nn = (index_t)PyArray_DIM((PyArrayObject *)a_i, 0);
    const index_t *i_n = (const index_t *)PyArray_DATA((PyArrayObject *)a_i);

    std::vector<index_t> seed(n + 1);
    first_neighbours(n, nn, i_n, seed.data());

    Py_DECREF(a_i);
    return array_1d_int(seed);
}

/* --------------------------------------------------------- get_jump_indicies */

PyObject *py_get_jump_indicies(PyObject *self, PyObject *args) {
    PyObject *py_sorted;

    if (!PyArg_ParseTuple(args, "O", &py_sorted)) return NULL;

    PyObject *a_sorted =
        PyArray_FROMANY(py_sorted, NPY_INT, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    if (!a_sorted) return NULL;

    index_t nn = (index_t)PyArray_DIM((PyArrayObject *)a_sorted, 0);
    const index_t *sorted =
        (const index_t *)PyArray_DATA((PyArrayObject *)a_sorted);

    std::vector<index_t> seed = get_jump_indicies(nn, sorted);

    Py_DECREF(a_sorted);
    return array_1d_int(seed);
}

/* -------------------------------------------------------------- triplet_list */

PyObject *py_triplet_list(PyObject *self, PyObject *args) {
    PyObject *py_fi, *py_absdist = NULL, *py_cutoff = NULL;

    if (!PyArg_ParseTuple(args, "O|OO", &py_fi, &py_absdist, &py_cutoff))
        return NULL;

    PyObject *a_fi = NULL, *a_absdist = NULL;

    a_fi = PyArray_FROMANY(py_fi, NPY_INT, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    if (!a_fi) goto fail;

    real_t cutoff;
    cutoff = 0.0;
    const real_t *absdist;
    absdist = NULL;

    if (py_cutoff || py_absdist) {
        if (!py_absdist || !py_cutoff) {
            PyErr_SetString(PyExc_TypeError,
                            "Cutoff and distances must be specified together.");
            goto fail;
        }
        a_absdist = PyArray_FROMANY(py_absdist, NPY_DOUBLE, 1, 1,
                                    NPY_ARRAY_C_CONTIGUOUS);
        if (!a_absdist) {
            PyErr_SetString(PyExc_TypeError,
                            "Distances must be an array of floats.");
            goto fail;
        }
        absdist = (const real_t *)PyArray_DATA((PyArrayObject *)a_absdist);
        if (PyFloat_Check(py_cutoff)) {
            cutoff = PyFloat_AsDouble(py_cutoff);
        } else {
            PyErr_SetString(PyExc_NotImplementedError,
                            "Cutoff must be a single float.");
            goto fail;
        }
    }

    {
        index_t n_first = (index_t)PyArray_DIM((PyArrayObject *)a_fi, 0);
        const index_t *first_i =
            (const index_t *)PyArray_DATA((PyArrayObject *)a_fi);

        std::vector<index_t> ij_t, ik_t;
        triplet_list(n_first, first_i, absdist, cutoff, ij_t, ik_t);

        PyObject *py_ij = array_1d_int(ij_t);
        PyObject *py_ik = array_1d_int(ik_t);
        if (!py_ij || !py_ik) {
            Py_XDECREF(py_ij);
            Py_XDECREF(py_ik);
            goto fail;
        }

        Py_DECREF(a_fi);
        Py_XDECREF(a_absdist);

        PyObject *py_ret = PyTuple_New(2);
        PyTuple_SET_ITEM(py_ret, 0, py_ij);
        PyTuple_SET_ITEM(py_ret, 1, py_ik);
        return py_ret;
    }

fail:
    Py_XDECREF(a_fi);
    Py_XDECREF(a_absdist);
    return NULL;
}
