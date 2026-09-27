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
 * in src/libneighbours; this file only marshals arrays in and out. Every entry
 * point runs inside guarded() (bind_py_common.hh), which turns C++ exceptions
 * into Python exceptions, and holds its references in PyRef so early returns
 * cannot leak.
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL MATSCIPY_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

#include <cstring>
#include <vector>

#include "bind_py_common.hh"
#include "bind_py_neighbours.hh"

#include "error.hh"
#include "first_neighbours.hh"
#include "neighbour_list.hh"
#include "triplet_list.hh"
#include "types.hh"

using namespace matscipy;
using namespace matscipy_py;

namespace {

/* ----------------------------------------------------------- neighbour_list */

PyObject *neighbour_list_impl(PyObject *, PyObject *args) {
    PyObject *py_cell_origin, *py_cell, *py_inv_cell, *py_pbc, *py_r;
    PyObject *py_quantities, *py_cutoffs, *py_types = NULL;

    if (!PyArg_ParseTuple(args, "O!OOOOOO|O", &PyUnicode_Type, &py_quantities,
                          &py_cell_origin, &py_cell, &py_inv_cell, &py_pbc,
                          &py_r, &py_cutoffs, &py_types))
        return NULL;

    GeometryArrays g;
    if (parse_geometry(py_cell_origin, py_cell, py_inv_cell, py_pbc, py_r,
                       py_types, -1, g) != 0)
        return NULL;

    PyRef a_cutoffs;
    real_t cutoff = 0.0;
    const real_t *per_atom = NULL, *per_type_sq = NULL;
    index_t ncutoffs = 0;
    std::vector<real_t> per_type_storage;
    if (resolve_cutoff(py_cutoffs, g.nat, a_cutoffs, &cutoff, &per_atom,
                       &per_type_sq, &ncutoffs, per_type_storage) != 0)
        return NULL;

    PyRef py_bquantities(PyUnicode_AsASCIIString(py_quantities));
    if (!py_bquantities) {
        PyErr_SetString(PyExc_TypeError, "Conversion to ASCII string failed.");
        return NULL;
    }
    const char *quantities = PyBytes_AS_STRING(py_bquantities.get());
    int flags = 0;
    if (quantity_flags(quantities, &flags) != 0) return NULL;

    NeighbourList nl;
    error_t status = neighbour_list(flags, g.origin_data(), g.cell_data(),
                                    g.inv_data(), g.periodic, g.nat, g.pos_data(),
                                    cutoff, per_atom, per_type_sq, ncutoffs,
                                    g.types_data(), nl);
    if (status != NL_SUCCESS) {
        raise_core_error(status);
        return NULL;
    }

    /* Build the output tuple in the requested order. */
    const Py_ssize_t nq = static_cast<Py_ssize_t>(std::strlen(quantities));
    PyRef py_ret(PyTuple_New(nq));
    if (!py_ret) return NULL;
    Py_ssize_t pos = 0;
    for (const char *q = quantities; *q; q++, pos++) {
        PyObject *item = NULL;
        switch (*q) {
            case 'i': item = array_1d_index(nl.first); break;
            case 'j': item = array_1d_index(nl.secnd); break;
            case 'D': item = array_2d_real(nl.distvec, 3); break;
            case 'd': item = array_1d_real(nl.absdist); break;
            case 'S': item = array_2d_index(nl.shift, 3); break;
        }
        if (!item) return NULL;
        PyTuple_SET_ITEM(py_ret.get(), pos, item); /* steals reference */
    }

    if (nq == 1) {
        PyObject *only = PyTuple_GET_ITEM(py_ret.get(), 0);
        Py_INCREF(only);
        return only;
    }
    return py_ret.release();
}

/* ---------------------------------------------------------- first_neighbours */

PyObject *first_neighbours_impl(PyObject *, PyObject *args) {
    Py_ssize_t n_arg;
    PyObject *py_i;

    if (!PyArg_ParseTuple(args, "nO", &n_arg, &py_i)) return NULL;
    const index_t n = static_cast<index_t>(n_arg);

    PyRef a_i(PyArray_FROMANY(py_i, NPY_INDEX_T, 1, 1, NPY_ARRAY_C_CONTIGUOUS));
    if (!a_i) return NULL;

    const index_t nn = static_cast<index_t>(PyArray_DIM(a_i.array(), 0));
    const index_t *i_n = static_cast<const index_t *>(PyArray_DATA(a_i.array()));

    /* The core rejects n < 0 and bad indices before touching seed; size the
       buffer defensively so an invalid n cannot throw first. */
    std::vector<index_t> seed(n < 0 ? 0 : static_cast<size_t>(n) + 1);
    error_t status = first_neighbours(n, nn, i_n, seed.data());
    if (status != NL_SUCCESS) {
        raise_core_error(status);
        return NULL;
    }
    return array_1d_index(seed);
}

/* --------------------------------------------------------- get_jump_indicies */

PyObject *get_jump_indicies_impl(PyObject *, PyObject *args) {
    PyObject *py_sorted;

    if (!PyArg_ParseTuple(args, "O", &py_sorted)) return NULL;

    PyRef a_sorted(
        PyArray_FROMANY(py_sorted, NPY_INDEX_T, 1, 1, NPY_ARRAY_C_CONTIGUOUS));
    if (!a_sorted) return NULL;

    const index_t nn = static_cast<index_t>(PyArray_DIM(a_sorted.array(), 0));
    const index_t *sorted =
        static_cast<const index_t *>(PyArray_DATA(a_sorted.array()));

    std::vector<index_t> seed;
    error_t status = get_jump_indicies(nn, sorted, seed);
    if (status != NL_SUCCESS) {
        raise_core_error(status);
        return NULL;
    }
    return array_1d_index(seed);
}

/* -------------------------------------------------------------- triplet_list */

PyObject *triplet_list_impl(PyObject *, PyObject *args) {
    PyObject *py_fi, *py_absdist = NULL, *py_cutoff = NULL;

    if (!PyArg_ParseTuple(args, "O|OO", &py_fi, &py_absdist, &py_cutoff))
        return NULL;

    PyRef a_fi(PyArray_FROMANY(py_fi, NPY_INDEX_T, 1, 1, NPY_ARRAY_C_CONTIGUOUS));
    if (!a_fi) return NULL;

    PyRef a_absdist;
    real_t cutoff = 0.0;
    const real_t *absdist = NULL;
    index_t n_absdist = 0;

    if (py_cutoff || py_absdist) {
        if (!py_absdist || !py_cutoff) {
            PyErr_SetString(PyExc_TypeError,
                            "Cutoff and distances must be specified together.");
            return NULL;
        }
        a_absdist = PyArray_FROMANY(py_absdist, NPY_DOUBLE, 1, 1,
                                    NPY_ARRAY_C_CONTIGUOUS);
        if (!a_absdist) {
            PyErr_SetString(PyExc_TypeError,
                            "Distances must be an array of floats.");
            return NULL;
        }
        absdist = static_cast<const real_t *>(PyArray_DATA(a_absdist.array()));
        n_absdist = static_cast<index_t>(PyArray_DIM(a_absdist.array(), 0));
        const int scalar = scalar_to_double(py_cutoff, &cutoff);
        if (scalar < 0) return NULL;
        if (scalar == 0) {
            PyErr_SetString(PyExc_TypeError, "Cutoff must be a single number.");
            return NULL;
        }
    }

    const index_t n_first = static_cast<index_t>(PyArray_DIM(a_fi.array(), 0));
    const index_t *first_i =
        static_cast<const index_t *>(PyArray_DATA(a_fi.array()));

    std::vector<index_t> ij_t, ik_t;
    error_t status =
        triplet_list(n_first, first_i, n_absdist, absdist, cutoff, ij_t, ik_t);
    if (status != NL_SUCCESS) {
        raise_core_error(status);
        return NULL;
    }

    PyRef py_ij(array_1d_index(ij_t));
    PyRef py_ik(array_1d_index(ik_t));
    if (!py_ij || !py_ik) return NULL;
    return PyTuple_Pack(2, py_ij.get(), py_ik.get());
}

}  // namespace

/* Exported entry points: exception-guarded wrappers around the bodies above. */

PyObject *py_neighbour_list(PyObject *self, PyObject *args) {
    return guarded(neighbour_list_impl, self, args);
}

PyObject *py_first_neighbours(PyObject *self, PyObject *args) {
    return guarded(first_neighbours_impl, self, args);
}

PyObject *py_get_jump_indicies(PyObject *self, PyObject *args) {
    return guarded(get_jump_indicies_impl, self, args);
}

PyObject *py_triplet_list(PyObject *self, PyObject *args) {
    return guarded(triplet_list_impl, self, args);
}
