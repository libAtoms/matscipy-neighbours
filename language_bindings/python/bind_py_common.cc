/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL MATSCIPY_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

#include <cmath>
#include <cstring>

#include "bind_py_common.hh"

#include "error.hh"
#include "types.hh"

using matscipy::index_t;
using matscipy::real_t;

namespace matscipy_py {

void raise_core_error(matscipy::error_t status) {
    PyObject *type = status == matscipy::NL_INVALID_ARGUMENT ? PyExc_ValueError
                                                              : PyExc_RuntimeError;
    PyErr_SetString(type, matscipy::has_error ? matscipy::error_string
                                              : "Unknown core error.");
}

/* ----------------------------------------------------------- geometry */

const real_t *GeometryArrays::origin_data() const {
    return static_cast<const real_t *>(PyArray_DATA(origin.array()));
}
const real_t *GeometryArrays::cell_data() const {
    return static_cast<const real_t *>(PyArray_DATA(cell.array()));
}
const real_t *GeometryArrays::inv_data() const {
    return static_cast<const real_t *>(PyArray_DATA(inv.array()));
}
const real_t *GeometryArrays::pos_data() const {
    return static_cast<const real_t *>(PyArray_DATA(pos.array()));
}
const index_t *GeometryArrays::types_data() const {
    return types ? static_cast<const index_t *>(PyArray_DATA(types.array()))
                 : NULL;
}

namespace {

/* FROMANY + shape check. `d0`/`d1` < 0 mean "any". Sets TypeError. */
bool convert(PyObject *obj, int typenum, int ndim, npy_intp d0, npy_intp d1,
             const char *what, PyRef &out) {
    out = PyArray_FROMANY(obj, typenum, ndim, ndim, NPY_ARRAY_C_CONTIGUOUS);
    if (!out) return false;
    PyArrayObject *a = out.array();
    bool ok = PyArray_NDIM(a) == ndim;
    if (ok && d0 >= 0) ok = PyArray_DIM(a, 0) == d0;
    if (ok && ndim > 1 && d1 >= 0) ok = PyArray_DIM(a, 1) == d1;
    if (!ok) {
        if (ndim == 1) {
            PyErr_Format(PyExc_TypeError, "%s must have shape (%zd,), got (%zd,).",
                         what, (Py_ssize_t)d0, (Py_ssize_t)PyArray_DIM(a, 0));
        } else {
            PyErr_Format(PyExc_TypeError,
                         "%s must have shape (%s, %zd), got (%zd, %zd).", what,
                         d0 >= 0 ? "3" : "n", (Py_ssize_t)d1,
                         (Py_ssize_t)PyArray_DIM(a, 0),
                         (Py_ssize_t)PyArray_DIM(a, 1));
        }
        return false;
    }
    return true;
}

}  // namespace

int parse_geometry(PyObject *py_origin, PyObject *py_cell, PyObject *py_inv,
                   PyObject *py_pbc, PyObject *py_pos, PyObject *py_types,
                   npy_intp nat_override, GeometryArrays &g) {
    if (!convert(py_origin, NPY_DOUBLE, 1, 3, -1, "cell_origin", g.origin) ||
        !convert(py_cell, NPY_DOUBLE, 2, 3, 3, "cell", g.cell) ||
        !convert(py_inv, NPY_DOUBLE, 2, 3, 3, "inv_cell", g.inv) ||
        !convert(py_pbc, NPY_BOOL, 1, 3, -1, "pbc", g.pbc) ||
        !convert(py_pos, NPY_DOUBLE, 2, -1, 3, "positions", g.pos)) {
        return -1;
    }
    const npy_intp host_rows = PyArray_DIM(g.pos.array(), 0);
    if (nat_override >= 0) {
        if (host_rows != 0 && host_rows != nat_override) {
            PyErr_SetString(PyExc_TypeError,
                            "Host and device position arrays disagree on the "
                            "number of atoms.");
            return -1;
        }
        g.nat = static_cast<index_t>(nat_override);
    } else {
        g.nat = static_cast<index_t>(host_rows);
    }
    if (py_types && py_types != Py_None) {
        if (!convert(py_types, NPY_INDEX_T, 1, -1, -1, "types", g.types)) {
            return -1;
        }
        if (PyArray_DIM(g.types.array(), 0) != static_cast<npy_intp>(g.nat)) {
            PyErr_SetString(PyExc_TypeError,
                            "Position and type arrays must have identical first "
                            "dimension.");
            return -1;
        }
    }
    const npy_bool *pb = static_cast<const npy_bool *>(PyArray_DATA(g.pbc.array()));
    for (int k = 0; k < 3; k++) g.periodic[k] = pb[k] != 0;
    return 0;
}

/* ------------------------------------------------------------- cutoffs */

int scalar_to_double(PyObject *obj, real_t *out) {
    const bool is_scalar =
        PyArray_IsAnyScalar(obj) ||
        (PyArray_Check(obj) && PyArray_NDIM((PyArrayObject *)obj) == 0);
    if (!is_scalar) return 0;
    *out = PyFloat_AsDouble(obj);
    if (*out == -1.0 && PyErr_Occurred()) return -1;
    return 1;
}

int resolve_cutoff(PyObject *py_cut, index_t nat, PyRef &a_cut, real_t *cutoff,
                   const real_t **per_atom, const real_t **per_type_sq,
                   index_t *ncutoffs, std::vector<real_t> &storage) {
    *cutoff = 0.0;
    *per_atom = NULL;
    *per_type_sq = NULL;
    *ncutoffs = 0;

    const int scalar = scalar_to_double(py_cut, cutoff);
    if (scalar < 0) return -1;
    if (scalar > 0) return 0;

    a_cut = PyArray_FROMANY(py_cut, NPY_DOUBLE, 1, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!a_cut) return -1;
    PyArrayObject *a = a_cut.array();
    const int ndim = PyArray_NDIM(a);
    const npy_intp dim0 = PyArray_DIM(a, 0);
    const real_t *cd = static_cast<const real_t *>(PyArray_DATA(a));

    if (ndim == 1) {
        if (dim0 != static_cast<npy_intp>(nat)) {
            PyErr_SetString(PyExc_TypeError,
                            "One-dimensional cutoff array must have length that "
                            "corresponds to position array.");
            return -1;
        }
        for (npy_intp k = 0; k < dim0; k++) {
            if (!(cd[k] >= 0) || !std::isfinite(cd[k])) {
                PyErr_SetString(PyExc_ValueError,
                                "Per-atom cutoff radii must be finite and "
                                "non-negative.");
                return -1;
            }
            *cutoff = std::max(*cutoff, 2 * cd[k]);
        }
        *per_atom = cd;
        return 0;
    }

    if (PyArray_DIM(a, 1) != dim0) {
        PyErr_SetString(PyExc_TypeError,
                        "Two-dimensional cutoff array must be square.");
        return -1;
    }
    *ncutoffs = static_cast<index_t>(dim0);
    storage.resize(static_cast<size_t>(dim0) * dim0);
    for (size_t k = 0; k < storage.size(); k++) {
        if (!(cd[k] >= 0) || !std::isfinite(cd[k])) {
            PyErr_SetString(PyExc_ValueError,
                            "Per-type cutoffs must be finite and non-negative.");
            return -1;
        }
        *cutoff = std::max(*cutoff, cd[k]);
        storage[k] = cd[k] * cd[k];
    }
    *per_type_sq = storage.data();
    return 0;
}

int quantity_flags(const char *q, int *flags) {
    *flags = 0;
    for (; *q; q++) {
        switch (*q) {
            case 'i': *flags |= matscipy::QUANTITY_FIRST; break;
            case 'j': *flags |= matscipy::QUANTITY_SECOND; break;
            case 'D': *flags |= matscipy::QUANTITY_DISTVEC; break;
            case 'd': *flags |= matscipy::QUANTITY_ABSDIST; break;
            case 'S': *flags |= matscipy::QUANTITY_SHIFT; break;
            default:
                PyErr_SetString(PyExc_ValueError, "Unsupported quantity specified.");
                return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------- result arrays */

namespace {

template <typename T>
PyObject *copy_to_array(const std::vector<T> &v, int typenum, int ndim,
                        npy_intp ncols) {
    npy_intp dims[2] = {static_cast<npy_intp>(v.size()), 1};
    if (ndim == 2) {
        dims[0] = static_cast<npy_intp>(v.size()) / ncols;
        dims[1] = ncols;
    }
    PyObject *a = PyArray_SimpleNew(ndim, dims, typenum);
    if (a && !v.empty()) {
        std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(a)), v.data(),
                    v.size() * sizeof(T));
    }
    return a;
}

}  // namespace

PyObject *array_1d_index(const std::vector<index_t> &v) {
    return copy_to_array(v, NPY_INDEX_T, 1, 1);
}
PyObject *array_2d_index(const std::vector<index_t> &v, npy_intp ncols) {
    return copy_to_array(v, NPY_INDEX_T, 2, ncols);
}
PyObject *array_1d_real(const std::vector<real_t> &v) {
    return copy_to_array(v, NPY_DOUBLE, 1, 1);
}
PyObject *array_2d_real(const std::vector<real_t> &v, npy_intp ncols) {
    return copy_to_array(v, NPY_DOUBLE, 2, ncols);
}

}  // namespace matscipy_py
