/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Helpers shared by the C++ binding sources (bind_py_neighbours.cc and
 * bind_py_dlpack.cc): the exception guard around every entry point, the
 * core-error -> Python-exception mapping, owned-reference and argument
 * validation helpers, and the NumPy result constructors. C++ only; module.c
 * never includes this header.
 */

#ifndef BIND_PY_COMMON_HH
#define BIND_PY_COMMON_HH

#include <Python.h>
#include <numpy/ndarraytypes.h>

#include <exception>
#include <new>
#include <stdexcept>
#include <vector>

#include "types.hh"

namespace matscipy_py {

/* NumPy type number matching the core's index_t. */
constexpr int NPY_INDEX_T = sizeof(matscipy::index_t) == 8 ? NPY_INT64 : NPY_INT32;

/* Run a binding body and convert any C++ exception escaping it into a Python
   exception. CPython's call machinery is C: an exception crossing it would
   terminate the interpreter. */
template <typename F>
PyObject *guarded(F body, PyObject *self, PyObject *args) noexcept {
    try {
        return body(self, args);
    } catch (const std::bad_alloc &) {
        PyErr_NoMemory();
    } catch (const std::length_error &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::out_of_range &e) {
        PyErr_SetString(PyExc_IndexError, e.what());
    } catch (const std::overflow_error &e) {
        PyErr_SetString(PyExc_OverflowError, e.what());
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Unknown C++ exception in _matscipy_neighbours.");
    }
    return NULL;
}

/* Raise the core error as a Python exception: NL_INVALID_ARGUMENT ->
   ValueError, anything else -> RuntimeError. Always sets an exception. */
void raise_core_error(matscipy::error_t status);

/* Owning reference to a Python object (Py_XDECREF on destruction). */
class PyRef {
    PyObject *p_ = nullptr;

   public:
    PyRef() = default;
    explicit PyRef(PyObject *p) : p_(p) {}
    PyRef(const PyRef &) = delete;
    PyRef &operator=(const PyRef &) = delete;
    ~PyRef() { Py_XDECREF(p_); }
    PyRef &operator=(PyObject *p) {
        Py_XDECREF(p_);
        p_ = p;
        return *this;
    }
    PyObject *get() const { return p_; }
    PyArrayObject *array() const { return reinterpret_cast<PyArrayObject *>(p_); }
    explicit operator bool() const { return p_ != nullptr; }
    /* Give up ownership (for returning to Python). */
    PyObject *release() {
        PyObject *p = p_;
        p_ = nullptr;
        return p;
    }
};

/* The geometry arrays of a neighbour-list call, converted to contiguous NumPy
   arrays of the core's types and validated:
     cell_origin (3,)  cell (3,3)  inv_cell (3,3)  pbc (3,)  positions (n,3)
     types (n,) or absent.
   All references are owned. */
struct GeometryArrays {
    PyRef origin, cell, inv, pbc, pos, types;
    matscipy::index_t nat = 0;
    bool periodic[3] = {false, false, false};

    const matscipy::real_t *origin_data() const;
    const matscipy::real_t *cell_data() const;
    const matscipy::real_t *inv_data() const;
    const matscipy::real_t *pos_data() const;
    const matscipy::index_t *types_data() const; /* NULL if no types */
};

/* Convert and validate the geometry arguments. `py_types` may be NULL or None.
   `nat_override >= 0` means the atom count comes from a device tensor; the
   host positions must then have 0 or nat_override rows. Returns 0, or -1 with
   a Python exception (TypeError) set. */
int parse_geometry(PyObject *py_origin, PyObject *py_cell, PyObject *py_inv,
                   PyObject *py_pbc, PyObject *py_pos, PyObject *py_types,
                   npy_intp nat_override, GeometryArrays &g);

/* Convert a 0-d number (Python int/float, NumPy scalar or 0-d array) to a
   double. Returns 1 on success, 0 if `obj` is not a scalar, -1 with a Python
   exception set if it is a scalar that cannot be converted. */
int scalar_to_double(PyObject *obj, matscipy::real_t *out);

/* Resolve the cutoff argument: scalar, per-atom radii (n,) or per-type matrix
   (m,m). On the array forms `a_cut` owns the converted array (it must outlive
   the pointers). Array entries must be finite and non-negative; the scalar's
   range is checked by the core. Returns 0, or -1 with an exception set. */
int resolve_cutoff(PyObject *py_cut, matscipy::index_t nat, PyRef &a_cut,
                   matscipy::real_t *cutoff, const matscipy::real_t **per_atom,
                   const matscipy::real_t **per_type_sq,
                   matscipy::index_t *ncutoffs,
                   std::vector<matscipy::real_t> &storage);

/* "ijdDS" -> Quantity flags. Returns 0, or -1 with ValueError set. */
int quantity_flags(const char *q, int *flags);

/* Fresh NumPy arrays holding copies of core result buffers. */
PyObject *array_1d_index(const std::vector<matscipy::index_t> &v);
PyObject *array_2d_index(const std::vector<matscipy::index_t> &v, npy_intp ncols);
PyObject *array_1d_real(const std::vector<matscipy::real_t> &v);
PyObject *array_2d_real(const std::vector<matscipy::real_t> &v, npy_intp ncols);

}  // namespace matscipy_py

#endif
