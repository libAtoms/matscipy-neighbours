/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Phase 4 — DLPack export. Builds the neighbour list on the requested backend
 * and hands each output array to Python as a DLPack "dltensor" capsule that
 * OWNS the underlying buffer (host std::vector or device Array<T, CudaSpace>).
 * numpy/cupy `from_dlpack` then wrap it zero-copy; the consumer takes ownership
 * and our deleter frees the buffer when the consumer is done. No host round-trip
 * for the device path — results stay on the GPU.
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL MATSCIPY_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "bind_py_dlpack.hh"
#include "dlpack.h"

#include "error.hh"
#include "memory_space.hh"
#include "neighbour_list.hh"
#include "neighbour_list_gpu.hh"
#include "types.hh"

using namespace matscipy;

namespace {

/* Owns the buffer behind a DLManagedTensor and the shape array. `keep` holds a
   shared_ptr to the std::vector / Array, released when the manager is deleted. */
struct DLPackManager {
    DLManagedTensor mt;
    int64_t shape[2];
    std::shared_ptr<void> keep;
};

void managed_deleter(DLManagedTensor *self) {
    delete reinterpret_cast<DLPackManager *>(self->manager_ctx);
}

/* PyCapsule destructor: only frees if the consumer never claimed the capsule
   (its name is still "dltensor"; a consumer renames it to "used_dltensor"). */
void capsule_destructor(PyObject *cap) {
    if (PyCapsule_IsValid(cap, "dltensor")) {
        auto *mt = static_cast<DLManagedTensor *>(
            PyCapsule_GetPointer(cap, "dltensor"));
        if (mt && mt->deleter) mt->deleter(mt);
    }
}

PyObject *make_capsule(std::shared_ptr<void> keep, void *data, int ndim,
                       int64_t d0, int64_t d1, uint8_t code, uint8_t bits,
                       DLDeviceType dev, int dev_id) {
    auto *m = new DLPackManager();
    m->shape[0] = d0;
    m->shape[1] = d1;
    m->keep = std::move(keep);
    m->mt.dl_tensor.data = data;
    m->mt.dl_tensor.device.device_type = dev;
    m->mt.dl_tensor.device.device_id = dev_id;
    m->mt.dl_tensor.ndim = ndim;
    m->mt.dl_tensor.dtype.code = code;
    m->mt.dl_tensor.dtype.bits = bits;
    m->mt.dl_tensor.dtype.lanes = 1;
    m->mt.dl_tensor.shape = m->shape;
    m->mt.dl_tensor.strides = nullptr;  /* compact row-major */
    m->mt.dl_tensor.byte_offset = 0;
    m->mt.manager_ctx = m;
    m->mt.deleter = managed_deleter;
    PyObject *cap = PyCapsule_New(&m->mt, "dltensor", capsule_destructor);
    if (!cap) delete m;
    return cap;
}

constexpr uint8_t kIntBits = sizeof(index_t) * 8;
constexpr uint8_t kRealBits = sizeof(real_t) * 8;

template <typename T>
PyObject *host_capsule(std::vector<T> &&v, int ndim, int64_t d0, int64_t d1,
                       uint8_t code, uint8_t bits) {
    auto keep = std::make_shared<std::vector<T>>(std::move(v));
    return make_capsule(keep, keep->data(), ndim, d0, d1, code, bits, kDLCPU, 0);
}

#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
template <typename T>
PyObject *device_capsule(Array<T, DeviceSpace> &&a, int ndim, int64_t d0,
                         int64_t d1, uint8_t code, uint8_t bits, int dev_id) {
    auto keep = std::make_shared<Array<T, DeviceSpace>>(std::move(a));
    void *data = keep->data();
    /* DeviceType codes equal the DLPack device codes by construction. */
    const auto dev = static_cast<DLDeviceType>(static_cast<int>(DeviceSpace::device));
    return make_capsule(keep, data, ndim, d0, d1, code, bits, dev, dev_id);
}
#endif

}  // namespace

PyObject *py_neighbour_list_dlpack(PyObject *self, PyObject *args) {
    PyObject *py_quant, *py_origin, *py_cell, *py_inv, *py_pbc, *py_pos, *py_cut;
    PyObject *py_types = NULL;
    int backend = 0;                  /* 0 = CPU/host, 1 = GPU/device */
    unsigned long long r_device_ptr = 0;  /* nonzero => device positions ptr */
    int nat_override = -1;
    int device_id = -1;               /* GPU to run on / report; -1 = current */

    if (!PyArg_ParseTuple(args, "O!OOOOOO|OiKii", &PyUnicode_Type, &py_quant,
                          &py_origin, &py_cell, &py_inv, &py_pbc, &py_pos,
                          &py_cut, &py_types, &backend, &r_device_ptr,
                          &nat_override, &device_id))
        return NULL;

#if !(defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP))
    if (backend != 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "GPU backend requested but the extension was built "
                        "without a GPU backend (-DENABLE_CUDA=ON).");
        return NULL;
    }
#endif

    PyObject *a_origin = NULL, *a_cell = NULL, *a_inv = NULL, *a_pbc = NULL;
    PyObject *a_pos = NULL, *a_cut = NULL, *a_types = NULL, *py_ret = NULL;
    PyObject *py_bquant = NULL;

    a_origin = PyArray_FROMANY(py_origin, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    a_cell = PyArray_FROMANY(py_cell, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    a_inv = PyArray_FROMANY(py_inv, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    a_pbc = PyArray_FROMANY(py_pbc, NPY_BOOL, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    a_pos = PyArray_FROMANY(py_pos, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!a_origin || !a_cell || !a_inv || !a_pbc || !a_pos) goto fail;
    if (py_types && py_types != Py_None) {
        a_types = PyArray_FROMANY(py_types, NPY_INT, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
        if (!a_types) goto fail;
    }

    {
        index_t nat = nat_override >= 0
                          ? (index_t)nat_override
                          : (index_t)PyArray_DIM((PyArrayObject *)a_pos, 0);

        /* Resolve the cutoff (scalar / per-atom 1d / per-type 2d), as in the
           numpy binding. */
        real_t cutoff = 0.0;
        const real_t *per_atom = NULL;
        const real_t *per_type_sq = NULL;
        index_t ncutoffs = 0;
        std::vector<real_t> per_type_storage;
        if (PyFloat_Check(py_cut)) {
            cutoff = PyFloat_AsDouble(py_cut);
        } else {
            a_cut = PyArray_FROMANY(py_cut, NPY_DOUBLE, 1, 2,
                                    NPY_ARRAY_C_CONTIGUOUS);
            if (!a_cut) goto fail;
            int ndim = PyArray_NDIM((PyArrayObject *)a_cut);
            npy_intp dim0 = PyArray_DIM((PyArrayObject *)a_cut, 0);
            const real_t *cdata =
                (const real_t *)PyArray_DATA((PyArrayObject *)a_cut);
            if (ndim == 1) {
                for (npy_intp k = 0; k < dim0; k++)
                    cutoff = std::max(cutoff, 2 * cdata[k]);
                per_atom = cdata;
            } else {
                ncutoffs = (index_t)dim0;
                per_type_storage.resize((size_t)ncutoffs * ncutoffs);
                for (size_t k = 0; k < per_type_storage.size(); k++) {
                    cutoff = std::max(cutoff, cdata[k]);
                    per_type_storage[k] = cdata[k] * cdata[k];
                }
                per_type_sq = per_type_storage.data();
            }
        }

        /* Quantity flags + the requested character order. */
        py_bquant = PyUnicode_AsASCIIString(py_quant);
        if (!py_bquant) goto fail;
        const char *quantities = PyBytes_AS_STRING(py_bquant);
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

        const real_t *origin =
            (const real_t *)PyArray_DATA((PyArrayObject *)a_origin);
        const real_t *cell = (const real_t *)PyArray_DATA((PyArrayObject *)a_cell);
        const real_t *inv = (const real_t *)PyArray_DATA((PyArrayObject *)a_inv);
        const npy_bool *pb = (const npy_bool *)PyArray_DATA((PyArrayObject *)a_pbc);
        bool pbc[3] = {(bool)pb[0], (bool)pb[1], (bool)pb[2]};
        const real_t *r_host =
            (const real_t *)PyArray_DATA((PyArrayObject *)a_pos);
        const index_t *types =
            a_types ? (const index_t *)PyArray_DATA((PyArrayObject *)a_types)
                    : NULL;

        const int nq = (int)Py_SAFE_DOWNCAST(strlen(quantities), size_t, int);
        py_ret = PyTuple_New(nq);
        if (!py_ret) goto fail;

        if (backend == 0) {
            /* CPU backend: host buffers wrapped as kDLCPU capsules. */
            NeighbourList nl;
            if (neighbour_list(flags, origin, cell, inv, pbc, nat, r_host, cutoff,
                               per_atom, per_type_sq, ncutoffs, types,
                               nl) != NL_SUCCESS) {
                if (has_error) PyErr_SetString(PyExc_RuntimeError, error_string);
                goto fail;
            }
            const int64_t np = nl.npairs;
            int pos = 0;
            for (const char *q = quantities; *q; q++, pos++) {
                PyObject *cap = NULL;
                switch (*q) {
                    case 'i': cap = host_capsule(std::move(nl.first), 1, np, 1,
                                                 kDLInt, kIntBits); break;
                    case 'j': cap = host_capsule(std::move(nl.secnd), 1, np, 1,
                                                 kDLInt, kIntBits); break;
                    case 'D': cap = host_capsule(std::move(nl.distvec), 2, np, 3,
                                                 kDLFloat, kRealBits); break;
                    case 'd': cap = host_capsule(std::move(nl.absdist), 1, np, 1,
                                                 kDLFloat, kRealBits); break;
                    case 'S': cap = host_capsule(std::move(nl.shift), 2, np, 3,
                                                 kDLInt, kIntBits); break;
                }
                if (!cap) goto fail;
                PyTuple_SET_ITEM(py_ret, pos, cap);
            }
        } else {
#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
            /* GPU backend: results stay on the device, wrapped as kDLCUDA. */
            const real_t *r = r_device_ptr ? (const real_t *)(uintptr_t)r_device_ptr
                                            : r_host;
            const bool r_is_dev = r_device_ptr != 0;
            NeighbourListDevice dev;
            if (neighbour_list_gpu_device(flags, origin, cell, inv, pbc, nat, r,
                                          r_is_dev, cutoff, per_atom, per_type_sq,
                                          ncutoffs, types, CellOrder::Linear,
                                          device_id, dev) != NL_SUCCESS) {
                if (has_error) PyErr_SetString(PyExc_RuntimeError, error_string);
                goto fail;
            }
            const int64_t np = dev.npairs;
            const int dev_id = device_id >= 0 ? device_id : current_device_id();
            int pos = 0;
            for (const char *q = quantities; *q; q++, pos++) {
                PyObject *cap = NULL;
                switch (*q) {
                    case 'i': cap = device_capsule(std::move(dev.first), 1, np, 1,
                                                   kDLInt, kIntBits, dev_id); break;
                    case 'j': cap = device_capsule(std::move(dev.secnd), 1, np, 1,
                                                   kDLInt, kIntBits, dev_id); break;
                    case 'D': cap = device_capsule(std::move(dev.distvec), 2, np, 3,
                                                   kDLFloat, kRealBits, dev_id); break;
                    case 'd': cap = device_capsule(std::move(dev.absdist), 1, np, 1,
                                                   kDLFloat, kRealBits, dev_id); break;
                    case 'S': cap = device_capsule(std::move(dev.shift), 2, np, 3,
                                                   kDLInt, kIntBits, dev_id); break;
                }
                if (!cap) goto fail;
                PyTuple_SET_ITEM(py_ret, pos, cap);
            }
#endif
        }
    }

    Py_XDECREF(py_bquant);
    Py_XDECREF(a_cut);
    Py_XDECREF(a_origin);
    Py_XDECREF(a_cell);
    Py_XDECREF(a_inv);
    Py_XDECREF(a_pbc);
    Py_XDECREF(a_pos);
    Py_XDECREF(a_types);
    return py_ret;

fail:
    Py_XDECREF(py_ret);
    Py_XDECREF(py_bquant);
    Py_XDECREF(a_cut);
    Py_XDECREF(a_origin);
    Py_XDECREF(a_cell);
    Py_XDECREF(a_inv);
    Py_XDECREF(a_pbc);
    Py_XDECREF(a_pos);
    Py_XDECREF(a_types);
    return NULL;
}

PyObject *py_coordination_dlpack(PyObject *self, PyObject *args) {
#if !(defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP))
    (void)args;
    PyErr_SetString(PyExc_RuntimeError,
                    "GPU coordination requires a GPU backend (-DENABLE_CUDA=ON).");
    return NULL;
#else
    PyObject *py_origin, *py_cell, *py_inv, *py_pbc, *py_pos, *py_cut;
    PyObject *py_types = NULL;
    unsigned long long r_device_ptr = 0;
    int nat_override = -1, device_id = -1;

    if (!PyArg_ParseTuple(args, "OOOOOO|OKii", &py_origin, &py_cell, &py_inv,
                          &py_pbc, &py_pos, &py_cut, &py_types, &r_device_ptr,
                          &nat_override, &device_id))
        return NULL;

    PyObject *a_origin = NULL, *a_cell = NULL, *a_inv = NULL, *a_pbc = NULL;
    PyObject *a_pos = NULL, *a_cut = NULL, *a_types = NULL, *ret = NULL;
    a_origin = PyArray_FROMANY(py_origin, NPY_DOUBLE, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    a_cell = PyArray_FROMANY(py_cell, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    a_inv = PyArray_FROMANY(py_inv, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    a_pbc = PyArray_FROMANY(py_pbc, NPY_BOOL, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
    a_pos = PyArray_FROMANY(py_pos, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!a_origin || !a_cell || !a_inv || !a_pbc || !a_pos) goto cfail;
    if (py_types && py_types != Py_None) {
        a_types = PyArray_FROMANY(py_types, NPY_INT, 1, 1, NPY_ARRAY_C_CONTIGUOUS);
        if (!a_types) goto cfail;
    }
    {
        index_t nat = nat_override >= 0
                          ? (index_t)nat_override
                          : (index_t)PyArray_DIM((PyArrayObject *)a_pos, 0);
        real_t cutoff = 0.0;
        const real_t *per_atom = NULL, *per_type_sq = NULL;
        index_t ncutoffs = 0;
        std::vector<real_t> pt_storage;
        if (PyFloat_Check(py_cut)) {
            cutoff = PyFloat_AsDouble(py_cut);
        } else {
            a_cut = PyArray_FROMANY(py_cut, NPY_DOUBLE, 1, 2, NPY_ARRAY_C_CONTIGUOUS);
            if (!a_cut) goto cfail;
            int ndim = PyArray_NDIM((PyArrayObject *)a_cut);
            npy_intp dim0 = PyArray_DIM((PyArrayObject *)a_cut, 0);
            const real_t *cd = (const real_t *)PyArray_DATA((PyArrayObject *)a_cut);
            if (ndim == 1) {
                for (npy_intp k = 0; k < dim0; k++) cutoff = std::max(cutoff, 2 * cd[k]);
                per_atom = cd;
            } else {
                ncutoffs = (index_t)dim0;
                pt_storage.resize((size_t)ncutoffs * ncutoffs);
                for (size_t k = 0; k < pt_storage.size(); k++) {
                    cutoff = std::max(cutoff, cd[k]);
                    pt_storage[k] = cd[k] * cd[k];
                }
                per_type_sq = pt_storage.data();
            }
        }
        const real_t *origin = (const real_t *)PyArray_DATA((PyArrayObject *)a_origin);
        const real_t *cell = (const real_t *)PyArray_DATA((PyArrayObject *)a_cell);
        const real_t *inv = (const real_t *)PyArray_DATA((PyArrayObject *)a_inv);
        const npy_bool *pb = (const npy_bool *)PyArray_DATA((PyArrayObject *)a_pbc);
        bool pbc[3] = {(bool)pb[0], (bool)pb[1], (bool)pb[2]};
        const real_t *r = r_device_ptr
                              ? (const real_t *)(uintptr_t)r_device_ptr
                              : (const real_t *)PyArray_DATA((PyArrayObject *)a_pos);
        const index_t *types =
            a_types ? (const index_t *)PyArray_DATA((PyArrayObject *)a_types) : NULL;

        NeighbourListDevice dev;
        if (neighbour_count_gpu_device(origin, cell, inv, pbc, nat, r,
                                       r_device_ptr != 0, cutoff, per_atom,
                                       per_type_sq, ncutoffs, types, device_id,
                                       dev) != NL_SUCCESS) {
            if (has_error) PyErr_SetString(PyExc_RuntimeError, error_string);
            goto cfail;
        }
        const int dev_id = device_id >= 0 ? device_id : current_device_id();
        ret = device_capsule(std::move(dev.counts), 1, nat, 1, kDLInt, kIntBits,
                             dev_id);
        if (!ret) goto cfail;
    }
    Py_XDECREF(a_cut);
    Py_XDECREF(a_origin);
    Py_XDECREF(a_cell);
    Py_XDECREF(a_inv);
    Py_XDECREF(a_pbc);
    Py_XDECREF(a_pos);
    Py_XDECREF(a_types);
    return ret;

cfail:
    Py_XDECREF(ret);
    Py_XDECREF(a_cut);
    Py_XDECREF(a_origin);
    Py_XDECREF(a_cell);
    Py_XDECREF(a_inv);
    Py_XDECREF(a_pbc);
    Py_XDECREF(a_pos);
    Py_XDECREF(a_types);
    return NULL;
#endif
}
