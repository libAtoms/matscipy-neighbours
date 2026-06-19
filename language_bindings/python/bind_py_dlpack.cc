/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * Phase 4 — DLPack import/export. Builds the neighbour list on the requested
 * backend and hands each output array to Python as a DLPack "dltensor" capsule
 * that OWNS its buffer (host std::vector or device Array<T, DeviceSpace>); any
 * DLPack consumer (numpy/cupy/torch/jax) then wraps it zero-copy. Device input
 * is read the same way — via the array's __dlpack__ — so the GPU path is not
 * tied to cupy and never round-trips positions through the host.
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

/* ----------------------------------------------------------- DLPack export */

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

/* ----------------------------------------------------------- DLPack import */

/* A device positions array imported through DLPack. Owns the consumed managed
   tensor; call release() once the (synchronous) build has read the data. */
struct ImportedDLPack {
    DLManagedTensor *mt = nullptr;
    const real_t *data = nullptr;
    int device_type = 0;
    int device_id = 0;
    npy_intp nat = 0;
    void release() {
        if (mt && mt->deleter) mt->deleter(mt);
        mt = nullptr;
    }
};

/* Import an (n, 3) float64 device array via its __dlpack__. Returns 0 and fills
   `imp` (owning the tensor) on success; -1 with a Python error set otherwise. */
int import_positions_dlpack(PyObject *arr, ImportedDLPack *imp) {
    PyObject *cap = PyObject_CallMethod(arr, "__dlpack__", NULL);
    if (!cap) return -1;
    if (!PyCapsule_IsValid(cap, "dltensor")) {
        PyErr_SetString(PyExc_TypeError,
                        "device positions did not yield an unversioned DLPack "
                        "capsule");
        Py_DECREF(cap);
        return -1;
    }
    auto *mt = static_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(cap, "dltensor"));
    if (!mt) {
        Py_DECREF(cap);
        return -1;
    }
    const DLTensor &t = mt->dl_tensor;
    bool ok_dtype =
        t.dtype.code == kDLFloat && t.dtype.bits == 64 && t.dtype.lanes == 1;
    bool ok_shape = t.ndim == 2 && t.shape[1] == 3;
    bool ok_contig = t.strides == nullptr ||
                     (t.strides[0] == 3 && t.strides[1] == 1);
    if (!ok_dtype || !ok_shape || !ok_contig) {
        PyErr_SetString(PyExc_TypeError,
                        "device positions must be a C-contiguous float64 array "
                        "of shape (n, 3)");
        if (mt->deleter) mt->deleter(mt);  /* release the just-produced tensor */
        Py_DECREF(cap);
        return -1;
    }
    imp->mt = mt;
    imp->data = reinterpret_cast<const real_t *>(
        static_cast<char *>(t.data) + t.byte_offset);
    imp->device_type = static_cast<int>(t.device.device_type);
    imp->device_id = t.device.device_id;
    imp->nat = t.shape[0];
    /* Consume: the producer's capsule destructor must not also free it. */
    PyCapsule_SetName(cap, "used_dltensor");
    Py_DECREF(cap);
    return 0;
}

/* ----------------------------------------------------------- shared parsing */

/* Resolve the cutoff argument (scalar / per-atom 1d / per-type 2d). On the array
   forms `*a_cut` is set to a new reference the caller must DECREF. */
int resolve_cutoff(PyObject *py_cut, PyObject **a_cut, real_t *cutoff,
                   const real_t **per_atom, const real_t **per_type_sq,
                   index_t *ncutoffs, std::vector<real_t> &storage) {
    *cutoff = 0.0;
    *per_atom = nullptr;
    *per_type_sq = nullptr;
    *ncutoffs = 0;
    if (PyFloat_Check(py_cut)) {
        *cutoff = PyFloat_AsDouble(py_cut);
        return 0;
    }
    *a_cut = PyArray_FROMANY(py_cut, NPY_DOUBLE, 1, 2, NPY_ARRAY_C_CONTIGUOUS);
    if (!*a_cut) return -1;
    int ndim = PyArray_NDIM((PyArrayObject *)*a_cut);
    npy_intp dim0 = PyArray_DIM((PyArrayObject *)*a_cut, 0);
    const real_t *cd = (const real_t *)PyArray_DATA((PyArrayObject *)*a_cut);
    if (ndim == 1) {
        for (npy_intp k = 0; k < dim0; k++) *cutoff = std::max(*cutoff, 2 * cd[k]);
        *per_atom = cd;
    } else {
        *ncutoffs = (index_t)dim0;
        storage.resize((size_t)*ncutoffs * *ncutoffs);
        for (size_t k = 0; k < storage.size(); k++) {
            *cutoff = std::max(*cutoff, cd[k]);
            storage[k] = cd[k] * cd[k];
        }
        *per_type_sq = storage.data();
    }
    return 0;
}

int quantity_flags(const char *q, int *flags) {
    *flags = 0;
    for (; *q; q++) {
        switch (*q) {
            case 'i': *flags |= QUANTITY_FIRST; break;
            case 'j': *flags |= QUANTITY_SECOND; break;
            case 'D': *flags |= QUANTITY_DISTVEC; break;
            case 'd': *flags |= QUANTITY_ABSDIST; break;
            case 'S': *flags |= QUANTITY_SHIFT; break;
            default:
                PyErr_SetString(PyExc_ValueError, "Unsupported quantity specified.");
                return -1;
        }
    }
    return 0;
}

}  // namespace

PyObject *py_neighbour_list_dlpack(PyObject *self, PyObject *args) {
    PyObject *py_quant, *py_origin, *py_cell, *py_inv, *py_pbc, *py_pos, *py_cut;
    PyObject *py_types = NULL;
    int backend = 0;          /* 0 = CPU/host, 1 = GPU/device */
    PyObject *py_in = NULL;   /* device positions object (has __dlpack__) or None */
    int device_id = -1;       /* GPU to run on / report; -1 = current */

    if (!PyArg_ParseTuple(args, "O!OOOOOO|OiOi", &PyUnicode_Type, &py_quant,
                          &py_origin, &py_cell, &py_inv, &py_pbc, &py_pos,
                          &py_cut, &py_types, &backend, &py_in, &device_id))
        return NULL;

    const bool device_in = py_in && py_in != Py_None;

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
    ImportedDLPack imp;

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
    if (device_in && import_positions_dlpack(py_in, &imp) != 0) goto fail;

    {
        index_t nat = device_in ? (index_t)imp.nat
                                : (index_t)PyArray_DIM((PyArrayObject *)a_pos, 0);

        real_t cutoff = 0.0;
        const real_t *per_atom = NULL, *per_type_sq = NULL;
        index_t ncutoffs = 0;
        std::vector<real_t> per_type_storage;
        if (resolve_cutoff(py_cut, &a_cut, &cutoff, &per_atom, &per_type_sq,
                           &ncutoffs, per_type_storage) != 0)
            goto fail;

        py_bquant = PyUnicode_AsASCIIString(py_quant);
        if (!py_bquant) goto fail;
        const char *quantities = PyBytes_AS_STRING(py_bquant);
        int flags = 0;
        if (quantity_flags(quantities, &flags) != 0) goto fail;

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

        const int nq = (int)strlen(quantities);
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
            /* GPU backend: results stay on the device, wrapped as device capsules.
               Device input is used in place; otherwise host positions upload. */
            NeighbourListRequest req;
            req.quantities = flags;
            req.cell_origin = origin;
            req.cell = cell;
            req.inv_cell = inv;
            req.pbc = pbc;
            req.nat = nat;
            req.positions = device_in ? imp.data : r_host;
            req.positions_on_device = device_in;
            req.cutoff = cutoff;
            req.per_atom_cutoff = per_atom;
            req.per_type_cutoff_sq = per_type_sq;
            req.ncutoffs = ncutoffs;
            req.types = types;
            req.device_id = device_in ? imp.device_id : device_id;

            NeighbourListDevice dev;
            error_t st = neighbour_list_gpu_device(req, dev);
            imp.release();  /* input consumed; result lives in `dev` */
            if (st != NL_SUCCESS) {
                if (has_error) PyErr_SetString(PyExc_RuntimeError, error_string);
                goto fail;
            }
            const int64_t np = dev.npairs;
            const int dev_id = req.device_id >= 0 ? req.device_id
                                                  : current_device_id();
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

    imp.release();
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
    imp.release();
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
    PyObject *py_types = NULL, *py_in = NULL;
    int device_id = -1;

    if (!PyArg_ParseTuple(args, "OOOOOO|OOi", &py_origin, &py_cell, &py_inv,
                          &py_pbc, &py_pos, &py_cut, &py_types, &py_in,
                          &device_id))
        return NULL;

    const bool device_in = py_in && py_in != Py_None;

    PyObject *a_origin = NULL, *a_cell = NULL, *a_inv = NULL, *a_pbc = NULL;
    PyObject *a_pos = NULL, *a_cut = NULL, *a_types = NULL, *ret = NULL;
    ImportedDLPack imp;

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
    if (device_in && import_positions_dlpack(py_in, &imp) != 0) goto cfail;
    {
        index_t nat = device_in ? (index_t)imp.nat
                                : (index_t)PyArray_DIM((PyArrayObject *)a_pos, 0);
        real_t cutoff = 0.0;
        const real_t *per_atom = NULL, *per_type_sq = NULL;
        index_t ncutoffs = 0;
        std::vector<real_t> pt_storage;
        if (resolve_cutoff(py_cut, &a_cut, &cutoff, &per_atom, &per_type_sq,
                           &ncutoffs, pt_storage) != 0)
            goto cfail;

        const real_t *origin = (const real_t *)PyArray_DATA((PyArrayObject *)a_origin);
        const real_t *cell = (const real_t *)PyArray_DATA((PyArrayObject *)a_cell);
        const real_t *inv = (const real_t *)PyArray_DATA((PyArrayObject *)a_inv);
        const npy_bool *pb = (const npy_bool *)PyArray_DATA((PyArrayObject *)a_pbc);
        bool pbc[3] = {(bool)pb[0], (bool)pb[1], (bool)pb[2]};
        const real_t *r_host = (const real_t *)PyArray_DATA((PyArrayObject *)a_pos);
        const index_t *types =
            a_types ? (const index_t *)PyArray_DATA((PyArrayObject *)a_types) : NULL;

        NeighbourListRequest req;
        req.cell_origin = origin;
        req.cell = cell;
        req.inv_cell = inv;
        req.pbc = pbc;
        req.nat = nat;
        req.positions = device_in ? imp.data : r_host;
        req.positions_on_device = device_in;
        req.cutoff = cutoff;
        req.per_atom_cutoff = per_atom;
        req.per_type_cutoff_sq = per_type_sq;
        req.ncutoffs = ncutoffs;
        req.types = types;
        req.device_id = device_in ? imp.device_id : device_id;

        NeighbourListDevice dev;
        error_t st = neighbour_count_gpu_device(req, dev);
        imp.release();
        if (st != NL_SUCCESS) {
            if (has_error) PyErr_SetString(PyExc_RuntimeError, error_string);
            goto cfail;
        }
        const int dev_id = req.device_id >= 0 ? req.device_id : current_device_id();
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
    imp.release();
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
