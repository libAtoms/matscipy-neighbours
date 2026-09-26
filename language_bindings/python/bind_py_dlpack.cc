/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 * Copyright (2014-2026) James Kermode, University of Warwick
 *                       Lars Pastewka, University of Freiburg
 *                       and others (see toplevel AUTHORS file)
 *
 * DLPack import/export. Builds the neighbour list on the requested backend and
 * hands each output array to Python as a DLPack "dltensor" capsule that OWNS its
 * buffer (host std::vector or device Array<T, DeviceSpace>); any DLPack consumer
 * (numpy/cupy/torch/jax) then wraps it zero-copy. Device input is read the same
 * way — via the array's __dlpack__ — so the GPU path is framework-agnostic and
 * never round-trips positions through the host.
 *
 * Every entry point runs inside guarded() (bind_py_common.hh) and holds its
 * references in PyRef / RAII structs, so early returns and C++ exceptions can
 * neither leak nor terminate the interpreter.
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL MATSCIPY_ARRAY_API
#define NO_IMPORT_ARRAY
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include <numpy/arrayobject.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "bind_py_common.hh"
#include "bind_py_dlpack.hh"
#include "dlpack.h"

#include "error.hh"
#include "memory_space.hh"
#include "neighbour_list.hh"
#include "neighbour_list_gpu.hh"
#include "types.hh"

using namespace matscipy;
using namespace matscipy_py;

namespace {

/* ----------------------------------------------------------- DLPack export */

/* Owns the buffer behind a DLManagedTensor and the shape array. `keep` holds a
   shared_ptr to the std::vector / Array, released when the manager is deleted. */
struct DLPackManager {
    DLManagedTensor mt;
    int64_t shape[3];
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
                       int64_t d0, int64_t d1, int64_t d2, uint8_t code,
                       uint8_t bits, DLDeviceType dev, int dev_id) {
    auto *m = new DLPackManager();
    m->shape[0] = d0;
    m->shape[1] = d1;
    m->shape[2] = d2;
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
                       uint8_t code, uint8_t bits, int64_t d2 = 1) {
    auto keep = std::make_shared<std::vector<T>>(std::move(v));
    return make_capsule(keep, keep->data(), ndim, d0, d1, d2, code, bits, kDLCPU,
                        0);
}

#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
template <typename T>
PyObject *device_capsule(Array<T, DeviceSpace> &&a, int ndim, int64_t d0,
                         int64_t d1, uint8_t code, uint8_t bits, int dev_id,
                         int64_t d2 = 1) {
    auto keep = std::make_shared<Array<T, DeviceSpace>>(std::move(a));
    void *data = keep->data();
    /* DeviceType codes equal the DLPack device codes by construction. */
    const auto dev = static_cast<DLDeviceType>(static_cast<int>(DeviceSpace::device));
    return make_capsule(keep, data, ndim, d0, d1, d2, code, bits, dev, dev_id);
}
#endif

/* ----------------------------------------------------------- DLPack import */

/* A device positions array imported through DLPack. Owns the consumed managed
   tensor; release() (or destruction) frees it once the synchronous build has
   read the data. */
struct ImportedDLPack {
    DLManagedTensor *mt = nullptr;
    const real_t *data = nullptr;
    int device_type = 0;
    int device_id = 0;
    npy_intp nat = -1;  /* -1: no device input */
    ImportedDLPack() = default;
    ImportedDLPack(const ImportedDLPack &) = delete;
    ImportedDLPack &operator=(const ImportedDLPack &) = delete;
    ~ImportedDLPack() { release(); }
    void release() {
        if (mt && mt->deleter) mt->deleter(mt);
        mt = nullptr;
    }
};

/* Import an (n, 3) float64 device array via its __dlpack__. Returns 0 and fills
   `imp` (owning the tensor) on success; -1 with a Python error set otherwise. */
int import_positions_dlpack(PyObject *arr, ImportedDLPack *imp) {
    PyRef cap(PyObject_CallMethod(arr, "__dlpack__", NULL));
    if (!cap) return -1;
    if (!PyCapsule_IsValid(cap.get(), "dltensor")) {
        PyErr_SetString(PyExc_TypeError,
                        "device positions did not yield an unversioned DLPack "
                        "capsule");
        return -1;
    }
    auto *mt = static_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(cap.get(), "dltensor"));
    if (!mt) return -1;
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
        /* Not consumed: the capsule keeps its "dltensor" name and its own
           destructor frees the managed tensor. */
        return -1;
    }
    imp->mt = mt;
    imp->data = reinterpret_cast<const real_t *>(
        static_cast<char *>(t.data) + t.byte_offset);
    imp->device_type = static_cast<int>(t.device.device_type);
    imp->device_id = t.device.device_id;
    imp->nat = t.shape[0];
    /* Consume: the producer's capsule destructor must not also free it. */
    PyCapsule_SetName(cap.get(), "used_dltensor");
    return 0;
}

/* ------------------------------------------------------------ shared input */

/* Everything the three entry points parse in common: the (optional) device
   positions, the validated host geometry, and the cutoff specification. */
struct Inputs {
    ImportedDLPack imp;
    GeometryArrays g;
    PyRef a_cut;
    real_t cutoff = 0.0;
    const real_t *per_atom = NULL;
    const real_t *per_type_sq = NULL;
    index_t ncutoffs = 0;
    std::vector<real_t> per_type_storage;

    bool device_in() const { return imp.nat >= 0; }
    const real_t *positions() const { return device_in() ? imp.data : g.pos_data(); }

    /* Returns 0, or -1 with an exception set. */
    int parse(PyObject *py_origin, PyObject *py_cell, PyObject *py_inv,
              PyObject *py_pbc, PyObject *py_pos, PyObject *py_cut,
              PyObject *py_types, PyObject *py_in, int backend) {
        const bool have_device = py_in && py_in != Py_None;
        if (have_device && backend == 0) {
            PyErr_SetString(PyExc_TypeError,
                            "device-resident positions require the GPU backend.");
            return -1;
        }
        if (have_device && import_positions_dlpack(py_in, &imp) != 0) return -1;
        if (parse_geometry(py_origin, py_cell, py_inv, py_pbc, py_pos, py_types,
                           have_device ? imp.nat : -1, g) != 0)
            return -1;
        return resolve_cutoff(py_cut, g.nat, a_cut, &cutoff, &per_atom,
                              &per_type_sq, &ncutoffs, per_type_storage);
    }

#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
    NeighbourListRequest request(int quantities, int device_id) const {
        NeighbourListRequest req;
        req.quantities = quantities;
        req.cell_origin = g.origin_data();
        req.cell = g.cell_data();
        req.inv_cell = g.inv_data();
        req.pbc = g.periodic;
        req.nat = g.nat;
        req.positions = positions();
        req.positions_on_device = device_in();
        req.cutoff = cutoff;
        req.per_atom_cutoff = per_atom;
        req.per_type_cutoff_sq = per_type_sq;
        req.ncutoffs = ncutoffs;
        req.types = g.types_data();
        req.device_id = device_in() ? imp.device_id : device_id;
        return req;
    }
#endif
};

#if !(defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP))
PyObject *no_gpu_backend() {
    PyErr_SetString(PyExc_RuntimeError,
                    "GPU backend requested but the extension was built without a "
                    "GPU backend (-DENABLE_CUDA=ON).");
    return NULL;
}
#endif

/* ------------------------------------------------------------ entry points */

PyObject *neighbour_list_dlpack_impl(PyObject *, PyObject *args) {
    PyObject *py_quant, *py_origin, *py_cell, *py_inv, *py_pbc, *py_pos, *py_cut;
    PyObject *py_types = NULL;
    int backend = 0;          /* 0 = CPU/host, 1 = GPU/device */
    PyObject *py_in = NULL;   /* device positions object (has __dlpack__) or None */
    int device_id = -1;       /* GPU to run on / report; -1 = current */

    if (!PyArg_ParseTuple(args, "O!OOOOOO|OiOi", &PyUnicode_Type, &py_quant,
                          &py_origin, &py_cell, &py_inv, &py_pbc, &py_pos,
                          &py_cut, &py_types, &backend, &py_in, &device_id))
        return NULL;

#if !(defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP))
    if (backend != 0) return no_gpu_backend();
#endif

    Inputs in;
    if (in.parse(py_origin, py_cell, py_inv, py_pbc, py_pos, py_cut, py_types,
                 py_in, backend) != 0)
        return NULL;

    PyRef py_bquant(PyUnicode_AsASCIIString(py_quant));
    if (!py_bquant) return NULL;
    const char *quantities = PyBytes_AS_STRING(py_bquant.get());
    int flags = 0;
    if (quantity_flags(quantities, &flags) != 0) return NULL;

    const Py_ssize_t nq = static_cast<Py_ssize_t>(std::strlen(quantities));
    PyRef py_ret(PyTuple_New(nq));
    if (!py_ret) return NULL;

    if (backend == 0) {
        /* CPU backend: host buffers wrapped as kDLCPU capsules. */
        NeighbourList nl;
        error_t st = neighbour_list(flags, in.g.origin_data(), in.g.cell_data(),
                                    in.g.inv_data(), in.g.periodic, in.g.nat,
                                    in.g.pos_data(), in.cutoff, in.per_atom,
                                    in.per_type_sq, in.ncutoffs,
                                    in.g.types_data(), nl);
        if (st != NL_SUCCESS) {
            raise_core_error(st);
            return NULL;
        }
        const int64_t np = nl.npairs;
        Py_ssize_t pos = 0;
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
            if (!cap) return NULL;
            PyTuple_SET_ITEM(py_ret.get(), pos, cap);
        }
    } else {
#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
        /* GPU backend: results stay on the device, wrapped as device capsules.
           Device input is used in place; otherwise host positions upload. */
        NeighbourListRequest req = in.request(flags, device_id);
        NeighbourListDevice dev;
        error_t st = neighbour_list_gpu_device(req, dev);
        in.imp.release();  /* input consumed; result lives in `dev` */
        if (st != NL_SUCCESS) {
            raise_core_error(st);
            return NULL;
        }
        const int64_t np = dev.npairs;
        const int dev_id = req.device_id >= 0 ? req.device_id
                                              : current_device_id();
        Py_ssize_t pos = 0;
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
            if (!cap) return NULL;
            PyTuple_SET_ITEM(py_ret.get(), pos, cap);
        }
#endif
    }
    return py_ret.release();
}

PyObject *coordination_dlpack_impl(PyObject *, PyObject *args) {
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

    Inputs in;
    if (in.parse(py_origin, py_cell, py_inv, py_pbc, py_pos, py_cut, py_types,
                 py_in, /*backend=*/1) != 0)
        return NULL;

    NeighbourListRequest req = in.request(0, device_id);
    NeighbourListDevice dev;
    error_t st = neighbour_count_gpu_device(req, dev);
    in.imp.release();
    if (st != NL_SUCCESS) {
        raise_core_error(st);
        return NULL;
    }
    const int dev_id = req.device_id >= 0 ? req.device_id : current_device_id();
    return device_capsule(std::move(dev.counts), 1, in.g.nat, 1, kDLInt,
                          kIntBits, dev_id);
#endif
}

/* Dense fixed-capacity (n x K) neighbour list: returns (idx, dist, count) as
   DLPack capsules plus an overflow flag. CPU backend yields host capsules;
   GPU backend yields device capsules. */
PyObject *neighbour_matrix_dlpack_impl(PyObject *, PyObject *args) {
    PyObject *py_origin, *py_cell, *py_inv, *py_pbc, *py_pos, *py_cut;
    int max_neighbours = 0;
    PyObject *py_types = NULL, *py_in = NULL;
    int backend = 0, device_id = -1;

    if (!PyArg_ParseTuple(args, "OOOOOOi|OiOi", &py_origin, &py_cell, &py_inv,
                          &py_pbc, &py_pos, &py_cut, &max_neighbours, &py_types,
                          &backend, &py_in, &device_id))
        return NULL;

#if !(defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP))
    if (backend != 0) return no_gpu_backend();
#endif

    Inputs in;
    if (in.parse(py_origin, py_cell, py_inv, py_pbc, py_pos, py_cut, py_types,
                 py_in, backend) != 0)
        return NULL;

    const index_t K = max_neighbours;
    const int64_t n64 = in.g.nat, K64 = K;
    PyRef cap_idx, cap_dist, cap_count;
    bool overflow = false;

    if (backend == 0) {
        NeighbourMatrix nm;
        error_t st = neighbour_matrix(in.g.origin_data(), in.g.cell_data(),
                                      in.g.inv_data(), in.g.periodic, in.g.nat,
                                      in.g.pos_data(), in.cutoff, in.per_atom,
                                      in.per_type_sq, in.ncutoffs,
                                      in.g.types_data(), K, nm);
        if (st != NL_SUCCESS) {
            raise_core_error(st);
            return NULL;
        }
        overflow = nm.overflow;
        cap_idx = host_capsule(std::move(nm.idx), 2, n64, K64, kDLInt, kIntBits);
        cap_dist = host_capsule(std::move(nm.dist), 3, n64, K64, kDLFloat,
                                kRealBits, 3);
        cap_count = host_capsule(std::move(nm.count), 1, n64, 1, kDLInt, kIntBits);
    } else {
#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
        NeighbourListRequest req = in.request(0, device_id);
        NeighbourMatrixDevice dev;
        error_t st = neighbour_matrix_gpu_device(req, K, dev);
        in.imp.release();
        if (st != NL_SUCCESS) {
            raise_core_error(st);
            return NULL;
        }
        overflow = dev.overflow;
        const int dev_id = req.device_id >= 0 ? req.device_id
                                              : current_device_id();
        cap_idx = device_capsule(std::move(dev.idx), 2, n64, K64, kDLInt,
                                 kIntBits, dev_id);
        cap_dist = device_capsule(std::move(dev.dist), 3, n64, K64, kDLFloat,
                                  kRealBits, dev_id, 3);
        cap_count = device_capsule(std::move(dev.count), 1, n64, 1, kDLInt,
                                   kIntBits, dev_id);
#endif
    }
    if (!cap_idx || !cap_dist || !cap_count) return NULL;
    return PyTuple_Pack(4, cap_idx.get(), cap_dist.get(), cap_count.get(),
                        overflow ? Py_True : Py_False);
}

}  // namespace

/* Exported entry points: exception-guarded wrappers around the bodies above. */

PyObject *py_neighbour_list_dlpack(PyObject *self, PyObject *args) {
    return guarded(neighbour_list_dlpack_impl, self, args);
}

PyObject *py_coordination_dlpack(PyObject *self, PyObject *args) {
    return guarded(coordination_dlpack_impl, self, args);
}

PyObject *py_neighbour_matrix_dlpack(PyObject *self, PyObject *args) {
    return guarded(neighbour_matrix_dlpack_impl, self, args);
}
