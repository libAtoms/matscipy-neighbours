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
#include "segment_sum.hh"
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

/* Whether this build's runtime can address memory on DLPack device type
   `dev_type`: CUDA or CUDA-managed for the CUDA build, ROCm for the HIP build.
   Anything else would be an illegal access. Sets a TypeError naming `what` if
   not. */
bool check_device_access(int dev_type, const char *what) {
#if defined(MATSCIPY_ENABLE_CUDA)
    const bool ok_device = dev_type == kDLCUDA || dev_type == kDLCUDAManaged;
    const char *backend = "CUDA";
#elif defined(MATSCIPY_ENABLE_HIP)
    const bool ok_device = dev_type == kDLROCM;
    const char *backend = "HIP";
#else
    const bool ok_device = false;
    const char *backend = "no GPU";
#endif
    if (!ok_device) {
        PyErr_Format(PyExc_TypeError,
                     "%s live on DLPack device type %d, which the %s backend of "
                     "this build cannot access",
                     what, dev_type, backend);
    }
    return ok_device;
}

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
    /* The pointer is handed to this build's runtime. */
    if (!check_device_access(static_cast<int>(t.device.device_type),
                             "device positions"))
        return -1;
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

/* Import a 1-D int64 device array (an index array such as a pair list's i)
   via its __dlpack__. Returns 0 and fills `imp` (owning the tensor, nat = its
   length) and `*data` on success; -1 with a Python error set otherwise. */
int import_index_dlpack(PyObject *arr, ImportedDLPack *imp,
                        const index_t **data) {
    PyRef cap(PyObject_CallMethod(arr, "__dlpack__", NULL));
    if (!cap) return -1;
    if (!PyCapsule_IsValid(cap.get(), "dltensor")) {
        PyErr_SetString(PyExc_TypeError,
                        "device index array did not yield an unversioned "
                        "DLPack capsule");
        return -1;
    }
    auto *mt = static_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(cap.get(), "dltensor"));
    if (!mt) return -1;
    const DLTensor &t = mt->dl_tensor;
    const bool ok = t.dtype.code == kDLInt && t.dtype.bits == kIntBits &&
                    t.dtype.lanes == 1 && t.ndim == 1 &&
                    (t.strides == nullptr || t.strides[0] == 1 || t.shape[0] <= 1);
    if (!ok) {
        PyErr_SetString(PyExc_TypeError,
                        "device index array must be a contiguous 1-D int64 "
                        "array");
        return -1;
    }
    if (!check_device_access(static_cast<int>(t.device.device_type),
                             "device index array"))
        return -1;
    imp->mt = mt;
    *data = reinterpret_cast<const index_t *>(static_cast<char *>(t.data) +
                                              t.byte_offset);
    imp->device_type = static_cast<int>(t.device.device_type);
    imp->device_id = t.device.device_id;
    imp->nat = t.shape[0];
    PyCapsule_SetName(cap.get(), "used_dltensor");
    return 0;
}

/* A C-contiguous tensor of any dtype and device imported through DLPack (host
   or device). Owns the consumed managed tensor until destruction. */
struct ImportedTensor {
    DLManagedTensor *mt = nullptr;
    void *data = nullptr;
    DLDataType dtype{};
    std::vector<int64_t> shape;
    int device_type = 0;
    int device_id = 0;
    ImportedTensor() = default;
    ImportedTensor(const ImportedTensor &) = delete;
    ImportedTensor &operator=(const ImportedTensor &) = delete;
    ~ImportedTensor() {
        if (mt && mt->deleter) mt->deleter(mt);
    }
    bool on_host() const { return device_type == kDLCPU; }
};

/* Import `arr` (anything with __dlpack__) as a C-contiguous tensor named
   `what` in error messages. Device tensors must be addressable by this build's
   runtime. Returns 0, or -1 with a Python error set. */
int import_tensor_dlpack(PyObject *arr, const char *what, ImportedTensor *imp) {
    PyRef cap(PyObject_CallMethod(arr, "__dlpack__", NULL));
    if (!cap) return -1;
    if (!PyCapsule_IsValid(cap.get(), "dltensor")) {
        PyErr_Format(PyExc_TypeError,
                     "%s did not yield an unversioned DLPack capsule", what);
        return -1;
    }
    auto *mt = static_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(cap.get(), "dltensor"));
    if (!mt) return -1;
    const DLTensor &t = mt->dl_tensor;
    bool contiguous = t.dtype.lanes == 1;
    bool empty = false;  /* no elements: the strides are meaningless */
    for (int k = 0; k < t.ndim; k++) empty = empty || t.shape[k] == 0;
    if (contiguous && t.strides && !empty) {
        int64_t expect = 1;
        for (int k = t.ndim - 1; k >= 0; k--) {
            if (t.shape[k] > 1 && t.strides[k] != expect) contiguous = false;
            expect *= t.shape[k];
        }
    }
    if (!contiguous) {
        PyErr_Format(PyExc_TypeError, "%s must be C-contiguous", what);
        return -1;
    }
    const int dev_type = static_cast<int>(t.device.device_type);
    if (dev_type != kDLCPU && !check_device_access(dev_type, what)) return -1;
    imp->mt = mt;
    imp->data = static_cast<char *>(t.data) + t.byte_offset;
    imp->dtype = t.dtype;
    imp->shape.assign(t.shape, t.shape + t.ndim);
    imp->device_type = dev_type;
    imp->device_id = t.device.device_id;
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

/* Dense fixed-capacity (n x K) neighbour list: returns (idx, dist, shift,
   count) as DLPack capsules plus an overflow flag; `quantities` (QUANTITY_*
   flags) selects dist and/or shift, the other is None. CPU backend yields host
   capsules; GPU backend yields device capsules. */
PyObject *neighbour_matrix_dlpack_impl(PyObject *, PyObject *args) {
    PyObject *py_origin, *py_cell, *py_inv, *py_pbc, *py_pos, *py_cut;
    int max_neighbours = 0;
    PyObject *py_types = NULL, *py_in = NULL;
    int backend = 0, device_id = -1, quantities = QUANTITY_DISTVEC;

    if (!PyArg_ParseTuple(args, "OOOOOOi|OiOii", &py_origin, &py_cell, &py_inv,
                          &py_pbc, &py_pos, &py_cut, &max_neighbours, &py_types,
                          &backend, &py_in, &device_id, &quantities))
        return NULL;
    const bool wD = quantities & QUANTITY_DISTVEC;
    const bool wS = quantities & QUANTITY_SHIFT;

#if !(defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP))
    if (backend != 0) return no_gpu_backend();
#endif

    Inputs in;
    if (in.parse(py_origin, py_cell, py_inv, py_pbc, py_pos, py_cut, py_types,
                 py_in, backend) != 0)
        return NULL;

    const index_t K = max_neighbours;
    const int64_t n64 = in.g.nat, K64 = K;
    PyRef cap_idx, cap_dist, cap_shift, cap_count;
    bool overflow = false;

    if (backend == 0) {
        NeighbourMatrix nm;
        error_t st = neighbour_matrix(in.g.origin_data(), in.g.cell_data(),
                                      in.g.inv_data(), in.g.periodic, in.g.nat,
                                      in.g.pos_data(), in.cutoff, in.per_atom,
                                      in.per_type_sq, in.ncutoffs,
                                      in.g.types_data(), K, nm,
                                      CellOrder::Linear, quantities);
        if (st != NL_SUCCESS) {
            raise_core_error(st);
            return NULL;
        }
        overflow = nm.overflow;
        cap_idx = host_capsule(std::move(nm.idx), 2, n64, K64, kDLInt, kIntBits);
        if (wD)
            cap_dist = host_capsule(std::move(nm.dist), 3, n64, K64, kDLFloat,
                                    kRealBits, 3);
        if (wS)
            cap_shift = host_capsule(std::move(nm.shift), 3, n64, K64, kDLInt,
                                     kIntBits, 3);
        cap_count = host_capsule(std::move(nm.count), 1, n64, 1, kDLInt, kIntBits);
    } else {
#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
        NeighbourListRequest req = in.request(0, device_id);
        NeighbourMatrixDevice dev;
        error_t st = neighbour_matrix_gpu_device(req, K, dev, quantities);
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
        if (wD)
            cap_dist = device_capsule(std::move(dev.dist), 3, n64, K64, kDLFloat,
                                      kRealBits, dev_id, 3);
        if (wS)
            cap_shift = device_capsule(std::move(dev.shift), 3, n64, K64,
                                       kDLInt, kIntBits, dev_id, 3);
        cap_count = device_capsule(std::move(dev.count), 1, n64, 1, kDLInt,
                                   kIntBits, dev_id);
#endif
    }
    if (!cap_idx || !cap_count || (wD && !cap_dist) || (wS && !cap_shift))
        return NULL;
    return PyTuple_Pack(5, cap_idx.get(), wD ? cap_dist.get() : Py_None,
                        wS ? cap_shift.get() : Py_None, cap_count.get(),
                        overflow ? Py_True : Py_False);
}

/* Row-start array of a device pair list's sorted first-index array, computed
   on the array's GPU and returned there as a DLPack capsule. */
PyObject *first_neighbours_dlpack_impl(PyObject *, PyObject *args) {
#if !(defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP))
    (void)args;
    return no_gpu_backend();
#else
    Py_ssize_t n_arg;
    PyObject *py_i;
    if (!PyArg_ParseTuple(args, "nO", &n_arg, &py_i)) return NULL;
    ImportedDLPack imp;
    const index_t *i_n = nullptr;
    if (import_index_dlpack(py_i, &imp, &i_n) != 0) return NULL;
    Array<index_t, DeviceSpace> seed;
    error_t st = first_neighbours_gpu_device(static_cast<index_t>(n_arg),
                                             static_cast<index_t>(imp.nat), i_n,
                                             seed, imp.device_id);
    imp.release();
    if (st != NL_SUCCESS) {
        raise_core_error(st);
        return NULL;
    }
    const int64_t len = static_cast<int64_t>(seed.size());
    return device_capsule(std::move(seed), 1, len, 1, kDLInt, kIntBits,
                          imp.device_id);
#endif
}

/* Per-segment sums of `values` (rows x d) over the row starts `seed`, plus the
   total if requested; host or device, matching the input. */
template <typename T>
PyObject *segment_sum_typed(const ImportedTensor &values,
                            const ImportedTensor &seed, int64_t nrows,
                            int64_t d, bool want_total, uint8_t code) {
    const index_t n = static_cast<index_t>(seed.shape[0]) - 1;
    const auto *s = static_cast<const index_t *>(seed.data);
    const auto *v = static_cast<const T *>(values.data);
    constexpr uint8_t bits = sizeof(T) * 8;
    PyRef cap_out, cap_total;
    if (values.on_host()) {
        std::vector<T> out(static_cast<std::size_t>(n) * d), total(d);
        error_t st = segment_sum<T>(n, s, nrows, d, v, out.data(),
                                    want_total ? total.data() : nullptr);
        if (st != NL_SUCCESS) {
            raise_core_error(st);
            return NULL;
        }
        cap_out = host_capsule(std::move(out), 2, n, d, code, bits);
        if (want_total) cap_total = host_capsule(std::move(total), 1, d, 1, code, bits);
    } else {
#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
        Array<T, DeviceSpace> out, total;
        error_t st = segment_sum_gpu_device<T>(n, s, nrows, d, v, out,
                                               want_total ? &total : nullptr,
                                               values.device_id);
        if (st != NL_SUCCESS) {
            raise_core_error(st);
            return NULL;
        }
        cap_out = device_capsule(std::move(out), 2, n, d, code, bits,
                                 values.device_id);
        if (want_total)
            cap_total = device_capsule(std::move(total), 1, d, 1, code, bits,
                                       values.device_id);
#else
        return no_gpu_backend();
#endif
    }
    if (!cap_out || (want_total && !cap_total)) return NULL;
    return PyTuple_Pack(2, cap_out.get(), want_total ? cap_total.get() : Py_None);
}

PyObject *segment_sum_dlpack_impl(PyObject *, PyObject *args) {
    PyObject *py_values, *py_seed;
    int want_total = 0;
    if (!PyArg_ParseTuple(args, "OO|p", &py_values, &py_seed, &want_total))
        return NULL;
    ImportedTensor values, seed;
    if (import_tensor_dlpack(py_values, "values", &values) != 0) return NULL;
    if (import_tensor_dlpack(py_seed, "seed", &seed) != 0) return NULL;
    if (seed.shape.size() != 1 || seed.shape[0] < 1 ||
        seed.dtype.code != kDLInt || seed.dtype.bits != kIntBits) {
        PyErr_SetString(PyExc_TypeError,
                        "seed must be a 1-D int64 array of length n + 1");
        return NULL;
    }
    if (values.shape.empty()) {
        PyErr_SetString(PyExc_TypeError, "values must have at least one dimension");
        return NULL;
    }
    if (values.device_type != seed.device_type ||
        (!values.on_host() && values.device_id != seed.device_id)) {
        PyErr_SetString(PyExc_ValueError,
                        "values and seed must be on the same device");
        return NULL;
    }
    const int64_t nrows = values.shape[0];
    int64_t d = 1;
    for (std::size_t k = 1; k < values.shape.size(); k++) d *= values.shape[k];
    const uint8_t code = values.dtype.code, bits = values.dtype.bits;
    if (code == kDLFloat && bits == 64)
        return segment_sum_typed<double>(values, seed, nrows, d, want_total, code);
    if (code == kDLFloat && bits == 32)
        return segment_sum_typed<float>(values, seed, nrows, d, want_total, code);
    if (code == kDLInt && bits == 64)
        return segment_sum_typed<std::int64_t>(values, seed, nrows, d, want_total, code);
    if (code == kDLInt && bits == 32)
        return segment_sum_typed<std::int32_t>(values, seed, nrows, d, want_total, code);
    PyErr_SetString(PyExc_TypeError,
                    "values must be float32, float64, int32 or int64");
    return NULL;
}

/* Return the GPU memory cached by the library's allocator to the driver
   (no-op without a GPU backend). */
PyObject *empty_gpu_cache_impl(PyObject *, PyObject *) {
#if defined(MATSCIPY_ENABLE_CUDA) || defined(MATSCIPY_ENABLE_HIP)
    empty_gpu_cache();
#endif
    Py_RETURN_NONE;
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

PyObject *py_first_neighbours_dlpack(PyObject *self, PyObject *args) {
    return guarded(first_neighbours_dlpack_impl, self, args);
}

PyObject *py_segment_sum_dlpack(PyObject *self, PyObject *args) {
    return guarded(segment_sum_dlpack_impl, self, args);
}

PyObject *py_empty_gpu_cache(PyObject *self, PyObject *args) {
    return guarded(empty_gpu_cache_impl, self, args);
}
