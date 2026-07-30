"""Experimental: device-resident neighbour-list adapter (optional capability).

This adapts matscipy-neighbours' native DLPack device cell list to ASE's
*experimental* device neighbour-list protocols
(:mod:`ase._4.plugins.neighborlist_device`): :class:`DeviceNeighborList` and
:class:`DeviceNeighborResult`.  A device-resident calculator discovers it via
``isinstance(backend, DeviceNeighborList)`` and exchanges edge data **on-device
via DLPack**, with no host round-trip per step.

GPU-only (CUDA/HIP).  The Verlet-skin **update check** (:meth:`needs_rebuild`
and the build-time reference snapshot) runs entirely in the native C++/CUDA
kernels -- no CuPy.  CuPy is still used only by :meth:`build_device` to return
the edge arrays as CuPy arrays and for the padded mask; a torch/JAX consumer
that only drives the update loop never imports it.  The host-resident path is
unchanged and lives in ``matscipy_neighbours._ase_plugin.neighbor_list``;
routing a GPU model through that host path costs a host round-trip every rebuild
-- residency requires *this* device capability.

``differentiable`` is ``False``: the kernel emits ``D`` numerically, not inside
an autograd graph.  Autograd consumers should request only ``i, j, S`` and
recompute ``D`` themselves.

Implementation notes
--------------------
* The :meth:`needs_rebuild` reduction (max squared displacement vs the cached
  build-time positions) is the native ``_ext.needs_rebuild`` kernel: a CUB/
  hipCUB transform-then-``DeviceReduce::Max`` over per-atom squared displacement
  plus an on-device threshold against ``skin**2``, returning a 1-element uint8
  device DLPack scalar.  No value is read back to the host (residency
  preserved).  The reference positions are snapshotted on the device by
  ``_ext.clone_device`` into a reusable handle.
* The **padded** path (``max_capacity`` given) wraps ``neighbour_matrix``'s
  dense ``(idx, dist, count)`` output and surfaces the overflow flag instead of
  raising, so a compiled consumer can reallocate rather than crash.  The dense
  format carries no cell shifts, so the canonical COO ``get("i"/"j"/"S"/"D")``
  contract is guaranteed on the **tight** path only; the padded result exposes
  ``get("idx"/"dist"/"count")`` plus :meth:`mask`.  (COO-vs-dense is a flagged
  design fork.)
* The **padded** path (``max_capacity`` given) wraps ``neighbour_matrix``'s
  dense ``(idx, dist, count)`` output and surfaces the overflow flag instead of
  raising, so a compiled consumer can reallocate rather than crash.  The dense
  format carries no cell shifts, so the canonical COO ``get("i"/"j"/"S"/"D")``
  contract is guaranteed on the **tight** path only; the padded result exposes
  ``get("idx"/"dist"/"count")`` plus :meth:`mask`.  (COO-vs-dense is a flagged
  design fork.)
"""
from __future__ import annotations

import numpy as np

from . import neighbours as _nb
from .neighbours import _DLPACK_CPU, _DLPACK_CUDA, _ext, neighbour_list


def _cupy():
    import cupy as cp  # imported lazily; this adapter is GPU-only
    return cp


def _to_host(x):
    """Bring a small (cell-sized) array to host as a contiguous float ndarray.

    Cells are 3x3, so this is negligible and not the residency concern (which is
    the O(n_atoms) positions / O(n_edges) edges).  Accepts host arrays, CuPy
    arrays, or any DLPack object.
    """
    if x is None:
        return None
    dev = getattr(x, '__dlpack_device__', None)
    if dev is not None and dev()[0] != _DLPACK_CPU:
        cp = _cupy()
        return np.ascontiguousarray(cp.asnumpy(cp.from_dlpack(x)), dtype=float)
    return np.ascontiguousarray(np.asarray(x), dtype=float)


class MatscipyDeviceResult:
    """On-device neighbour data; satisfies ``DeviceNeighborResult``.

    Arrays are CuPy device arrays (DLPack-exporting).  A returned view aliases
    backend buffers and is only valid until the next ``build_device``.
    """

    def __init__(self, arrays, *, padded, did_overflow, n_edges, mask=None):
        self._arrays = arrays          # name -> device array (DLPack-exporting)
        self._padded = bool(padded)
        self._did_overflow = did_overflow
        self._n_edges = n_edges
        self._mask = mask

    @property
    def n_edges(self):
        # Logical edge count; an int per the protocol.  For the dense path this
        # is a device->host reduction of the per-atom counts, evaluated lazily.
        if not isinstance(self._n_edges, int):
            self._n_edges = int(self._n_edges)
        return self._n_edges

    @property
    def did_overflow(self):
        return self._did_overflow

    @property
    def padded(self):
        return self._padded

    def get(self, quantity):
        """Return one quantity as a DLPack-exporting device array (keyed by NAME).

        Tight path: ``"i"``, ``"j"``, ``"S"``, ``"D"`` (whichever were built).
        Padded path: ``"idx"``, ``"dist"``, ``"count"`` (the dense layout has no
        shifts, so COO ``"S"`` is unavailable -- use the tight path for COO).
        """
        try:
            return self._arrays[quantity]
        except KeyError:
            available = ', '.join(sorted(self._arrays)) or '(none)'
            raise KeyError(
                f'quantity {quantity!r} not available on this result; '
                f'have {available}. '
                + ('The padded (max_capacity) path exposes the dense '
                   'idx/dist/count layout, which has no cell shifts; request '
                   'the tight path (max_capacity=None) for COO i/j/S/D.'
                   if self._padded else
                   'It was not among the requested quantities.')) from None

    def mask(self):
        return self._mask


class MatscipyDeviceNeighborList:
    """matscipy-neighbours device backend; satisfies ``DeviceNeighborList``."""

    differentiable = False

    def __init__(self, device_id=0):
        if not getattr(_ext, '_has_gpu', 0):
            raise RuntimeError(
                'matscipy-neighbours was built without a GPU backend '
                '(-DENABLE_CUDA=ON / -DENABLE_HIP=ON); the device capability '
                'is unavailable.')
        self._device_type = int(getattr(_ext, '_device_type', _DLPACK_CUDA))
        self._device_id = int(device_id)
        self._ref = None  # cached build-time positions (device), for needs_rebuild

    @property
    def device(self):
        return (self._device_type, self._device_id)

    def build_device(self, positions, cell, pbc, cutoff, quantities='ijS', *,
                     self_interaction=False, max_capacity=None, stream=None):
        if self_interaction:
            raise NotImplementedError(
                'the matscipy-neighbours backend does not support '
                'self_interaction=True')
        cp = _cupy()
        pos = cp.from_dlpack(positions)           # device array (zero-copy view)
        self._device_id = int(pos.device.id)
        # Snapshot build-time positions on the device for needs_rebuild, via the
        # native C++ kernel (a reusable, opaque handle -- not a one-shot DLPack
        # capsule -- since the consumer may overwrite its positions in place each
        # step). No CuPy copy.
        self._ref = _ext.clone_device(pos)
        cell_h = _to_host(cell)
        pbc_t = tuple(bool(b) for b in pbc)

        if max_capacity is None:
            return self._build_tight(pos, cell_h, pbc_t, cutoff, quantities, cp)
        return self._build_padded(pos, cell_h, pbc_t, cutoff, max_capacity, cp)

    def _build_tight(self, pos, cell_h, pbc_t, cutoff, quantities, cp):
        invalid = set(quantities) - set('ijdDS')
        if invalid or not quantities:
            raise ValueError(
                f'quantities must be a non-empty subset of "ijdDS"; '
                f'got {quantities!r}.')
        outs = neighbour_list(quantities, positions=pos, cell=cell_h,
                              pbc=pbc_t, cutoff=cutoff, array_namespace=cp)
        if len(quantities) == 1:
            outs = (outs,)
        # Keyed by NAME regardless of the order neighbour_list returned them.
        arrays = {name: arr for name, arr in zip(quantities, outs)}
        n_edges = int(arrays[quantities[0]].shape[0])
        return MatscipyDeviceResult(arrays, padded=False, did_overflow=False,
                                    n_edges=n_edges)

    def _build_padded(self, pos, cell_h, pbc_t, cutoff, max_capacity, cp):
        # Reproduce neighbour_matrix's body but surface the overflow flag rather
        # than raising, so a compiled consumer can reallocate (JAX-MD style).
        nat = int(pos.shape[0])
        co, ce, inv, pb, nums = _nb._host_metadata(
            cell_h, pbc_t, None, None, nat, positions=None)
        rc, nums = _nb._resolve_cutoff(cutoff, nums)
        host_pos = np.empty((0, 3), dtype=float)
        idx_cap, dist_cap, count_cap, overflow = _ext.neighbour_matrix_dlpack(
            co, ce, inv, pb, host_pos, rc, int(max_capacity), nums,
            1, pos, self._device_id)
        out_dev = (self._device_type, self._device_id)
        idx, dist, count = (
            cp.from_dlpack(_nb.DLPackTensor(c, out_dev))
            for c in (idx_cap, dist_cap, count_cap))
        # Boolean (max_capacity,) per-row-flattened mask of valid entries.
        col = cp.arange(int(max_capacity))
        mask = col[None, :] < count[:, None]
        arrays = {'idx': idx, 'dist': dist, 'count': count}
        return MatscipyDeviceResult(
            arrays, padded=True, did_overflow=bool(overflow),
            n_edges=count.sum(), mask=mask)

    def needs_rebuild(self, positions, *, skin, stream=None):
        """On-device flag: 1 iff max squared displacement > skin**2.

        Runs the native ``_ext.needs_rebuild`` kernel (CUB/hipCUB max-reduction
        of per-atom squared displacement vs the cached reference, thresholded on
        device); returns a 1-element uint8 device DLPack tensor.  Nothing is read
        back to the host -- residency is preserved.  An eager consumer syncs this
        scalar at a single documented point (``NeighborListBuilder.update_device``
        -> ``bool(...)``, served by :meth:`DLPackTensor.__bool__`); a compiled
        consumer adopts it with ``from_dlpack`` and branches in-graph.
        """
        if self._ref is None:
            raise RuntimeError('call build_device(...) before needs_rebuild(...)')
        cap = _ext.needs_rebuild(positions, self._ref, float(skin))
        return _nb.DLPackTensor(cap, (self._device_type, self._device_id))
