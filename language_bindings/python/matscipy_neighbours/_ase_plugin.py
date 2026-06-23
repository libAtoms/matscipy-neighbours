"""ASE v4 neighbour-list plugin for matscipy-neighbours.

Registers the compiled neighbour list as an ``ase.plugins`` backend, so it can
be selected *explicitly* via :func:`ase.neighborlist.get_neighbor_list` (or, in
due course, a calculator's ``neighbor_list=`` option).  Selection is never
automatic -- this only makes the backend *available* under the name
``"matscipy-neighbours"``.

This module is the ``ase.plugins`` entry point (it exposes ``__ase_plugins__``).
The registration is guarded so that installing this package alongside an ASE
that predates the v4 plugin API registers nothing rather than breaking plugin
discovery.  The heavy import of the compiled extension happens lazily, inside
the adapter, so merely building the plugin collection does not import it.
"""
from __future__ import annotations


def neighbor_list(quantities, atoms, cutoff, *, self_interaction=False):
    """matscipy-neighbours neighbour list adapted to ASE's protocol.

    ``matscipy_neighbours.neighbour_list`` returns the same ``(i, j, d, D, S)``
    flat arrays and quantity letters as ``ase.neighborlist.neighbor_list`` but
    has no ``self_interaction`` option (it never returns pure self-pairs).  Per
    the plugin contract a request it cannot honour is rejected rather than
    silently differing, so ``self_interaction=True`` raises.
    """
    if self_interaction:
        raise NotImplementedError(
            'the matscipy-neighbours backend does not support '
            'self_interaction=True')
    from matscipy_neighbours import neighbour_list as _neighbour_list

    return _neighbour_list(quantities, atoms, cutoff)


def device_neighbor_list(device_id=0):
    """Return the *experimental* device-resident neighbour-list backend.

    The result satisfies ASE's experimental ``DeviceNeighborList`` protocol
    (:mod:`ase._4.plugins.neighborlist_device`) and builds/maintains edge data
    on-device, exchanged via DLPack.  GPU-only (CUDA/HIP) and requires CuPy.

    This device capability is intentionally *separate* from the host
    ``neighbor_list`` registration above: the host ``ase.plugins``
    ``NeighborListPlugin`` carries no device slot yet, so a consumer obtains the
    device backend through this factory and checks
    ``isinstance(backend, DeviceNeighborList)``.  (Folding an optional
    ``device_implementation=`` into ``NeighborListPlugin`` upstream would let it
    be advertised through the same plugin record -- a noted upstream ask.)
    """
    from matscipy_neighbours._device import MatscipyDeviceNeighborList

    return MatscipyDeviceNeighborList(device_id=device_id)


try:
    from ase._4.plugins.neighborlist import NeighborListPlugin
except ImportError:
    # ASE without the v4 plugin API: register nothing, but do not break the
    # discovery of other plugins.
    __ase_plugins__: set = set()
else:
    __ase_plugins__ = {
        NeighborListPlugin(
            'matscipy-neighbours',
            long_name='matscipy-neighbours compiled neighbour list',
            citation='https://github.com/libAtoms/matscipy-neighbours',
            implementation='matscipy_neighbours._ase_plugin.neighbor_list',
        ),
    }
