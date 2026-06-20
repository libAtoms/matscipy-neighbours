# C++ API

The core lives in `src/libneighbours/` (namespace `matscipy`) and is
Python-free. Numeric types are aliased in `types.hh`: `index_t` (32-bit signed)
and `real_t` (double).

## Building a neighbour list

```cpp
#include "neighbour_list.hh"

matscipy::NeighbourList nl;
matscipy::error_t err = matscipy::neighbour_list(
    QUANTITY_FIRST | QUANTITY_SECOND | QUANTITY_DISTVEC,
    cell_origin, cell, inv_cell, pbc, nat, positions, cutoff,
    /*per_atom_cutoff=*/nullptr, /*per_type_cutoff_sq=*/nullptr,
    /*ncutoffs=*/0, /*types=*/nullptr, nl);
```

- `cell_origin[3]`, `cell[9]` (row-major lattice vectors), `inv_cell[9]`,
  `pbc[3]`, `positions[3*nat]`.
- Quantity flags: `QUANTITY_FIRST`, `QUANTITY_SECOND`, `QUANTITY_DISTVEC`,
  `QUANTITY_ABSDIST`, `QUANTITY_SHIFT`.
- Cutoffs: a scalar `cutoff`, or `per_atom_cutoff[nat]`, or a
  `per_type_cutoff_sq[ncutoffs*ncutoffs]` matrix with `types[nat]`.
- A trailing `CellOrder order = CellOrder::Linear` selects the cell layout
  (`Linear` or `Morton`).

### `NeighbourList`

Holds the result; only the requested quantities are filled:

```cpp
struct NeighbourList {
    std::vector<index_t> first;    // [npairs]
    std::vector<index_t> secnd;    // [npairs]
    std::vector<real_t>  distvec;  // [3*npairs]
    std::vector<real_t>  absdist;  // [npairs]
    std::vector<index_t> shift;    // [3*npairs]
    index_t npairs;
};
```

Typed `Span` views (`first_view()`, `distvec_view()`, …) and per-pair accessors
(`distvec_at(p)`, `shift_at(p)`) give bounds-checked-free access without raw
`.data()`.

## Dense fixed-capacity list

`neighbour_matrix(...)` returns a `NeighbourMatrix` whose shape is static — each
atom's neighbours fill a row of an `n x max_neighbours` matrix
(`idx[n*K]`, `dist[n*K*3]`, `count[n]`, plus an `overflow` flag). Unused slots
are 0; mask with `count`. This suits fixed-shape consumers (e.g. JAX). The GPU
form `neighbour_matrix_gpu_device(req, K, out)` returns a `NeighbourMatrixDevice`
with `Array<T, DeviceSpace>` buffers. `overflow` is set when an atom's degree
exceeds `K` (retry with a larger capacity).

## `CellGrid`

`cell_list.hh` provides the spatial index. It bundles the **grid definition**
(origin, cell/inverse, resolution `n1/n2/n3`, edge vectors, box widths `len`,
periodicity) with the **binning** (`sorted_atom` plus a dense `cell_first`/
`cell_count` CSR, or a hashed compact table for huge/sparse grids). The cutoff is
not part of the grid — it is a query parameter that the grid's resolution only
bounds.

```cpp
matscipy::CellGrid cg;
matscipy::cell_grid_geometry(origin, cell, inv_cell, pbc, cutoff, cg);  // definition
matscipy::build_cell_grid(raw_cell, nat, CellOrder::Linear, cg);        // binning
```

`DenseQuery` / `SparseQuery` map a neighbour-cell coordinate to a contiguous
slice of `sorted_atom`.

## GPU backend

Available when built with `-DENABLE_CUDA=ON` or `-DENABLE_HIP=ON`
(`neighbour_list_gpu.hh`).

```cpp
// Host-out: same signature as the CPU call, results copied back to NeighbourList.
neighbour_list_gpu(quantities, cell_origin, cell, inv_cell, pbc, nat,
                   positions, cutoff, per_atom_cutoff, per_type_cutoff_sq,
                   ncutoffs, types, nl, order);

// Device-out: results stay on the GPU in a NeighbourListDevice.
NeighbourListRequest req;          // bundles geometry, positions, cutoffs, order, device
req.cell_origin = origin; req.cell = cell; req.inv_cell = inv_cell;
req.pbc = pbc; req.nat = nat; req.positions = device_ptr;
req.positions_on_device = true; req.cutoff = cutoff;
req.quantities = QUANTITY_FIRST | QUANTITY_SECOND;
NeighbourListDevice dev;
neighbour_list_gpu_device(req, dev);   // dev.first/.secnd/... are device buffers
```

- `NeighbourListRequest::positions_on_device` uses an existing device pointer in
  place (no host upload); `device_id` selects the GPU.
- `NeighbourListDevice` mirrors `NeighbourList` with `Array<T, DeviceSpace>`
  buffers plus a `counts` array; it offers the same `Span` views.
- `neighbour_count_gpu_device(req, dev)` fills only `dev.counts` (per-atom
  neighbour counts) without materialising the pair arrays.

## Memory spaces

`memory_space.hh` provides `Array<T, Space>` (a move-only owning buffer) with
`HostSpace` / `CudaSpace` / `HipSpace` tags and `deep_copy` between spaces. The
`DeviceType` codes match DLPack (`kDLCPU=1`, `kDLCUDA=2`, `kDLROCm=10`).
