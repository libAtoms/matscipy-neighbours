# Algorithm & references

## Pipeline

A neighbour-list build proceeds as data-parallel steps, identical in structure
on the CPU and the GPU:

1. **Cell index.** Each atom's position maps to integer cell coordinates
   `(k, l, m) = floor(inv_cell · (r − origin) · n)`. The grid resolution
   `n1/n2/n3` is chosen so each cell is about one cutoff wide, so only the
   immediately neighbouring cells are scanned.
2. **Binning.** Atoms are grouped per cell so that the atoms of one cell occupy
   a contiguous slice of a `sorted_atom` array, via one of two backends:
    - **Dense CSR** — `cell_first[c]` / `cell_count[c]` indexed by the linear
      cell index `c = c1 + n1·(c2 + n2·c3)`, built by a counting-sort histogram
      and prefix sum. Used for bounded grids.
    - **Hashed compact** — an open-addressing hash table keyed by the 64-bit
      linear cell index, storing offsets only for non-empty cells. Used for
      huge/sparse (vacuum) grids, where a dense array would be wasteful or
      overflow. Selected automatically.
3. **Cell ordering.** `CellOrder::Linear` lays cells out in row-major (z-major)
   order; `CellOrder::Morton` orders them along a Z-curve by interleaving the
   cell-coordinate bits, which can improve spatial locality. Linear is the
   default.
4. **Two-pass output.** A first pass counts each atom's neighbours; a prefix sum
   turns the counts into write offsets and the exact total; the buffers are
   allocated once; a second pass fills them. Each atom writes a disjoint slice,
   so the passes parallelise without contention.

Within a cell pair the distance test enforces the cutoff (global, per-atom, or
per-type), and each emitted pair carries the shift `S` with
`D == r[j] − r[i] + S · cell`. The output is ordered by the first index `i`.

The CPU and GPU paths share one definition of the cell index, hash and Morton
keys, the cell lookup, and the neighbour-cell traversal; they differ only in how
the binning is built (a serial counting sort / `std::sort` versus parallel
atomic histograms and a vendor radix sort).

## References

The cell-list design follows two papers on parallel neighbour search for
smoothed particle hydrodynamics:

- **M. Ihmsen, N. Akinci, M. Becker, M. Teschner.** *A Parallel SPH
  Implementation on Multi-Core CPUs.* Computer Graphics Forum, 2010.
- **S. Band, C. Gissler, M. Teschner.** *Compressed Neighbour Lists for SPH.*
  Computer Graphics Forum, 2019.

### What is used from each

| Component here | Origin |
|----------------|--------|
| Dense linear cell list (default) | Ihmsen *index sort*: sorted particle array + per-cell first-index, built by counting sort / parallel reduction |
| `Morton` cell ordering | Ihmsen *Z-index sort* / Band's Morton cell index (bit-interleaving), with cell runs marked in the sorted list |
| Hashed compact backend | Ihmsen *compact / spatial hashing*: a hash table keyed by the cell index, memory scaling with non-empty cells |
| Neighbour-cell traversal | Ihmsen's straightforward scan of the surrounding cells |

Both papers emphasise that the pipeline has no data dependencies, which is what
makes the GPU port a backend swap rather than a new algorithm.

### What is deliberately not used

- **Band's BigMin/LitMax range query** and **compact cell array** (a prefix-sum
  list of only non-empty cells): the implementation uses the surrounding-cell
  scan, and represents non-empty-cell-only storage with hashing instead.
- **Band's neighbour-list compression** (delta + variable-byte encoding): the
  API returns full pair arrays, so there is nothing to compress.

### Beyond the source papers

The papers target a single SPH support radius on an axis-aligned box. This
implementation additionally handles triclinic cells, per-direction periodicity,
and scalar / per-atom / per-type cutoffs, with the `D == r[j] − r[i] + S · cell`
shift contract and `i`-sorted output.
