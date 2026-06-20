"""matscipy-neighbours — neighbour lists for particle simulations.

This pure-Python package wraps the ``_matscipy_neighbours`` C-extension and
exposes the same public ``neighbour_list`` API as upstream
`matscipy.neighbours <https://github.com/libAtoms/matscipy>`_, so existing
matscipy code keeps working.
"""

from .neighbours import (
    DLPackTensor,
    coordination,
    first_neighbours,
    get_jump_indicies,
    mic,
    neighbour_list,
    neighbour_matrix,
    triplet_list,
)

__all__ = [
    "neighbour_list",
    "neighbour_matrix",
    "first_neighbours",
    "get_jump_indicies",
    "triplet_list",
    "mic",
    "coordination",
    "DLPackTensor",
]
