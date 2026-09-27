"""matscipy-neighbours — neighbour lists for particle simulations.

This pure-Python package wraps the ``_matscipy_neighbours`` C-extension and
exposes the same public ``neighbour_list`` API as upstream
`matscipy.neighbours <https://github.com/libAtoms/matscipy>`_, so existing
matscipy code keeps working.
"""

from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("matscipy-neighbours")
except PackageNotFoundError:  # pragma: no cover - running from the build tree
    __version__ = "unknown"
