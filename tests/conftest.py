"""Test configuration for matscipy-neighbours.

Locates the compiled ``_matscipy_neighbours`` extension (built into the CMake
build tree) and makes it importable, so the suite can be run either via
``ctest`` (which sets PYTHONPATH) or directly with ``pytest tests/`` after a
build.
"""

import pathlib
import sys
import sysconfig

# Make the freshly-built extension importable when running pytest directly.
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Match this interpreter's ABI-tagged name, the abi3 name, and the plain
# `.so` that CMake's Python_add_library emits. (Avoid a bare wildcard so we
# don't pick up a stale extension built for a different Python version.)
_patterns = [
    f"_matscipy_neighbours{_EXT_SUFFIX}",
    "_matscipy_neighbours.abi3.so",
    "_matscipy_neighbours.so",
]
_candidates = [
    p
    for pat in _patterns
    for p in _ROOT.rglob(pat)
    if ".git" not in p.parts
]
if _candidates:
    # Most recently modified build wins.
    _candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    sys.path.insert(0, str(_candidates[0].parent))

# Make the pure-Python wrapper package importable from the source tree
# (it imports the compiled extension found above).
_pkg_dir = _ROOT / "language_bindings" / "python"
if _pkg_dir.is_dir():
    sys.path.insert(0, str(_pkg_dir))

# test_neighbours.py is the verbatim upstream matscipy suite. It depends on ase,
# matscipytest, the matscipy Python package and data files that are not part of
# this repository, so it cannot run here. Keep it as reference, exclude it from
# collection.
collect_ignore = ["test_neighbours.py"]
