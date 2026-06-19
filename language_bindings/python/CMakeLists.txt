# ======================================================================
# matscipy-neighbours — Python C-extension (_matscipy_neighbours)
# ======================================================================

# The extension mixes C (module.c) and C++ (bind_py_neighbours.cc + the core).
Python_add_library(_matscipy_neighbours MODULE
    bind_py_neighbours.cc
    bind_py_dlpack.cc
    module.c
)

target_include_directories(_matscipy_neighbours PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(_matscipy_neighbours PRIVATE neighbours Python::NumPy)

# Put the built module at the top of the build tree so it is easy to import.
set_target_properties(_matscipy_neighbours PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
