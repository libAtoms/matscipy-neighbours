# ======================================================================
# matscipy-neighbours — XLA FFI handlers (_matscipy_neighbours_xla)
# ======================================================================
# A plain shared library of FFI handler symbols, loaded with ctypes and
# registered by matscipy_neighbours.jax (no Python C-API, no jaxlib link: the
# handlers only use the header-only XLA FFI API).  Needs JAX_FFI_INCLUDE_DIR,
# i.e. `python -c "import jax.ffi; print(jax.ffi.include_dir())"`.
set(_srcs xla_neighbours.cc)
if(ENABLE_CUDA)
    list(APPEND _srcs xla_neighbours_cuda.cu)
endif()
add_library(_matscipy_neighbours_xla SHARED ${_srcs})
target_include_directories(_matscipy_neighbours_xla PRIVATE ${JAX_FFI_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(_matscipy_neighbours_xla PRIVATE neighbours)
target_compile_features(_matscipy_neighbours_xla PRIVATE cxx_std_17)
if(ENABLE_CUDA)
    set_target_properties(_matscipy_neighbours_xla PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_RUNTIME_LIBRARY Shared)
endif()
set_target_properties(_matscipy_neighbours_xla PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" PREFIX "")
install(TARGETS _matscipy_neighbours_xla
        LIBRARY DESTINATION matscipy_neighbours
        RUNTIME DESTINATION matscipy_neighbours)
