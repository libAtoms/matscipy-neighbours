# ======================================================================
# matscipy-neighbours — Python-free C++ core library
# ======================================================================

add_library(neighbours STATIC
    error.cc
    tools.cc
    memory_space.cc
    cell_list.cc
    neighbour_list.cc
    first_neighbours.cc
    triplet_list.cc
)

# GPU backend sources (single-source kernels + device memory backend). Added to
# the same target and compiled by nvcc/hipcc; the CPU build omits them entirely.
if(ENABLE_CUDA)
    target_sources(neighbours PRIVATE memory_space_cuda.cu)
    set_source_files_properties(memory_space_cuda.cu PROPERTIES LANGUAGE CUDA)
    # The header guards device code on MATSCIPY_ENABLE_CUDA; PUBLIC so every
    # consumer (extension, tests) sees the same Array<T, CudaSpace> definition.
    target_compile_definitions(neighbours PUBLIC MATSCIPY_ENABLE_CUDA)
    set_target_properties(neighbours PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
elseif(ENABLE_HIP)
    target_sources(neighbours PRIVATE memory_space_cuda.cu)
    set_source_files_properties(memory_space_cuda.cu PROPERTIES LANGUAGE HIP)
    target_compile_definitions(neighbours PUBLIC MATSCIPY_ENABLE_HIP)
endif()

target_include_directories(neighbours PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(neighbours PUBLIC cxx_std_17)

# The static core is linked into the Python extension (a shared module), so it
# must be position-independent.
set_target_properties(neighbours PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Optional shared-memory parallelism. The two-pass neighbour build is written so
# the per-atom loops are race-free; the pragmas are no-ops without OpenMP, so the
# serial build stays correct.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(neighbours PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "neighbours: OpenMP enabled")
else()
    message(STATUS "neighbours: OpenMP not found, building serial core")
endif()
