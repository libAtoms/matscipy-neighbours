# ======================================================================
# matscipy-neighbours — Python-free C++ core library
# ======================================================================

add_library(neighbours STATIC
    error.cc
    tools.cc
    cell_list.cc
    neighbour_list.cc
    first_neighbours.cc
    triplet_list.cc
)

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
