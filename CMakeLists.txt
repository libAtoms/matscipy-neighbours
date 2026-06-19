# ======================================================================
# matscipy-neighbours - Neighbour list for particle simulations
# https://github.com/libAtoms/matscipy-neighbours
#
# CMake build:
#   neighbours              static, Python-free C++ core (src/libneighbours)
#   _matscipy_neighbours    Python C-extension linking the core
# ======================================================================

cmake_minimum_required(VERSION 3.18)

project(matscipy_neighbours
        LANGUAGES C CXX
        DESCRIPTION "Neighbour list for particle simulations")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_COVERAGE "Build with coverage instrumentation (gcov/llvm-cov)" OFF)
option(BUILD_TESTING "Build the test suites" ON)

# GPU backends (Phase 3). Off by default so the CPU build needs no toolkit.
# Exactly one may be enabled at a time; both share the single-source kernels.
option(ENABLE_CUDA "Build the CUDA GPU backend (needs nvcc)" OFF)
option(ENABLE_HIP "Build the HIP GPU backend (needs hipcc)" OFF)

if(ENABLE_CUDA AND ENABLE_HIP)
    message(FATAL_ERROR "Enable only one of ENABLE_CUDA / ENABLE_HIP at a time.")
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Default to a broad set of real-world architectures if the user did not
    # pin one; override with -DCMAKE_CUDA_ARCHITECTURES=NN.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 52 70 80 native)
    endif()
    message(STATUS "matscipy-neighbours: CUDA backend ON "
                   "(arch ${CMAKE_CUDA_ARCHITECTURES})")
endif()

if(ENABLE_HIP)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    message(STATUS "matscipy-neighbours: HIP backend ON")
endif()

if(ENABLE_COVERAGE)
    # Applies to the core, the extension and the C++ tests.
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

# ----------------------------------------------------------------------
# Python-free core
# ----------------------------------------------------------------------
add_subdirectory(src/libneighbours)

# ----------------------------------------------------------------------
# Python extension
# ----------------------------------------------------------------------
find_package(Python REQUIRED
             COMPONENTS Interpreter Development.Module NumPy)

add_subdirectory(language_bindings/python)

# ----------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------
option(BUILD_BENCHMARKS "Build the core benchmark executable" ON)
if(BUILD_BENCHMARKS)
    add_executable(bench_core benchmarks/bench_core.cc)
    target_link_libraries(bench_core PRIVATE neighbours)
endif()

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
if(BUILD_TESTING)
    enable_testing()

    # --- C++ core tests (GoogleTest) ---------------------------------------
    # Prefer a system/Homebrew GoogleTest; otherwise fetch it (needs network,
    # e.g. in CI).
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(test_core tests/test_core.cc)
    target_link_libraries(test_core PRIVATE neighbours GTest::gtest_main)

    add_executable(test_memory_space tests/test_memory_space.cc)
    target_link_libraries(test_memory_space PRIVATE neighbours GTest::gtest_main)
    if(ENABLE_CUDA)
        # The device round-trip test calls the CUDA runtime directly; let nvcc
        # compile it so <cuda_runtime.h> and cudaGetDeviceCount are available.
        set_source_files_properties(tests/test_memory_space.cc
                                    PROPERTIES LANGUAGE CUDA)
    elseif(ENABLE_HIP)
        set_source_files_properties(tests/test_memory_space.cc
                                    PROPERTIES LANGUAGE HIP)
    endif()

    include(GoogleTest)
    gtest_discover_tests(test_core)
    gtest_discover_tests(test_memory_space)

    # --- Python tests (pytest) ---------------------------------------------
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE _have_pytest
        OUTPUT_QUIET ERROR_QUIET
    )

    if(_have_pytest EQUAL 0)
        add_test(
            NAME pytest
            COMMAND ${Python_EXECUTABLE} -m pytest -v
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        )
        # Make both the built extension and the pure-Python wrapper importable.
        set_tests_properties(pytest PROPERTIES
            ENVIRONMENT
            "PYTHONPATH=$<TARGET_FILE_DIR:_matscipy_neighbours>:${CMAKE_CURRENT_SOURCE_DIR}/language_bindings/python"
        )
    else()
        message(WARNING
            "pytest not found for ${Python_EXECUTABLE}; the 'pytest' test will "
            "not be registered. Install pytest or reconfigure with "
            "-DPython_EXECUTABLE=<python-with-pytest>.")
    endif()
endif()
