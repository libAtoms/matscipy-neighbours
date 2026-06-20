# ======================================================================
# Lennard-Jones Langevin droplet examples (CPU and GPU)
# ======================================================================

add_executable(lj_langevin_cpu lj_langevin_cpu.cc)
target_link_libraries(lj_langevin_cpu PRIVATE neighbours)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lj_langevin_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_CUDA)
    add_executable(lj_langevin_gpu lj_langevin_gpu.cu)
    set_source_files_properties(lj_langevin_gpu.cu PROPERTIES LANGUAGE CUDA)
    target_link_libraries(lj_langevin_gpu PRIVATE neighbours)
    # cuRAND for the per-atom Langevin noise.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(lj_langevin_gpu PRIVATE CUDA::curand)
endif()
