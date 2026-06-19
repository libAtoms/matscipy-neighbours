/*
 * matscipy-neighbours — Neighbour list for particle simulations
 * https://github.com/libAtoms/matscipy-neighbours
 *
 * SPDX-License-Identifier: MIT
 *
 * Unit tests for the memory-space abstraction (Array<T, Space> + deep_copy).
 * Host paths run everywhere; the device round-trip is compiled only when a GPU
 * backend is enabled, and skips at runtime if no device is present.
 */

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "memory_space.hh"

#ifdef MATSCIPY_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

using namespace matscipy;

TEST(MemorySpace, DeviceTypeCodesMatchDLPack) {
    /* Each space's device code matches the DLPack device type:
       kDLCPU=1, kDLCUDA=2, kDLROCm=10. */
    EXPECT_EQ(static_cast<int>(HostSpace::device), 1);
    EXPECT_EQ(static_cast<int>(CudaSpace::device), 2);
    EXPECT_EQ(static_cast<int>(HipSpace::device), 10);
}

TEST(MemorySpace, HostAllocateAndAccess) {
    Array<int> a(5);
    EXPECT_EQ(a.size(), 5u);
    EXPECT_FALSE(a.empty());
    ASSERT_NE(a.data(), nullptr);
    for (std::size_t i = 0; i < a.size(); i++) a.data()[i] = static_cast<int>(i);
    for (std::size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(a.data()[i], static_cast<int>(i));
}

TEST(MemorySpace, EmptyArrayHasNullData) {
    Array<double> a;
    EXPECT_EQ(a.size(), 0u);
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(a.data(), nullptr);
}

TEST(MemorySpace, ShrinkKeepsBufferGrowReallocates) {
    Array<int> a(10);
    int *base = a.data();
    a.resize(4);  /* shrink: same buffer, lower logical size */
    EXPECT_EQ(a.size(), 4u);
    EXPECT_EQ(a.data(), base);
    a.resize(8);  /* still within capacity 10 */
    EXPECT_EQ(a.size(), 8u);
    EXPECT_EQ(a.data(), base);
}

TEST(MemorySpace, MoveTransfersOwnership) {
    Array<int> a(3);
    a.data()[0] = 42;
    int *raw = a.data();
    Array<int> b(std::move(a));
    EXPECT_EQ(b.data(), raw);
    EXPECT_EQ(b.data()[0], 42);
    EXPECT_EQ(a.data(), nullptr);  /* NOLINT: moved-from is intentionally empty */
    EXPECT_EQ(a.size(), 0u);
}

TEST(MemorySpace, DeepCopyHostToHost) {
    Array<double> src(6), dst(6);
    for (std::size_t i = 0; i < src.size(); i++) src.data()[i] = 1.5 * i;
    deep_copy(dst, src);
    for (std::size_t i = 0; i < dst.size(); i++) EXPECT_EQ(dst.data()[i], 1.5 * i);
}

TEST(MemorySpace, DeepCopyEmptyIsNoop) {
    Array<int> src, dst;
    deep_copy(dst, src);  /* must not touch null pointers */
    SUCCEED();
}

#ifdef MATSCIPY_ENABLE_CUDA
TEST(MemorySpace, DeviceRoundTrip) {
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        GTEST_SKIP() << "no CUDA device available";
    }
    const std::size_t n = 1024;
    Array<int> host_in(n), host_out(n);
    Array<int, CudaSpace> dev(n);
    std::iota(host_in.data(), host_in.data() + n, 0);

    deep_copy(dev, host_in);       /* H2D */
    deep_copy(host_out, dev);      /* D2H */
    for (std::size_t i = 0; i < n; i++)
        ASSERT_EQ(host_out.data()[i], static_cast<int>(i));
}

TEST(MemorySpace, DeviceToDeviceCopy) {
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        GTEST_SKIP() << "no CUDA device available";
    }
    const std::size_t n = 256;
    Array<double> host_in(n), host_out(n);
    Array<double, CudaSpace> d1(n), d2(n);
    for (std::size_t i = 0; i < n; i++) host_in.data()[i] = 2.0 * i + 1.0;

    deep_copy(d1, host_in);   /* H2D */
    deep_copy(d2, d1);        /* D2D */
    deep_copy(host_out, d2);  /* D2H */
    for (std::size_t i = 0; i < n; i++)
        ASSERT_EQ(host_out.data()[i], 2.0 * i + 1.0);
}
#endif
