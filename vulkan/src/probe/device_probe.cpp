// vulkan/src/probe/device_probe.cpp - the port's first run on a card: the facts it plans against and the
// two capabilities it needs ASSERTED BY EXECUTION, not by extension presence alone:
//   * OpUDot (integer dot product) computes 10 for (1,2,3,4).(1,1,1,1) - the dp4a every quantized kernel
//     uses (strata's CUDA sources call ggml_cuda_dp4a 109 times);
//   * a 32-wide forced subgroup on the hardware's native width (64 on RDNA1-era GFX1013): ballot +
//     shuffleXor come back as wave32 would give - strata's wave32 semantics (hip_compat's shuffles) made
//     portable to Vulkan.
// The SPIR-V for both is compiled from probe kernels at build time by glslc; no downloads, like the rest.
#include "strata/vulkan_runtime.hpp"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "probe_dot.spv.h"
#include "probe_subgroup.spv.h"
using namespace strata;

int main() {
    const vk::DeviceInfo& d = vk::device_info();      // throws with a named reason on an unusable card
    std::printf("probe: %s: subgroup %u (forcing 32), wg invocations %u, max allocation %llu MiB, "
                "dot=%d ssc=%d bda=%d uma=%d\n",
                d.name.c_str(), d.subgroup_size, d.max_workgroup_invocations,
                (unsigned long long)(d.max_allocation_size >> 20),
                (int)d.integer_dot_product, (int)d.subgroup_size_control,
                (int)d.buffer_device_address, (int)d.unified_memory);
    if (!d.integer_dot_product || !d.subgroup_size_control) return 1;

    // ---- A: dp4a. Load and run probe_dot.comp; expect the buffer to come back 10.
    vk::Kernel kdot = vk::kernel_load("probe_dot", probe_dot_spv, probe_dot_spv_len, 1, 0, 64, false);
    vk::Arena& a = vk::Arena::instance();
    uint64_t out = a.alloc(64 * 4);
    { const uint64_t sz[1] = {256}; vk::kernel_launch_sized(kdot, 1, &out, sz, 1, nullptr); }
    int val = -1;
    vk::copy_d2h(&val, (void*)(uintptr_t)out, 4, nullptr);
    std::printf("probe: dp4a kernel: got %d (expect 10): %s\n", val, val == 10 ? "OK" : "FAIL");

    // ---- B: forced-32 subgroup: expect 321 (width 32; shuffleXor(1,16) reads lane 16's 2 -> +1).
    vk::Kernel ksg = vk::kernel_load("probe_subgroup", probe_subgroup_spv, probe_subgroup_spv_len, 1, 0, 64, true);
    uint64_t out2 = a.alloc(64 * 4);
    { const uint64_t sz[1] = {256}; vk::kernel_launch_sized(ksg, 1, &out2, sz, 1, nullptr); }
    int val2 = -1;
    vk::copy_d2h(&val2, (void*)(uintptr_t)out2, 4, nullptr);
    std::printf("probe: forced-32 subgroup kernel: got %d (expect 321): %s\n", val2, val2 == 321 ? "OK" : "FAIL");

    const bool pass = (val == 10) && (val2 == 321);
    std::printf("vk_device_probe %s on %s\n", pass ? "PASS" : "FAIL", d.name.c_str());
    return pass ? 0 : 1;
}
