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
#include <stdexcept>
#include <vector>

#include "probe_dot.spv.h"
#include "probe_subgroup.spv.h"
#include "probe_wg64k.spv.h"
#include "probe_spans.spv.h"
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

    // ---- C: 64 KiB workgroup-shared array: a 16384 x i32 Workgroup OpVariable (exactly this driver's
    // maxComputeSharedMemorySize = 65536, the LDS budget the RDNA3 HIP-port kernels assume) written by
    // 256 threads, barrier, read back in-kernel: every thread checks wg[0], wg[8192], wg[16383] and
    // stores 1/0. The pipeline compiles AND executes only if the driver grants the full 64 KiB.
    vk::Kernel kwg = vk::kernel_load("probe_wg64k", probe_wg64k_spv, probe_wg64k_spv_len, 1, 0, 256, false);
    uint64_t out3 = a.alloc(256 * 4);
    { const uint64_t sz3[1] = {1024}; vk::kernel_launch_sized(kwg, 1, &out3, sz3, 1, nullptr); }
    std::vector<int> wg_res(256, -1);
    vk::copy_d2h(wg_res.data(), (void*)(uintptr_t)out3, 256 * 4, nullptr);
    int cval = 1;
    for (int r : wg_res) if (r != 1) { cval = 0; break; }
    std::printf("probe: 64K workgroup array: got %d (expect 1): %s\n", cval, cval == 1 ? "OK" : "FAIL");

    // ---- D: far-apart arena spans. Two 600 MiB regions - over RADV's 512 MiB max allocation, so each
    // chains into 2 pieces (512 MiB + ~86 MiB). Patterns are written at region A's offset 0 (first piece)
    // and region B's (bytes - 4 KiB) (deep in the LAST piece of the chain): both copies and the kernel's
    // piece-local sub-span bindings exercise piece_of() lookup at both ends of a chained region.
    // Sub-check: an H2D copy that STRADDLES a piece boundary must throw with 'piece' in its message -
    // that named rejection IS the design (the launcher/copy refuses instead of reading wrong bytes).
    const uint64_t MiB = 1ull << 20;
    const uint64_t region = 600 * MiB;
    uint64_t ra = a.alloc(region);          // throws "out of VRAM" if the 256 MiB slack rule fails;
    uint64_t rb = a.alloc(region);          // then rerun at 300 MiB each and note it (not needed on BC-250)
    std::vector<int> patA(1024), patB(256);
    for (size_t j = 0; j < patA.size(); j++) patA[j] = (int)(j * 3 + 7);
    for (size_t j = 0; j < patB.size(); j++) patB[j] = 7000000 + (int)j;
    vk::copy_h2d((void*)(uintptr_t)ra, patA.data(), 1024 * 4, nullptr);                    // head of piece 0
    vk::copy_h2d((void*)(uintptr_t)(rb + region - 1024 * 4), patB.data(), 256 * 4, nullptr); // tail of last piece
    bool straddle_rejected = false;
    try {
        int filler[1024]; for (size_t j = 0; j < 1024; j++) filler[j] = (int)j;
        // half on piece 0, half on piece 1 of region A (512 MiB boundary)
        vk::copy_h2d((void*)(uintptr_t)(ra + 512 * MiB - 1024), filler, 4096, nullptr);
    } catch (const std::exception& e) {
        straddle_rejected = std::strstr(e.what(), "piece") != nullptr;
        std::printf("probe: piece-straddle copy rejected by name: %s\n", e.what());
    }
    // Kernel binds three piece-local sub-spans: patA (in A's first piece), patB (in B's last piece), out.
    vk::Kernel ksp = vk::kernel_load("probe_spans", probe_spans_spv, probe_spans_spv_len, 3, 0, 256, false);
    uint64_t out4 = a.alloc(1024 * 4);
    { const uint64_t bufs[3] = {ra, rb + region - 4096, out4};
      const uint64_t szs[3] = {4096, 1024, 4096};
      vk::kernel_launch_sized(ksp, 4, bufs, szs, 3, nullptr); }
    std::vector<int> span_res(1024, -1);
    vk::copy_d2h(span_res.data(), (void*)(uintptr_t)out4, 1024 * 4, nullptr);
    int spans = 1;
    for (int r : span_res) if (r != 1) { spans = 0; break; }
    const bool dval = spans && straddle_rejected;
    std::printf("probe: far-apart arena spans + piece-straddle copy rejected: %s\n", dval ? "OK" : "FAIL");

    const bool pass = (val == 10) && (val2 == 321) && (cval == 1) && dval;
    std::printf("vk_device_probe %s on %s\n", pass ? "PASS" : "FAIL", d.name.c_str());
    return pass ? 0 : 1;
}
