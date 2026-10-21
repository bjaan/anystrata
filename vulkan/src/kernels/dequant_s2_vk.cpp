// vulkan/src/kernels/dequant_s2_vk.cpp - the Vulkan launcher for include/strata/kernels/dequant_s2.hpp.
#include "strata/kernels/dequant_s2.hpp"
#include "strata/vulkan_runtime.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int QK = 64;
#include "dequant_s2.spv.h"
}  // namespace

void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks) {
    if (n_blocks <= 0) return;
    static const vk::Kernel k = vk::kernel_load(
        "dequant_s2", dequant_s2_spv, dequant_s2_spv_len, 3, 4, 256, false);
    const uint64_t bufs[3]  = {(uint64_t)(uintptr_t)codes, (uint64_t)(uintptr_t)scales,
                               (uint64_t)(uintptr_t)out};
    const uint64_t sizes[3] = {(uint64_t)n_blocks * (QK / 4), (uint64_t)n_blocks * 4,
                               (uint64_t)n_blocks * QK * 4};
    const uint32_t pc[1] = {(uint32_t)n_blocks};
    vk::kernel_launch_sized(k, (int)((n_blocks + 255) / 256), bufs, sizes, 3, pc);
}

}  // namespace strata::kernels
