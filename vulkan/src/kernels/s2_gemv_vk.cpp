// vulkan/src/kernels/s2_gemv_vk.cpp - the Vulkan launcher for include/strata/kernels/s2_gemv.hpp.
//
// Same contract as src/kernels/cuda/s2_gemv.cu: the pointers are arena offsets (the Vulkan backend's device
// pointers), and y[o] = sum_i fp16(x[i]) * W[i][o] over the S2 planes. The launcher computes the same
// grid as the CUDA one (threads = 128 = the .comp's local_size_x; blocks = ceil(n_out / 128)).
#include "strata/kernels/s2_gemv.hpp"
#include "strata/vulkan_runtime.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int QK = 64;
#include "s2_gemv.spv.h"      // s2_gemv_spv[] / s2_gemv_spv_len, embedded by tools/embed_spv.py
}  // namespace

void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
             int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK != 0) {
        std::fprintf(stderr, "s2_gemv: n_in %lld is not a multiple of %d\n", (long long)n_in, QK);
        std::exit(1);
    }
    static const vk::Kernel k = vk::kernel_load(
        "s2_gemv", s2_gemv_spv, s2_gemv_spv_len, /*storage buffers*/ 4, /*push bytes*/ 8,
        /*workgroup*/ 128, /*force 32-wide subgroups*/ false);
    const int64_t nb = n_in / QK;
    const uint64_t bufs[4]  = {(uint64_t)(uintptr_t)x, (uint64_t)(uintptr_t)codes,
                               (uint64_t)(uintptr_t)scales, (uint64_t)(uintptr_t)y};
    const uint64_t sizes[4] = {(uint64_t)n_in * 2, (uint64_t)n_out * nb * (QK / 4),
                               (uint64_t)n_out * nb * 4, (uint64_t)n_out * 4};
    const uint32_t pc[2] = {(uint32_t)n_in, (uint32_t)n_out};
    vk::kernel_launch_sized(k, (int)((n_out + 127) / 128), bufs, sizes, 4, pc);
}

}  // namespace strata::kernels
