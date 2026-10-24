// vulkan/src/poly/kernels/bf16_to_f32_vk.cpp - the Vulkan launcher for strata::poly::bf16_to_f32
// (include/strata/poly/kernels.hpp:27; the CUDA body is src/poly/kernels.cu:154-156 + the
// wrapper at :2212: grid(n,256) x 256, out[i] = bitcast(uint32(w[i]) << 16)). The widening is
// exact - no rounding class exists - so bitwise parity is by construction; the launch mirrors
// the CUDA geometry through one u32 word per thread (elements 2i, 2i+1).
//
// Demanded by the GSQ-hybrid long-context gate: engine.cpp:2081 converts a BF16 dense matrix
// to the f32 wf_ scratch when a prompt batch reaches kStageRows (the 485-token 16 KiB-context
// run; the 18-token gate stays under the threshold and never reaches it).
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "bf16_to_f32.spv.h"

const vk::Kernel& pl_bf16_to_f32() {
    static const vk::Kernel k = vk::kernel_load("bf16_to_f32", bf16_to_f32_spv, bf16_to_f32_spv_len,
                                                2, 4, 256, false);
    return k;
}
}  // namespace

void bf16_to_f32(const uint16_t* w, float* out, size_t n, void* stream) {
    if (n == 0) return;
    if ((strata_vk_compat::off(w)) % 4 || (strata_vk_compat::off(out)) % 4)
        throw std::invalid_argument("bf16_to_f32: the arena offsets must be word-aligned");
    (void) stream;   // the compat queue is single-stream (store_f16 precedent)
    const uint32_t n32 = (uint32_t)n;                    // the model's matrices stay under 2^32 elems
    const uint64_t bufs[2] = {strata_vk_compat::off(w), strata_vk_compat::off(out)};
    const uint64_t sizes[2] = {n * 2, n * 4};
    vk::kernel_launch_sized(pl_bf16_to_f32(), (int)((n + 511) / 512), bufs, sizes, 2, &n32);
}

}  // namespace strata::poly
