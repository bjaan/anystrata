// vulkan/src/poly/kernels/likeliest_vk.cpp - the Vulkan launcher for strata::poly::likeliest
// (include/strata/poly/kernels.hpp:234; the CUDA body is src/poly/kernels.cu:1827 + the wrapper
// at :2663, <<<1, 1024>>> - ALWAYS one block of 1024 threads whatever n is, mirrored by the
// .comp's local_size_x = 1024 so the strided loops' iteration counts match exactly).
//
// The CUDA kernel writes the Likeliest struct into PINNED HOST memory through unified addressing
// (the engine's h_guess_); a Vulkan kernel sees only arena buffers, so the launcher writes to a
// fixed 8-byte arena scratch and copies it back with copy_d2h (synchronous under the runtime's
// one-queue model, matching the engine's cudaStreamSynchronize right after the call).
// n <= 0: the CUDA kernel would run with an empty strided loop - token 0 (at == n), prob 1/0 =
// +inf; the launcher reproduces that on the host rather than launching with a zero-size buffer.
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <limits>

namespace strata::poly {
namespace {
#include "likeliest.spv.h"

const vk::Kernel& pl_likeliest() {
    static const vk::Kernel k = vk::kernel_load("likeliest", likeliest_spv, likeliest_spv_len,
                                                2, 16, 1024, false);
    return k;
}
}  // namespace

void likeliest(const float* logits, int n, Likeliest* out, void* stream) {
    if (n <= 0) { out->token = 0; out->prob = std::numeric_limits<float>::infinity(); return; }
    static uint64_t scratch = 0;
    if (!scratch) scratch = vk::Arena::instance().alloc(8);
    const uint64_t bufs[2] = {strata_vk_compat::off(logits), scratch};
    // The .comp guards every read with `i < n` INSIDE the body (a bare strided header loop is
    // hoisted by glslc -O into a do-while whose first read folds garbage past the buffer on
    // RADV - caught by the n=5 parity shape), so the exact n*4 bound is safe.
    const uint64_t sizes[2] = {(uint64_t)n * 4, 8};
    const uint32_t pc[4] = {(uint32_t)n, 0u, 0u, 0u};
    vk::kernel_launch_sized(pl_likeliest(), 1, bufs, sizes, 2, pc);
    vk::copy_d2h(out, (void*)scratch, 8, stream);
}

}  // namespace strata::poly
