// vulkan/src/poly/kernels/store_f16_vk.cpp - the Vulkan launcher for strata::poly::store_f16
// (include/strata/poly/kernels.hpp:68; the CUDA body is src/poly/kernels.cu:516
// store_f16_kernel + the wrapper at :2283, which writes (half*)cache + pos*n).
//
// BITWISE target: the .comp packs f16 with the f16_rn_bits helper (quantize_q8_1.comp's,
// proven byte-exact against __float2half by native_mmvq_vk_parity) into u32 words - NOT
// packHalf2x16, which truncates on RADV (PORT-NOTES law 1). n values are n/2 words; an odd
// n, or a pos*n element offset that is odd, cannot be word-addressed and throws rather
// than fake (the model's row widths are even; q8_0 rows are multiples of 32).
// store_f16_at (kernels.cu:522) reads the row from the card at launch time: not demanded
// by the trap chain yet - it stays a stub (PORT-STATUS records it).
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "store_f16.spv.h"

const vk::Kernel& pl_store_f16() {
    static const vk::Kernel k = vk::kernel_load("store_f16", store_f16_spv, store_f16_spv_len,
                                                2, 4, 256, false);
    return k;
}
}  // namespace

void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream) {
    if (rows <= 0 || n <= 0) return;
    const size_t all = (size_t)n * (size_t)rows;
    if (n % 2 || (all % 2) || ((size_t)pos * (size_t)n) % 2)
        throw std::invalid_argument("store_f16: an odd element offset cannot be word-addressed here");
    const uint64_t xo = strata_vk_compat::off(x);
    const uint64_t co = strata_vk_compat::off(cache) + (uint64_t)pos * (uint64_t)n * 2;
    const uint32_t words = (uint32_t)(all / 2);
    const uint64_t bufs[2] = {xo, co};
    // x's bound rounds up to a whole word so the last thread's second read is in range
    // (the product is masked by `words`; only the padding value is read, never stored).
    const uint64_t xbytes = ((uint64_t)all + 1u) / 2u * 8u;   // 2 floats per word, 8 bytes each
    const uint64_t sizes[2] = {xbytes, (all / 2) * 4};
    vk::kernel_launch_sized(pl_store_f16(), (int)((words + 255) / 256), bufs, sizes, 2, &words);
}

}  // namespace strata::poly
