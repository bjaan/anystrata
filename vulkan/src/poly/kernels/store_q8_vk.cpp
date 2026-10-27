// vulkan/src/poly/kernels/store_q8_vk.cpp - the Vulkan launcher for strata::poly::store_q8
// (include/strata/poly/kernels.hpp:73; the CUDA body src/poly/kernels.cu:531 store_q8_kernel,
// wrapper 2292). The at variant (store_q8_at, :553) stays a trap, like its store_f16_at twin:
// graph recording is off (--no-graphs), no path reads a card-time row yet (PORT-STATUS).
//
// BITWISE target: q8_rint mirrors CUDA rintf (ties to EVEN; GLSL round() ties away); the scale
// is f16_rn_bits(amax/127) (the helper native_mmvq_vk_parity proved byte-exact vs __float2half);
// codes are clamp(rint(x*(1/d)), +-127) with the EXPLICIT reciprocal the CUDA body computes.
//
// Layout laws (store_f16.comp's words; PORT-NOTES: glslc rejects byte arrays in std430):
//   codes: a block's 32 int8 live in 8 uint words, little-endian lanes (word j = values 4j..4j+3);
//   scales: two consecutive f16 a word (low, high); one thread does TWO blocks so each scale
//   word is written once. nb = n/32 must be EVEN for the pairing (every q8-capable model has
//   n a multiple of 64; qwen35moe n_kv*head_dim = 256). A throw beats a fake (the CUDA wrapper
//   throws on its own n%32 law too, kernels.cu:2293).
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "store_q8.spv.h"

const vk::Kernel& pl_store_q8() {
    static const vk::Kernel k = vk::kernel_load("store_q8", store_q8_spv, store_q8_spv_len,
                                                4, 16, 256, false);
    return k;
}
}  // namespace

void store_q8(const float* x, int8_t* codes, uint16_t* scales, int n, int64_t pos, int rows, void* stream) {
    if ((n & 31) != 0) throw std::invalid_argument("store_q8: n is not a multiple of 32");
    if ((n & 63) != 0) throw std::invalid_argument("store_q8: n/32 must be even here (scale-word pairing)");
    const size_t all = (size_t)n * (size_t)rows;
    if (!all) return;
    const uint64_t nb = (uint64_t)(n >> 5);
    const uint32_t pairs = (uint32_t)(nb * (size_t)rows / 2);
    const struct { uint32_t n, nb, pairs; int32_t ring; } pc = {(uint32_t)n, (uint32_t)nb, pairs, -1};
    // positional: bindings start AT pos (the byte-offset math the CUDA wrapper does on its pointers)
    const uint64_t bufs[4] = {strata_vk_compat::off(x),
                              strata_vk_compat::off(codes) + (uint64_t)pos * (uint64_t)n,
                              strata_vk_compat::off(scales) + (uint64_t)pos * nb * 2,
                              strata_vk_compat::off(x)};
    const uint64_t sizes[4] = {all * 4, (size_t)rows * (size_t)n, (size_t)rows * nb * 2, 8};
    vk::kernel_launch_sized(pl_store_q8(), (int)((pairs + 255) / 256), bufs, sizes, 4, &pc);
}

}  // namespace strata::poly
