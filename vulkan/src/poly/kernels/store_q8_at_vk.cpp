// vulkan/src/poly/kernels/store_q8_at_vk.cpp - the Vulkan launcher for strata::poly::store_q8_at
// (include/strata/poly/kernels.hpp:75; CUDA body src/poly/kernels.cu:553, wrapper :2301). The
// graph-path sibling of store_q8_vk.cpp: the target cache row comes from the device cell (the
// engine writes h_at_/at_ outside the captured graph), so one replay serves every token's
// position. Same q8 laws as store_q8.comp (q8_rint ties-to-even, f16_rn_bits scales, 4-int8
// code words, 2-f16 scale words, n/32 even); int31 rows via the at cell's low word.
// Whole-cache bindings bounded by their arena piece tails (engine allocations never straddle
// pieces and card-time rows stay below the allocation end).
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "store_q8_at.spv.h"

const vk::Kernel& pl_store_q8_at() {
    static const vk::Kernel k = vk::kernel_load("store_q8_at", store_q8_at_spv, store_q8_at_spv_len,
                                                4, 16, 256, false);
    return k;
}
}  // namespace

void store_q8_at(const float* x, int8_t* codes, uint16_t* scales, int n, const long long* at, int rows, int ring,
                 void* stream) {
    if ((n & 31) != 0) throw std::invalid_argument("store_q8_at: n is not a multiple of 32");
    if ((n & 63) != 0) throw std::invalid_argument("store_q8_at: n/32 must be even here (scale-word pairing)");
    const size_t all = (size_t)n * (size_t)rows;
    if (!all) return;
    const uint64_t nb = (uint64_t)(n >> 5);
    const uint32_t pairs = (uint32_t)(nb * (size_t)rows / 2);
    const struct { uint32_t n, nb, pairs; int32_t ring; } pc = {(uint32_t)n, (uint32_t)nb, pairs, (int32_t)ring};
    const uint64_t xo = strata_vk_compat::off(x), co = strata_vk_compat::off(codes),
                   so = strata_vk_compat::off(scales), ao = strata_vk_compat::off((const void*)at);
    const uint64_t ctail = vk::piece_tail(co), stail = vk::piece_tail(so);
    if (!ctail || !stail) throw std::runtime_error("store_q8_at: a cache binding is not inside an arena piece");
    const uint64_t bufs[4] = {xo, co, so, ao};
    const uint64_t sizes[4] = {all * 4, ctail, stail, 8};
    vk::kernel_launch_sized(pl_store_q8_at(), (int)((pairs + 255) / 256), bufs, sizes, 4, &pc);
}

}  // namespace strata::poly
