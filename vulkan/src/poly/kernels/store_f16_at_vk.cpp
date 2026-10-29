// vulkan/src/poly/kernels/store_f16_at_vk.cpp - the Vulkan launcher for strata::poly::store_f16_at
// (include/strata/poly/kernels.hpp; CUDA body src/poly/kernels.cu:522 store_f16_at_kernel + the
// wrapper at :2288, which writes (half*)cache at rows read from the card cell `at`). The graph
// path (engine.cpp's replayed decode arms: attn_store with at != null) demands it: the row is
// the device-time h_at_ cell the engine copies OUTSIDE the graph, so one captured launch serves
// every replay with different positions - the whole point of the _at arms.
//
// BITWISE target: the same f16_rn_bits path store_f16_vk.cpp pins; the row math of CUDA :525
// (ring > 0 ? (*at + r) % ring : *at + r) with the int31 law of attend_score.comp: *at read as
// the low u32 word (the launcher guards at + rows < 2^31, mirroring the runtime's own int31
// position law; engine positions are context rows, far below it).
// Bounds: the cache binding starts at the cache base and is bounded to the piece tail (the
// engine's allocations never straddle pieces; every card-time row the cell can name is below
// the allocation end); x is rows*n floats; the AT binding is the 8-byte cell.
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "store_f16_at.spv.h"

const vk::Kernel& pl_store_f16_at() {
    static const vk::Kernel k = vk::kernel_load("store_f16_at", store_f16_at_spv, store_f16_at_spv_len,
                                                3, 16, 256, false);
    return k;
}
}  // namespace

void store_f16_at(const float* x, uint16_t* cache, int n, const long long* at, int rows, int ring, void* stream) {
    if (rows <= 0 || n <= 0) return;
    const size_t all = (size_t)n * (size_t)rows;
    if (all % 2) throw std::invalid_argument("store_f16_at: an odd element count cannot be word-addressed here");
    if (uintptr_t(x) % 4 || uintptr_t(cache) % 4 || uintptr_t(at) % 4)
        throw std::invalid_argument("store_f16_at: unaligned word-addressed buffers");
    const uint64_t pairs = all / 2, row_pairs = (uint64_t)n / 2;
    if (pairs > 0x7FFFFFFFull) throw std::invalid_argument("store_f16_at: too many f16 words for one launch");
    const int32_t rp = (int32_t)ring;
    const struct { uint32_t pairs, row_pairs, at_lo; int32_t ring; } pc =
        {(uint32_t)pairs, (uint32_t)row_pairs, 0u, rp};
    const uint64_t xo = strata_vk_compat::off(x), co = strata_vk_compat::off(cache),
                   ao = strata_vk_compat::off((const void*)at);
    const uint64_t bufs[3] = {xo, co, ao};
    const uint64_t ctail = vk::piece_tail(co);
    if (!ctail) throw std::runtime_error("store_f16_at: the cache binding is not inside an arena piece");
    const uint64_t sizes[3] = {all * 4, ctail, 8};
    vk::kernel_launch_sized(pl_store_f16_at(), (int)((pairs + 255) / 256), bufs, sizes, 3, &pc);
}

}  // namespace strata::poly
