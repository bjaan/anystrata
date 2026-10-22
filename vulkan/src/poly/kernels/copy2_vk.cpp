// vulkan/src/poly/kernels/copy2_vk.cpp - the Vulkan launcher for strata::poly::copy2
// (include/strata/poly/kernels.hpp:143; the CUDA body is src/poly/kernels.cu:859 copy2_kernel +
// the wrapper at :2395).
//
// copy2 moves two vectors of 4-byte words in one launch, thread i < n0 copies pair 0, the next n1
// copy pair 1. The CUDA kernel reads both pinned host and device pointers through unified
// addressing; a Vulkan kernel sees only arena buffers, so the launcher classifies each pointer
// (arena device pointer = offset + kPtrBias under the arena's cursor; everything else is host RAM
// from cudaHostAlloc/cudaMallocHost or ordinary malloc) and dispatches per pair:
//   * all four pointers arena-side: ONE copy2.comp launch reproducing the kernel's two-range mapping;
//   * any host-side pointer: the staged copies (copy_h2d/copy_d2h walk piece boundaries themselves);
//   * both host: plain memcpy.
// Semantics identical to CUDA's (word copies, n0/n1 in 4-byte words; a null pair carries n=0).
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "copy2.spv.h"     // copy2_spv[] / copy2_spv_len, embedded by tools/embed_spv.py

bool is_device(const void* p) {
    const uint64_t o = strata_vk_compat::off(p);
    return o < strata::vk::Arena::instance().used();   // an arena offset (bias already stripped)
}
}  // namespace

void copy2(const void* x0, void* y0, int n0, const void* x1, void* y1, int n1, void* stream) {
    if (n0 < 0 || n1 < 0) throw std::runtime_error("copy2: negative element count");
    const bool live0 = n0 > 0, live1 = n1 > 0;
    if (!live0 && !live1) return;

    const bool dev0 = live0 && is_device(x0) && is_device(y0);
    const bool dev1 = live1 && is_device(x1) && is_device(y1);

    if ((dev0 || !live0) && (dev1 || !live1)) {
        // every live pair is arena-side: one dispatch, the CUDA kernel's exact two-range mapping.
        static const vk::Kernel k = vk::kernel_load(
            "copy2", copy2_spv, copy2_spv_len, /*storage buffers*/ 4, /*push bytes*/ 8,
            /*workgroup*/ 256, /*force 32-wide subgroups*/ false);
        // bindings: [x0,y0,x1,y1]; a dead pair aliases the live one (its branch never runs).
        const uint64_t a0 = live0 ? strata_vk_compat::off(x0) : strata_vk_compat::off(x1);
        const uint64_t b0 = live0 ? strata_vk_compat::off(y0) : strata_vk_compat::off(y1);
        const uint64_t a1 = live1 ? strata_vk_compat::off(x1) : a0;
        const uint64_t b1 = live1 ? strata_vk_compat::off(y1) : b0;
        const uint64_t bufs[4]  = {a0, b0, a1, b1};
        const uint64_t sizes[4] = {live0 ? (uint64_t)n0 * 4 : 4, live0 ? (uint64_t)n0 * 4 : 4,
                                   live1 ? (uint64_t)n1 * 4 : 4, live1 ? (uint64_t)n1 * 4 : 4};
        const uint32_t pc[2] = {(uint32_t)(live0 ? n0 : 0), (uint32_t)(live1 ? n1 : 0)};
        vk::kernel_launch_sized(k, (n0 + n1 + 255) / 256, bufs, sizes, 4, pc);
        return;
    }

    // any pair mixing host memory: staged per-pair copies (correct through the piece chain).
    auto pair = [&](const void* x, void* y, int n) {
        if (n <= 0) return;
        const size_t bytes = (size_t)n * 4;
        const bool dx = is_device(x), dy = is_device(y);
        if (dx && dy) {    // D2D through a host scratch (y is a device pointer: never a memcpy target)
            std::vector<uint8_t> scratch(bytes);
            vk::copy_d2h(scratch.data(), (void*)(uintptr_t)strata_vk_compat::off(x), bytes, stream);
            vk::copy_h2d((void*)(uintptr_t)strata_vk_compat::off(y), scratch.data(), bytes, stream);
        }
        else if (dx)       vk::copy_d2h(y, (void*)(uintptr_t)strata_vk_compat::off(x), bytes, stream);
        else if (dy)       vk::copy_h2d((void*)(uintptr_t)strata_vk_compat::off(y), x, bytes, stream);
        else               std::memcpy(y, x, bytes);
    };
    pair(x0, y0, n0);
    pair(x1, y1, n1);
}

}  // namespace strata::poly
