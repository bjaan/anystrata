// vulkan/src/poly/kernels/conv4_silu_vk.cpp - the Vulkan launcher for strata::poly::conv4_silu
// (include/strata/poly/kernels.hpp:240; the CUDA bodies are src/poly/kernels.cu:1864 walk kernel,
// :1899 rows kernel, :1972 state kernel, wrapper at :2673).
//
// The wrapper's arm choice is transcribed EXACTLY (rows >= CONV_ROWS(16) && no snapshots -> rows +
// state kernels; else the walk kernel): the sum's partition differs per arm and the walk kernel's
// per-thread token loop is where a snapshot call goes. Three compiled variants:
// conv4_walk (CONV_VARIANT=0), conv4_rows (CONV_VARIANT=1), conv4_state.
//
// A null snap is bound as out with snaps=0, so the walk kernel's snapshot arm never fires (Vulkan
// needs legal bindings for all 5 slots).
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "conv4_walk.spv.h"
#include "conv4_rows.spv.h"
#include "conv4_state.spv.h"

const vk::Kernel& pl_walk() {
    static const vk::Kernel k = vk::kernel_load("conv4_walk", conv4_walk_spv, conv4_walk_spv_len, 5, 32, 256, false);
    return k;
}
const vk::Kernel& pl_rows() {
    static const vk::Kernel k = vk::kernel_load("conv4_rows", conv4_rows_spv, conv4_rows_spv_len, 5, 32, 256, false);
    return k;
}
const vk::Kernel& pl_state() {
    static const vk::Kernel k = vk::kernel_load("conv4_state", conv4_state_spv, conv4_state_spv_len, 2, 8, 256, false);
    return k;
}
constexpr int CONV_ROWS_HINT = 16;   // CUDA kernels.cu:1896 CONV_ROWS - the wrapper's arm switch
}  // namespace

void conv4_silu(const float* x, float* state, const float* w, float* out, int channels, int rows,
                void* stream, float* snap, size_t stride, unsigned snaps) {
    if (rows <= 0) return;
    if (channels <= 0) throw std::runtime_error("conv4_silu: channels must be positive");
    const uint64_t xo = strata_vk_compat::off(x), so = strata_vk_compat::off(state);
    const uint64_t wo = strata_vk_compat::off(w), oo = strata_vk_compat::off(out);
    const uint64_t no = strata_vk_compat::off(snap ? snap : out);   // bound-always; snaps gates use
    const uint64_t blocks = (uint64_t)((channels + 255) / 256);
    const size_t xspan = (size_t)rows * channels * 4, ospan = xspan;
    if (rows >= CONV_ROWS_HINT && (!snap || !snaps)) {
        const uint64_t bufs[5] = {xo, so, wo, oo, no};
        const uint64_t sizes[5] = {xspan, (uint64_t)channels * 3 * 4, (uint64_t)channels * 4 * 4,
                                   ospan, ospan};
        const uint32_t pc[8] = {(uint32_t)channels, (uint32_t)rows, 0, 0, 0, 0, 0, 0};
        if (blocks * (uint64_t)rows > 0x7FFFFFFFLL)
            throw std::runtime_error("conv4_silu: grid overflows the flattened dispatch");
        vk::kernel_launch_sized(pl_rows(), (int)(blocks * (uint64_t)rows), bufs, sizes, 5, pc);
        const uint64_t sb[2] = {xo, so};
        const uint64_t ss[2] = {xspan, (uint64_t)channels * 3 * 4};
        const uint32_t spc[2] = {(uint32_t)channels, (uint32_t)rows};
        vk::kernel_launch_sized(pl_state(), (int)blocks, sb, ss, 2, spc);
        return;
    }
    // The walk arm: state in place, snapshots per snap_of().
    if (stride >> 32) throw std::runtime_error("conv4_silu: snapshot stride exceeds uint32");
    // Touched extent: slot j (j < popcount(snaps)) writes 3*channel floats at snap + j*stride,
    // so the last write ends at (popcount-1)*stride + 3*channels floats -- NOT popcount*stride
    // (the extra tail pushed the span past the arena's piece boundary once speculation wrote
    // snapshots; Task 2.19, same fix as gdn_step's state-snap arm).
    const uint64_t conv_state = (uint64_t)channels * 3 * 4;
    const uint64_t snap_span = (snap && stride && snaps)
        ? ((uint64_t)__builtin_popcount(snaps) - 1u) * (uint64_t)stride * 4 + conv_state
        : ospan;
    const uint64_t bufs[5] = {xo, so, wo, oo, no};
    const uint64_t sizes[5] = {xspan, (uint64_t)channels * 3 * 4, (uint64_t)channels * 4 * 4,
                               ospan, snap_span};
    const uint32_t pc[8] = {(uint32_t)channels, (uint32_t)rows, 0, 0,
                            (uint32_t)(stride & 0xFFFFFFFFu), (uint32_t)(stride >> 32), snaps, 0};
    vk::kernel_launch_sized(pl_walk(), (int)blocks, bufs, sizes, 5, pc);
}

}  // namespace strata::poly
