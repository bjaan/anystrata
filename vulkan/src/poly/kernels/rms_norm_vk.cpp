// vulkan/src/poly/kernels/rms_norm_vk.cpp - the Vulkan launcher for strata::poly::rms_norm
// (include/strata/poly/kernels.hpp:32; the CUDA body is src/poly/kernels.cu:159 rms_norm_kernel +
// the wrapper at :2216).
//
// One workgroup per row; the workgroup size IS the CUDA launcher's sum_threads(n) (128/256/512/1024),
// because the strided ss loop's iteration partition is part of the fp contract: the parity test's
// tolerance is only 1e-5 relative, and matching the partition keeps the accumulation order identical
// modulo contraction. Hence four compiled variants (-DRMS_THREADS=N in CMake) and a size-selected
// pipeline here.
//
// The CUDA block_sum's warp_sum is a __shfl_xor chain at WARP=32; gfx1013 subgroups are 64-wide by
// default, and an xor chain with offsets 16..1 does NOT cross a 32-lane aligned pair boundary —
// but with 64-wide subgroups the partner lanes live in the other half of the subgroup and the chain
// still sums correctly WITHIN each 32-lane aligned block (xor of offsets <= 16 never leaves it), so
// the sum is per-warp either way. part[tid>>5] indexes by the LOCAL invocation id (same on both),
// and the final part[0..threads/32) walk and the multiply/scale loop are index-order, not subgroup-
// order. The kernel therefore needs no forced 32-wide subgroups; the .comp documents the argument.
//
// The w == nullptr case: CUDA passes a null w and the kernel branches. Vulkan needs four legal
// bindings regardless: bind x's buffer for w with has_w=0.
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "rms_norm_128.spv.h"
#include "rms_norm_256.spv.h"
#include "rms_norm_512.spv.h"
#include "rms_norm_1024.spv.h"

// Load (once) and return the pipeline for workgroup size `threads`.
const vk::Kernel& for_threads(int threads) {
    switch (threads) {
        case 128: {
            static const vk::Kernel k = vk::kernel_load("rms_norm_128", rms_norm_128_spv,
                                                        rms_norm_128_spv_len, 3, 16, 128, false);
            return k;
        }
        case 256: {
            static const vk::Kernel k = vk::kernel_load("rms_norm_256", rms_norm_256_spv,
                                                        rms_norm_256_spv_len, 3, 16, 256, false);
            return k;
        }
        case 512: {
            static const vk::Kernel k = vk::kernel_load("rms_norm_512", rms_norm_512_spv,
                                                        rms_norm_512_spv_len, 3, 16, 512, false);
            return k;
        }
        case 1024: {
            static const vk::Kernel k = vk::kernel_load("rms_norm_1024", rms_norm_1024_spv,
                                                        rms_norm_1024_spv_len, 3, 16, 1024, false);
            return k;
        }
        default: throw std::runtime_error("rms_norm: workgroup size must be a sum_threads value");
    }
}
}  // namespace

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream,
              int y_stride) {
    if (rows <= 0 || n <= 0) return;
    const int threads = sum_threads(n);
    const int ys = y_stride ? y_stride : n;
    const bool has_w = w != nullptr;

    const uint64_t xo = strata_vk_compat::off(x), yo = strata_vk_compat::off(y);
    const uint64_t wo = has_w ? strata_vk_compat::off(w) : xo;
    const uint64_t bufs[3] = {xo, wo, yo};
    // The .comp indexes bx as rows*n and by as rows*ys; sizes bound the whole span each buffer may touch.
    const uint64_t sizes[3] = {(uint64_t)rows * n * 4, has_w ? (uint64_t)n * 4 : (uint64_t)rows * n * 4,
                               (uint64_t)rows * ys * 4};
    uint32_t eps_bits; std::memcpy(&eps_bits, &eps, 4);
    const uint32_t pc[4] = {(uint32_t)n, (uint32_t)ys, (uint32_t)(has_w ? 1 : 0), eps_bits};
    vk::kernel_launch_sized(for_threads(threads), rows, bufs, sizes, 3, pc);
}

}  // namespace strata::poly
