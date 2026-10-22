// vulkan/src/poly/kernels/gemv_vk.cpp - the Vulkan launcher for strata::poly::gemv_f32
// (include/strata/poly/kernels.hpp:19; the CUDA body is src/poly/kernels.cu:75 gemv_kernel +
// the gemv<> wrapper at :103 and the entry at :2179).
//
// Five compiled variants (-DGEMV_THREADS=N for 32/128/256/512/1024); the launcher picks the size
// EXACTLY as the CUDA wrapper does (n_in >= 128 ? sum_threads(n_in) : 32) — the strided loop's
// partition is part of the fp contract. The CUDA grid dim3(n_out, rows) is flattened to groups_x =
// n_out*rows (independent workgroups, so the sum a block produces is unchanged).
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "gemv_32.spv.h"
#include "gemv_128.spv.h"
#include "gemv_256.spv.h"
#include "gemv_512.spv.h"
#include "gemv_1024.spv.h"

const vk::Kernel& for_threads(int threads) {
    switch (threads) {
        case 32:   { static const vk::Kernel k = vk::kernel_load("gemv_32",   gemv_32_spv,   gemv_32_spv_len,   3, 16, 32,   false); return k; }
        case 128:  { static const vk::Kernel k = vk::kernel_load("gemv_128",  gemv_128_spv,  gemv_128_spv_len,  3, 16, 128,  false); return k; }
        case 256:  { static const vk::Kernel k = vk::kernel_load("gemv_256",  gemv_256_spv,  gemv_256_spv_len,  3, 16, 256,  false); return k; }
        case 512:  { static const vk::Kernel k = vk::kernel_load("gemv_512",  gemv_512_spv,  gemv_512_spv_len,  3, 16, 512,  false); return k; }
        case 1024: { static const vk::Kernel k = vk::kernel_load("gemv_1024", gemv_1024_spv, gemv_1024_spv_len, 3, 16, 1024, false); return k; }
        default: throw std::runtime_error("gemv: workgroup size must be 32 or a sum_threads value");
    }
}
}  // namespace

void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
              void* stream) {
    if (rows <= 0 || n_in <= 0 || n_out <= 0) return;
    if (rows_per_x <= 0 || n_out % rows_per_x != 0)
        throw std::runtime_error("gemv_f32: rows_per_x must divide n_out");
    const int threads = n_in >= 128 ? sum_threads(n_in) : 32;   // the CUDA wrapper's choice, exactly
    const long long groups = (long long)n_out * rows;
    if (groups > 0x7FFFFFFFLL) throw std::runtime_error("gemv_f32: n_out*rows overflows the dispatch");

    const uint64_t wo = strata_vk_compat::off(w), xo = strata_vk_compat::off(x), yo = strata_vk_compat::off(y);
    const uint64_t bufs[3] = {wo, xo, yo};
    // Spans the .comp may touch: w n_out*n_in; x rows*(n_out/rows_per_x)*n_in; y rows*n_out.
    const uint64_t sizes[3] = {(uint64_t)n_out * n_in * 4,
                               (uint64_t)rows * (n_out / rows_per_x) * n_in * 4,
                               (uint64_t)rows * n_out * 4};
    const uint32_t pc[4] = {(uint32_t)n_in, (uint32_t)n_out, (uint32_t)rows_per_x, 0u};
    vk::kernel_launch_sized(for_threads(threads), (int)groups, bufs, sizes, 3, pc);
}

}  // namespace strata::poly
