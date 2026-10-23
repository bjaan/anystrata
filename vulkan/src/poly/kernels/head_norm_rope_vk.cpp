// vulkan/src/poly/kernels/head_norm_rope_vk.cpp - the Vulkan launcher for
// strata::poly::head_norm_rope (include/strata/poly/kernels.hpp:166; the CUDA body is
// src/poly/kernels.cu:994 head_norm_rope_kernel + the wrapper at :2442).
//
// One workgroup per token-and-head; the workgroup size IS the CUDA launcher's
// `dim >= 256 ? 256 : 128` (2445) - the strided ss loop's partition is fp contract, so
// the build compiles two variants (-DHNR_THREADS=128|256).
//
// null arms (w, freq, axes): Vulkan needs legal bindings for every buffer; bind x's span
// with the has_* flag clear, exactly like rms_norm_vk.cpp does for w. `at` is the engine's
// on-card row counter (at_ + 1, two i64 words little-endian: the low word first); the
// kernel reads word 0 and stays in int32 range (sequence rows), so no shaderInt64.
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace strata::poly {
namespace {
#include "head_norm_rope_128.spv.h"
#include "head_norm_rope_256.spv.h"

const vk::Kernel& for_threads(int threads) {
    switch (threads) {
        case 128: {
            static const vk::Kernel k = vk::kernel_load("head_norm_rope_128", head_norm_rope_128_spv,
                                                        head_norm_rope_128_spv_len, 7, 64, 128, false);
            return k;
        }
        case 256: {
            static const vk::Kernel k = vk::kernel_load("head_norm_rope_256", head_norm_rope_256_spv,
                                                        head_norm_rope_256_spv_len, 7, 64, 256, false);
            return k;
        }
        default: throw std::runtime_error("head_norm_rope: workgroup size must be 128 or 256");
    }
}
inline uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
}  // namespace

void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream, const float* freq,
                    const int32_t* axes, int sections, const long long* at) {
    if (rows <= 0 || heads <= 0 || dim <= 0) return;
    if (n_rot % 2 || n_rot > dim) throw std::invalid_argument("head_norm_rope: bad n_rot");
    if (axes && sections <= 0) throw std::invalid_argument("head_norm_rope: axes need sections");
    const int blocks = rows * heads;
    const int threads = dim >= 256 ? 256 : 128;   // the CUDA launcher's choice (2445)

    const uint64_t xo = strata_vk_compat::off(x), yo = strata_vk_compat::off(y);
    // The kernel reads x up to (blocks-1)*x_stride + dim floats (CUDA 1000/1003).
    const uint64_t xspan = ((uint64_t)(blocks - 1) * (uint64_t)x_stride + (uint64_t)dim) * 4;
    const uint64_t wo = w ? strata_vk_compat::off(w) : xo;
    const uint64_t fo = freq ? strata_vk_compat::off(freq) : xo;
    const uint64_t ao = axes ? strata_vk_compat::off(axes) : xo;
    const uint64_t to = at ? strata_vk_compat::off(at) : xo;
    // The mantissa table m[j] = (float) std::pow((double) theta_scale, (double) j): CUDA's
    // powf(theta_scale, j) (kernels.cu:1016) is a correctly-rounded-to-few-ulp libdevice
    // result; the hardware v_log/v_exp chain is not, so the launcher pow's host-side once.
    // The bump arena never frees: cache one upload per (theta bits, half) pair - the model
    // uses one theta_scale for its whole run.
    static std::mutex mant_m;
    static std::vector<std::pair<std::pair<uint32_t, int>, uint64_t>> mant;   // ((bits,half), offset)
    uint64_t mo = 0;
    {
        std::lock_guard<std::mutex> lk(mant_m);
        const uint32_t tb = bits(theta_scale);
        for (auto& e : mant) if (e.first.first == tb && e.first.second == n_rot / 2) mo = e.second;
        if (!mo) {
            const int half = n_rot / 2;
            std::vector<float> m((size_t)half);
            const double td = theta_scale;
            for (int j = 0; j < half; ++j) m[j] = (float) std::pow(td, (double) j);
            mo = vk::Arena::instance().alloc((size_t)half * 4);
            vk::copy_h2d((void*)mo, m.data(), (size_t)half * 4, nullptr);
            mant.push_back({{tb, half}, mo});
        }
    }
    const uint64_t bufs[7] = {xo, wo, yo, ao, fo, to, mo};
    const uint64_t sizes[7] = {xspan, w ? (uint64_t)dim * 4 : xspan, (uint64_t)blocks * dim * 4,
                               axes ? (uint64_t)rows * 3 * 4 : xspan,
                               freq ? (uint64_t)(n_rot / 2) * 4 : xspan,
                               at ? 8u : xspan, (uint64_t)(n_rot / 2) * 4};
    uint32_t pc[16] = {};
    pc[0] = (uint32_t)dim; pc[1] = (uint32_t)x_stride; pc[2] = (uint32_t)heads; pc[3] = (uint32_t)n_rot;
    pc[4] = bits(theta_scale); pc[5] = bits(eps);
    pc[6] = (uint32_t)(uint64_t)pos; pc[7] = (uint32_t)((uint64_t)pos >> 32);
    pc[8] = (uint32_t)sections; pc[9] = w ? 1u : 0u; pc[10] = axes ? 1u : 0u; pc[11] = freq ? 1u : 0u;
    pc[12] = at ? 1u : 0u;
    vk::kernel_launch_sized(for_threads(threads), blocks, bufs, sizes, 7, pc);
}

}  // namespace strata::poly
