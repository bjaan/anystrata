// vulkan/src/tests/poly_kernels_vk_parity.cpp - unit parity for the Task-2.1 trio:
// strata::poly::copy2, strata::poly::rms_norm, strata::kernels::native_quantize_q8_1.
//
// The poly kernels have no in-tree _parity.cpp (their coverage is poly_expert_parity, which needs a
// whole model), so this writes the reference the tree's own way: a scalar transcription of the CUDA
// body in the same term order (the .comp headers cite the exact line numbers), run against the
// device through the arena. Tolerance law (PORT-STATUS): copy2 and native_quantize_q8_1 are
// BYTE-EXACT (word copies; the quantizer is exact integer arithmetic after one roundf — no rounding
// freedom anywhere); rms_norm holds rel 1e-5 like the gemv family (fp32 contraction may differ
// between glslc's a*b+c and RADV's fma).
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata;

namespace {

// ---- host f16 encode/decode matching the compat/cuda_fp16.h rules (round-to-nearest-even, finite inputs)
uint16_t f32_to_f16(float f) {
    uint32_t b; std::memcpy(&b, &f, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    int32_t exp = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t man = b & 0x7FFFFFu;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);                 // fp16 overflow -> inf (not exercised)
    if (exp <= 0) return (uint16_t)sign;                              // underflow -> 0 (inputs stay normal)
    // round-to-nearest-even on the 13 dropped bits
    uint32_t hman = man >> 13; uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (hman & 1u))) hman++;
    if (hman == 0x400u) { hman = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7C00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | hman);
}
float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) f = sign << 31;
    else f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    float out; std::memcpy(&out, &f, 4); return out;
}

int fails = 0;
void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) fails++;
}

// ---- the CUDA-body references, transcribed
void ref_rms_norm(const std::vector<float>& x, const std::vector<float>* w, std::vector<float>& y,
                  int n, int rows, float eps, int y_stride) {
    for (int r = 0; r < rows; ++r) {
        double ss = 0.0;                                   // host reference sums exactly; the rel 1e-5
        for (int i = 0; i < n; ++i) { float v = x[(size_t)r * n + i]; ss += (double)v * v; }   // bound absorbs order
        const float scale = (float)(1.0 / std::sqrt(ss / n + eps));
        for (int i = 0; i < n; ++i) {
            const float v = x[(size_t)r * n + i];
            y[(size_t)r * y_stride + i] = w ? v * scale * (*w)[i] : v * scale;
        }
    }
}

struct Q81Ref { uint16_t d_bits, sum_bits; int8_t qs[32]; };
std::vector<Q81Ref> ref_quant(const std::vector<float>& x) {   // n = 32k
    std::vector<Q81Ref> out(x.size() / 32);
    for (size_t blk = 0; blk < out.size(); ++blk) {
        float amax = 0.f;
        float l[32];                                        // the warp_sum xor tree, lane by lane,
        for (int j = 0; j < 32; ++j) { const float v = x[blk * 32 + j]; amax = std::fmax(amax, std::fabs(v)); l[j] = v; }
        for (int o = 16; o > 0; o >>= 1) { float p[32]; for (int j = 0; j < 32; ++j) p[j] = l[j];
          for (int j = 0; j < 32; ++j) l[j] = l[j] + p[j ^ o]; }   // shuffle tree: all reads see the OLD round
        const float sum = l[0];
        float d = amax / 127.0f;
        if (std::fabs(d) > 65504.f) d = std::copysign(65504.f, d);       // q8_1_finite
        float sd = sum; if (std::fabs(sd) > 65504.f) sd = std::copysign(65504.f, sd);
        Q81Ref b; b.d_bits = f32_to_f16(d); b.sum_bits = f32_to_f16(sd);
        for (int j = 0; j < 32; ++j) {
            const float v = x[blk * 32 + j];
            if (amax == 0.f) { b.qs[j] = 0; continue; }
            float q = std::roundf(v / d);
            q = q > 127.f ? 127.f : (q < -127.f ? -127.f : q);
            b.qs[j] = (int8_t)q;
        }
        out[blk] = b;
    }
    return out;
}

}  // namespace

int main() {
    try { vk::device_info(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "poly_kernels_vk_parity: %s\nSKIP\n", e.what());
        return 77;
    }
    vk::Arena& a = vk::Arena::instance();

    // ---- copy2: two word ranges in one launch, incl. the dead-pair arm the engine uses (copy2(x,y,n,null,null,0)).
    {
        const int n0 = 1024, n1 = 300;
        std::vector<uint32_t> x0(n0), x1(n1);
        std::mt19937 rng(7);
        for (auto& v : x0) v = rng();
        for (auto& v : x1) v = rng();
        const uint64_t a_x0 = a.alloc(n0 * 4), a_y0 = a.alloc(n0 * 4);
        const uint64_t a_x1 = a.alloc(n1 * 4), a_y1 = a.alloc(n1 * 4);
        vk::copy_h2d((void*)a_x0, x0.data(), n0 * 4, nullptr);
        vk::copy_h2d((void*)a_x1, x1.data(), n1 * 4, nullptr);
        poly::copy2(strata_vk_compat::wrap(a_x0), strata_vk_compat::wrap(a_y0), n0,
                    strata_vk_compat::wrap(a_x1), strata_vk_compat::wrap(a_y1), n1, nullptr);
        std::vector<uint32_t> y0(n0, 0), y1(n1, 0);
        vk::copy_d2h(y0.data(), (void*)a_y0, n0 * 4, nullptr);
        vk::copy_d2h(y1.data(), (void*)a_y1, n1 * 4, nullptr);
        check(y0 == x0 && y1 == x1, "copy2 both pairs, device side");
        // mixed arm: pinned host source -> device (the engine's at_ staging shape)
        std::vector<uint32_t> hx(64, 0xDEADBEEFu);
        poly::copy2(hx.data(), strata_vk_compat::wrap(a_y1), 64, nullptr, nullptr, 0, nullptr);
        vk::copy_d2h(y1.data(), (void*)a_y1, n1 * 4, nullptr);
        check(std::equal(hx.begin(), hx.end(), y1.begin()) && y1[64] == x1[64],
              "copy2 host->device pair, dead second pair (untouched tail keeps prior values)");
    }

    // ---- rms_norm: the model's shape (n=2048, eps 1e-6), 40 rows, with and without weights,
    //      y_stride != n once (the engine's in-place-offset use), all sum_threads sizes exercised.
    {
        const int n = 2048, rows = 40; const float eps = 9.99999974752427e-07f;
        std::vector<float> x((size_t)rows * n), w(n), y((size_t)rows * n);
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> u(-4.f, 4.f);
        for (auto& v : x) v = u(rng);
        for (auto& v : w) v = u(rng);
        const uint64_t ax = a.alloc(x.size() * 4), aw = a.alloc(w.size() * 4), ay = a.alloc(y.size() * 4);
        vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
        vk::copy_h2d((void*)aw, w.data(), w.size() * 4, nullptr);
        poly::rms_norm((const float*)strata_vk_compat::wrap(ax), (const float*)strata_vk_compat::wrap(aw), (float*)strata_vk_compat::wrap(ay),
                       n, rows, eps, nullptr, 0);
        std::vector<float> got(y.size(), 0.f);
        vk::copy_d2h(got.data(), (void*)ay, got.size() * 4, nullptr);
        std::vector<float> ref; ref_rms_norm(x, &w, ref = std::vector<float>(y.size()), n, rows, eps, n);
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i)
            worst = std::fmax(worst, std::fabs((double)got[i] - ref[i]) / std::fmax(1e-3, std::fabs((double)ref[i])));
        std::printf("  rms_norm w: worst rel %.3e (tol 1e-5)\n", worst);
        check(worst < 1e-5, "rms_norm n=2048 (threads 512) weighted");
        // unweighted arm + y_stride arm at a different size (small n -> 128 threads)
        poly::rms_norm((const float*)strata_vk_compat::wrap(ax), nullptr, (float*)strata_vk_compat::wrap(ay), 256, rows, eps, nullptr, n);
        got.assign(y.size(), 0.f);
        vk::copy_d2h(got.data(), (void*)ay, got.size() * 4, nullptr);
        ref_rms_norm(x, nullptr, ref, 256, rows, eps, n);
        worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i)
            if ((size_t)i % n < 256)
                worst = std::fmax(worst, std::fabs((double)got[i] - ref[i]) / std::fmax(1e-3, std::fabs((double)ref[i])));
        std::printf("  rms_norm unw+stride: worst rel %.3e\n", worst);
        check(worst < 1e-5, "rms_norm n=256 unweighted, y_stride=2048 (128 threads)");
    }

    // ---- native_quantize_q8_1: BYTE-EXACT on d, sum, and all 32 qs, over a ramp with negatives,
    //      zeros, and a saturation case (xi/d beyond 127 cannot happen by construction but the amax==0
    //      and the +-127 clamp branches are forced), n_in=32*40, ncols=1.
    {
        const int n = 32 * 40;
        std::vector<float> x(n);
        std::mt19937 rng(23);
        for (int i = 0; i < n; ++i) x[i] = (float)((int)(rng() % 2001) - 1000) / 250.0f;
        for (int blk = 0; blk < n / 32; blk += 7) for (int j = 0; j < 32; ++j) x[blk * 32 + j] = 0.f;  // all-zero blocks
        for (int j = 0; j < 32; ++j) x[1 * 32 + j] = (j == 0) ? -500.0f : 1.0f;                        // clamp branch: v/d = -127...
        const uint64_t ax = a.alloc(n * 4), aq = a.alloc((size_t)(n / 32) * 36);
        vk::copy_h2d((void*)ax, x.data(), n * 4, nullptr);
        vk::fill_d((void*)aq, 0xFE, (size_t)(n / 32) * 36, nullptr);
        // engine-shaped pointers: the launcher strips the kPtrBias, so pass wrapped device pointers
        // (this exact call caught a raw-offset mistake in the first cut: 674144-64 -> crossed piece).
        kernels::native_quantize_q8_1((const float*)strata_vk_compat::wrap(ax), strata_vk_compat::wrap(aq),
                                      n, 1, strata_vk_compat::wrap(1000));
        // NOTE the stream: validate_stream demands non-null; any token works (one queue underneath).
        std::vector<uint8_t> q((size_t)(n / 32) * 36);
        vk::copy_d2h(q.data(), (void*)aq, q.size(), nullptr);
        const auto ref = ref_quant(x);
        int bad = 0;
        for (size_t blk = 0; blk < ref.size(); ++blk) {
            const uint8_t* p = q.data() + blk * 36;
            uint16_t db, sb; std::memcpy(&db, p, 2); std::memcpy(&sb, p + 2, 2);
            if (db != ref[blk].d_bits || sb != ref[blk].sum_bits) { if (bad < 4) std::printf("  blk %zu ds: got %04x/%04x want %04x/%04x\n", blk, db, sb, ref[blk].d_bits, ref[blk].sum_bits); ++bad; continue; }
            for (int j = 0; j < 32; ++j) if ((int8_t)p[4 + j] != ref[blk].qs[j]) {
                if (bad < 4) std::printf("  blk %zu qs[%d]: got %d want %d (x=%.3f d=%.6f)\n", blk, j, (int)(int8_t)p[4 + j], (int)ref[blk].qs[j], x[blk * 32 + j], f16_to_f32(ref[blk].d_bits));
                ++bad; break;
            }
        }
        std::printf("  native_quantize_q8_1: %zu blocks, mismatches %d\n", ref.size(), bad);
        check(bad == 0, "native_quantize_q8_1 byte-exact");
    }

    std::printf("poly_kernels_vk_parity %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
