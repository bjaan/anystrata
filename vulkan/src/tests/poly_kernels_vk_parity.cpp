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
#include <limits>
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

    // ---- gemv_f32 (Task 2.4): reference = gemv_kernel's partition transcribed (CUDA kernels.cu
    //      :75): one output row a block of sum_threads(n_in) (or 32 when n_in < 128), a strided
    //      loop, warp_sum, then the ordered part[] walk. The host reference sums the SAME per-thread
    //      strided order in fp32, so the only freedom is contraction (a*b vs fma): the gemv-family
    //      rel 1e-5 bound (s2_gemv_vk_parity / bf16_gemv_parity's tol). Shapes: the model's n_in
    //      2048 (512 threads) + arms that force each variant size (1024->1024, 4096->1024? no:
    //      sum_threads(4096)=1024, 1000 -> 256, 100 -> 32) + rows_per_x groups.
    {
        auto ref_gemv = [](const std::vector<float>& w, const std::vector<float>& x,
                           std::vector<float>& y, int n_in, int n_out, int rows_per_x, int rows,
                           int threads) {
            const int part = (threads + 31) / 32;
            for (int o = 0; o < n_out; ++o)
                for (int tok = 0; tok < rows; ++tok) {
                    const size_t wr = (size_t)o * n_in;
                    const size_t xr = ((size_t)tok * (n_out / rows_per_x) + o / rows_per_x) * n_in;
                    // thread t's fp32 strided sum: i = t, t+threads, ... in CUDA's order
                    std::vector<float> tacc(threads, 0.f);
                    for (int i = 0; i < n_in; ++i) tacc[i % threads] += w[wr + i] * x[xr + i];
                    // warp_sum per 32-lane warp: xor tree 16..1 (per-round copies, law #4), then
                    // the ordered part[0..part) walk.
                    std::vector<float> ws(part, 0.f);
                    for (int p = 0; p < part; ++p) {
                        float l[32];
                        for (int j = 0; j < 32; ++j) l[j] = (size_t)p * 32 + j < (size_t)threads ? tacc[p * 32 + j] : 0.f;
                        for (int off = 16; off > 0; off >>= 1) {
                            float pr[32]; for (int j = 0; j < 32; ++j) pr[j] = l[j];
                            for (int j = 0; j < 32; ++j) l[j] = l[j] + pr[j ^ off];
                        }
                        ws[p] = l[0];
                    }
                    float s = 0.f;
                    for (int p = 0; p < part; ++p) s += ws[p];
                    y[(size_t)tok * n_out + o] = s;
                }
        };
        struct Case { int n_in, n_out, rows_per_x, rows; };
        const Case cases[] = {
            {2048, 8192, 8192, 1},      // the model's dense projection shape (512 threads)
            {2048, 2048, 512, 2},       // rows_per_x grouping (per-head MLA reads)
            {4096, 128, 128, 1},        // 1024-thread variant
            {1000, 64, 64, 1},          // 256-thread variant (1000*4>... sum_threads -> 256)
            {100, 32, 32, 3},           // narrow-input arm: threads 32
        };
        std::mt19937 rng(31);
        std::uniform_real_distribution<float> u(-2.f, 2.f);
        for (const auto& c : cases) {
            const int threads = c.n_in >= 128 ? poly::sum_threads(c.n_in) : 32;
            std::vector<float> w((size_t)c.n_out * c.n_in), x((size_t)c.rows * (c.n_out / c.rows_per_x) * c.n_in),
                                 y((size_t)c.rows * c.n_out, 0.f), ref(y.size());
            for (auto& v : w) v = u(rng);
            for (auto& v : x) v = u(rng);
            const uint64_t aw = a.alloc(w.size() * 4), ax = a.alloc(x.size() * 4), ay = a.alloc(y.size() * 4);
            vk::copy_h2d((void*)aw, w.data(), w.size() * 4, nullptr);
            vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
            poly::gemv_f32((const float*)strata_vk_compat::wrap(aw), (const float*)strata_vk_compat::wrap(ax),
                           (float*)strata_vk_compat::wrap(ay), c.n_in, c.n_out, c.rows_per_x, c.rows, nullptr);
            vk::copy_d2h(y.data(), (void*)ay, y.size() * 4, nullptr);
            ref_gemv(w, x, ref, c.n_in, c.n_out, c.rows_per_x, c.rows, threads);
            double worst = 0.0; int bit = 1;
            for (size_t i = 0; i < y.size(); ++i) {
                if (std::memcmp(&y[i], &ref[i], 4) != 0) bit = 0;
                worst = std::fmax(worst, std::fabs((double)y[i] - ref[i]) / std::fmax(1e-3, std::fabs((double)ref[i])));
            }
            std::printf("  gemv_f32 n_in=%d n_out=%d rpx=%d rows=%d thr=%d: worst rel %.3e, bitwise %s\n",
                        c.n_in, c.n_out, c.rows_per_x, c.rows, threads, worst, bit ? "yes" : "no");
            check(worst < 1e-5, "gemv_f32 rel 1e-5 (gemv-family tol)");
        }
    }

    // ---- gemv_bf16 (Task 2.16): the bf16-weight twin. Host weights are encoded with the tree's
    //      bf16_from_f32 (bf16_bits.hpp:37 round-to-nearest-even) and the reference decodes with
    //      f32_from_bf16's exact shift (bf16_bits.hpp:50) — the same pair the .comp's word-pick +
    //      uintBitsToFloat(bits<<16) mirrors for ldw (CUDA kernels.cu:72). The partition is the
    //      f32 gemv's, so the sum inherits its bitwise behaviour; the rel 1e-5 bound is kept as
    //      the family gate. Same shapes as the f32 block.
    {
        auto ref_gemv_bf16 = [](const std::vector<uint16_t>& w, const std::vector<float>& x,
                                std::vector<float>& y, int n_in, int n_out, int rows_per_x, int rows,
                                int threads) {
            const int part = (threads + 31) / 32;
            for (int o = 0; o < n_out; ++o)
                for (int tok = 0; tok < rows; ++tok) {
                    const size_t wr = (size_t)o * n_in;
                    const size_t xr = ((size_t)tok * (n_out / rows_per_x) + o / rows_per_x) * n_in;
                    std::vector<float> tacc(threads, 0.f);
                    for (int i = 0; i < n_in; ++i) {
                        // f32_from_bf16: exact (uint32)h << 16 reinterpreted
                        uint32_t b = (uint32_t)w[wr + i] << 16; float wf;
                        std::memcpy(&wf, &b, 4);
                        tacc[i % threads] += wf * x[xr + i];
                    }
                    std::vector<float> ws(part, 0.f);
                    for (int p = 0; p < part; ++p) {
                        float l[32];
                        for (int j = 0; j < 32; ++j) l[j] = (size_t)p * 32 + j < (size_t)threads ? tacc[p * 32 + j] : 0.f;
                        for (int off = 16; off > 0; off >>= 1) {
                            float pr[32]; for (int j = 0; j < 32; ++j) pr[j] = l[j];
                            for (int j = 0; j < 32; ++j) l[j] = l[j] + pr[j ^ off];
                        }
                        ws[p] = l[0];
                    }
                    float s = 0.f;
                    for (int p = 0; p < part; ++p) s += ws[p];
                    y[(size_t)tok * n_out + o] = s;
                }
        };
        // bf16_from_f32 (bf16_bits.hpp:37) transcribed: NaN quieting not exercised (uniform finite).
        auto bf16_enc = [](float f) {
            uint32_t i; std::memcpy(&i, &f, 4);
            if ((i & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((i >> 16) | 64u);
            i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
            return (uint16_t)(i >> 16);
        };
        struct Case { int n_in, n_out, rows_per_x, rows; };
        const Case cases[] = {
            {2048, 8192, 8192, 1},      // the model's dense projection shape (512 threads)
            {2048, 2048, 512, 2},       // rows_per_x grouping
            {4096, 128, 128, 1},        // 1024-thread variant
            {1000, 64, 64, 1},          // 256-thread variant
            {100, 32, 32, 3},           // narrow-input arm: threads 32 (odd n_in: word-crossing)
            {257, 16, 16, 1},           // odd n_in at 128 threads: a row's bf16s straddle words
        };
        std::mt19937 rng(131);
        std::uniform_real_distribution<float> u(-2.f, 2.f);
        for (const auto& c : cases) {
            const int threads = c.n_in >= 128 ? poly::sum_threads(c.n_in) : 32;
            std::vector<float> xf((size_t)c.rows * (c.n_out / c.rows_per_x) * c.n_in);
            for (auto& v : xf) v = u(rng);
            std::vector<uint16_t> w((size_t)c.n_out * c.n_in);
            for (auto& v : w) v = bf16_enc(u(rng));
            std::vector<float> y((size_t)c.rows * c.n_out, 0.f), ref(y.size());
            const size_t wbytes = w.size() * 2;
            const uint64_t aw = a.alloc(wbytes), ax = a.alloc(xf.size() * 4), ay = a.alloc(y.size() * 4);
            vk::copy_h2d((void*)aw, w.data(), wbytes, nullptr);
            vk::copy_h2d((void*)ax, xf.data(), xf.size() * 4, nullptr);
            poly::gemv_bf16((const uint16_t*)strata_vk_compat::wrap(aw), (const float*)strata_vk_compat::wrap(ax),
                            (float*)strata_vk_compat::wrap(ay), c.n_in, c.n_out, c.rows_per_x, c.rows, nullptr);
            vk::copy_d2h(y.data(), (void*)ay, y.size() * 4, nullptr);
            ref_gemv_bf16(w, xf, ref, c.n_in, c.n_out, c.rows_per_x, c.rows, threads);
            double worst = 0.0; int bit = 1;
            for (size_t i = 0; i < y.size(); ++i) {
                if (std::memcmp(&y[i], &ref[i], 4) != 0) bit = 0;
                worst = std::fmax(worst, std::fabs((double)y[i] - ref[i]) / std::fmax(1e-3, std::fabs((double)ref[i])));
            }
            std::printf("  gemv_bf16 n_in=%d n_out=%d rpx=%d rows=%d thr=%d: worst rel %.3e, bitwise %s\n",
                        c.n_in, c.n_out, c.rows_per_x, c.rows, threads, worst, bit ? "yes" : "no");
            check(bit == 1, "gemv_bf16 bitwise vs the f32_from_bf16 partition reference");
            check(worst < 1e-5, "gemv_bf16 rel 1e-5 (gemv-family tol)");
        }
    }

    // ---- conv4_silu (Task 2.4): elementwise 4-tap window + SiLU, no reductions: the CUDA arms
    //      (rows>=16 no-snap -> rows+state kernels; else the token walk, with snapshots) are
    //      independent of partition for the SUM (same left-to-right term order in every arm), so
    //      the reference is one scalar transcription; only exp() may differ (CUDA expf vs GLSL
    //      exp). The conv4 window has no mul-into-accumulator at all in the CUDA source's sense --
    //      s0*w0 + s1*w1 + s2*w2 + v*w3 IS a fused chain on both (PORT-NOTES law #6), so try
    //      bitwise first; the fallback is the family's rel 1e-5.
    {
        auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
        auto silu = [&](float v) { return v * sig(v); };
        struct CRef { std::vector<float> out, out_u, st; std::vector<std::vector<float>> snaps; };
        auto ref_conv = [&](const std::vector<float>& x, const std::vector<float>& st_in,
                            const std::vector<float>& w, int channels, int rows,
                            const std::vector<float>& /*snap unused here*/, int n_snap, size_t stride,
                            unsigned snaps) {
            CRef r; r.out.assign(x.size(), 0.f); r.out_u.assign(x.size(), 0.f);
            std::vector<float> st = st_in;
            for (int c = 0; c < channels; ++c) {
                float s0 = st[3*c], s1 = st[3*c+1], s2 = st[3*c+2];
                const float* wc = &w[4*size_t(c)];
                for (int row = 0; row < rows; ++row) {
                    const float v = x[size_t(row)*channels + c];
                    // Two contraction arms of the SAME left-to-right term order: the CUDA build's
                    // --use_fast_math fuses (fma chain, PORT-NOTES law #6); glslc emits mul/add and
                    // RADV's NIR may or may not contract per chain. The pass rule is the tree's
                    // rel 1e-5 against EITHER arm (never looser than the gemv-family bound).
                    // The CUDA build's --use_fast_math fuses this left-to-right chain (PORT-NOTES
                    // law #6): y = fma(v,w3, fma(s2,w2, fma(s1,w1, s0*w0))). On THIS box RADV
                    // lowers even an explicit OpExtInst Fma back to mul+add (measured: identical
                    // device outputs for both spellings), so the device follows the UNFUSED arm;
                    // both arms are reported and the pass rule is the tree's rel 1e-5 against
                    // the closer arm (never looser than the gemv-family bound).
                    float y = std::fma(v, wc[3], std::fma(s2, wc[2], std::fma(s1, wc[1], s0 * wc[0])));
                    float yu = ((s0 * wc[0] + s1 * wc[1]) + s2 * wc[2]) + v * wc[3];
                    r.out[size_t(row)*channels + c] = silu(y);
                    r.out_u[size_t(row)*channels + c] = silu(yu);
                    s0 = s1; s1 = s2; s2 = v;
                    if (n_snap && row < 32 && ((snaps >> row) & 1u)) {
                        int slot = __builtin_popcount(snaps & ((1u << row) - 1u));
                        r.snaps.push_back({(float)slot, (float)c, s0, s1, s2});
                    }
                }
                st[3*c] = s0; st[3*c+1] = s1; st[3*c+2] = s2;
            }
            r.st = st;
            return r;
        };
        // the model's shape: qwen35moe ssm inner_size 6144 channels, decode rows=1 (walk arm);
        // plus a prompt arm rows=20 (rows+state kernels); plus a snapshots arm (walk with snap).
        struct Case { int channels; int rows; unsigned snaps; };
        const Case cases[] = { {6144, 1, 0}, {6144, 20, 0}, {2048, 4, 0b00000101u} };
        std::mt19937 rng(41);
        std::uniform_real_distribution<float> u(-3.f, 3.f);
        for (const auto& cs : cases) {
            const int channels = cs.channels, rows = cs.rows;
            std::vector<float> x((size_t)rows * channels), st((size_t)channels * 3), w((size_t)channels * 4),
                                 out((size_t)rows * channels, 0.f);
            for (auto& v : x) v = u(rng);
            for (auto& v : st) v = u(rng);
            for (auto& v : w) v = u(rng);
            const int n_snap = cs.snaps ? __builtin_popcount(cs.snaps) : 0;
            const size_t stride = (size_t)channels * 3;
            std::vector<float> snap(n_snap ? (size_t)n_snap * stride : 0, 0.f);
            const uint64_t ax = a.alloc(x.size() * 4), as_ = a.alloc(st.size() * 4), aw = a.alloc(w.size() * 4),
                           ao = a.alloc(out.size() * 4),
                           asnap = n_snap ? a.alloc(snap.size() * 4) : 0;
            vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
            vk::copy_h2d((void*)as_, st.data(), st.size() * 4, nullptr);
            vk::copy_h2d((void*)aw, w.data(), w.size() * 4, nullptr);
            vk::copy_h2d((void*)ao, out.data(), out.size() * 4, nullptr);
            if (n_snap) vk::fill_d((void*)asnap, 0, snap.size() * 4, nullptr);
            poly::conv4_silu((const float*)strata_vk_compat::wrap(ax), (float*)strata_vk_compat::wrap(as_),
                             (const float*)strata_vk_compat::wrap(aw), (float*)strata_vk_compat::wrap(ao),
                             channels, rows, nullptr,
                             n_snap ? (float*)strata_vk_compat::wrap(asnap) : nullptr, stride, cs.snaps);
            std::vector<float> got(out.size(), 0.f), st_got(st.size(), 0.f), snap_got(snap.size(), 0.f);
            vk::copy_d2h(got.data(), (void*)ao, got.size() * 4, nullptr);
            vk::copy_d2h(st_got.data(), (void*)as_, st_got.size() * 4, nullptr);
            if (n_snap) vk::copy_d2h(snap_got.data(), (void*)asnap, snap_got.size() * 4, nullptr);
            const auto ref = ref_conv(x, st, w, channels, rows, snap, n_snap, stride, cs.snaps);
            int bad = 0, bit = 1, bit_u = 1;
            double worst = 0.0, worst_u = 0.0;
            auto arm = [&](const char* name, const std::vector<float>& r, double& wrst, int& bitx) {
                for (size_t i = 0; i < got.size(); ++i) {
                    if (std::memcmp(&got[i], &r[i], 4) != 0) bitx = 0;
                    wrst = std::fmax(wrst, std::fabs((double)got[i] - r[i]) / std::fmax(1e-3, std::fabs((double)r[i])));
                }
                std::printf("  conv4 %s: bitwise %s, worst rel %.3e (tree tol 1e-5)\n", name,
                            bitx ? "yes" : "no", wrst);
            };
            arm("fused arm  ", ref.out, worst, bit);
            arm("unfused arm", ref.out_u, worst_u, bit_u);
            // The pass rule: the device must match ONE of the contraction arms under the tree's
            // rel 1e-5 bound (bitwise to an arm is the ideal; a fused/unfused gap alone is not a
            // partition mismatch). Pick the closer arm's bound.
            const bool closer_fused = worst <= worst_u;
            bad = (closer_fused ? worst : worst_u) >= 1e-5 ? 1 : 0;
            // walk arm advances state in place; the rows arm's state comes from conv4_state_kernel
            // (last three x, or old where rows < 3) -- rows>=16 means always x.
            std::vector<float> want_st = st;
            if (rows >= 16) {
                for (int c = 0; c < channels; ++c) { want_st[3*c] = x[size_t(rows-3)*channels+c];
                    want_st[3*c+1] = x[size_t(rows-2)*channels+c]; want_st[3*c+2] = x[size_t(rows-1)*channels+c]; }
            } else {
                for (auto& v : want_st) v = 0.f;   // walk arm: recompute below from the ref
                want_st = ref.st;
            }
            for (size_t i = 0; i < st_got.size(); ++i)
                if (std::memcmp(&st_got[i], &want_st[i], 4) != 0) { ++bad; if (bad < 4) std::printf("  conv4 state[%zu]: got %.9g want %.9g\n", i, st_got[i], want_st[i]); }
            int snapbad = 0;
            for (auto& sn : ref.snaps) {   // sn = {slot, channel, s0, s1, s2}
                const float* m = snap_got.data() + (size_t)sn[0] * stride + 3 * (size_t)sn[1];
                if (std::memcmp(m, &sn[2], 12) != 0) {
                    if (snapbad < 3) std::printf("  snap slot %d c %d: got %.6g,%.6g,%.6g want %.6g,%.6g,%.6g\n",
                                                 (int)sn[0], (int)sn[1], m[0], m[1], m[2], sn[2], sn[3], sn[4]);
                    ++snapbad;
                }
            }
            std::printf("  conv4_silu channels=%d rows=%d snaps=%X: bad %d, snap mismatches %d\n",
                        channels, rows, cs.snaps, bad, snapbad);
            check(bad == 0 && snapbad == 0, "conv4_silu out+state+snap match (closer contraction arm within rel 1e-5)");
        }
    }

    // ---- Task 2.7: the elementwise batch (axpy, swiglu_clamp, mul_sigmoid, add_scaled).
    {
        auto arm_bits = [](const std::vector<float>& got, const std::vector<float>& ref) {
            int bit = 1; double worst = 0.0;
            for (size_t i = 0; i < got.size(); ++i) {
                if (std::memcmp(&got[i], &ref[i], 4) != 0) bit = 0;
                worst = std::fmax(worst, std::fabs((double)got[i] - ref[i]) / std::fmax(1e-3, std::fabs((double)ref[i])));
            }
            return std::pair<int, double>(bit, worst);   // bit: 1 bitwise, 0 not
        };
        std::mt19937 rng(47);
        std::uniform_real_distribution<float> u(-3.f, 3.f);

        // axpy at the model's 2048-dim arm, negatives and zeros included; add=true fuses to
        // fma(a,x,acc) on CUDA -- measure the fused and unfused arms, bitwise to the closer one.
        for (int add : {1, 0}) {
            const int n = 2048;
            std::vector<float> x(n), acc(n), a0(n);
            for (auto& v : x) v = u(rng);
            for (auto& v : acc) v = u(rng);
            a0 = acc;
            if (add) for (int i = 0; i < n; i += 7) x[i] = 0.f;
            const float sc = -0.7340033f;
            const uint64_t ax_ = a.alloc(n * 4), aa = a.alloc(n * 4);
            vk::copy_h2d((void*)ax_, x.data(), n * 4, nullptr);
            vk::copy_h2d((void*)aa, acc.data(), n * 4, nullptr);
            poly::axpy((const float*)strata_vk_compat::wrap(ax_), sc,
                       (float*)strata_vk_compat::wrap(aa), n, add != 0, nullptr);
            std::vector<float> got(n, 0.f);
            vk::copy_d2h(got.data(), (void*)aa, n * 4, nullptr);
            std::vector<float> ref(n), ref_u(n);
            for (int i = 0; i < n; ++i) {
                ref[i] = add ? (float)((double)a0[i] + (double)sc * x[i])   // fma arm (host exact)
                             : sc * x[i];
                ref_u[i] = add ? a0[i] + sc * x[i] : sc * x[i];              // mul+add arm
            }
            if (!add) ref = ref_u;
            auto bf = arm_bits(got, ref), bu = arm_bits(got, ref_u);
            std::printf("  axpy n=%d add=%d: bitwise-fused %s, bitwise-unfused %s, worst rel %.3e\n",
                        n, add, bf.first ? "yes" : "no", bu.first ? "yes" : "no",
                        std::fmin(bf.second, bu.second));
            check(bf.first || bu.first, "axpy matches one contraction arm bitwise (scale arm exact)");
        }

        // add_scaled: 3 vectors into 3 acc rows, n=1024 (ffn width-ish), one shared x.
        {
            const int n = 1024, count = 3;
            poly::ScaledRows rows;
            std::vector<std::vector<float>> x(count, std::vector<float>(n)), acc(count, std::vector<float>(n));
            for (int k = 0; k < count; ++k) { for (auto& v : x[k]) v = u(rng); for (auto& v : acc[k]) v = u(rng); }
            rows.w[0] = 0.5f; rows.w[1] = -1.25f; rows.w[2] = 0.0625f;
            std::vector<uint64_t> ax(count), aa(count);
            for (int k = 0; k < count; ++k) {
                ax[k] = a.alloc(n * 4); aa[k] = a.alloc(n * 4);
                vk::copy_h2d((void*)ax[k], x[k].data(), n * 4, nullptr);
                vk::copy_h2d((void*)aa[k], acc[k].data(), n * 4, nullptr);
                rows.x[k] = (const float*)strata_vk_compat::wrap(ax[k]);
                rows.acc[k] = (float*)strata_vk_compat::wrap(aa[k]);
            }
            poly::add_scaled(rows, count, n, nullptr);
            int bad = 0;
            for (int k = 0; k < count; ++k) {
                std::vector<float> got(n, 0.f);
                vk::copy_d2h(got.data(), (void*)aa[k], n * 4, nullptr);
                std::vector<float> ref(n), ref_u(n);
                for (int i = 0; i < n; ++i) {
                    ref[i] = (float)((double)acc[k][i] + (double)rows.w[k] * x[k][i]);
                    ref_u[i] = acc[k][i] + rows.w[k] * x[k][i];
                }
                auto b1 = arm_bits(got, ref).first, b2 = arm_bits(got, ref_u).first;
                if (!(b1 || b2)) ++bad;
            }
            std::printf("  add_scaled count=%d n=%d: bad rows %d\n", count, n, bad);
            check(bad == 0, "add_scaled per-row matches a contraction arm bitwise");
        }

        // swiglu_clamp at the ffn arm: n=1024, limit=7 (the model's softcap-style clamp); the
        // exp-vs-expf gap is the only non-bitwise freedom -> rel 1e-5 (silu family bound).
        {
            const int n = 1024; const float limit = 7.0f;
            std::vector<float> gate(n), up(n);
            for (auto& v : gate) v = u(rng) * 4.f;   // cross the clamp both ways
            for (auto& v : up) v = u(rng) * 4.f;
            gate[0] = limit + 1.f; up[0] = -limit - 1.f; gate[1] = 0.f; up[1] = 0.f;
            const uint64_t ag = a.alloc(n * 4), au = a.alloc(n * 4), ao = a.alloc(n * 4);
            vk::copy_h2d((void*)ag, gate.data(), n * 4, nullptr);
            vk::copy_h2d((void*)au, up.data(), n * 4, nullptr);
            vk::fill_d((void*)ao, 0, n * 4, nullptr);
            poly::swiglu_clamp((const float*)strata_vk_compat::wrap(ag), (const float*)strata_vk_compat::wrap(au),
                               (float*)strata_vk_compat::wrap(ao), n, limit, nullptr);
            std::vector<float> got(n, 0.f);
            vk::copy_d2h(got.data(), (void*)ao, n * 4, nullptr);
            std::vector<float> ref(n);
            for (int i = 0; i < n; ++i) {
                float g = std::fmin(gate[i], limit);
                float uu = std::fmin(std::fmax(up[i], -limit), limit);
                ref[i] = g / (1.0f + std::exp(-g)) * uu;
            }
            auto b = arm_bits(got, ref);
            std::printf("  swiglu_clamp n=%d limit=%g: bitwise %s, worst rel %.3e (tree tol 1e-5)\n",
                        n, limit, b.first ? "yes" : "no", b.second);
            check(b.second < 1e-5, "swiglu_clamp within rel 1e-5 (exp vs expf)");
        }

        // swiglu_clamp_rows (Task 2.8): rows of gate|up pairs (CUDA :873), n 1024 and an odd 300
        // (the tail guard), rows 3: same exp-vs-expf caveat as swiglu_clamp -> rel 1e-5.
        for (int n : {1024, 300}) {
            const int rows = 3; const float limit = 7.0f;
            std::vector<float> gu(size_t(rows) * 2 * n), out(size_t(rows) * n);
            for (auto& v : gu) v = u(rng) * 4.f;
            gu[0] = limit + 1.f; gu[n] = -limit - 1.f;
            const uint64_t ag = a.alloc(gu.size() * 4), ao = a.alloc(out.size() * 4);
            vk::copy_h2d((void*)ag, gu.data(), gu.size() * 4, nullptr);
            vk::fill_d((void*)ao, 0, out.size() * 4, nullptr);
            poly::swiglu_clamp_rows((const float*)strata_vk_compat::wrap(ag),
                                    (float*)strata_vk_compat::wrap(ao), n, rows, limit, nullptr);
            std::vector<float> got(out.size(), 0.f);
            vk::copy_d2h(got.data(), (void*)ao, got.size() * 4, nullptr);
            std::vector<float> ref(out.size());
            for (int r = 0; r < rows; ++r)
                for (int i = 0; i < n; ++i) {
                    const float g = std::fmin(gu[size_t(r) * 2 * n + i], limit);
                    const float up = std::fmin(std::fmax(gu[size_t(r) * 2 * n + n + i], -limit), limit);
                    ref[size_t(r) * n + i] = g / (1.0f + std::exp(-g)) * up;
                }
            auto b = arm_bits(got, ref);
            std::printf("  swiglu_clamp_rows n=%d rows=%d limit=%g: bitwise %s, worst rel %.3e (tree tol 1e-5)\n",
                        n, rows, limit, b.first ? "yes" : "no", b.second);
            check(b.second < 1e-5, "swiglu_clamp_rows within rel 1e-5 (exp vs expf)");
        }

        // mul_sigmoid: rows=16 (heads) x n=128, g_stride=256 (bigger than n: strided gate rows),
        // plus the one_gate arm. sigmoid = 1/(1+expf(-x)) -> same transcendental caveat.
        for (int og : {0, 1}) {
            const int n = 128, rows = 16, g_stride = 256;
            std::vector<float> acc((size_t)rows * n), g((size_t)rows * g_stride);
            for (auto& v : acc) v = u(rng);
            for (auto& v : g) v = u(rng) * 2.f;
            const uint64_t aa = a.alloc(acc.size() * 4), ag = a.alloc(g.size() * 4);
            vk::copy_h2d((void*)aa, acc.data(), acc.size() * 4, nullptr);
            vk::copy_h2d((void*)ag, g.data(), g.size() * 4, nullptr);
            poly::mul_sigmoid((float*)strata_vk_compat::wrap(aa),
                              (const float*)strata_vk_compat::wrap(ag), n, rows, g_stride, og != 0, nullptr);
            std::vector<float> got(acc.size(), 0.f);
            vk::copy_d2h(got.data(), (void*)aa, got.size() * 4, nullptr);
            std::vector<float> ref(acc.size());
            for (int r = 0; r < rows; ++r) {
                const float gv = g[(size_t)r * g_stride + (og ? 0 : 0)];  // one_gate=1 -> col 0
                double worst = 0.0; int bit = 1;
                for (int i = 0; i < n; ++i) {
                    const float gate = g[(size_t)r * g_stride + (og ? 0 : i)];
                    ref[(size_t)r * n + i] = acc[(size_t)r * n + i] * (1.0f / (1.0f + std::exp(-gate)));
                    (void)gv;
                }
            }
            auto b = arm_bits(got, ref);
            std::printf("  mul_sigmoid n=%d rows=%d g_stride=%d one_gate=%d: bitwise %s, worst rel %.3e\n",
                        n, rows, g_stride, og, b.first ? "yes" : "no", b.second);
            check(b.second < 1e-5, "mul_sigmoid within rel 1e-5 (exp vs expf)");
        }
    }

    // ---- Task 2.10: head_norm_rope — the CUDA body (kernels.cu:994-1022) transcribed in the
    //      same term order. The ss sum is host-double (the rel 1e-5 bound absorbs the fp32
    //      strided/warp partition, rms_norm precedent); pow/cos/sin land on RADV's hardware
    //      approximations vs CUDA's libdevice on the real GPU, so rel 1e-5 is the contract
    //      here (swiglu/mul_sigmoid precedent), NOT bitwise. Arms: plain rope, freq, mrope
    //      axes (the model's sections form), on-card `at`, x_stride != dim (the q arm),
    //      unweighted, and the n_rot < dim pass-through tail.
    {
        auto ref_hnr = [](const std::vector<float>& x, int x_stride, const float* w, std::vector<float>& y,
                          int dim, int heads, int n_rot, float theta_scale, long long pos, int rows, float eps,
                          const float* freq, const int32_t* axes, int sections) {
            const int half = n_rot / 2;
            for (int blk = 0; blk < rows * heads; ++blk) {
                const size_t base = (size_t)blk * x_stride;
                double ss = 0.0;
                for (int i = 0; i < dim; ++i) { float v = x[base + i]; ss += (double)v * v; }
                const float scale = (float)(1.0 / std::sqrt(ss / dim + eps));
                const int tok = blk / heads;
                const float own = (float)(pos + tok);
                for (int i = 0; i < dim; ++i) {
                    float out;
                    if (i >= n_rot) {
                        const float v = x[base + i];
                        out = w ? (v * scale) * w[i] : v * scale;
                    } else {
                        const int j = i < half ? i : i - half;
                        const float p = axes ? (float)axes[(size_t)tok * 3 + (j % sections) % 3] : own;
                        const float turn = p * (float) std::pow((double) theta_scale, (double) j);  // launcher table
                        const float theta = freq ? turn / freq[j] : turn;
                        const float x0 = w ? (x[base + j] * scale) * w[j] : x[base + j] * scale;
                        const float x1 = w ? (x[base + j + half] * scale) * w[j + half] : x[base + j + half] * scale;
                        const float cs = (float) std::cos((double) theta), sn = (float) std::sin((double) theta);
                        out = i < half ? (float)((double) x0 * cs - (double) x1 * sn)
                                       : (float)((double) x0 * sn + (double) x1 * cs);
                    }
                    y[(size_t)blk * dim + i] = out;
                }
            }
        };
        const int dim = 128, heads = 4, rows = 3, n_rot = 128;   // the 128-thread arm
        const float theta_scale = 0.93f, eps = 1e-6f;
        std::vector<float> x((size_t)rows * heads * dim), w(dim), fr(n_rot / 2);
        std::mt19937 rng(210); std::uniform_real_distribution<float> u(-3.f, 3.f);
        for (auto& v : x) v = u(rng);
        for (auto& v : w) v = u(rng);
        for (auto& v : fr) v = u(rng) < 0.f ? 1.f : 1.f / (1.f + std::fabs(u(rng)));
        const uint64_t ax = a.alloc(x.size() * 4), aw = a.alloc(w.size() * 4);
        const uint64_t af = a.alloc(fr.size() * 4), ay = a.alloc(x.size() * 4);
        vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
        vk::copy_h2d((void*)aw, w.data(), w.size() * 4, nullptr);
        vk::copy_h2d((void*)af, fr.data(), fr.size() * 4, nullptr);
        std::vector<int32_t> axes((size_t)rows * 3, 0);
        for (int t = 0; t < rows; ++t) { axes[t*3] = 7 + t; axes[t*3+1] = 2 + t; axes[t*3+2] = 5 - t; }
        const uint64_t aax = a.alloc(axes.size() * 4);
        vk::copy_h2d((void*)aax, axes.data(), axes.size() * 4, nullptr);
        // host-built pos table for the at arm (two u32 words, little-endian, like the engine's at_)
        long long at_pos = 4093;   // the launch-replay arm: within the documented contract (the .comp's
                           // Cody-Waite word costs grow with q ~ pos/2pi: measured <= 3e-7 to
                           // pos 4000, the tree's max-context tier; PORT-STATUS records the bound)
        const uint64_t aat = a.alloc(8);
        vk::copy_h2d((void*)aat, &at_pos, 8, nullptr);
        const auto run_cmp = [&](const char* what, int xs, const float* wp, const float* fp,
                                 const int32_t* axp, int sec, long long pos, const long long* atp) {
            poly::head_norm_rope((const float*)strata_vk_compat::wrap(ax), xs,
                                 wp ? (const float*)strata_vk_compat::wrap(aw) : nullptr,
                                 (float*)strata_vk_compat::wrap(ay), dim, heads, n_rot, theta_scale,
                                 pos, rows, eps, nullptr,
                                 fp ? (const float*)strata_vk_compat::wrap(af) : nullptr,
                                 axp ? (const int32_t*)strata_vk_compat::wrap(aax) : nullptr, sec,
                                 atp ? (const long long*)strata_vk_compat::wrap(aat) : nullptr);
            std::vector<float> got(x.size(), 0.f), ref(x.size(), 0.f);
            vk::copy_d2h(got.data(), (void*)ay, got.size() * 4, nullptr);
            // the host reference reads x/w/freq/axes exactly as the launcher handed them
            ref_hnr(x, xs, wp ? w.data() : nullptr, ref, dim, heads, n_rot, theta_scale,
                    atp ? at_pos : pos, rows, eps, fp ? fr.data() : nullptr, axp, sec);
            double worst = 0.0;
            for (size_t i = 0; i < got.size(); ++i)
                worst = std::fmax(worst, std::fabs((double)got[i] - ref[i]) /
                                       std::fmax(1.0, std::fabs((double)ref[i])));
            std::printf("  head_norm_rope %s: worst rel %.3e (tol 1e-5)\n", what, worst);
            check(worst < 1e-5, what);
        };
        run_cmp("head_norm_rope plain rope weighted", dim, &w[0], nullptr, nullptr, 1, 42, nullptr);
        run_cmp("head_norm_rope unweighted", dim, nullptr, nullptr, nullptr, 1, 42, nullptr);
        run_cmp("head_norm_rope freq arm", dim, &w[0], &fr[0], nullptr, 1, 42, nullptr);
        run_cmp("head_norm_rope mrope axes", dim, &w[0], nullptr, axes.data(), 3, 42, nullptr);
        run_cmp("head_norm_rope on-card at", dim, &w[0], nullptr, nullptr, 1, 0, &at_pos);
    }
    // 256-thread arm, n_rot < dim: the tail passes through normed only (kernels.cu:1010-1012).
    {
        const int dim = 256, heads = 2, rows = 2, n_rot = 128;
        const float theta_scale = 0.9f, eps = 1e-6f;
        // x holds the qg_ shape: blocks heads at x_stride = 2*dim apart (the padding is real).
        std::vector<float> x((size_t)rows * heads * 2 * dim), w(dim);
        std::mt19937 rng(211); std::uniform_real_distribution<float> u(-3.f, 3.f);
        for (auto& v : x) v = u(rng);
        for (auto& v : w) v = u(rng);
        const uint64_t ax = a.alloc(x.size() * 4), aw = a.alloc(w.size() * 4);
        const uint64_t ay = a.alloc((size_t)rows * heads * dim * 4);
        vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
        vk::copy_h2d((void*)aw, w.data(), w.size() * 4, nullptr);
        poly::head_norm_rope((const float*)strata_vk_compat::wrap(ax), 2 * dim,
                             (const float*)strata_vk_compat::wrap(aw), (float*)strata_vk_compat::wrap(ay),
                             dim, heads, n_rot, theta_scale, 900, rows, eps, nullptr, nullptr, nullptr, 1, nullptr);
        std::vector<float> got((size_t)rows * heads * dim, 0.f), ref(got.size(), 0.f);
        vk::copy_d2h(got.data(), (void*)ay, got.size() * 4, nullptr);
        {   // the reference walks x with the same stride the launcher read
            const int xs = 2 * dim;
            double ss;
            for (int b = 0; b < rows * heads; ++b) {
                ss = 0.0;
                for (int i = 0; i < dim; ++i) { float v = x[(size_t)b * xs + i]; ss += (double)v * v; }
                const float scale = (float)(1.0 / std::sqrt(ss / dim + eps));
                const int half = n_rot / 2;
                const float own = (float)(900 + b / heads);
                for (int i = 0; i < dim; ++i) {
                    if (i >= n_rot) { ref[(size_t)b * dim + i] = (x[(size_t)b*xs + i] * scale) * w[i]; continue; }
                    const int j = i < half ? i : i - half;
                    const float turn = own * (float) std::pow((double) theta_scale, (double) j);  // launcher table
                    const float x0 = (x[(size_t)b*xs + j] * scale) * w[j];
                    const float x1 = (x[(size_t)b*xs + j + half] * scale) * w[j + half];
                    const float cs = (float) std::cos((double) turn), sn = (float) std::sin((double) turn);
                    ref[(size_t)b * dim + i] = i < half ? (float)((double) x0 * cs - (double) x1 * sn)
                                                        : (float)((double) x0 * sn + (double) x1 * cs);
                }
            }
        }
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i)
            worst = std::fmax(worst, std::fabs((double)got[i] - ref[i]) / std::fmax(1.0, std::fabs((double)ref[i])));
        std::printf("  head_norm_rope 256 threads n_rot<dim x_stride: worst rel %.3e\n", worst);
        check(worst < 1e-5, "head_norm_rope dim=256 (256 threads), n_rot=128 tail, x_stride=2*dim");
    }

    // ---- Task 2.10: store_f16 — BITWISE expected: f16_rn_bits is the same RN encoder the
    //      mmvq parity suite proved byte-exact; no other arithmetic is involved.
    {
        const int n = 320, rows = 5; const int64_t pos = 7;
        std::vector<float> x((size_t)n * rows);
        std::mt19937 rng(212); std::uniform_real_distribution<float> u(-60000.f, 60000.f);
        for (auto& v : x) v = u(rng);
        x[0] = 0.f; x[1] = -0.f; x[2] = 1.f; x[3] = -1.f;
        std::vector<uint16_t> cache((size_t)(pos + rows) * n, 0xA5A5);
        const uint64_t ax = a.alloc(x.size() * 4), ac = a.alloc(cache.size() * 2);
        vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
        vk::copy_h2d((void*)ac, cache.data(), cache.size() * 2, nullptr);
        poly::store_f16((const float*)strata_vk_compat::wrap(ax), (uint16_t*)strata_vk_compat::wrap(ac),
                        n, pos, rows, nullptr);
        std::vector<uint16_t> got(cache.size(), 0);
        vk::copy_d2h(got.data(), (void*)ac, got.size() * 2, nullptr);
        size_t bad = 0;
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < n; ++i)
                if (got[(size_t)(pos + r) * n + i] != f32_to_f16(x[(size_t)r * n + i])) bad++;
        check(bad == 0, "store_f16 byte-exact f16 (f16_rn arm)");
        // the rows before pos keep their prior bytes (the kernel's window is pos*n..pos*n+all)
        bool untouched = true;
        for (size_t i = 0; i < (size_t)pos * n && untouched; ++i) if (got[i] != 0xA5A5) untouched = false;
        check(untouched, "store_f16 leaves the rows before pos alone");
    }

    // ---- q8_0 KV: store_q8 — BITWISE expected vs the CUDA body (kernels.cu:531): per 32-value
    //      block amax over fabsf, d = amax/127f (host SSE: correctly rounded, like CUDA's), id =
    //      d>0 ? 1/d : 0, code = clamp(rintf(x*id), -127, 127) — rintf ties to EVEN, which the
    //      tie block below pins (GLSL round() would tie away). Scale word f16_rn_bits(d); codes
    //      little-endian 4 a uint word (the layout store_f16.comp records).
    {
        const int n = 256, rows = 6; const int64_t pos = 11;   // n/32 = 8 blocks a row (even)
        std::vector<float> x((size_t)n * rows);
        std::mt19937 rng(908); std::uniform_real_distribution<float> u(-60000.f, 60000.f);
        for (auto& v : x) v = u(rng);
        // block 0 of row 0: the WHOLE block in tie range (amax 127 -> d = 1, id = 1): exact
        // ties 2.5 -> 2, 3.5 -> 4, -2.5 -> -2, -3.5 -> -4 under rintf; round() would give +-3.
        for (int j = 0; j < 32; ++j) x[j] = 0.f;
        x[0] = 127.f; x[1] = 2.5f; x[2] = 3.5f; x[3] = -2.5f; x[4] = -3.5f;
        const size_t nb = (size_t)(n >> 5);
        std::vector<int8_t> codes((size_t)(pos + rows) * n, (int8_t)0x5A);
        std::vector<uint16_t> scales((size_t)(pos + rows) * nb, 0xA5A5);
        const uint64_t ax = a.alloc(x.size() * 4);
        const uint64_t ac = a.alloc(codes.size()), as = a.alloc(scales.size() * 2);
        vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
        vk::copy_h2d((void*)ac, codes.data(), codes.size(), nullptr);
        vk::copy_h2d((void*)as, scales.data(), scales.size() * 2, nullptr);
        poly::store_q8((const float*)strata_vk_compat::wrap(ax), (int8_t*)strata_vk_compat::wrap(ac),
                       (uint16_t*)strata_vk_compat::wrap(as), n, pos, rows, nullptr);
        std::vector<int8_t> gc(codes.size(), 0); std::vector<uint16_t> gs(scales.size(), 0);
        vk::copy_d2h(gc.data(), (void*)ac, gc.size(), nullptr);
        vk::copy_d2h(gs.data(), (void*)as, gs.size() * 2, nullptr);
        size_t bad_c = 0, bad_s = 0;
        for (int r = 0; r < rows; ++r)
            for (size_t b = 0; b < nb; ++b) {
                const float* xr = &x[(size_t)r * n + b * 32];
                float amax = 0.f;
                for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(xr[j]));
                const float d = amax / 127.0f;
                const float id = d > 0.f ? 1.0f / d : 0.0f;
                if (gs[(size_t)(pos + r) * nb + b] != f32_to_f16(d)) bad_s++;
                for (int j = 0; j < 32; ++j) {
                    float q = std::rintf(xr[j] * id);
                    q = std::fmin(std::fmax(q, -127.0f), 127.0f);
                    if (gc[(size_t)(pos + r) * n + b * 32 + j] != (int8_t)q) bad_c++;
                }
            }
        check(bad_c == 0, "store_q8 codes byte-exact vs the CUDA body (rint ties-to-even)");
        check(bad_s == 0, "store_q8 scales byte-exact f16(d)");
        // the ties arm, asserted positively: a ties-away round() would show +-3 here
        check(gc[(size_t)pos * n + 1] == 2 && gc[(size_t)pos * n + 2] == 4 &&
              gc[(size_t)pos * n + 3] == (int8_t)-2 && gc[(size_t)pos * n + 4] == (int8_t)-4,
              "store_q8 rint ties: 2.5->2 3.5->4 -2.5->-2 -3.5->-4");
        // outside the window: the prior bytes hold (codes before pos*n, scales before pos*nb)
        bool keep = true;
        for (size_t i = 0; i < (size_t)pos * n && keep; ++i) if (gc[i] != (int8_t)0x5A) keep = false;
        for (size_t i = 0; i < (size_t)pos * nb && keep; ++i) if (gs[i] != 0xA5A5) keep = false;
        check(keep, "store_q8 leaves the rows before pos alone");
    }

    // ---- Task 2.19: likeliest — the CUDA body (kernels.cu:1827) transcribed exactly: the
    //      strided max/sum partitions of a 1024-thread block, the xor trees 16..1, the in-order
    //      part[0..31] folds. TOKEN is an integer: equality expected (first index of the max,
    //      ties included). prob = 1/sum rides the exp-vs-expf rel 1e-5 bound (elementwise law).
    {
        auto ref_likeliest = [](const std::vector<float>& x) {
            const int n = (int)x.size();
            const float NINF = -std::numeric_limits<float>::infinity();
            // thread t's strided partial max (CUDA :1831-1832)
            std::vector<float> ms(1024, NINF);
            for (int t = 0; t < 1024; ++t)
                for (int i = t; i < n; i += 1024) ms[t] = std::fmax(ms[t], x[i]);
            // block_max (:58): per-warp xor butterfly -- EVERY lane updates each round against
            // its (lane^o) partner (reads see the OLD round); then lane 0 of each warp writes
            // part[wid]; thread 0 folds m = part[0]; fmaxf over part[1..31] in order.
            for (int o = 16; o > 0; o >>= 1) {
                std::vector<float> old = ms;
                for (int t = 0; t < 1024; ++t) ms[t] = std::fmax(ms[t], old[(t & ~31) | ((t & 31) ^ o)]);
            }
            float m = ms[0];
            for (int k = 1; k < 32; ++k) m = std::fmax(m, ms[k * 32]);
            // second strided pass (:1836-1839): at = min over x[i] == m (integer, order-free);
            // per-thread partial sums of expf over x[i] > m - 20.
            int at = n;
            std::vector<float> sums(1024, 0.0f);
            for (int t = 0; t < 1024; ++t)
                for (int i = t; i < n; i += 1024) {
                    if (x[i] == m) at = std::min(at, i);
                    if (x[i] > m - 20.0f) sums[t] += std::expf(x[i] - m);
                }
            // warp butterfly again (float +=, all lanes per round), part = lane 0 of each warp,
            // then block_sum's fold s = 0; s += part[k] for k = 0..31 IN ORDER (:46-47).
            for (int o = 16; o > 0; o >>= 1) {
                std::vector<float> old = sums;
                for (int t = 0; t < 1024; ++t) sums[t] = sums[t] + old[(t & ~31) | ((t & 31) ^ o)];
            }
            float s = 0.0f;
            for (int k = 0; k < 32; ++k) s += sums[k * 32];
            poly::Likeliest r; r.token = at < n ? at : 0; r.prob = 1.0f / s;
            return r;
        };
        std::mt19937 rng(97);
        std::uniform_real_distribution<float> u(-30.f, 30.f);
        const int shapes[] = {5, 1024, 151936 / 4, 151936};   // tiny, exact block, sub-block, full-vocab-ish
        for (const int n : shapes) {
            std::vector<float> x(n);
            for (auto& v : x) v = u(rng);
            if (n > 10) { const float mx = u(rng); x[n / 3] = x[n / 7] = x[n - 1] = mx;
                          for (auto& v : x) v = std::fmin(v, mx); }  // tie at the max: the FIRST index wins
            const uint64_t ax = a.alloc((size_t)n * 4), ao = a.alloc(8);
            vk::copy_h2d((void*)ax, x.data(), (size_t)n * 4, nullptr);
            poly::Likeliest got{ -1, 0.f };
            poly::likeliest((const float*)strata_vk_compat::wrap(ax), n, &got, nullptr);
            const poly::Likeliest ref = ref_likeliest(x);
            const bool tok = got.token == ref.token;
            const double rel = std::fabs((double)got.prob - ref.prob) / std::fmax(1e-9f, std::fabs((double)ref.prob));
            if (!tok) std::printf("  likeliest n=%d token got %d want %d\n", n, got.token, ref.token);
            std::printf("  likeliest n=%d: token %d (ref %d), prob rel %.3e\n", n, got.token, ref.token, rel);
            check(tok && rel < 1e-5, "likeliest token-equal + prob rel 1e-5");
        }
    }

    std::printf("poly_kernels_vk_parity %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
