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

    std::printf("poly_kernels_vk_parity %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
