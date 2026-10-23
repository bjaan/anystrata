// vulkan/src/tests/poly_attend_vk_parity.cpp - parity for strata::poly::attend (Task 2.11),
// the general fp16-KV chain: attn_score_reg_kernel<4,1,false> + attn_softmax_kernel +
// attn_value_t<1> / attn_value (src/poly/kernels.cu:1155, 1220, 1499, 1315) - the model's arm
// (dim 128, per 8: Qwen3.6's n_head 16 / kv_heads 2).
//
// TWO host references, per the contract:
//  ORDER: an exact scalar transcription of the THREE kernels' partitions - the 32-lane dot
//    split (lane l values 4l..4l+3) with the fma chains (law 6, nvcc -fmad), the scores_sum<4>
//    xor tree with per-round copies (law 4), the softmax's strided passes + 8-warp part[] walk,
//    and the value share-rotation walk (k = 0..31 in order). This is the bitwise-plausible arm;
//    the ONLY freedom is exp-vs-expf (PORT-NOTES law 9: GLSL exp lowers to RADV's approximation,
//    CUDA's expf is software ~1 ULP), so the bound is the tree's rel 1e-5 (gdn/elementwise
//    precedent), never a fudge.
//  DIRECT: a naive no-tiling softmax attention (plain sequential dot, plain sum) - catches a
//    partition/order bug the ordered arm could share. Also rel 1e-5.
//
// Cases ride the launcher's own arms: rows=1 with chunk=1 (the PLAIN value arm, wt null - CUDA
// 2511's predicate is false when c == chunk), rows=1 with chunk=4 (the wt arm lights:
// kv_heads*1*8 <= 3*16 -> attn_value_t<1>), and a rows=5 chunk=8 launch (two chunk-loop
// iterations replayed by the launcher: an 8-row window's first 5 tokens... actually c=5<8 with
// one launch: wt arm + blocks=grid(40,8)=5). pos values sit under chunk (the engine keeps
// pos < chunk in decode: chunk_ = kScoreFloats/(n_head*max_context) = 256 at the model's
// shape, contexts under it here).
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

int fails = 0;
void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) fails++;
}

float frandom(std::mt19937& rng) {
    return (float)((int)(rng() % 2001) - 1000) / 1000.0f;
}

// host float -> f16 bits, RN-even (the mirror of f16_rn_bits in store_f16.comp): builds the
// cache inputs.
uint16_t f16_rn(float f) {
    uint32_t b; std::memcpy(&b, &f, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    int exp = int((b >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = b & 0x7FFFFFu;
    if (exp >= 31) return uint16_t(sign | 0x7C00u);
    if (exp <= 0) {
        man |= 0x800000u;
        const uint32_t sh = uint32_t(1 - exp) + 13u;
        uint32_t h = (sh >= 32u) ? 0u : (man >> sh);
        const uint32_t rem = (sh >= 32u) ? 0u : (man & ((1u << sh) - 1u));
        if (sh < 32u && (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u)))) h++;
        return uint16_t(sign | h);
    }
    uint32_t hman = man >> 13, rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (hman & 1u))) { hman++; if (hman == 0x400u) { hman = 0; exp++; if (exp >= 31) return uint16_t(sign | 0x7C00u); } }
    return uint16_t(sign | (uint32_t(exp) << 10) | hman);
}

// exact half -> float, the bit build both .comp files use (subnormals normalized, no rounding)
float f16_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0u) {
        if (m == 0u) { const uint32_t b = sign << 31; float z; std::memcpy(&z, &b, 4); return z; }
        uint32_t mb = uint32_t(m) << 13; int ee = -14;
        while (!(mb & 0x800000u)) { mb <<= 1; --ee; }
        const uint32_t b = (sign << 31) | (uint32_t(ee + 127) << 23) | (mb & 0x7FFFFFu);
        float r; std::memcpy(&r, &b, 4); return r;
    }
    if (e == 31u) { const uint32_t b = (sign << 31) | 0x7F800000u | (m << 13); float r; std::memcpy(&r, &b, 4); return r; }
    const uint32_t bb = (sign << 31) | ((e + 112u) << 23) | (m << 13);
    float r; std::memcpy(&r, &bb, 4); return r;
}

// CUDA's warp_sum tree over 32 lanes with per-round copies (law 4; the poly_gdn test's helper).
float ref_warp_sum(float lane[32]) {
    float l[32]; std::memcpy(l, lane, sizeof l);
    for (int off = 16; off > 0; off >>= 1) {
        float p[32]; std::memcpy(p, l, sizeof l);
        for (int j = 0; j < 32; ++j) l[j] = l[j] + p[j ^ off];
    }
    return l[0];
}

// CUDA block_max/block_sum (:58/:42) for the softmax's 256-thread block: the strided pass per
// thread, the xor tree per warp, then part[0..7] in order.
float ref_block_reduce(const std::vector<float>& terms, int n_threads, bool max) {
    std::vector<float> part(n_threads / 32, max ? -INFINITY : 0.0f);
    for (int w = 0; w < n_threads / 32; ++w) {
        float l[32] = {};
        for (int k = 0; k < 32; ++k) l[k] = terms[size_t(w) * 32 + k];
        for (int off = 16; off > 0; off >>= 1) {
            float p[32]; std::memcpy(p, l, sizeof l);
            for (int k = 0; k < 32; ++k) l[k] = max ? std::fmax(l[k], p[k ^ off]) : l[k] + p[k ^ off];
        }
        part[w] = l[0];
    }
    float acc = part[0];
    for (size_t k = 1; k < part.size(); ++k) acc = max ? std::fmax(acc, part[k]) : acc + part[k];
    return acc;
}

// One case: the model's arm, window=0, ring=0, whole=false, at=null.
struct Case { long long pos; int rows, chunk; int n_head, kv_heads, dim; const char* what; };

void run(const Case& cs) {
    const int per = cs.n_head / cs.kv_heads;
    const float scale = 1.0f / std::sqrt((float) cs.dim);
    const long long nrows = cs.pos + cs.rows;
    std::mt19937 rng(99 + cs.rows * 7 + size_t(cs.pos));
    std::vector<float> q(size_t(cs.rows) * cs.n_head * cs.dim);
    std::vector<uint16_t> kc(size_t(nrows) * cs.kv_heads * cs.dim), vc = kc;
    for (auto& v : q) v = frandom(rng) * 0.4f;
    // include some subnormal and zero cache values: the decode must be exact there too
    for (size_t i = 0; i < kc.size(); ++i) {
        const int r = int(rng() % 64);
        if (r == 0) kc[i] = 0;
        else if (r == 1) kc[i] = uint16_t(rng() % 0x400u);          // f16 subnormal
        else kc[i] = f16_rn(frandom(rng));
    }
    vc = kc;
    for (size_t i = 0; i < vc.size(); ++i) {
        const int r = int(rng() % 64);
        if (r == 0) vc[i] = 0;
        else if (r == 1) vc[i] = uint16_t(rng() % 0x400u);
        else vc[i] = f16_rn(frandom(rng) * 0.7f);
    }

    // ---- ORDER reference: the three kernels' exact partitions (window=0, ring=0, at=null),
    // replaying the launcher's chunk loop: each launch's score rows lie `keys` apart.
    const int span = int(cs.pos + cs.rows);              // whole=false: the full causal span
    // the launcher's reg-variant pick (mirrors the CUDA wrapper: kernels.cu:2488-2499)
    const int N = (per % 8 == 0 && cs.dim == 256) ? 8 : (per % 4 == 0) ? 4 : 0;
    const int STEPS = cs.dim / 128;
    if (!N) { std::printf("  SKIP %s: no reg variant\n", cs.what); return; }
    std::vector<float> score(size_t(cs.rows) * cs.n_head * span, 0.f);
    std::vector<float> stride_of(size_t(cs.rows), 0.f);
    for (int first = 0, li = 0; first < cs.rows; first += cs.chunk, ++li) {
        const int c = std::min(cs.chunk, cs.rows - first);
        const int keys = int(cs.pos) + first + c;        // lo=0: span of this launch
        stride_of[first] = float(keys);                  // the launch's score stride
        // score kernel: per unit (k0-block, group), per th-step (token t, head base h)
        const int units = (keys + 7) / 8 * cs.kv_heads;
        for (int u = 0; u < units; ++u) {
            const int g = u % cs.kv_heads, k0 = u / cs.kv_heads * 8;
            const int rowsk = std::min(8, keys - k0);
            for (int th = 0; th < c * per; th += N) {
                const int t = first + th / per, h = g * per + th % per;
                const float* qq = q.data() + (size_t(t) * cs.n_head + h) * cs.dim;
                for (int r = 0; r < rowsk; ++r) {
                    const uint16_t* kr = kc.data() + (size_t(k0 + r) * cs.kv_heads + g) * cs.dim;
                    float acc[32][8];
                    for (int l = 0; l < 32; ++l)
                        for (int hh = 0; hh < N; ++hh) {
                            float a = 0.f;
                            for (int st = 0; st < STEPS; ++st)
                                for (int j = 0; j < 4; ++j) {
                                    const int i = 128 * st + 4 * l + j;
                                    a = std::fma(qq[hh * cs.dim + i], f16_to_float(kr[i]), a);
                                }
                            acc[l][hh] = a;
                        }
                    // scores_sum<N> with two-level (pre-round) partner reads
                    float cur[32][8];
                    for (int l = 0; l < 32; ++l) for (int i = 0; i < 8; ++i) cur[l][i] = acc[l][i];
                    for (int off = 16, m = N; off > 0; off >>= 1) {
                        float nxt[32][8];
                        for (int l = 0; l < 32; ++l) for (int i = 0; i < 8; ++i) nxt[l][i] = cur[l][i];
                        for (int l = 0; l < 32; ++l) {
                            const bool upper = (l & off) != 0;
                            const int pl = l ^ off;
                            if (m > 1) {
                                const int half = m / 2;
                                // CUDA's m>1 arm writes v[0..half) only; the higher slots
                                // stale-carry exactly as the hardware's registers do.
                                for (int i = 0; i < half; ++i) {
                                    const float keep = upper ? cur[l][i + half] : cur[l][i];
                                    const float psend = upper ? cur[pl][i + half] : cur[pl][i];
                                    nxt[l][i] = keep + psend;
                                }
                            } else {
                                nxt[l][0] = cur[l][0] + cur[pl][0];
                            }
                        }
                        if (m > 1) m /= 2;   // CUDA's m = half: once per round, uniform
                        for (int l = 0; l < 32; ++l) for (int i = 0; i < 8; ++i) cur[l][i] = nxt[l][i];
                    }
                    for (int l = 0; l < 32; ++l) if ((l & (32 / N - 1)) == 0) {
                        // idx folds the m>1 rounds' bits high-first: bit16, bit8, (+bit4 for N=8)
                        const int which = N >= 8 ? ((l >> 4) & 1) * 4 + ((l >> 3) & 1) * 2 + ((l >> 2) & 1)
                                                 : ((l >> 4) & 1) * 2 + ((l >> 3) & 1);
                        score[(size_t(t) * cs.n_head + h + which) * span + k0 + r] = cur[l][0] * scale;
                    }
                }
            }
        }
    }
    // softmax kernel (in place over the span; window=0: own=0, n = pos+t+1 = span_t)
    for (int t = 0; t < cs.rows; ++t) {
        const int n = int(cs.pos) + t + 1;
        for (int h = 0; h < cs.n_head; ++h) {
            float* rowp = score.data() + (size_t(t) * cs.n_head + h) * span;
            // strided pass: thread tid takes j = tid, tid+256...: mx per thread, then block_max
            // (rows past n stay 0 from the fill: CUDA writes zeros there but they never enter
            // the mx/sum loops, and the value walk multiplies the 0 weights, matching CUDA)
            std::vector<float> mxth(256, -INFINITY);
            for (int j = 0; j < n; ++j) mxth[j % 256] = std::fmax(mxth[j % 256], rowp[j]);
            const float mx = ref_block_reduce(mxth, 256, true);
            std::vector<float> smth(256, 0.f);
            for (int j = 0; j < n; ++j) rowp[j] = std::exp(rowp[j] - mx);
            for (int j = 0; j < n; ++j) smth[j % 256] += rowp[j];
            const float inv = 1.0f / ref_block_reduce(smth, 256, false);
            for (int j = 0; j < n; ++j) rowp[j] *= inv;
            for (int j = n; j < span; ++j) rowp[j] = 0.0f;   // CUDA 1241-1242: the rows the token does not attend
        }
    }
    // value walk (both arms sum the same products in the same order: the strided share walk
    // then k=0..31; the weights differ only in WHERE they lie, not their values)
    std::vector<float> ref_out(size_t(cs.rows) * cs.n_head * cs.dim, 0.f);
    for (int t = 0; t < cs.rows; ++t)
        for (int h = 0; h < cs.n_head; ++h) {
            const int g = h / per;
            const float* wrow = score.data() + (size_t(t) * cs.n_head + h) * span;
            for (int d = 0; d < cs.dim; ++d) {
                // one thread = lane d; its share s walks rows j = s, s+32, ... of the span and
                // acc[s] += w[j] * value(row j, d) in that order; the block then adds share
                // buckets k = 0..31 (window=0: own=0, so no rotation).
                float acc[32] = {};
                for (int s = 0; s < 32; ++s)
                    for (int j = s; j < span; j += 32) {
                        const float xv = f16_to_float(vc[(size_t(j) * cs.kv_heads + g) * cs.dim + d]);
                        acc[s] = std::fma(wrow[j], xv, acc[s]);
                    }
                // rotation: own=0 for window=0, so bucket index (s + 32 - 0) % 32 = s: k=0..31 order
                float sum = 0.f;
                for (int k = 0; k < 32; ++k) sum += acc[k];
                ref_out[(size_t(t) * cs.n_head + h) * cs.dim + d] = sum;
            }
        }

    // ---- DIRECT reference (naive, no tiling)
    std::vector<float> naive(size_t(cs.rows) * cs.n_head * cs.dim, 0.f);
    for (int t = 0; t < cs.rows; ++t)
        for (int h = 0; h < cs.n_head; ++h) {
            const int g = h / per, n = int(cs.pos) + t + 1;
            std::vector<float> sc(n);
            float mx = -INFINITY;
            for (int r = 0; r < n; ++r) {
                float s = 0.f;
                for (int d = 0; d < cs.dim; ++d)
                    s += q[(size_t(t) * cs.n_head + h) * cs.dim + d] * f16_to_float(kc[(size_t(r) * cs.kv_heads + g) * cs.dim + d]);
                sc[r] = s * scale;
                mx = std::fmax(mx, sc[r]);
            }
            double sum = 0.0;
            for (int r = 0; r < n; ++r) sum += std::exp(sc[r] - mx);
            const float inv = float(1.0 / sum);
            for (int d = 0; d < cs.dim; ++d) {
                float o = 0.f;
                for (int r = 0; r < n; ++r)
                    o += sc[r] * 0 + float(std::exp(sc[r] - mx)) * inv * f16_to_float(vc[(size_t(r) * cs.kv_heads + g) * cs.dim + d]);
                naive[(size_t(t) * cs.n_head + h) * cs.dim + d] = o;
            }
        }

    // ---- the device run
    vk::Arena& a = vk::Arena::instance();
    // +8: the launcher bounds the cache spans with the u64-read slack (a lane on the last odd
    // half touches the next word), exactly like the CUDA body.
    const uint64_t aq = a.alloc(q.size() * 4), ak = a.alloc(kc.size() * 2 + 8), av = a.alloc(vc.size() * 2 + 8),
                   as = a.alloc(size_t(cs.chunk) * cs.n_head * span * 4 + size_t(cs.kv_heads) * ((cs.rows * per + 7) / 8) * span * 8 * 4),
                   ao = a.alloc(size_t(cs.rows) * cs.n_head * cs.dim * 4);
    vk::copy_h2d((void*)aq, q.data(), q.size() * 4, nullptr);
    vk::copy_h2d((void*)ak, kc.data(), kc.size() * 2, nullptr);
    vk::fill_d((void*)(ak + kc.size() * 2), 0, 8, nullptr);
    vk::copy_h2d((void*)av, vc.data(), vc.size() * 2, nullptr);
    vk::fill_d((void*)(av + vc.size() * 2), 0, 8, nullptr);
    vk::fill_d((void*)as, 0, size_t(cs.chunk) * cs.n_head * span * 4 + size_t(cs.kv_heads) * ((cs.rows * per + 7) / 8) * span * 8 * 4, nullptr);
    vk::fill_d((void*)ao, 0, size_t(cs.rows) * cs.n_head * cs.dim * 4, nullptr);
    std::vector<float> got(size_t(cs.rows) * cs.n_head * cs.dim, 0.f);
    try {
        poly::attend((float*)strata_vk_compat::wrap(aq), (uint16_t*)strata_vk_compat::wrap(ak),
                     (uint16_t*)strata_vk_compat::wrap(av), cs.pos, cs.rows,
                     (float*)strata_vk_compat::wrap(as), cs.chunk,
                     (float*)strata_vk_compat::wrap(ao), cs.dim, cs.n_head, cs.kv_heads, scale, nullptr);
        vk::copy_d2h(got.data(), (void*)ao, got.size() * 4, nullptr);
    } catch (const std::exception& e) {
        std::printf("  FAIL %s: launcher threw: %s\n", cs.what, e.what());
        fails++;
        return;
    }

    auto rel = [](const std::vector<float>& g, const std::vector<float>& r) {
        double d = 0.0, m = 0.0;
        for (size_t i = 0; i < g.size(); ++i) { d += std::fabs((double)g[i] - r[i]); m += std::fabs((double)g[i]); }
        return d / (m > 1e-30 ? m : 1e-30);
    };
    const double ro = rel(got, ref_out), rn = rel(got, naive);
    // per-element: worst |dev - ref| against the tensor's OWN max (the gdn_parity.cpp:88 form —
    // a fixed 1e-3 floor amplifies few-ULP noise at cancelling softmax cells, which the tree's
    // rel_l1 exists to avoid).
    double worst = 0.0, rmax = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::fmax(worst, std::fabs((double)got[i] - ref_out[i]));
        rmax = std::fmax(rmax, std::fabs(ref_out[i]));
    }
    std::printf("  attend %s: out rel_l1 order %.3e naive %.3e; worst abs %.3e (max ref %.3f) target rel_l1 <=1e-5\n",
                cs.what, ro, rn, worst, rmax);
    check(ro < 1e-5 && rn < 1e-5 && worst < 1e-5, cs.what);
}

}  // namespace

int main() {
    try { vk::device_info(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "poly_attend_vk_parity: %s\nSKIP\n", e.what());
        return 77;
    }
    const Case cases[] = {
        {17, 1, 1, 16, 2, 128, "dim128 decode rows=1 chunk=1 (plain value arm)"},
        {31, 1, 4, 16, 2, 128, "dim128 decode rows=1 chunk=4 (wt arm, attn_value_t<1>)"},
        {24, 5, 8, 16, 2, 128, "dim128 chunked launch rows=5 chunk=8"},
        {63, 1, 1, 16, 2, 256, "MODEL dim256 per8 rows=1 chunk=1 (plain value arm)"},
        {40, 1, 4, 16, 2, 256, "MODEL dim256 rows=1 chunk=4 (wt arm, attn_value_t<1>)"},
        {33, 3, 8, 16, 2, 256, "MODEL dim256 chunked rows=3 chunk=8"},
    };
    for (const auto& cs : cases) run(cs);
    std::printf("poly_attend_vk_parity %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
