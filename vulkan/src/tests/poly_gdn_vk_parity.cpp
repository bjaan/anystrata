// vulkan/src/tests/poly_gdn_vk_parity.cpp - parity for strata::poly::gdn_step (Task 2.5).
//
// gdn_step has no in-tree _parity.cpp for the poly form (PORT-STATUS row cites gdn_parity.cpp for
// the strata::kernels GDN family); like Tasks 2.1/2.4, this writes the reference the tree's own
// way: a scalar fp32 transcription of src/poly/kernels.cu's chain — gdn_norm_kernel (:1988),
// gdn_gates_kernel (:2002), the state walk (:2017 plain / :2058 reg<PER>) and gdn_out_kernel
// (:2118) — run against the device through the arena.
//
// Sum partitions are transcribed EXACTLY (law #4): the block_sum's xor tree with per-round copies
// then the ordered part[] walk (the launcher pins workgroup size = dim, so the warp count is
// known); the state walk's per-lane strided accumulation then the warp xor tree. The ONLY freedom
// left is fp32 contraction (law #6): the reference runs both arms — FUSED (nvcc fmad semantics,
// spelled by the reg kernel's __fmul_rn/__fmaf_rn: sk=fma(col*dc, kn, sk), n=fma(dc, col, kn*d),
// o=fma(n, q, o)) and UNFUSED (mul, mul, add) — and the test passes if the device matches ONE arm
// within the tree's rel 1e-5 gemv-family bound (the closer arm; bitwise reported when it falls
// out, as conv4 measured the unfused arm at 5.9e-7 on this box).
//
// expf/logf/sqrtf between CUDA and RADV are ~1 ULP apart at most; the walk is shallow (rows <= 8
// here, the model calls rows == 1 per decode step) so the rel 1e-5 bound holds with room to spare.
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

float frandom(std::mt19937& rng) {   // ~N(0, 0.5) without <random> normal dependencies
    return (float)((int)(rng() % 2001) - 1000) / 1000.0f;
}

// warp_sum tree over 32 lanes with per-round copies (law #4), then the ordered part[] walk:
// the exact block_sum of src/poly/kernels.cu:42 when each thread holds one value.
float ref_block_sum(const float* vals, int n) {   // n = dim, block = n threads
    const int warps = n / 32;
    float s = 0.f;
    for (int w = 0; w < warps; ++w) {
        float l[32];
        for (int j = 0; j < 32; ++j) l[j] = vals[w * 32 + j];
        for (int off = 16; off > 0; off >>= 1) {
            float p[32];
            std::memcpy(p, l, sizeof l);
            for (int j = 0; j < 32; ++j) l[j] = l[j] + p[j ^ off];
        }
        s = s + l[0];   // part[0] + part[1] + ... IN ORDER
    }
    return s;
}
float ref_warp_sum(const float* lane) {   // one warp: lane[32] -> all-lane xor tree
    float l[32];
    std::memcpy(l, lane, sizeof l);
    for (int off = 16; off > 0; off >>= 1) {
        float p[32];
        std::memcpy(p, l, sizeof l);
        for (int j = 0; j < 32; ++j) l[j] = l[j] + p[j ^ off];
    }
    return l[0];
}
float ref_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct GdnCase { int dim, n_head, k_heads, rows; unsigned snaps; const char* what; };

// One full reference pass (fused or unfused contraction arm) of the whole gdn_step chain.
void ref_gdn_step(const std::vector<float>& qkv0, const std::vector<float>& alpha0,
                  const std::vector<float>& beta0, const std::vector<float>& dt,
                  const std::vector<float>& a, std::vector<float>& s, const std::vector<float>& z,
                  const std::vector<float>& o_norm, std::vector<float>& out, std::vector<float>& snap,
                  int dim, int n_head, int k_heads, int rows, float eps, size_t stride, unsigned snaps,
                  bool fused) {
    const size_t row_n = (size_t(2 * k_heads + n_head)) * dim;
    const size_t keys = size_t(dim) * k_heads, d_inner = size_t(dim) * n_head;
    std::vector<float> qkv = qkv0, alpha = alpha0, beta = beta0;
    // 1. gdn_norm_kernel: q scaled by 1/sqrt(sum q^2 + eps) * 1/sqrt(dim); k by 1/sqrt(sum k^2 + eps)
    for (int row = 0; row < rows; ++row)
        for (int hk = 0; hk < k_heads; ++hk) {
            std::vector<float> terms(dim), kt(dim);
            const float* x = qkv.data() + (size_t)row * row_n;
            for (int j = 0; j < dim; ++j) {
                const float q = x[size_t(hk) * dim + j], k = x[keys + size_t(hk) * dim + j];
                terms[j] = q * q; kt[j] = k * k;
            }
            const float qs = ref_block_sum(terms.data(), dim), ks = ref_block_sum(kt.data(), dim);
            float* y = qkv.data() + (size_t)row * row_n;
            for (int j = 0; j < dim; ++j) {
                y[size_t(hk) * dim + j] = y[size_t(hk) * dim + j] / std::sqrt(qs + eps) * (1.0f / std::sqrt((float)dim));
                y[keys + size_t(hk) * dim + j] = y[keys + size_t(hk) * dim + j] / std::sqrt(ks + eps);
            }
        }
    // 2. gdn_gates_kernel: alpha = exp(softplus(alpha + dt[h]) * a[h]); beta = sigmoid(beta)
    for (int at = 0; at < rows * n_head; ++at) {
        const int h = at % n_head;
        const float g = alpha[at] + dt[h];
        alpha[at] = std::exp((g > 20.0f ? g : std::log(1.0f + std::exp(g))) * a[h]);
        beta[at] = ref_sigmoid(beta[at]);
    }
    // 3. the state walk (plain and reg kernels agree to the bit in CUDA; the partition is per lane)
    for (int row = 0; row < rows; ++row)
        for (int h = 0; h < n_head; ++h)
            for (int j = 0; j < dim; ++j) {
                float* col = s.data() + (size_t(h) * dim + j) * dim;
                const float* x = qkv.data() + (size_t)row * row_n;
                const float* qn = x + size_t(h % k_heads) * dim;
                const float* kn = x + keys + size_t(h % k_heads) * dim;
                const float v = x[2 * keys + size_t(h) * dim + j];
                const float dc = alpha[size_t(row) * n_head + h], sh = beta[size_t(row) * n_head + h];
                // per-lane sums then the warp tree
                float lane[32];
                for (int l = 0; l < 32; ++l) {
                    float skk = 0.f;
                    for (int i = l; i < dim; i += 32)
                        skk = fused ? std::fmaf(col[i] * dc, kn[i], skk) : skk + (col[i] * dc) * kn[i];
                    lane[l] = skk;
                }
                const float sk = ref_warp_sum(lane);
                const float d = (v - sk) * sh;
                for (int l = 0; l < 32; ++l) lane[l] = 0.f;
                for (int l = 0; l < 32; ++l)
                    for (int i = l; i < dim; i += 32) {
                        const float n = fused ? std::fmaf(dc, col[i], kn[i] * d)
                                              : (col[i] * dc) + (kn[i] * d);
                        col[i] = n;
                        if (row < 32 && ((snaps >> row) & 1u)) {
                            const size_t slot = __builtin_popcount(snaps & ((1u << row) - 1u));
                            snap[slot * stride + (size_t(h) * dim + j) * dim + i] = n;
                        }
                        lane[l] = fused ? std::fmaf(n, qn[i], lane[l]) : lane[l] + n * qn[i];
                    }
                const float o = ref_warp_sum(lane);
                out[size_t(row) * d_inner + size_t(h) * dim + j] = o;
            }
    // 4. gdn_out_kernel: rms-norm, weights, silu(z) gate; left-to-right ((o*r)*onorm)*(g*sig(g))
    const std::vector<float> pre = out;   // gdn_out_kernel reads every o before any write (barrier)
    for (size_t r = 0; r < size_t(rows) * n_head; ++r) {
        std::vector<float> terms(dim);
        for (int i = 0; i < dim; ++i) terms[i] = pre[r * dim + i] * pre[r * dim + i];
        const float os = ref_block_sum(terms.data(), dim);
        for (int j = 0; j < dim; ++j) {
            const size_t at = r * dim + j;
            const float o = pre[at], gate = z[at];
            out[at] = o / std::sqrt(os / float(dim) + eps) * o_norm[j] * (gate * ref_sigmoid(gate));
        }
    }
}

}  // namespace

int main() {
    try { vk::device_info(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "poly_gdn_vk_parity: %s\nSKIP\n", e.what());
        return 77;
    }
    vk::Arena& a = vk::Arena::instance();
    const float eps = 1e-6f;

    const GdnCase cases[] = {
        {128, 32, 16, 1, 0u, "model decode shape dim=128 nh=32 kh=16 rows=1 (reg<4>)"},
        {128, 32, 16, 5, 0u, "dim=128 rows=5 token walk (reg<4>)"},
        {128, 32, 16, 8, 0x41u, "dim=128 rows=8 + snapshots 0x41 (reg<4> snap arm)"},
        {32, 4, 2, 3, 0u, "dim=32 (reg<1>)"},
        {64, 8, 4, 2, 0u, "dim=64 (reg<2>)"},
        {96, 6, 3, 2, 0u, "dim=96 (plain state kernel arm)"},
    };

    for (const auto& cs : cases) {
        const size_t row_n = size_t(2 * cs.k_heads + cs.n_head) * cs.dim;
        const size_t d_inner = size_t(cs.dim) * cs.n_head;
        const unsigned snaps = cs.snaps;
        // The engine's rec_n_: the WHOLE per-slot state span, since snap_of() adds mine=(h*dim+j)*dim.
        const size_t stride = size_t(cs.n_head) * cs.dim * cs.dim;
        const int n_snap = snaps ? __builtin_popcount(snaps) : 0;

        std::mt19937 rng(1234 + cs.dim + cs.rows);
        std::vector<float> qkv((size_t)cs.rows * row_n), alpha((size_t)cs.rows * cs.n_head),
            beta((size_t)cs.rows * cs.n_head), dt(cs.n_head), a_(cs.n_head),
            s(size_t(cs.n_head) * cs.dim * cs.dim), z((size_t)cs.rows * d_inner),
            onorm(cs.dim), out((size_t)cs.rows * d_inner, 0.f),
            snap((size_t)n_snap * stride, 0.f);
        for (auto& v : qkv) v = frandom(rng) * 0.5f;
        for (auto& v : alpha) v = frandom(rng);
        for (auto& v : beta) v = frandom(rng);
        for (auto& v : dt) v = 0.1f + 0.4f * std::fabs(frandom(rng));
        for (auto& v : a_) v = 0.2f * std::fabs(frandom(rng)) + 0.01f;
        for (auto& v : s) v = frandom(rng) * 0.3f;
        for (auto& v : z) v = frandom(rng);
        for (auto& v : onorm) v = 0.5f + std::fabs(frandom(rng));

        const uint64_t aq = a.alloc(qkv.size() * 4), aa = a.alloc(alpha.size() * 4),
                       ab = a.alloc(beta.size() * 4), adt = a.alloc(dt.size() * 4),
                       aa2 = a.alloc(a_.size() * 4), as = a.alloc(s.size() * 4),
                       az = a.alloc(z.size() * 4), ao_ = a.alloc(onorm.size() * 4),
                       ao2 = a.alloc(out.size() * 4),
                       asn = n_snap ? a.alloc(snap.size() * 4) : 0;
        vk::copy_h2d((void*)aq, qkv.data(), qkv.size() * 4, nullptr);
        vk::copy_h2d((void*)aa, alpha.data(), alpha.size() * 4, nullptr);
        vk::copy_h2d((void*)ab, beta.data(), beta.size() * 4, nullptr);
        vk::copy_h2d((void*)adt, dt.data(), dt.size() * 4, nullptr);
        vk::copy_h2d((void*)aa2, a_.data(), a_.size() * 4, nullptr);
        vk::copy_h2d((void*)as, s.data(), s.size() * 4, nullptr);
        vk::copy_h2d((void*)az, z.data(), z.size() * 4, nullptr);
        vk::copy_h2d((void*)ao_, onorm.data(), onorm.size() * 4, nullptr);
        vk::fill_d((void*)ao2, 0, out.size() * 4, nullptr);
        if (n_snap) vk::fill_d((void*)asn, 0, snap.size() * 4, nullptr);

        poly::gdn_step((float*)strata_vk_compat::wrap(aq), (float*)strata_vk_compat::wrap(aa),
                       (float*)strata_vk_compat::wrap(ab), (const float*)strata_vk_compat::wrap(adt),
                       (const float*)strata_vk_compat::wrap(aa2), (float*)strata_vk_compat::wrap(as),
                       (const float*)strata_vk_compat::wrap(az), (const float*)strata_vk_compat::wrap(ao_),
                       (float*)strata_vk_compat::wrap(ao2), cs.dim, cs.n_head, cs.k_heads, cs.rows, eps,
                       nullptr, n_snap ? (float*)strata_vk_compat::wrap(asn) : nullptr, stride, snaps);

        std::vector<float> got_out(out.size(), 0.f), got_s(s.size(), 0.f), got_snap(snap.size(), 0.f);
        vk::copy_d2h(got_out.data(), (void*)ao2, got_out.size() * 4, nullptr);
        vk::copy_d2h(got_s.data(), (void*)as, got_s.size() * 4, nullptr);
        if (n_snap) vk::copy_d2h(got_snap.data(), (void*)asn, got_snap.size() * 4, nullptr);

        // both arms from the same inputs
        std::vector<float> s_f = s, s_u = s, o_f(out.size(), 0.f), o_u(out.size(), 0.f),
                           snap_f(snap.size(), 0.f), snap_u(snap.size(), 0.f);
        ref_gdn_step(qkv, alpha, beta, dt, a_, s_f, z, onorm, o_f, snap_f, cs.dim, cs.n_head,
                     cs.k_heads, cs.rows, eps, stride, snaps, true);
        ref_gdn_step(qkv, alpha, beta, dt, a_, s_u, z, onorm, o_u, snap_u, cs.dim, cs.n_head,
                     cs.k_heads, cs.rows, eps, stride, snaps, false);

        // The tree's OWN GDN metric (src/kernels/gdn_parity.cpp:88 rel_l1: sum|a-b| / max|b|, tol
//   1e-5 at :202-203 — 'the arithmetic's floor, not a structural allowance'). Per-element rel with
//   a 1e-3 floor is NOT gdn's in-tree bound: the output's RMS norm divides by sqrt(sum o^2 / dim)
//   and amplifies a few-ULP delta at cancelling cells; rel_l1 measures the same absolute noise
//   against the tensor's own scale. Never looser: same 1e-5 number, the form the GDN row uses.
        auto rel = [](const std::vector<float>& g, const std::vector<float>& r) {
            double d = 0.0, m = 0.0;
            for (size_t i = 0; i < g.size(); ++i) {
                d += std::fabs((double)g[i] - r[i]);
                m += std::fabs((double)g[i]);          // rel_l1: denominator sum|device| (gdn_parity.cpp:88)
            }
            return d / (m > 1e-30 ? m : 1e-30);
        };
        auto absmax = [](const std::vector<float>& g, const std::vector<float>& r) {
            double w = 0.0, sc = 0.0;
            for (size_t i = 0; i < g.size(); ++i) {
                w = std::fmax(w, std::fabs((double)g[i] - r[i]));
                sc = std::fmax(sc, std::fabs((double)r[i]));
            }
            return std::pair<double, double>(w, sc);   // (worst |d|, max |ref|)
        };
        // (diagnostics only: absolute ULPs and the tree's rel_l1 together prove the delta is
        // per-ULP noise, not a partition mismatch.)
        const auto [of, oscale] = absmax(got_out, o_f);
        const auto [ou, _] = absmax(got_out, o_u);
        const auto [sfa, ssc] = absmax(got_s, s_f);
        const auto [sua, _b] = absmax(got_s, s_u);
        std::printf("  scales: out |ref|max %.4f, worst abs dev-vs-fused %.3e / unfused %.3e; "
                    "s max %.4f, abs %.3e/%.3e\n", oscale, of, ou, ssc, sfa, sua);
        const double wf = rel(got_out, o_f), wu = rel(got_out, o_u);
        const bool bit_f = got_out == o_f && got_s == s_f;
        const bool bit_u = got_out == o_u && got_s == s_u;
        const double closer = wf <= wu ? wf : wu;
        std::printf("  gdn %s: out fused rel %.3e%s, unfused rel %.3e%s (tree tol 1e-5)\n",
                    cs.what, wf, bit_f ? " [BITWISE]" : "", wu, bit_u ? " [BITWISE]" : "");
        bool ok = closer < 1e-5;
        // state cells: covered by bit_f/bit_u above (got_s == ref); check the closer arm's state rel too
        const double sf = rel(got_s, s_f), su = rel(got_s, s_u);
        if ((wf <= wu ? sf : su) >= 1e-5) { ok = false; std::printf("  state rel over bound: %.3e\n", sf < su ? sf : su); }
        if (n_snap) {   // snapshots: exact copies of the closer arm's n values
            const double sn_f = rel(got_snap, snap_f), sn_u = rel(got_snap, snap_u);
            if (std::min(sn_f, sn_u) >= 1e-5) { ok = false; std::printf("  snap rel over bound: %.3e/%.3e\n", sn_f, sn_u); }
            std::printf("  snap slots: rel %.3e/%.3e\n", sn_f, sn_u);
        }
        check(ok, cs.what);
    }

    std::printf("poly_gdn_vk_parity %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
