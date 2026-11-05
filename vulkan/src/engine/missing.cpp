// vulkan/src/engine/missing.cpp - what a kernel entry point with no Vulkan implementation yet does when
// called. The trap pattern is metal/src/engine/missing.cpp's, on this backend's names:
//
// Default: say which one and stop (exit code 86), so a run never continues on outputs nobody wrote.
// STRATA_VK_STUBS=log: say it once per entry point and return (outputs untouched) - for listing which
// entry points a code path reaches, never for answers.
//
// The trap DEFINITIONS themselves are generated (stubs_gen.cpp, vulkan/tools/gen_stubs.py) from the
// engine's own headers; this file owns the entry and the handful of functions the ledger marks n/a or
// that the host loop asks unconditionally - the metal file's shape.
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

namespace strata::vk_engine {

void missing(const char* name) {
    static const bool log_only = [] {
        const char* v = std::getenv("STRATA_VK_STUBS");
        return v != nullptr && std::string(v) == "log";
    }();
    if (!log_only) {
        std::fprintf(stderr, "strata-vk: %s has no Vulkan implementation yet\n", name);
        std::fflush(stderr);
        std::_Exit(86);
    }
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(name).second) std::fprintf(stderr, "strata-vk: STUB %s\n", name);
}

}  // namespace strata::vk_engine

// ---- the functions the trap roster does NOT cover, defined honestly here
#include <atomic>
#include <unordered_map>
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/iq_kernels.hpp"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace strata::kernels {

// no control vector on Vulkan (--control-vector-scaled is not ported): the empty one, `loaded()` false
const Cvec& cvec() {
    static const Cvec none{};
    return none;
}

// Capability queries, host-side by contract (the CUDA build answers from a pure type table). They are
// transcribed from src/kernels/cuda/native_mmvq.cu (native_mmvq_supported's switch and slots_kind's) so
// the ENGINE's format decisions — keep the GGUF matrices as-is, cache them, batch them — stay what the
// CUDA build's are, and the run reaches the launch traps (native_mmvq & co. stay trapped in
// stubs_gen.cpp) in the model's own demand order instead of re-routing the whole load to bf16 copies.
// native_q8_1_bytes / native_q8_1_il_bytes are pure byte arithmetic (Q81Block: half2 ds + int8 qs[32],
// 36 bytes; Q8K = 32; the interleave copy doubles it) — the scratch sizing the load does with them is
// no device work. native_mmvq_il_supported: FALSE — the interleaved MMVQ variants are not on this
// backend's port plan at the CUDA table's shape (Phase 2 picks the plain path).
// The native (IQ) expert tier's capability query: pure type-table arithmetic, transcribed from
// src/kernels/cuda/iq_kernels.cu:3570 (native_expert_supported: is_iq x gu_qk x d_qk x the 256-block
// shape rules; qk values from the Fmt<T> table, STRATA_Q6K_EXPERTS=1 as this tree's option defaults).
// Same rule as native_mmvq_supported above: trapping a load-path QUESTION would route the model through
// a different expert tier than the CUDA build picks; the ANSWER's kernels (native_expert_*) stay
// reachable through the CPU pool (strata_vk_kernels_cpu) — this query asks about HOST work, not CUDA.
bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 ||
           t == 11 || t == 12 || t == 13 || t == 14 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8 || t == 40;
}
int gu_qk(int t) {
    switch (t) {
        case 16: case 17: case 18: case 21: case 22: case 23: case 29: case 12: case 13: case 14: return 256;
        case 42: case 40: return 64;
        case 6: case 2: case 3: case 8: return 32;
        default: return 0;
    }
}
int d_qk(int t) {   // STRATA_D_FMTS(20 23 42 7 6 2 3 8 13 14 40) with the Fmt<T>::qk values
    switch (t) {
        case 23: case 13: case 14: return 256;
        case 42: case 40: return 64;
        case 20: case 7: case 6: case 2: case 3: case 8: return 32;
        default: return 0;
    }
}
size_t iq_row_bytes(int t, int64_t n) noexcept {   // verbatim from iq_kernels.cu (host byte arithmetic)
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 2: return (size_t) (n / 32) * sizeof(block_q4_0);
        case 3: return (size_t) (n / 32) * sizeof(block_q4_1);
        case 40: return (size_t) (n / 64) * sizeof(block_nvfp4);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 30: return (size_t) n * 2;   // BF16: the token embedding only
        default: return 0;
    }
}
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    // pure byte arithmetic from iq_kernels.cu:3577 (the row bytes come from iq_row_bytes, which the
    // ggml type table answers; NVFP4's +16 tail per the CUDA comment)
    NativeExpertLayout L;
    L.gu_type = gu_type; L.d_type = d_type; L.n_embd = n_embd; L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    if (gu_type == 40 || d_type == 40) { L.tail_off = L.bytes; L.bytes += 16; }
    return L;
}
size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) +
           ((((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1)) + 255) & ~(size_t) 255);
}

// The rope table builders and the process's rope config: pure host float64 math, transcribed
// verbatim from src/kernels/cuda/rope.cu:45-86 (the same loop order the CUDA build runs — "float64
// throughout, in the reference's order" — so the tables are bitwise what CUDA computes; yarn's ramp/
// corr_dims/freq_scale are header-inline in rope_scaling.hpp, included above). One writer before
// session_init, readers after: the CUDA file's contract, unchanged here.
namespace { RopeScaling g_rope_scaling; }
void rope_scaling_set(const RopeScaling& scaling) { g_rope_scaling = scaling; }
const RopeScaling& rope_scaling() { return g_rope_scaling; }
void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}
void build_rope_table(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    if (sc.type == RopeScalingType::None) { build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab, sin_tab); return; }
    const int half = n_rot / 2;
    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(sc.freq_base, -2.0 * (double) i / (double) n_rot);
            const double extrap = (double) p * inv;
            const double interp = fs * extrap;
            double ang = interp;
            if (correct) {
                const double ramp = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[(size_t) p * half + i] = (float) (std::cos(ang) * ms);
            sin_tab[(size_t) p * half + i] = (float) (std::sin(ang) * ms);
        }
    }
}

// The MMVQ weight-size question the pack load asks per matrix (native_mmvq.cu:2046ff, verbatim):
// pure byte arithmetic + the same shape validation the ported launchers run (validate_shape's body,
// transcribed here; Q8K=32, MAX_NCOLS=8 from the CUDA file's constants).
std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 39: block_elems = 32; block_bytes = 17; break;
    case 10: block_elems = 256; block_bytes = 84; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    if (n_in <= 0 || n_in % block_elems != 0)
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out))
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    return row_bytes * std::size_t(n_out);
}

// The packed-Q8_0 launch optimization (persistent-warp weight layout, native_mmvq.cu): enabled is the
// env question verbatim; ELIGIBLE is this backend's honest "no packed kernel is built here" — the
// plain native_mmvq path (ported+parity-green) serves every shape. A caller that sets STRATA_Q8_PACKED=1
// anyway reaches register/pack_host, and those stay named traps (stubs_gen): enabling it changes nothing silently.
// ---- the GR / hyper-connection module's host surface (gr.cu, fused_gr.cu): config setters,
// capability questions and the workspace layout arithmetic — transcribed verbatim. The READ/WRITE
// kernels (gr_read/gr_write/fused_gr_read/... and the persist path) stay named traps: those ARE the
// fp32-fallback port work (PORT-STATUS row 36/38), and a selftest against the plain read needs them,
// so fused_gr_check's honest answer on this backend is the plain variant (kHcPlain), exactly what
// the CUDA build falls back to when the card fails its bit-for-bit check.
namespace { bool g_gr_fp32_act = false, g_gr_native_mmvf = false; }
void gr_set_fp32_activations(bool enabled) { g_gr_fp32_act = enabled; }
void gr_set_native_mmvf(bool enabled) { g_gr_native_mmvf = enabled; }
bool gr_native_mmvf_enabled() { return g_gr_native_mmvf; }
size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    const size_t sz[5] = {
        hc_dim * sizeof(float),                // 0: xn
        hc_dim * sizeof(uint16_t),             // 1: xq
        (size_t) s.hc_lr * sizeof(uint16_t),   // 2: lq
        hc_dim * sizeof(float),                // 3: gated
        (size_t) s.hc_lr * sizeof(float),      // 4: lo
    };
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) { al[k] = (sz[k] + 15) & ~(size_t) 15; bytes += al[k]; }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[5];
        for (int k = 0; k < 5; ++k) { ptr[k] = p; p += al[k]; }
        out.xn = (float*) ptr[0]; out.xq = (uint16_t*) ptr[1]; out.lq = (uint16_t*) ptr[2];
        out.gated = (float*) ptr[3]; out.lo = (float*) ptr[4];
    }
    return bytes;
}
// The PLE (n-gram embedding) module's host surface (ple.cu:199-226): the mode setters and the
// scratch-size question, transcribed verbatim (geometry constants NG_* are header-inline in
// ngram.hpp, shared with the host PleReader). ple_block/_projected/_history_advance and the
// postops launches are the kernels themselves — named traps, real port work.
namespace { bool ple_native_bf16_ = false, ple_native_postops_ = false; }
void ple_set_native_bf16(bool enabled) { ple_native_bf16_ = enabled; }
bool ple_native_bf16_enabled() { return ple_native_bf16_; }
void ple_set_native_postops(bool enabled) { ple_native_postops_ = enabled; }
bool ple_native_postops_enabled() { return ple_native_postops_; }
bool ple_block_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
uint64_t ple_block_scratch_bytes() {
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    const size_t e = (size_t) NG_N_EMBD * sizeof(uint16_t);
    return ((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + e + 256;
}

// ---- the engine's config-flag pairs (every one is a relaxed atomic bool or a plain assignment in
// the CUDA source; bodies transcribed one for one). These SELECT kernel paths (native_qsa, fused
// router, ...) - the selected kernels themselves all stay named traps, so the flags only decide
// WHICH trap a run reaches, exactly as on CUDA.
namespace {
std::atomic<bool> g_native_gdn{}, g_native_rope{}, g_native_router{}, g_native_qsa{},
                  g_native_qsa_indexer{}, g_native_qsa_score{}, g_native_moe_combine{};
bool g_se_native_bf16 = false, g_old_kernels = false, g_grouped_v1 = false, g_pa_imma_tristate = false;
int g_b6_rows = 0, g_multi_exact = 0;
}
void native_gdn_set_enabled(bool v) { g_native_gdn.store(v, std::memory_order_relaxed); }
bool native_gdn_enabled() { return g_native_gdn.load(std::memory_order_relaxed); }
void native_rope_set_enabled(bool v) { g_native_rope.store(v, std::memory_order_relaxed); }
bool native_rope_enabled() { return g_native_rope.load(std::memory_order_relaxed); }
void native_router_set_enabled(bool v) { g_native_router.store(v, std::memory_order_relaxed); }
bool native_router_enabled() { return g_native_router.load(std::memory_order_relaxed); }
bool native_router_topk_supported(int n_expert, int k) { return (n_expert == 512 && k == 10) || (n_expert == 256 && k == 8); }
void native_qsa_set_enabled(bool v) { g_native_qsa.store(v, std::memory_order_relaxed); }
bool native_qsa_enabled() { return g_native_qsa.load(std::memory_order_relaxed); }
void native_qsa_indexer_set_enabled(bool v) { g_native_qsa_indexer.store(v, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return g_native_qsa_indexer.load(std::memory_order_relaxed); }
void native_qsa_score_set_enabled(bool v) { g_native_qsa_score.store(v, std::memory_order_relaxed); }
bool native_qsa_score_enabled() { return g_native_qsa_score.load(std::memory_order_relaxed); }
void native_moe_combine_set_enabled(bool v) { g_native_moe_combine.store(v, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return g_native_moe_combine.load(std::memory_order_relaxed); }
bool native_moe_combine_hits_supported(const float* parts, const float* shared, const float* output,
                                       int64_t n_embd, int64_t k, int n_tok) {
    return k == 10 && shared != nullptr && n_tok >= 1 && n_tok <= 8 && n_embd > 0 && (n_embd & 3) == 0 &&
           (((uintptr_t) parts | (uintptr_t) shared | (uintptr_t) output) & 15u) == 0;
}
void shared_expert_set_native_bf16(bool enabled) { g_se_native_bf16 = enabled; }
bool shared_expert_native_bf16_enabled() { return g_se_native_bf16; }
void iq_set_old_kernels(bool old) { g_old_kernels = old; }
void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }
void native_mmvq_set_b6_rows(int mode) { g_b6_rows = mode; }
void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact ? 1 : 0; }
void qsa_prompt_attn_set_imma(int on) { g_pa_imma_tristate = on < 0 ? false : (on != 0); }   // tristate kept as intent
namespace { int g_expdec = 0; int g_exp_mode = -1; NativeExpertExpdecParams g_expdec_p{}; }
void native_expert_set_expdec(int on) { g_expdec = on ? 1 : 0; }
int native_expert_expdec() { return g_expdec; }
void native_expert_set_expdec_params(const NativeExpertExpdecParams& p) { g_expdec_p = p; }
NativeExpertExpdecParams native_expert_expdec_params() { return g_expdec_p; }
// takes: the expdec kernel variant's geometry predicate — false here (the variant's kernels are the traps;
// the plain grouped path is what a Vulkan run reaches): same honest shape as packed_eligible above.
bool native_expert_expdec_takes(const NativeExpertLayout&, bool& gu, bool& down) { gu = down = false; return false; }
void native_expert_set_mode(int m) { g_exp_mode = m; }   // bench knob; the launchers that read it are traps
bool native_q6_k_packed_enabled() {
    static const bool enabled = [] { const char* v = std::getenv("STRATA_Q6K_PACKED"); return v && v[0] == '1'; }();
    return enabled;
}
// native_q6_k_unpack (native_mmvq.cu:2560): the packed-plane registry's owner-side free — pure host
// map bookkeeping (no kernel). This backend never packs unless STRATA_Q6K_PACKED=1 routes the run to
// native_q6_k_pack, which stays a named trap (its plane build is host, but its selftest and the packed
// mmvq consumer are CUDA-kernel territory); the cleanup path in native_head.cpp/mtp.cpp calls unpack
// UNCONDITIONALLY, and on CUDA a never-packed weight simply misses the registry and returns. Same here.
namespace { std::unordered_map<const void*, int>& q6_registry() { static std::unordered_map<const void*, int> r; return r; } }
void native_q6_k_unpack(const void* weights) {
    auto found = q6_registry().find(weights);
    if (found == q6_registry().end()) return;
    q6_registry().erase(found);   // nothing was ever allocated here: the pack trap would have stopped the run first
}

// ---- the load-path byte-size questions (a run must ANSWER these to size its scratch before any kernel
// is demanded — trapping one aborts the load, so these are transcribed verbatim from the CUDA sources):
// scratch arithmetic (moe_hit/shared_expert/argmax/row_top/coupled_draft), the KV block layout, and the
// dequant type table. The launches that USE these answers stay named traps.
uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) + ((xh + 15) & ~15ull);
}
uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}
uint64_t argmax_rows_scratch_bytes(int n_rows) { return (uint64_t) n_rows * 1040ull; }   // ArgRow (verify_kernels.cu:810): 16B + 128f + 128i
uint64_t row_top_prob_scratch_bytes(int n_rows) { return (uint64_t) n_rows * 144ull; }    // TopRow (:857): 16B + 32f
size_t coupled_draft_scratch_bytes(int nv) {                                              // sampler.cu:1301-1315
    const int blocks = (nv + 4095) / 4096;   // kSplitBlockSpan = 4 warps x 1024
    if (nv <= 0 || blocks > 64) return 0;    // kSplitMaxBlocks
    return (size_t) blocks * 64ull * 8ull;   // kSelMax * sizeof(int2)
}
uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {   // kv_stream.cu:199 verbatim
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvHybrid)
        return rows * (uint64_t) s.head_dim + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 +
               rows * kv_q4_bytes_per_head((int) s.head_dim);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                          : rows * (uint64_t) s.head_dim * 2 * 2;
}
bool dequant_bf16_supported(int ggml_type) noexcept {   // the geometry() table, dequant_bf16.cu:324
    switch (ggml_type) {
    case 2: case 6: case 7: case 8: case 20: case 42: return true;
    case 11: case 12: case 13: case 14: case 23: return true;
    default: return false;
    }
}
void dequant_set_wide(int mode) { static int g = 0; g = mode; }   // the wide-path bench knob; launchers trap
// cvec: the contract's cvec() is the empty unloaded vector on this backend (above); loaded() false makes
// CUDA's own bodies take their early-return: enabled false, set_enabled a no-op.
// the rope/mrope table registries (native_rope.cu:130-171): per-device pointer+scaling stores, pure
// host state; only the native_rope launches read them and those stay traps (rope_table_for answers
// {} unless STRATA_ROPE_TABLE=1, exactly the CUDA build's opt-in rule).
constexpr int kRegDevices = 64;
namespace {
struct RopeReg { RopeTab tab; RopeScaling scaling; };
RopeReg g_rope_tab[kRegDevices] = {};
std::atomic<const int32_t*> g_mrope_tab[kRegDevices] = {};
int reg_dev() { int d = 0; if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kRegDevices) d = 0; return d; }
bool same_scaling(const RopeScaling& a, const RopeScaling& b) {
    return a.type == b.type && a.freq_base == b.freq_base && a.factor == b.factor && a.freq_scale_in == b.freq_scale_in &&
           a.orig_ctx == b.orig_ctx && a.ext_factor == b.ext_factor && a.attn_factor == b.attn_factor &&
           a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow;
}
bool rope_table_on() {
    static const bool on = [] { const char* e = std::getenv("STRATA_ROPE_TABLE"); return e != nullptr && e[0] == '1'; }();
    return on;
}
}
void mrope_table_set(const int32_t* device_table) { g_mrope_tab[reg_dev()].store(device_table, std::memory_order_relaxed); }
const int32_t* mrope_table() { return g_mrope_tab[reg_dev()].load(std::memory_order_relaxed); }
void rope_table_set(const float* cos_tab, const float* sin_tab, int max_pos, const RopeScaling& scaling) {
    g_rope_tab[reg_dev()] = RopeReg{RopeTab{cos_tab, sin_tab, max_pos}, scaling};
}
void rope_table_release(const float* cos_tab) {
    for (RopeReg& r : g_rope_tab) if (cos_tab != nullptr && r.tab.cos == cos_tab) r = RopeReg{};
}
RopeTab rope_table_for(const RopeScaling& scaling) {
    if (!rope_table_on()) return {};
    const RopeReg& r = g_rope_tab[reg_dev()];
    return r.tab.cos != nullptr && same_scaling(r.scaling, scaling) ? r.tab : RopeTab{};
}

bool cvec_enabled() { return false; }
void cvec_set_enabled(bool) {}
// the QSA decode-attention scratch question (qsa_decode_attn.cu:935, verbatim; HD=256 head_dim,
// CHUNK=64 cells per block — the file's geometry constants): sizing only; the step/merge launches
// that consume the scratch stay named traps (the port's Phase-3 Task 3.2 ordered-FP32 work).
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + 64 - 1) / 64;
    return (uint64_t) chunks * (uint64_t) s.n_head * (256 + 2) + 64;
}

namespace { constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3; }
void fused_gr_set_fast(int) {}   // a CUDA launch-mode knob (STRATA_GR_FAST build gate); nothing to hold
bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == 2560 && hc == 4 && hc_lr == 320;   // the compiled geometry, verbatim (N/HC/LR)
}
int fused_gr_variant() { return kHcPlain; }   // no bit-for-bit selftest ran here: the plain read
void fused_gr_check() {
    std::fprintf(stderr, "strata hc: Vulkan: the hyper-connection read runs as the plain one "
                         "(split/staged need their kernels; PORT-STATUS gr row todo)\n");
}
void fused_gr_persist_init() {}               // env-gated (STRATA_HC_PERSIST); the path traps by name if used

bool native_q8_0_packed_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q8_PACKED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}
bool native_q8_0_packed_eligible(int, int) { return false; }

bool iq_supported(int t) noexcept { return is_iq(t); }
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 30; }   // verbatim from iq_kernels.cu

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 10 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29 || ggml_type == 39;
}
bool native_mmvq_ids_supported(int ggml_type) {
    switch (ggml_type) {
    case 2: case 6: case 8: case 20: case 39: case 10: case 11: case 12: case 13: case 14: case 23: case 42:
    case 7: case 16: case 17: case 18: case 21: case 22: case 29: return true;   // slots_kind's two tables
    default: return false;
    }
}
bool native_mmvq_il_supported(int, int, int) { return false; }
std::size_t native_q8_1_bytes(int n_in, int ncols) {
    if (n_in <= 0 || n_in % 32 != 0)
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    if (ncols < 1 || ncols > 8) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
    return std::size_t(ncols) * std::size_t(n_in / 32) * 36;
}
std::size_t native_q8_1_il_bytes(int n_in, int ncols) { return native_q8_1_bytes(n_in, ncols) * 2; }

// PDL (programmatic dependent launch) is an NVIDIA launch-graph feature - the ledger's pdl row is n/a
// and this backend runs --no-graphs. verify.cpp evaluates pdl_supported() on EVERY run, so answering
// "not supported" (the CUDA build's own fallback on sm_<90) is what the contract needs; a trap here
// would abort before any kernel is demanded. The scope flag is a plain static.
bool& pdl_scope() { static bool s = false; return s; }
bool pdl_supported() { return false; }

}  // namespace strata::kernels
