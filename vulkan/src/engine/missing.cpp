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
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/iq_kernels.hpp"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include <cmath>
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
