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
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/native_mmvq.hpp"
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
