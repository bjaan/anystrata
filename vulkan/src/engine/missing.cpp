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
namespace strata::kernels {

// no control vector on Vulkan (--control-vector-scaled is not ported): the empty one, `loaded()` false
const Cvec& cvec() {
    static const Cvec none{};
    return none;
}

// PDL (programmatic dependent launch) is an NVIDIA launch-graph feature - the ledger's pdl row is n/a
// and this backend runs --no-graphs. verify.cpp evaluates pdl_supported() on EVERY run, so answering
// "not supported" (the CUDA build's own fallback on sm_<90) is what the contract needs; a trap here
// would abort before any kernel is demanded. The scope flag is a plain static.
bool& pdl_scope() { static bool s = false; return s; }
bool pdl_supported() { return false; }

}  // namespace strata::kernels
