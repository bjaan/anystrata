// vulkan/src/tests/vk_stubs_trap.cpp - the Task 1.2 proof: an unported kernel entry point names
// itself and stops the run. Two arms, both driven by the ctest case:
//   (default)   calling strata::kernels::rope_neox_apply aborts the process with exit code 86 and
//               prints the entry point's name on stderr;
//   (argv[1]="log", STRATA_VK_STUBS=log) the same call prints one STUB line and RETURNS (outputs
//               untouched) — the list-a-path mode, never for answers.
// The link itself is part of the proof: this TU references nothing but the stub library, so if the
// trap roster did not cover rope_neox_apply the link would fail with an undefined symbol.
#include "strata/kernels/rope.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#ifdef _WIN32
static int setenv(const char* n, const char* v, int overwrite) {
    if (!overwrite && std::getenv(n)) return 0;
    return _putenv_s(n, v);
}
#endif

int main(int argc, char** argv) {
    const bool log_arm = argc > 1 && std::string(argv[1]) == "log";
    if (log_arm) setenv("STRATA_VK_STUBS", "log", 1);
    // every argument is a host pointer the trap never dereferences — the contract types, not the data
    strata::kernels::rope_neox_apply(nullptr, nullptr, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
    if (!log_arm) {
        std::fprintf(stderr, "vk_stubs_trap: rope_neox_apply RETURNED — the trap did not stop the run\n");
        return 1;
    }
    // log mode: calling twice prints the STUB line once (per-entry dedup) and returns again
    strata::kernels::rope_neox_apply(nullptr, nullptr, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
    std::printf("vk_stubs_trap: the stub returned in log mode as contracted\n");
    return 0;
}
