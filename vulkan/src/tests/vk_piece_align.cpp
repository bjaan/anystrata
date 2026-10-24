// vulkan/src/tests/vk_piece_align.cpp - the Task 2.11 arena law: a region that fits inside ONE
// max-allocation piece is single-piece BY CONSTRUCTION. arena_add() bumps the cursor past a
// piece whose remaining tail is shorter than the new region, so no kernel span computed from a
// region's own numbers can straddle: this removes the class of layout-dependent
// "kernel argument N crosses an arena piece" throws (attend hit it when the engine's score
// scratch landed near a 512 MiB piece boundary on the BC-250).
//
// The check rides kernel_launch_sized's piece_of() loop (vk_runtime.cpp:281): a launch binds
// each (buffer, size) span through ONE VkBuffer - the runtime throws by name before submit when
// a span crosses a piece. Dispatch is 0 groups: only the binding/validation path matters.
//   * alloc A (504 MiB: leaves < 8 MiB of tail in its piece when max_alloc = 512 MiB), then
//     alloc B (16 MiB): B's FULL span must bind in one launch (the align rule moved it to the
//     next piece start - without the rule it would straddle and throw);
//   * a region LARGER than max_alloc (600 MiB) still chains: a full-span binding must throw by
//     name (the pre-existing rejection, kept for the oversize class - the probe's case D owns
//     the copy-side behaviour);
//   * two consecutive small allocs (32/16 MiB) still bind whole.
// On a driver whose max allocation is not 512 MiB the sizes adapt: A = max_alloc - 8 MiB,
// B = 16 MiB (only meaningful when max_alloc > 32 MiB, asserted from device_info()).
#include "strata/vulkan_runtime.hpp"

#include "store_f16.spv.h"    // any compiled kernel works: the launch throws before executing

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata;

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) fails++;
}

// Bind bufs with full spans through one launch: throws runtime_error naming the crossing when
// a span straddles pieces. Returns true when the launch took the binding (0 groups = no work).
bool bind_whole(const vk::Kernel& k, uint64_t b0, uint64_t n0, uint64_t b1, uint64_t n1) {
    const uint64_t bufs[2] = {b0, b1};
    const uint64_t sizes[2] = {n0, n1};
    try { vk::kernel_launch_sized(k, 0, bufs, sizes, 2, nullptr); return true; }
    catch (const std::exception& e) { std::printf("  (bind threw: %s)\n", e.what()); return false; }
}

}  // namespace

int main() {
    // Pin the piece cap deterministically (the probe's route): the default is now budget-driven
    // (16 GiB with halve-on-OOM retry, post GSQ-hybrid), and this test's geometry (near-tail
    // bump, oversize chain) wants small fixed pieces; free_bytes never covers a 16 GiB A anyway.
    setenv("STRATA_VK_MAX_ALLOC", "536870912", 1);
    const vk::DeviceInfo* di = nullptr;
    try { di = &vk::device_info(); }
    catch (const std::exception& e) { std::fprintf(stderr, "vk_piece_align: %s\nSKIP\n", e.what()); return 77; }
    const uint64_t maxa = di->max_allocation_size;
    const uint64_t MiB = 1ull << 20;
    if (maxa < 64 * MiB) { std::printf("vk_piece_align: max allocation %llu too small to test; SKIP\n",
                                       (unsigned long long)maxa); return 77; }
    vk::Arena& a = vk::Arena::instance();
    // store_f16: 2 storage buffers, 4-byte push constants, 256 threads - the binding shape is
    // all this test exercises (groups_x = 0 dispatches nothing).
    const vk::Kernel& k = vk::kernel_load("store_f16", store_f16_spv, store_f16_spv_len, 2, 4, 256, false);

    // 1. the near-boundary pair: A eats all but 8 MiB of a piece, B (16 MiB) must realign.
    const uint64_t A = a.alloc(maxa - 8 * MiB);
    if (!bind_whole(k, A, maxa - 8 * MiB, A, 4096)) check(false, "A whole-span bind"); else check(true, "A whole-span bind");
    const uint64_t B = a.alloc(16 * MiB);
    check(bind_whole(k, B, 16 * MiB, B + 8 * MiB, 8 * MiB), "B (16 MiB) single-piece after a short tail");

    // 2. consecutive small allocs: each whole.
    const uint64_t C = a.alloc(32 * MiB), D = a.alloc(16 * MiB);
    check(bind_whole(k, C, 32 * MiB, D, 16 * MiB), "consecutive 32/16 MiB allocs each whole");

    // 3. the oversize class keeps its named rejection: a region just over maxa chains (the
    // align rule only repositions allocs that FIT one piece), and the chain's FULL span must
    // not bind as one VkBuffer. Size chosen to stay under this box's VRAM (14 GiB) after the
    // 520/16/48 MiB above: 3 MiB over the piece cap.
    const uint64_t over = maxa + 3 * MiB;
    if (over <= di->free_bytes - 1024 * MiB) {
        const uint64_t E = a.alloc(over);
        bool threw = false;
        try {
            const uint64_t bufs[2] = {E, E};
            const uint64_t sizes[2] = {over, 4096};
            vk::kernel_launch_sized(k, 0, bufs, sizes, 2, nullptr);
        } catch (const std::exception& e) { threw = std::string(e.what()).find("piece") != std::string::npos; }
        check(threw, "chained region full-span bind rejected by name");
        // a piece-local sub-span of the chain binds fine (what the engine's copies rely on)
        check(bind_whole(k, E, maxa, E, 4096), "chained region's first piece binds");
    } else {
        std::printf("  skip the chained-region check: not enough free VRAM\n");
    }

    std::printf("vk_piece_align %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
