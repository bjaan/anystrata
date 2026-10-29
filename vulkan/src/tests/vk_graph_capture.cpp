// vulkan/src/tests/vk_graph_capture.cpp - the Phase-4.2 capture layer, pinned:
//  * a capture records real launches (store_f16 twice with different windows) into ONE CB;
//  * replaying it produces bits identical to the eager sequence (the capture uses the eager
//    per-dispatch barrier law, so ordering — and therefore results — cannot change);
//  * a graph replays many times; destroy frees it and a later launch throws;
//  * capturing() answers correctly around begin/end.
// The end-to-end proof stays with the model gate (the engine replays ~1,600 layer graphs per
// decoded token batch); this pins the plumbing so a regression here is not blamed on the engine.
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_compat/strata_vk.h"
#include "strata/vulkan_runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace vk = strata::vk;

int main() {
    try { vk::device_info(); }
    catch (const std::exception& e) { std::fprintf(stderr, "vk_graph_capture: %s\nSKIP\n", e.what()); return 77; }
    vk::Arena& a = vk::Arena::instance();
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) fails++; };

    const int n = 8, rows = 8;                       // 8 rows of 8 f32 -> 64 f16
    std::vector<float> x((size_t)n * rows), y((size_t)n * rows);
    for (size_t i = 0; i < x.size(); ++i) { x[i] = float(int(i) % 17) * 0.375f - 2.f; y[i] = float(int(i % 5)) * 1.5f; }
    const uint64_t ax = a.alloc(x.size() * 4), ay = a.alloc(y.size() * 4),
                   ac = a.alloc((size_t)n * rows * 2);
    vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);
    vk::copy_h2d((void*)ay, y.data(), y.size() * 4, nullptr);

    auto eager_pair = [&]() {   // two chained stores into the cache, then read it back
        vk::fill_d((void*)ac, 0, (size_t)n * rows * 2, nullptr);
        strata::poly::store_f16((const float*)strata_vk_compat::wrap(ax), (uint16_t*)strata_vk_compat::wrap(ac),
                                n, 0, 4, nullptr);                    // first half
        strata::poly::store_f16((const float*)strata_vk_compat::wrap(ay), (uint16_t*)strata_vk_compat::wrap(ac),
                                n, 4, 4, nullptr);                    // second half: SAME cache, RAW dep
        std::vector<uint16_t> out((size_t)n * rows, 0);
        vk::copy_d2h(out.data(), (void*)ac, out.size() * 2, nullptr);
        return out;
    };
    const std::vector<uint16_t> want = eager_pair();

    // ---- capture the fill + the pair: fills are recordable (vkCmdFillBuffer, the CUDA memset
    // analogue); kernels record with the eager per-dispatch barrier law.
    vk::graph_begin();
    check(vk::capturing(), "capturing() during an open capture");
    vk::fill_d((void*)ac, 0, (size_t)n * rows * 2, nullptr);          // recorded, not eager
    strata::poly::store_f16((const float*)strata_vk_compat::wrap(ax), (uint16_t*)strata_vk_compat::wrap(ac),
                            n, 0, 4, nullptr);
    strata::poly::store_f16((const float*)strata_vk_compat::wrap(ay), (uint16_t*)strata_vk_compat::wrap(ac),
                            n, 4, 4, nullptr);
    const uint32_t d = vk::graph_end();
    check(!vk::capturing(), "capturing() false after end");
    check(d == 2, "capture recorded both dispatches (the fill is not a dispatch)");
    const uint32_t gi = vk::capture_graph_count() - 1;
    check(vk::graph_dispatches(gi) == 2, "graph_dispatches reports the recorded count");

    // replay over a cleared cache: the graph does NOT include the clear (fills can't be captured);
    // clear first (eager), then launch, then compare.
    for (int round = 0; round < 3; ++round) {
        vk::fill_d((void*)ac, 0xAA, (size_t)n * rows * 2, nullptr);   // eager pre-clear: NOT in the graph
        vk::graph_launch(gi);                                          // graph = its own fill + the pair
        vk::device_sync();
        std::vector<uint16_t> got((size_t)n * rows, 0);
        vk::copy_d2h(got.data(), (void*)ac, got.size() * 2, nullptr);
        bool same = got == want;
        check(same, (std::string("replay ") + std::to_string(round) + " bit-identical to eager").c_str());
    }
    // destroy: a later launch throws rather than replaying freed memory
    vk::graph_destroy(gi);
    try { vk::graph_launch(gi); check(false, "launch after destroy throws"); }
    catch (const std::exception&) { check(true, "launch after destroy throws"); }

    // ---- recorded copies: a graph that carries an H2D (frozen bytes), a kernel that reads it,
    // and a D2H whose staged result lands on the host at the next drain — the router's exact
    // shape (copy2 device->pinned inside a capture body).
    {
        const uint64_t asrc = a.alloc(64 * 4);
        std::vector<float> src(64);
        for (size_t i = 0; i < 64; ++i) src[i] = float(i) * 0.5f;
        const uint64_t adst = a.alloc(64 * 4);
        vk::fill_d((void*)adst, 0, 64 * 4, nullptr);
        // capture: h2d(src -> adst), then store_f16 reads adst into the cache, then d2h(adst -> host)
        std::vector<float> host_back(64, -1.f);
        vk::graph_begin();
        vk::copy_h2d((void*)(uintptr_t)asrc, src.data(), 64 * 4, nullptr);   // recorded now (bytes frozen)
        vk::copy_d2h(host_back.data(), (void*)(uintptr_t)asrc, 64 * 4, nullptr);  // recorded, delivered at drain
        vk::graph_end();
        const uint32_t g2 = vk::capture_graph_count() - 1;
        // mutate the host source AFTER capture: the replay must still carry the FROZEN bytes
        for (size_t i = 0; i < 64; ++i) src[i] = 999.f;
        for (size_t i = 0; i < 64; ++i) host_back[i] = -1.f;
        vk::graph_launch(g2);
        vk::device_sync();   // the drain delivers the D2H
        bool ok = true;
        for (size_t i = 0; i < 64; ++i) if (host_back[i] != float(i) * 0.5f) ok = false;
        check(ok, "recorded h2d froze the capture-time bytes; recorded d2h delivered at drain");
        vk::graph_destroy(g2);
    }

    std::printf("vk_graph_capture %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
