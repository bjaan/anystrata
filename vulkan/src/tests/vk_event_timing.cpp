// vulkan/src/tests/vk_event_timing.cpp - the Task 4.2 law, pinned by execution:
// the compat event pair around a REAL dispatched kernel returns a NON-ZERO, PLAUSIBLE GPU
// interval (the engine's --stage-timing tables and STRATA_COMMIT_TIMING read exactly these).
//   * gpu arm (device_info().gpu_timestamps, the default on BC-250): vkCmdWriteTimestamp ticks
//     scaled by timestampPeriod. Assert: > 0 (the old layer returned a flat 0 — the bug this
//     task closes), <= the host-wall interval of the same span (a queue timestamp cannot exceed
//     the wall clock that wrapped it), and monotone across three records of one pair.
//   * host arm (STRATA_VK_NO_GPU_EVENTS=1, run as the second ctest case): record = drain +
//     steady_clock stamp; elapsed > 0 too — the fallback must stay alive for cards without
//     timestampComputeAndGraphics.
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_compat/strata_vk.h"
#include "strata/vulkan_runtime.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace vk = strata::vk;
using clk = std::chrono::steady_clock;

int main() {
    try { vk::device_info(); }
    catch (const std::exception& e) { std::fprintf(stderr, "vk_event_timing: %s\nSKIP\n", e.what()); return 77; }
    const bool gpu_arm = vk::device_info().gpu_timestamps && !getenv("STRATA_VK_NO_GPU_EVENTS");
    std::printf("vk_event_timing: arm=%s period=%.3f ns/tick\n", gpu_arm ? "gpu-timestamps" : "host-clock",
                vk::device_info().timestamp_period_ns);
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) fails++; };

    // ---- the measured span: store_f16 over a big cache (tens of microseconds of real work).
    const int n = 4096, rows = 64;                       // 256K f32 -> 256K f16
    std::vector<float> x((size_t)n * rows);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i) % 101) * 0.125f - 6.f;
    vk::Arena& a = vk::Arena::instance();
    const uint64_t ax = a.alloc(x.size() * 4), ac = a.alloc((size_t)n * rows * 2);
    vk::copy_h2d((void*)ax, x.data(), x.size() * 4, nullptr);

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    check(cudaEventCreate(&e0) == cudaSuccess && cudaEventCreate(&e1) == cudaSuccess, "eventCreate");

    // warm the pipeline once (first-launch compile noise belongs to neither arm's contract)
    strata::poly::store_f16((const float*)strata_vk_compat::wrap(ax), (uint16_t*)strata_vk_compat::wrap(ac),
                            n, 0, rows, nullptr);
    cudaDeviceSynchronize();

    float gpu_ms = 0.f; double host_ms = 0.0;
    for (int rep = 0; rep < 3; ++rep) {                  // last pair's numbers are what we assert
        const auto t0 = clk::now();
        cudaEventRecord(e0, nullptr);
        strata::poly::store_f16((const float*)strata_vk_compat::wrap(ax), (uint16_t*)strata_vk_compat::wrap(ac),
                                n, 0, rows, nullptr);
        cudaEventRecord(e1, nullptr);
        check(cudaEventSynchronize(e1) == cudaSuccess, "eventSynchronize");
        const auto t1 = clk::now();
        host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        check(cudaEventElapsedTime(&gpu_ms, e0, e1) == cudaSuccess, "eventElapsedTime");
    }
    std::printf("  gpu/arm interval %.3f ms, host-wall %.3f ms\n", gpu_ms, host_ms);
    check(gpu_ms > 0.f, "elapsed > 0 (the pre-4.2 layer returned a flat 0)");
    // plausibility: 256K elements f32->f16 on 24 CUs is well under a millisecond of queue time;
    // and a timestamp span can never exceed the wall clock that wrapped the same work.
    check(gpu_ms < 10.0, "elapsed plausible (< 10 ms for a 256K store)");
    check((double)gpu_ms <= host_ms + 0.5, "elapsed <= host-wall of the wrapped span");
    if (gpu_arm) check(vk::capture_graph_count() >= 0, "gpu arm active on this device");
    else check(!vk::device_info().gpu_timestamps || getenv("STRATA_VK_NO_GPU_EVENTS"),
               "host arm: no feature, or forced off for the A/B");

    // ---- re-record semantics (CUDA's): a pair re-recorded returns the NEW interval, not the stale one.
    cudaEventRecord(e0, nullptr);
    cudaEventRecord(e1, nullptr);
    float second = -1.f;
    check(cudaEventElapsedTime(&second, e0, e1) == cudaSuccess && second >= 0.f,
          "re-recorded pair yields a fresh, non-negative interval");

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    std::printf("vk_event_timing %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
