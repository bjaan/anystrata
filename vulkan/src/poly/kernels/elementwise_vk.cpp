// vulkan/src/poly/kernels/elementwise_vk.cpp - the Vulkan launchers for the Task-2.7 poly
// elementwise batch: strata::poly::axpy, swiglu_clamp, add_scaled, mul_sigmoid.
// CUDA bodies/wrappers (src/poly/kernels.cu): axpy :853/:2391 (grid(n,256)x256), swiglu_clamp
// :844/:2387 (grid(n,256)x256), add_scaled :867/:2402 (the 1..32 pass runs each vector through the
// axpy kernel exactly as the CUDA comment says -- "a thread adds its value of each vector in turn"
// == count sequential launches, one element per thread, no ordering interaction between passes),
// mul_sigmoid :1849/:2667 (dim3(grid(n,256), min(65535, rows-r)) replayed as one flattened launch
// per 65535-row chunk, buffers offset like the wrapper).
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "elementwise_axpy.spv.h"
#include "elementwise_swiglu.spv.h"
#include "elementwise_swiglu_rows.spv.h"
#include "elementwise_mul_sigmoid.spv.h"

const vk::Kernel& pl_axpy() {
    static const vk::Kernel k = vk::kernel_load("elementwise_axpy", elementwise_axpy_spv,
                                                elementwise_axpy_spv_len, 2, 16, 256, false);
    return k;
}
const vk::Kernel& pl_swiglu() {
    static const vk::Kernel k = vk::kernel_load("elementwise_swiglu", elementwise_swiglu_spv,
                                                elementwise_swiglu_spv_len, 3, 16, 256, false);
    return k;
}
const vk::Kernel& pl_swiglu_rows() {
    static const vk::Kernel k = vk::kernel_load("elementwise_swiglu_rows", elementwise_swiglu_rows_spv,
                                                elementwise_swiglu_rows_spv_len, 2, 16, 256, false);
    return k;
}
const vk::Kernel& pl_mul_sigmoid() {
    static const vk::Kernel k = vk::kernel_load("elementwise_mul_sigmoid", elementwise_mul_sigmoid_spv,
                                                elementwise_mul_sigmoid_spv_len, 2, 16, 256, false);
    return k;
}
inline uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
}  // namespace

// The engine's moe_add passes h_cpu_ — HOST pinned memory (engine.cpp:1865) — as x: CUDA reads
// it through unified addressing at LAUNCH time, so every replay reads the live host bytes. A
// Vulkan kernel binds arena buffers only, and a captured upload would FREEZE those bytes into
// the graph (found the hard way: graphs + the expert-swap regime -> every replay after the
// first added the capture-time zeros -> generated ids collapse to 0s). The faithful shape is
// EAGER: a host x forces this launch off the capture path entirely - drain pending work,
// memcpy the host bytes into launcher-owned arena scratch (live every replay: moe_add is
// called OUTSIDE the layer graphs' replayed region... engine.cpp:2754/2846/2959 call moe_add
// between replays, so eager here is exactly CUDA's ordering), then run the kernel. Bitwise
// identical math (same f32 loads, same add/fma arm). Scratch grows by whole reallocations.
void axpy(const float* x, float a, float* acc, int n, bool add, void* stream) {
    if (n <= 0) return;
    static uint64_t scr_off = 0; static int scr_n = 0;
    // arena offset below used() = a device pointer; anything else (a host malloc at ~140 TB)
    // is not: the copy2_vk.cpp classification, same numbers, same reason.
    const bool dev_x = strata_vk_compat::off(x) < vk::Arena::instance().used();
    if (!dev_x) {
        // INSIDE a capture this upload records as a memcpy NODE whose source is the LIVE host
        // pages (import_host_buf: CUDA memcpy-node semantics, pointers not values): every replay
        // re-reads h_cpu_ as the swap thread wrote it. Eagerly: the plain arena copy lane. Same
        // scratch growth either way.
        if (n > scr_n) {   // grow by whole reallocation; an OLD captured graph keeps the OLD
                           // scratch binding (its copy node writes it live), so old replays stay
                           // correct while new calls use the bigger region: bump arena, never free.
            scr_off = vk::Arena::instance().alloc((uint64_t)n * 4);
            scr_n = n;
        }
        if (getenv("STRATA_VK_CAPTURE_TRACE") && vk::capturing())
            std::fprintf(stderr, "cap-axpy: host-x upload recorded INSIDE capture n=%d scr=%llu\n",
                         n, (unsigned long long)scr_off);
        vk::copy_h2d((void*)(uintptr_t)scr_off, x, (uint64_t)n * 4, stream);
        x = (const float*)strata_vk_compat::wrap(scr_off);
    }
    const uint64_t bufs[2] = {strata_vk_compat::off(x), strata_vk_compat::off(acc)};
    const uint64_t sizes[2] = {(uint64_t)n * 4, (uint64_t)n * 4};
    const uint32_t pc[4] = {(uint32_t)n, add ? 1u : 0u, bits(a), 0u};
    vk::kernel_launch_sized(pl_axpy(), (n + 255) / 256, bufs, sizes, 2, pc);
}

void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream) {
    if (n <= 0) return;
    const uint64_t bufs[3] = {strata_vk_compat::off(gate), strata_vk_compat::off(up),
                              strata_vk_compat::off(out)};
    const uint64_t sizes[3] = {(uint64_t)n * 4, (uint64_t)n * 4, (uint64_t)n * 4};
    const uint32_t pc[4] = {(uint32_t)n, bits(limit), 0u, 0u};
    vk::kernel_launch_sized(pl_swiglu(), (n + 255) / 256, bufs, sizes, 3, pc);
}

void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream) {
    // CUDA :2407: grid(ceil(n/256), rows) x 256 — row-major gate|up pairs of n floats each. No
    // wrapper-side validation there; guard the degenerate shapes like the sibling launchers and
    // replay the flattened launch per 65535-row chunk (grid.y's ceiling; mul_sigmoid's pattern).
    if (n <= 0 || rows <= 0) return;
    const int blocks = (n + 255) / 256;
    for (int r = 0; r < rows; r += 65535) {
        const int chunk = rows - r < 65535 ? rows - r : 65535;
        const uint64_t bufs[2] = {strata_vk_compat::off(gate_up) + (uint64_t)r * 2 * n * 4,
                                  strata_vk_compat::off(out) + (uint64_t)r * n * 4};
        const uint64_t sizes[2] = {(uint64_t)chunk * 2 * n * 4, (uint64_t)chunk * n * 4};
        const uint32_t pc[4] = {(uint32_t)n, bits(limit), (uint32_t)blocks, 0u};
        vk::kernel_launch_sized(pl_swiglu_rows(), blocks * chunk, bufs, sizes, 2, pc);
    }
}

void add_scaled(const ScaledRows& rows, int count, int n, void* stream) {
    if (count < 1 || count > 32) throw std::invalid_argument("add_scaled: 1 to 32 vectors");
    if (n <= 0) return;
    // One axpy pass per vector, same order as the CUDA kernel's k-loop (per element, in-place acc).
    for (int k = 0; k < count; ++k) {
        const uint64_t bufs[2] = {strata_vk_compat::off(rows.x[k]), strata_vk_compat::off(rows.acc[k])};
        const uint64_t sizes[2] = {(uint64_t)n * 4, (uint64_t)n * 4};
        const uint32_t pc[4] = {(uint32_t)n, 1u, bits(rows.w[k]), 0u};
        vk::kernel_launch_sized(pl_axpy(), (n + 255) / 256, bufs, sizes, 2, pc);
    }
}

void mul_sigmoid(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream) {
    if (n <= 0 || rows <= 0) return;
    const int blocks = (n + 255) / 256;
    for (int r = 0; r < rows; r += 65535) {   // the wrapper's chunking, transcribed
        const int chunk = rows - r < 65535 ? rows - r : 65535;
        const uint64_t bufs[2] = {strata_vk_compat::off(acc) + (uint64_t)r * n * 4,
                                  strata_vk_compat::off(g) + (uint64_t)r * g_stride * 4};
        // spans from the chunk start; the tail rows of each buffer are what this launch touches.
        const uint64_t sizes[2] = {(uint64_t)chunk * n * 4, (uint64_t)chunk * g_stride * 4};
        const uint32_t pc[4] = {(uint32_t)n, (uint32_t)g_stride, one_gate ? 1u : 0u, (uint32_t)blocks};
        vk::kernel_launch_sized(pl_mul_sigmoid(), blocks * chunk, bufs, sizes, 2, pc);
    }
}

}  // namespace strata::poly
