// vulkan/src/kernels/native_mmvq_vk.cpp - the Vulkan launcher for
// include/strata/kernels/native_mmvq.hpp's native_mmvq, ported for ggml_type 16 (IQ2_XXS, the
// Qwen3.6-35B-A3B UD-IQ2_XXS experts) and ggml_type 13 (Q5_K, the dense attention projections).
// Every other type calls missing() with a name carrying its type id, so a run never computes with
// a wrong decoder (the CUDA switch, src/kernels/cuda/native_mmvq.cu:3110ff case 16 -> iq_mmvq,
// case 13 -> native_q5_k_mmvq).
//
// The CUDA wrapper's validation is kept whole (native_mmvq.cu:2026-2041 and the q5_k wrapper
// 2644-2651): n_in % block_elems (256 for iq/q5_k types), 1 <= ncols <= 8, non-null 4-aligned
// arena pointers, non-null stream, n_out > 0.
//
// Type 13 geometry (native_q5_k_mmvq, :2644-2669): the ncols>1 launch_multi<Q5KTraits> is pinned
// bitwise-equal per column to the ncols=1 kernel (:975-980), and the iq2_xxs precedent runs the
// per-column kernel for every ncols. The ncols=1 kernel picks SmallK by
// n_in/QK < VDR*WARPS*WARP/QI (=8): false (ROWS=1, grid n_out) for the model's n_in=2048,
// true (ROWS=4, grid ceil(n_out/4)) for small n_in. mmvq_q5_k.comp transcribes BOTH, selected by
// the 4th push constant (rows). STRATA_WAVE_MMVQ / wave routing is compiled out here as in Task
// 2.2 (the wave path swaps the partition; the pinned per-column equality above is what we port).
//
// iq2xxs_grid: the 256 uint64 table from ggml-common.h:560. The C/C++ GGML_TABLE arm makes it
// static-per-TU, so no symbol links here; the values are extracted verbatim to
// iq2xxs_grid_vk.h (tools note: regenerated from the same pinned llama.cpp checkout the engine
// links). One arena upload at first use, bound as the kernel's 4th storage buffer.
// Type 14 geometry (native_q6_k_mmvq, :2955-2981; kernel :806-842): same per-column pin as q5_k,
// VDR=1 so the SmallK pick is n_in/QK < WARPS*WARP/32 (=4): false (ROWS=1, grid n_out) for the
// model's n_in=2048, true (ROWS=4) below. Q6KBlock is 210 B, not 4-aligned: mmvq_q6_k.comp reads
// it byte-wise. The q6_packed_mmvq head (:2957, STRATA_Q6_PACKED=1) is a CUDA-only opt-in route;
// the default generic kernel is what the parity contract pins, as in Tasks 2.2/2.3.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"
#include "iq2xxs_grid_vk.h"
#include "iq3xxs_grid_vk.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::vk_engine { void missing(const char* name); }

namespace strata::kernels {
namespace {
#include "native_mmvq.spv.h"   // mmvq_iq2_xxs.comp
#include "mmvq_q5_k.spv.h"     // mmvq_q5_k.comp
#include "mmvq_q6_k.spv.h"     // mmvq_q6_k.comp
#include "mmvq_slots_iq2_xxs.spv.h"  // mmvq_slots_iq2_xxs.comp
#include "mmvq_slots_q5_k.spv.h"     // mmvq_slots_q5_k.comp
#include "mmvq_slots_q6_k.spv.h"     // mmvq_slots_q6_k.comp
#include "mmvq_iq3_xxs.spv.h"       // mmvq_iq3_xxs.comp (Task 2.9, type 18 IQ3_XXS)
#include "mmvq_iq4_xs.spv.h"        // mmvq_iq4_xs.comp (Task 2.12, type 23 IQ4_XS)

constexpr int IQ_BLOCK_ELEMS = 256;
using vk::kIq2xxsGrid;   // the grid table header (namespace strata::vk)
using vk::kIq3xxsGrid;   // the iq3xxs grid table header
constexpr int IQ_BLOCK_BYTES = 66;   // block_iq2_xxs
constexpr int IQ3_BLOCK_BYTES = 98;  // block_iq3_xxs (half d + uint8 qs[96])
constexpr int IQ4XS_BLOCK_BYTES = 146;  // block_iq4_xs (half d + uint16 scales_h + uint8 scales_l[4] + uint8 qs[128])
constexpr int Q5K_BLOCK_BYTES = 176; // Q5KBlock (native_mmvq.cu:129)
constexpr int Q6K_BLOCK_BYTES = 210; // Q6KBlock (native_mmvq.cu:96: ql[128]+qh[64]+scales[16]+d)
constexpr int Q8_BLOCK_BYTES = 36;   // block_q8_1
constexpr int THREADS = 128;         // block (32,4)
constexpr int MAX_NCOLS = 8;

void validate_shape(int n_in, int ncols, int block_elems) {
    if (n_in <= 0 || n_in % block_elems != 0)
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0)
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null stream");
}

// the arena copy of iq2xxs_grid, uploaded once (raw arena offset, like kernel_launch_sized takes)
uint64_t grid_buffer() {
    static const uint64_t p = [] {
        const uint64_t g = vk::Arena::instance().alloc(sizeof(kIq2xxsGrid));
        vk::copy_h2d(reinterpret_cast<void*>(static_cast<std::uintptr_t>(g)), kIq2xxsGrid,
                     sizeof(kIq2xxsGrid), nullptr);
        vk::device_sync();
        return g;
    }();
    return p;
}
// the arena copy of iq3xxs_grid (256 x uint32 = 1024 B), uploaded once - same route
uint64_t grid32_buffer() {
    static const uint64_t p = [] {
        const uint64_t g = vk::Arena::instance().alloc(sizeof(kIq3xxsGrid));
        vk::copy_h2d(reinterpret_cast<void*>(static_cast<std::uintptr_t>(g)), kIq3xxsGrid,
                     sizeof(kIq3xxsGrid), nullptr);
        vk::device_sync();
        return g;
    }();
    return p;
}
}  // namespace

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    if (ggml_type != 16 && ggml_type != 18 && ggml_type != 23 && ggml_type != 13 && ggml_type != 14) {
        // the CUDA wrapper would throw/exit here too — name the type instead (never compute with a
        // wrong decoder); shape is not validated for a type this port does not run.
        const std::string name = "strata::kernels::native_mmvq type " + std::to_string(ggml_type);
        strata::vk_engine::missing(name.c_str());
        return;
    }
    if (std::getenv("STRATA_VK_TRACE_MMVQ16") && ggml_type == 16)
        std::fprintf(stderr, "mmvq16 n_in %d n_out %d ncols %d\n", n_in, n_out, ncols);
    if (std::getenv("STRATA_VK_TRACE_MMVQ13") && ggml_type == 13)
        std::fprintf(stderr, "mmvq13 n_in %d n_out %d ncols %d\n", n_in, n_out, ncols);
    if (std::getenv("STRATA_VK_TRACE_MMVQ14") && ggml_type == 14)
        std::fprintf(stderr, "mmvq14 n_in %d n_out %d ncols %d\n", n_in, n_out, ncols);
    if (std::getenv("STRATA_VK_TRACE_MMVQ18") && ggml_type == 18)
        std::fprintf(stderr, "mmvq18 n_in %d n_out %d ncols %d\n", n_in, n_out, ncols);
    if (std::getenv("STRATA_VK_TRACE_MMVQ23") && ggml_type == 23)
        std::fprintf(stderr, "mmvq23 n_in %d n_out %d ncols %d\n", n_in, n_out, ncols);
    validate_shape(n_in, ncols, IQ_BLOCK_ELEMS);   // 256 elems for both iq2_xxs and q5_k
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const uint64_t nb = uint64_t(n_in) / IQ_BLOCK_ELEMS;
    const uint64_t x_sz = uint64_t(ncols) * nb * 8 * Q8_BLOCK_BYTES;   // n_in/32 blocks a column
    const uint64_t y_sz = uint64_t(ncols) * uint64_t(n_out) * 4;
    if (ggml_type == 13) {
        static const vk::Kernel k = vk::kernel_load(
            "native_mmvq_q5_k", mmvq_q5_k_spv, mmvq_q5_k_spv_len,
            /*storage buffers*/ 3, /*push bytes*/ 16, /*workgroup*/ THREADS,
            /*force 32-wide subgroups*/ true);
        const uint64_t w_sz = uint64_t(n_out) * nb * Q5K_BLOCK_BYTES;
        const uint64_t bufs[3] = {strata_vk_compat::off(weights), strata_vk_compat::off(x_q8_1),
                                  strata_vk_compat::off(y)};
        const uint64_t sizes[3] = {w_sz, x_sz, y_sz};
        // SmallK: n_in/QK < BLOCKS_PER_ITER (8) -> ROWS=4; else ROWS=1 (native_mmvq.cu:2662-2666)
        const uint32_t rows = (nb < 8u) ? 4u : 1u;
        const uint32_t pc[4] = {(uint32_t)n_in, (uint32_t)n_out, (uint32_t)ncols, rows};
        vk::kernel_launch_sized(k, rows == 4u ? int((n_out + 3) / 4) : n_out, bufs, sizes, 3, pc);
    } else if (ggml_type == 14) {
        static const vk::Kernel k = vk::kernel_load(
            "native_mmvq_q6_k", mmvq_q6_k_spv, mmvq_q6_k_spv_len,
            /*storage buffers*/ 3, /*push bytes*/ 16, /*workgroup*/ THREADS,
            /*force 32-wide subgroups*/ true);
        const uint64_t w_sz = uint64_t(n_out) * nb * Q6K_BLOCK_BYTES;
        const uint64_t bufs[3] = {strata_vk_compat::off(weights), strata_vk_compat::off(x_q8_1),
                                  strata_vk_compat::off(y)};
        const uint64_t sizes[3] = {w_sz, x_sz, y_sz};
        // SmallK: n_in/QK < BLOCKS_PER_ITER (WARPS*WARP/32 = 4, VDR=1) -> ROWS=4; else ROWS=1
        // (native_mmvq.cu:2974-2978)
        const uint32_t rows = (nb < 4u) ? 4u : 1u;
        const uint32_t pc[4] = {(uint32_t)n_in, (uint32_t)n_out, (uint32_t)ncols, rows};
        vk::kernel_launch_sized(k, rows == 4u ? int((n_out + 3) / 4) : n_out, bufs, sizes, 3, pc);
    } else if (ggml_type == 16) {
        static const vk::Kernel k = vk::kernel_load(
            "native_mmvq_iq2_xxs", native_mmvq_spv, native_mmvq_spv_len,
            /*storage buffers*/ 4, /*push bytes*/ 12, /*workgroup*/ THREADS,
            /*force 32-wide subgroups*/ true);
        const uint64_t w_sz = uint64_t(n_out) * nb * IQ_BLOCK_BYTES;
        const uint64_t bufs[4]  = {strata_vk_compat::off(weights), strata_vk_compat::off(x_q8_1),
                                   strata_vk_compat::off(y), grid_buffer()};
        const uint64_t sizes[4] = {w_sz, x_sz, y_sz, sizeof(kIq2xxsGrid)};
        const uint32_t pc[3] = {(uint32_t)n_in, (uint32_t)n_out, (uint32_t)ncols};
        vk::kernel_launch_sized(k, (int)((n_out + 3) / 4), bufs, sizes, 4, pc);
    } else if (ggml_type == 23) {  // IQ4_XS: Fmt<23> = {qk 256, ipb 8, step 4}, no grid table
        static const vk::Kernel k = vk::kernel_load(
            "native_mmvq_iq4_xs", mmvq_iq4_xs_spv, mmvq_iq4_xs_spv_len,
            /*storage buffers*/ 3, /*push bytes*/ 12, /*workgroup*/ THREADS,
            /*force 32-wide subgroups*/ true);
        const uint64_t w_sz = uint64_t(n_out) * nb * IQ4XS_BLOCK_BYTES;
        const uint64_t bufs[3]  = {strata_vk_compat::off(weights), strata_vk_compat::off(x_q8_1),
                                   strata_vk_compat::off(y)};
        const uint64_t sizes[3] = {w_sz, x_sz, y_sz};
        const uint32_t pc[3] = {(uint32_t)n_in, (uint32_t)n_out, (uint32_t)ncols};
        vk::kernel_launch_sized(k, (int)((n_out + 3) / 4), bufs, sizes, 3, pc);
    } else {  // ggml_type == 18 (IQ3_XXS): Fmt<18> = {qk 256, ipb 8, step 2} - same grid/launch as 16
        static const vk::Kernel k = vk::kernel_load(
            "native_mmvq_iq3_xxs", mmvq_iq3_xxs_spv, mmvq_iq3_xxs_spv_len,
            /*storage buffers*/ 4, /*push bytes*/ 12, /*workgroup*/ THREADS,
            /*force 32-wide subgroups*/ true);
        const uint64_t w_sz = uint64_t(n_out) * nb * IQ3_BLOCK_BYTES;
        const uint64_t bufs[4]  = {strata_vk_compat::off(weights), strata_vk_compat::off(x_q8_1),
                                   strata_vk_compat::off(y), grid32_buffer()};
        const uint64_t sizes[4] = {w_sz, x_sz, y_sz, sizeof(kIq3xxsGrid)};
        const uint32_t pc[3] = {(uint32_t)n_in, (uint32_t)n_out, (uint32_t)ncols};
        vk::kernel_launch_sized(k, (int)((n_out + 3) / 4), bufs, sizes, 4, pc);
    }
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(e));
}

// ---- native_mmvq_slots (Task 2.8) ----------------------------------------------
// The CUDA wrapper (native_mmvq.cu:3178-3190): false, with nothing launched, when slots_kind()
// is 0 - i.e. for every format this port has no kernel for (the unported mmvq types and, at the
// engine level, the iq family beyond 16). Types 16/13/14 dispatch to the slots kernels, which are
// per-slot transcriptions of the single-call kernels' exact partition, so matrix i's y[i] is
// BITWISE native_mmvq(type, w[i], x[i], y[i], n_in, n_out, 1) (the parity test asserts both
// against the host refs and against the single-launch outputs). native_mmvq_ids (the sibling
// contract, :3194) is NOT ported: it stays trap-by-number in stubs_gen.
bool native_mmvq_slots(int ggml_type, const MmvqSlots& slots, int count, int n_in, int n_out, void* stream) {
    if (ggml_type != 16 && ggml_type != 13 && ggml_type != 14) return false;   // CUDA's slots_kind == 0 leg
    if (n_out <= 0 || count < 1 || count > MAX_NCOLS)
        throw std::invalid_argument("native MMVQ requires n_out > 0 and 1 to 8 matrices");
    validate_shape(n_in, 1, IQ_BLOCK_ELEMS);        // n_in % 256 (per-matrix ncols == 1)
    for (int i = 0; i < count; ++i) {
        validate_pointer(slots.w[i]);
        validate_pointer(slots.x[i]);
        validate_pointer(slots.y[i]);
    }
    validate_stream(stream);
    if (std::getenv("STRATA_VK_TRACE_MMVQ_SLOTS"))
        std::fprintf(stderr, "mmvq_slots t %d count %d n_in %d n_out %d\n", ggml_type, count, n_in, n_out);
    const uint64_t nb = uint64_t(n_in) / IQ_BLOCK_ELEMS;
    const uint64_t x_sz = nb * 8 * Q8_BLOCK_BYTES;  // one Q8_1 column per slot
    const uint64_t y_sz = uint64_t(n_out) * 4;
    // 24 bindings (w,x,y per slot; +grid for iq2_xxs): unused slots alias slot 0's buffers with a
    // 4-byte span - the flattened grid runs exactly count slots' arms, so they are never touched.
    uint64_t bufs[25], sizes[25];
    const int n_buf = ggml_type == 16 ? 25 : 24;
    const uint64_t rb = uint64_t(n_out) * nb *
                        (ggml_type == 16 ? IQ_BLOCK_BYTES : ggml_type == 13 ? Q5K_BLOCK_BYTES : Q6K_BLOCK_BYTES);
    for (int i = 0; i < 8; ++i) {
        const int j = i < count ? i : 0;
        bufs[i]      = strata_vk_compat::off(slots.w[j]);
        bufs[8 + i]  = strata_vk_compat::off(slots.x[j]);
        bufs[16 + i] = strata_vk_compat::off(slots.y[j]);
        sizes[i]      = (i < count) ? rb : 4;
        sizes[8 + i]  = (i < count) ? x_sz : 4;
        sizes[16 + i] = (i < count) ? y_sz : 4;
    }
    if (ggml_type == 16) {
        static const vk::Kernel k = vk::kernel_load(
            "mmvq_slots_iq2_xxs", mmvq_slots_iq2_xxs_spv, mmvq_slots_iq2_xxs_spv_len,
            /*storage buffers*/ 25, /*push bytes*/ 16, /*workgroup*/ THREADS, true);
        bufs[24] = grid_buffer(); sizes[24] = sizeof(kIq2xxsGrid);
        const uint32_t gx = uint32_t((n_out + 3) / 4);   // mmvq_kernel's grid x, iq_kernels.cu:3506
        const uint32_t pc[4] = {(uint32_t)count, (uint32_t)n_in, (uint32_t)n_out, gx};
        vk::kernel_launch_sized(k, int(gx) * count, bufs, sizes, 25, pc);
    } else {
        // launch_slots (native_mmvq.cu:1758-1768): SmallK n_in/DIV < BPI -> ROWS=2, else ROWS=1
        const bool q5 = ggml_type == 13;
        const uint32_t rows = (nb < (q5 ? 8u : 4u)) ? 2u : 1u;
        const uint32_t gx = uint32_t((n_out + rows - 1) / rows);
        const uint32_t pc[5] = {(uint32_t)count, (uint32_t)n_in, (uint32_t)n_out, gx, rows};
        if (q5) {
            static const vk::Kernel k = vk::kernel_load(
                "mmvq_slots_q5_k", mmvq_slots_q5_k_spv, mmvq_slots_q5_k_spv_len,
                /*storage buffers*/ 24, /*push bytes*/ 20, /*workgroup*/ THREADS, true);
            vk::kernel_launch_sized(k, int(gx) * count, bufs, sizes, 24, pc);
        } else {
            static const vk::Kernel k = vk::kernel_load(
                "mmvq_slots_q6_k", mmvq_slots_q6_k_spv, mmvq_slots_q6_k_spv_len,
                /*storage buffers*/ 24, /*push bytes*/ 20, /*workgroup*/ THREADS, true);
            vk::kernel_launch_sized(k, int(gx) * count, bufs, sizes, 24, pc);
        }
    }
    return true;
}

}  // namespace strata::kernels
