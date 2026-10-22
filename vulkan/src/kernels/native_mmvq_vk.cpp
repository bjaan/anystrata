// vulkan/src/kernels/native_mmvq_vk.cpp - the Vulkan launcher for
// include/strata/kernels/native_mmvq.hpp's native_mmvq, ported for ggml_type 16 (IQ2_XXS) only —
// the Qwen3.6-35B-A3B UD-IQ2_XXS experts (gate/up and down). Every other type calls missing() with
// a name carrying its type id, so a run never computes with a wrong decoder (the CUDA switch,
// src/kernels/cuda/native_mmvq.cu:3110ff case 16 -> iq_mmvq -> launch_mmvq<16>).
//
// The CUDA wrapper's validation is kept whole (native_mmvq.cu:2026-2041): n_in % block_elems (256
// for iq types), 1 <= ncols <= 8, non-null 4-aligned arena pointers, non-null stream.
//
// launch_mmvq<16> geometry (iq_kernels.cu:2594-2608): grid = ceil(n_out/4), block = (32,4) = 128
// threads; with multi_exact on (the default) the Split/multi kernel answers, and iq_kernels.cu:699
// states each of its columns is BITWISE equal to the per-column mmvq_kernel — which is the kernel
// this launcher runs (mmvq_iq2_xxs.comp).
//
// iq2xxs_grid: the 256 uint64 table from ggml-common.h:560. The C/C++ GGML_TABLE arm makes it
// static-per-TU, so no symbol links here; the values are extracted verbatim to
// iq2xxs_grid_vk.h (tools note: regenerated from the same pinned llama.cpp checkout the engine
// links). One arena upload at first use, bound as the kernel's 4th storage buffer.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"
#include "iq2xxs_grid_vk.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::vk_engine { void missing(const char* name); }

namespace strata::kernels {
namespace {
#include "native_mmvq.spv.h"   // embedded by tools/embed_spv.py (mmvq_iq2_xxs.comp)

constexpr int IQ_BLOCK_ELEMS = 256;
using vk::kIq2xxsGrid;   // the grid table header (namespace strata::vk)
constexpr int IQ_BLOCK_BYTES = 66;   // block_iq2_xxs
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
}  // namespace

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    if (ggml_type != 16) {
        // the CUDA wrapper would throw/exit here too — name the type instead (never compute with a
        // wrong decoder); shape is not validated for a type this port does not run.
        const std::string name = "strata::kernels::native_mmvq type " + std::to_string(ggml_type);
        strata::vk_engine::missing(name.c_str());
        return;
    }
    if (std::getenv("STRATA_VK_TRACE_MMVQ16"))
        std::fprintf(stderr, "mmvq16 n_in %d n_out %d ncols %d\n", n_in, n_out, ncols);
    validate_shape(n_in, ncols, IQ_BLOCK_ELEMS);
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    static const vk::Kernel k = vk::kernel_load(
        "native_mmvq_iq2_xxs", native_mmvq_spv, native_mmvq_spv_len,
        /*storage buffers*/ 4, /*push bytes*/ 12, /*workgroup*/ THREADS,
        /*force 32-wide subgroups*/ true);
    const uint64_t nb   = uint64_t(n_in) / IQ_BLOCK_ELEMS;
    const uint64_t w_sz = uint64_t(n_out) * nb * IQ_BLOCK_BYTES;
    const uint64_t x_sz = uint64_t(ncols) * nb * 8 * Q8_BLOCK_BYTES;   // n_in/32 blocks a column
    const uint64_t y_sz = uint64_t(ncols) * uint64_t(n_out) * 4;
    const uint64_t bufs[4]  = {strata_vk_compat::off(weights), strata_vk_compat::off(x_q8_1),
                               strata_vk_compat::off(y), grid_buffer()};
    const uint64_t sizes[4] = {w_sz, x_sz, y_sz, sizeof(kIq2xxsGrid)};
    const uint32_t pc[3] = {(uint32_t)n_in, (uint32_t)n_out, (uint32_t)ncols};
    vk::kernel_launch_sized(k, (int)((n_out + 3) / 4), bufs, sizes, 4, pc);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(e));
}

}  // namespace strata::kernels
