// vulkan/src/kernels/native_quantize_q8_1_vk.cpp - the Vulkan launcher for
// include/strata/kernels/native_mmvq.hpp's native_quantize_q8_1 (the CUDA body is
// src/kernels/cuda/native_mmvq.cu:169 native_quantize_q8_1_kernel + the wrapper at :2010).
//
// The wrapper's contract kept whole: the same validate_shape / validate_pointer / validate_stream
// throws (the engine tests them: native_mmvq.cu:2026-2041 - n_in % 32, 1 <= ncols <= 8, non-null
// 4-aligned pointers, non-null stream), and the same flattening - columns are contiguous and n_in is
// a multiple of 32, so ncols columns quantize as ONE vector of n_in*ncols elements and every
// 32-element Q8_1 block stays inside one column. QUANT_THREADS = 256 (native_mmvq.cu:63).
// PDL (pdl_trigger/pdl_wait) is host-answered false by the stubs (pdl_supported), so the launch is
// plain here; the .comp carries no PDL instructions.
//
// force_subgroup32 = true: the CUDA body assumes warp 32 for the warp_max/warp_sum chains, and
// n_total % 32 == 0 guarantees whole warps return; at a forced 32-wide subgroup every shuffle
// partner is always live. (At 64-wide the chains with offsets <= 16 stay inside each aligned
// 32-lane half, so the sums are still per-warp - probe B proved the mechanism; forcing is the
// belt-and-braces choice and costs nothing on gfx1013, whose native CU wave is 32 lanes anyway.)
#include "strata/kernels/native_mmvq.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
#include "native_quantize_q8_1.spv.h"   // embedded by tools/embed_spv.py
constexpr int Q8K = 32;
constexpr int QUANT_THREADS = 256;
constexpr int MAX_NCOLS = 8;

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
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
}  // namespace

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    static const vk::Kernel k = vk::kernel_load(
        "native_quantize_q8_1", native_quantize_q8_1_spv, native_quantize_q8_1_spv_len,
        /*storage buffers*/ 2, /*push bytes*/ 4, /*workgroup*/ QUANT_THREADS,
        /*force 32-wide subgroups*/ true);
    const int n_total = n_in * ncols;
    const uint64_t bufs[2]  = {strata_vk_compat::off(x), strata_vk_compat::off(x_q8_1)};
    const uint64_t sizes[2] = {(uint64_t)n_total * 4, (uint64_t)(n_total / Q8K) * 36};  // Q81Block = 36 B
    const uint32_t pc[1] = {(uint32_t)n_total};
    vk::kernel_launch_sized(k, (int)((n_total + QUANT_THREADS - 1) / QUANT_THREADS), bufs, sizes, 2, pc);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(e));
}

}  // namespace strata::kernels
