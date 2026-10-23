// vulkan/src/poly/kernels/gdn_vk.cpp - the Vulkan launcher for strata::poly::gdn_step
// (include/strata/poly/kernels.hpp:254; the CUDA bodies are src/poly/kernels.cu:1988 gdn_norm_kernel,
// :2002 gdn_gates_kernel, :2017 gdn_state_kernel, :2058 gdn_state_reg_kernel<PER>, :2118
// gdn_out_kernel; the wrapper at :2699).
//
// The wrapper's shape is transcribed EXACTLY: gdn_norm <<<rows*k_heads, dim>>> (one workgroup size
// per dim: the block_sum part[] walk counts warps off blockDim), gdn_gates <<<grid(gates,256),256>>>,
// then switch (dim / WARP): 1/2/4/8 -> gdn_state_reg_kernel<PER>, default -> gdn_state_kernel, with
// blocks = n_head * ceil(dim/columns), threads = columns * WARP, columns = min(dim, 1024/WARP) = 32
// for every legal dim (32..256 in 32s) -> 1024 threads, one warp a column; finally
// gdn_out <<<rows*n_head, dim>>>.
//
// The state walk never splits into a rows/state pair (unlike conv4): every token feeds the next,
// so one launch walks the tokens; the reg variants hold each warp's column in registers across the
// walk (same sums, per the CUDA comment), the plain variant reads it from memory each token.
// A null snap is bound as out with snaps=0 so the snapshot arm never fires.
#include "strata/poly/kernels.hpp"
#include "strata/vulkan_runtime.hpp"
#include "strata/vulkan_compat/strata_vk.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {
#include "gdn_norm_32.spv.h"
#include "gdn_norm_64.spv.h"
#include "gdn_norm_96.spv.h"
#include "gdn_norm_128.spv.h"
#include "gdn_norm_160.spv.h"
#include "gdn_norm_192.spv.h"
#include "gdn_norm_224.spv.h"
#include "gdn_norm_256.spv.h"
#include "gdn_gates.spv.h"
#include "gdn_state_reg_1.spv.h"
#include "gdn_state_reg_2.spv.h"
#include "gdn_state_reg_4.spv.h"
#include "gdn_state_reg_8.spv.h"
#include "gdn_state_plain.spv.h"
#include "gdn_out_32.spv.h"
#include "gdn_out_64.spv.h"
#include "gdn_out_96.spv.h"
#include "gdn_out_128.spv.h"
#include "gdn_out_160.spv.h"
#include "gdn_out_192.spv.h"
#include "gdn_out_224.spv.h"
#include "gdn_out_256.spv.h"

// (buffers, push_constant_bytes, workgroup_size, force_subgroup32=false: every shuffle offset in
// these kernels is <= 16 and a 32-lane aligned block's xor-16 partner is in the same warp, the
// rms_norm reasoning.)
const vk::Kernel& pl_gates() {
    static const vk::Kernel k = vk::kernel_load("gdn_gates", gdn_gates_spv, gdn_gates_spv_len, 4, 8, 256, false);
    return k;
}
const vk::Kernel& pl_norm(int dim) {
    switch (dim) {
        case 32:  { static const vk::Kernel k = vk::kernel_load("gdn_norm_32",  gdn_norm_32_spv,  gdn_norm_32_spv_len,  1, 16, 32,  false); return k; }
        case 64:  { static const vk::Kernel k = vk::kernel_load("gdn_norm_64",  gdn_norm_64_spv,  gdn_norm_64_spv_len,  1, 16, 64,  false); return k; }
        case 96:  { static const vk::Kernel k = vk::kernel_load("gdn_norm_96",  gdn_norm_96_spv,  gdn_norm_96_spv_len,  1, 16, 96,  false); return k; }
        case 128: { static const vk::Kernel k = vk::kernel_load("gdn_norm_128", gdn_norm_128_spv, gdn_norm_128_spv_len, 1, 16, 128, false); return k; }
        case 160: { static const vk::Kernel k = vk::kernel_load("gdn_norm_160", gdn_norm_160_spv, gdn_norm_160_spv_len, 1, 16, 160, false); return k; }
        case 192: { static const vk::Kernel k = vk::kernel_load("gdn_norm_192", gdn_norm_192_spv, gdn_norm_192_spv_len, 1, 16, 192, false); return k; }
        case 224: { static const vk::Kernel k = vk::kernel_load("gdn_norm_224", gdn_norm_224_spv, gdn_norm_224_spv_len, 1, 16, 224, false); return k; }
        case 256: { static const vk::Kernel k = vk::kernel_load("gdn_norm_256", gdn_norm_256_spv, gdn_norm_256_spv_len, 1, 16, 256, false); return k; }
        default: throw std::runtime_error("gdn_step: norm workgroup size must be 32 to 256, in 32s");
    }
}
const vk::Kernel& pl_out(int dim) {
    switch (dim) {
        case 32:  { static const vk::Kernel k = vk::kernel_load("gdn_out_32",  gdn_out_32_spv,  gdn_out_32_spv_len,  3, 16, 32,  false); return k; }
        case 64:  { static const vk::Kernel k = vk::kernel_load("gdn_out_64",  gdn_out_64_spv,  gdn_out_64_spv_len,  3, 16, 64,  false); return k; }
        case 96:  { static const vk::Kernel k = vk::kernel_load("gdn_out_96",  gdn_out_96_spv,  gdn_out_96_spv_len,  3, 16, 96,  false); return k; }
        case 128: { static const vk::Kernel k = vk::kernel_load("gdn_out_128", gdn_out_128_spv, gdn_out_128_spv_len, 3, 16, 128, false); return k; }
        case 160: { static const vk::Kernel k = vk::kernel_load("gdn_out_160", gdn_out_160_spv, gdn_out_160_spv_len, 3, 16, 160, false); return k; }
        case 192: { static const vk::Kernel k = vk::kernel_load("gdn_out_192", gdn_out_192_spv, gdn_out_192_spv_len, 3, 16, 192, false); return k; }
        case 224: { static const vk::Kernel k = vk::kernel_load("gdn_out_224", gdn_out_224_spv, gdn_out_224_spv_len, 3, 16, 224, false); return k; }
        case 256: { static const vk::Kernel k = vk::kernel_load("gdn_out_256", gdn_out_256_spv, gdn_out_256_spv_len, 3, 16, 256, false); return k; }
        default: throw std::runtime_error("gdn_step: out workgroup size must be 32 to 256, in 32s");
    }
}
// The wrapper's switch (dim / WARP): reg<1>/<2>/<4>/<8>, the plain kernel for every other legal dim.
const vk::Kernel& pl_state(int dim, int* per) {
    switch (dim / 32) {
        case 1: *per = 1; { static const vk::Kernel k = vk::kernel_load("gdn_state_reg_1", gdn_state_reg_1_spv, gdn_state_reg_1_spv_len, 6, 32, 1024, false); return k; }
        case 2: *per = 2; { static const vk::Kernel k = vk::kernel_load("gdn_state_reg_2", gdn_state_reg_2_spv, gdn_state_reg_2_spv_len, 6, 32, 1024, false); return k; }
        case 4: *per = 4; { static const vk::Kernel k = vk::kernel_load("gdn_state_reg_4", gdn_state_reg_4_spv, gdn_state_reg_4_spv_len, 6, 32, 1024, false); return k; }
        case 8: *per = 8; { static const vk::Kernel k = vk::kernel_load("gdn_state_reg_8", gdn_state_reg_8_spv, gdn_state_reg_8_spv_len, 6, 32, 1024, false); return k; }
        default: *per = 0; { static const vk::Kernel k = vk::kernel_load("gdn_state_plain", gdn_state_plain_spv, gdn_state_plain_spv_len, 6, 32, 1024, false); return k; }
    }
}
}  // namespace

void gdn_step(float* qkv, float* alpha, float* beta, const float* dt, const float* a, float* s, const float* z,
              const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows, float norm_eps,
              void* stream, float* snap, size_t stride, unsigned snaps) {
    if (dim % 32 != 0 || dim > 256)
        throw std::invalid_argument("gdn_step: a head of 32 to 256 values, in 32s");
    if (rows <= 0) return;
    if (n_head <= 0 || k_heads <= 0) throw std::invalid_argument("gdn_step: head counts must be positive");
    const size_t row_n = (2 * size_t(k_heads) + n_head) * dim;
    const size_t gates = size_t(rows) * n_head, d_inner = size_t(dim) * n_head;
    const size_t keys = size_t(dim) * k_heads;
    const uint64_t qo = strata_vk_compat::off(qkv), ao = strata_vk_compat::off(alpha),
                   bo = strata_vk_compat::off(beta), dto = strata_vk_compat::off(dt),
                   ao2 = strata_vk_compat::off(a), so = strata_vk_compat::off(s),
                   zo = strata_vk_compat::off(z), ono = strata_vk_compat::off(o_norm),
                   oo = strata_vk_compat::off(out),
                   no = strata_vk_compat::off(snap ? snap : out);   // bound-always; snaps gates use
    uint32_t eps_bits; std::memcpy(&eps_bits, &norm_eps, 4);

    // 1. gdn_norm_kernel: <<<rows * k_heads, dim>>>
    {
        const uint64_t bufs[1] = {qo};
        const uint64_t sizes[1] = {row_n * (size_t)rows * 4};
        const uint32_t pc[4] = {(uint32_t)dim, (uint32_t)k_heads, (uint32_t)row_n, eps_bits};
        if ((long long)rows * k_heads > 0x7FFFFFFFLL) throw std::runtime_error("gdn_step: norm grid overflows");
        vk::kernel_launch_sized(pl_norm(dim), (int)(rows * k_heads), bufs, sizes, 1, pc);
    }
    // 2. gdn_gates_kernel: <<<grid(gates,256), 256>>>
    {
        const uint64_t bufs[4] = {ao, bo, dto, ao2};
        const uint64_t sizes[4] = {gates * 4, gates * 4, (uint64_t)n_head * 4, (uint64_t)n_head * 4};
        const uint32_t pc[2] = {(uint32_t)n_head, (uint32_t)gates};
        if (gates > 0x7FFFFFFFLL) throw std::runtime_error("gdn_step: gates grid overflows");
        vk::kernel_launch_sized(pl_gates(), int((gates + 255) / 256), bufs, sizes, 4, pc);
    }
    // 3. state walk: columns = min(dim, 32), blocks = n_head * ceil(dim/columns), threads = columns*32
    if (stride >> 32) throw std::runtime_error("gdn_step: snapshot stride exceeds uint32");
    {
        const int columns = dim < 32 ? dim : 32;   // 1024/WARP = 32
        int per = 0;
        const vk::Kernel& k = pl_state(dim, &per);
        const size_t blocks = (size_t)n_head * ((dim + columns - 1) / columns);
        const uint64_t bufs[6] = {qo, ao, bo, so, oo, no};
        // The snap arm writes each snapshot token's STATE (n_head*dim*dim floats) at snap +
        // j*stride, j < popcount(snaps): the touched extent is (popcount-1)*stride + layer, NOT
        // popcount*stride - declaring the whole stride tail (this launch's snap start already
        // sits layer-offset deep in the buffer) pushed the span past the arena piece it lives in
        // (Task 2.19: "kernel argument 5 of gdn_state_reg_4 crosses an arena piece" once the
        // model reached snapshot speculation).
        const size_t snap_layer = (uint64_t)n_head * dim * dim;
        const size_t snap_extent = (snap && stride && snaps)
                                       ? ((size_t)__builtin_popcount(snaps) - 1u) * stride + snap_layer
                                       : (size_t)0;
        const uint64_t sizes[6] = {row_n * (size_t)rows * 4, gates * 4, gates * 4,
                                   snap_layer * 4, (uint64_t)rows * d_inner * 4,
                                   snap_extent ? snap_extent * 4 : (uint64_t)rows * d_inner * 4};
        const uint32_t pc[8] = {(uint32_t)dim, (uint32_t)n_head, (uint32_t)k_heads, (uint32_t)rows,
                                (uint32_t)columns, (uint32_t)(stride & 0xFFFFFFFFu), (uint32_t)(stride >> 32), snaps};
        if (blocks > 0x7FFFFFFFLL) throw std::runtime_error("gdn_step: state grid overflows");
        vk::kernel_launch_sized(k, int(blocks), bufs, sizes, 6, pc);
        (void)keys; (void)per;
    }
    // 4. gdn_out_kernel: <<<rows * n_head, dim>>>
    {
        const uint64_t bufs[3] = {oo, zo, ono};
        const uint64_t sizes[3] = {(uint64_t)rows * d_inner * 4, (uint64_t)rows * d_inner * 4, (uint64_t)dim * 4};
        const uint32_t pc[4] = {(uint32_t)dim, (uint32_t)d_inner, eps_bits, 0u};
        if (gates > 0x7FFFFFFFLL) throw std::runtime_error("gdn_step: out grid overflows");
        vk::kernel_launch_sized(pl_out(dim), int(gates), bufs, sizes, 3, pc);
    }
}

}  // namespace strata::poly
